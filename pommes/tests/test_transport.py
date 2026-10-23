import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal as assert_equal_epsilon
from numpy.testing import assert_array_equal as assert_equal

from pommes.model.build_model import build_model


@pytest.fixture()
def parameters_transport(parameters_dispatch_invest, transport):
    p = xr.merge([parameters_dispatch_invest, transport])
    # Hurdle costs are weighted like other variable costs (x 8760 / modelled hours): scaled here so that the
    # 0.01 tie-breaker of the fixture keeps its weight per MWh of a single modelled hour in the objectives below
    p["transport_hurdle_costs"] = p.transport_hurdle_costs / 8760
    return p


@pytest.fixture()
def parameters_transport_single_horizon(parameters_transport):
    p = parameters_transport
    p = p.sel(
        conversion_tech=["wind_onshore"],
        transport_tech=["power_line"],
        link=["link_1"],
        hour=[0],
        resource=["electricity"],
        year_dec=[2040, 2050],
        year_inv=[2030],
        year_op=[2030],
    ).copy(deep=True)

    return p


@pytest.fixture()
def parameters_transport_multi_horizon(parameters_transport):
    p = parameters_transport.sel(
        conversion_tech=["wind_onshore"],
        link=["link_1"],
        hour=[2],
        resource=["electricity"],
        transport_tech=["power_line"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030],
    ).copy(deep=True)

    p["conversion_annuity_cost"] = 0
    p["conversion_power_capacity_investment_min"] = xr.DataArray(
        [[1000, 0], [0, 1000]], coords=[p.year_inv, p.area]
    )
    p["conversion_power_capacity_investment_max"] = xr.DataArray(
        [[1000, 0], [0, 1000]], coords=[p.year_inv, p.area]
    )
    return p


@pytest.fixture()
def parameters_transport_multi_horizon_3_zones(parameters_transport):
    p = parameters_transport.sel(
        conversion_tech=["wind_onshore"],
        hour=[2],
        resource=["electricity"],
        transport_tech=["power_line"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030],
    ).copy(deep=True)

    p = p.sel(area="area_1", drop=True)
    areas = np.array(["area_1", "area_2", "area_3"], dtype=str)
    coords_area3 = xr.DataArray(areas, coords=dict(area=areas))
    p = p.assign_coords(area=coords_area3)

    p["conversion_annuity_cost"] = 0
    p["conversion_power_capacity_investment_min"] = xr.DataArray(
        [[1000, 0, np.nan], [0, 1000, np.nan]], coords=[p.year_inv, p.area]
    )
    p["conversion_power_capacity_investment_max"] = xr.DataArray(
        [[1000, 0, np.nan], [0, 1000, np.nan]], coords=[p.year_inv, p.area]
    )
    p["transport_end_of_life"] = (
        p["transport_end_of_life"].copy().expand_dims(link=p.link)
    )
    return p


def test_transport_no_transport(parameters_transport_single_horizon):
    p = parameters_transport_single_horizon

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True).sum("year_dec")

    assert_equal(
        s.operation_conversion_power_capacity.to_numpy(), np.array([20, 20])
    )
    assert_equal(s.operation_conversion_power.to_numpy(), np.array([10, 10]))
    assert_equal(s.operation_transport_power_capacity.to_numpy(), np.array(0))
    assert_equal(s.operation_transport_power.to_numpy(), np.array(0))

    assert model.objective.value == 400.0


def test_transport_one_hour(parameters_transport_single_horizon):
    p = parameters_transport_single_horizon
    p = p.copy(deep=True)

    p.update(
        dict(
            conversion_power_capacity_investment_max=p.conversion_power_capacity_investment_max.expand_dims(
                area=p.area,
                conversion_tech=p.conversion_tech,
            ).copy(),
        )
    )

    p.conversion_power_capacity_investment_max.loc[
        dict(area="area_2", conversion_tech="wind_onshore")
    ] = 0

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True).sum("year_dec")

    assert_equal(
        s.operation_conversion_power_capacity.to_numpy(), np.array([40, 0])
    )
    assert_equal(s.operation_conversion_power.to_numpy(), np.array([20, 0]))
    assert_equal(s.operation_transport_power_capacity.to_numpy(), np.array(10))
    assert_equal(s.operation_transport_power.to_numpy(), np.array(10))

    assert model.objective.value == 550.0 + 10 * 0.01  # hurdle costs


def test_transport_no_transport_min_capacity(
    parameters_transport_single_horizon,
):
    p = parameters_transport_single_horizon.copy(deep=True)

    p["transport_power_capacity_investment_min"] = 5

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True).sum("year_dec")

    assert_equal(
        s.operation_conversion_power_capacity.to_numpy(), np.array([20, 20])
    )
    assert_equal(s.operation_conversion_power.to_numpy(), np.array([10, 10]))
    assert_equal(s.operation_transport_power_capacity.to_numpy(), np.array(5))
    assert_equal(s.operation_transport_power.to_numpy(), np.array(0))

    assert model.objective.value == 475.0


def test_transport_multi_hour(parameters_transport):
    p = parameters_transport
    p = p.sel(
        conversion_tech=["wind_onshore"],
        transport_tech=["power_line"],
        hour=[0, 1, 2],
        resource=["electricity"],
        year_dec=[2040, 2050],
        year_inv=[2030],
        year_op=[2030],
    )

    p = p.copy(deep=True)

    p["transport_fixed_cost"] = 0

    p.update(
        dict(
            conversion_availability=p.conversion_availability.expand_dims(
                area=p.area
            ).copy()
        )
    )

    p.conversion_availability.loc[dict(conversion_tech="wind_onshore")] = [
        [0.6, 0.3, 0.1],
        [0.4, 0.7, 0.9],
    ]

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True).sum("year_dec")

    assert_equal_epsilon(
        s.operation_conversion_power_capacity.to_numpy(),
        np.array([20, 20]),
        decimal=12,
    )
    assert_equal(
        s.operation_conversion_power.to_numpy(),
        np.array([[12, 6, 2], [8, 14, 18]]),
    )
    assert_equal_epsilon(
        s.operation_transport_power_capacity.to_numpy(), np.array([2, 8]), 12
    )
    assert_equal_epsilon(
        s.operation_transport_power.to_numpy(),
        np.array([[2, 0, 0], [0, 4, 8]]),
        decimal=12,
    )

    # 3 modelled hours stand for the year: each MWh carried counts 8760 / 3 times at 0.01 / 8760
    assert model.objective.value == pytest.approx(450.0 + 0.01 * 14 / 3)


def test_transport_multi_horizon(parameters_transport_multi_horizon):
    p = parameters_transport_multi_horizon

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_transport_costs.to_numpy(), np.array([[25, 25], [25, 25]])
    )

    np.testing.assert_array_equal(
        s.planning_transport_power_capacity.to_numpy(),
        np.array([[np.nan, np.nan], [10, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_transport_power_capacity.to_numpy(), np.array([10, 10])
    )

    np.testing.assert_array_equal(
        s.operation_transport_power.to_numpy(),
        np.array([10, 0]),
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.to_numpy(),
        np.array([[80, 90], [0, 90]]),
    )

    assert model.objective.value == 5 * 10 * 2 + 10 * 10 * 2 + 0.01 * 10


def test_transport_early_decommissioning(parameters_transport_multi_horizon):
    p = parameters_transport_multi_horizon.copy(deep=True)

    p["transport_early_decommissioning"] = True

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_transport_costs.to_numpy(), np.array([[25, 25], [25, 25]])
    )

    np.testing.assert_array_equal(
        s.planning_transport_power_capacity.to_numpy(),
        np.array([[10, np.nan], [0, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_transport_power_capacity.to_numpy(), np.array([10, 0])
    )

    np.testing.assert_array_equal(
        s.operation_transport_power.to_numpy(), np.array([10, 0])
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.to_numpy(),
        np.array([[80, 90], [0, 90]]),
    )

    assert model.objective.value == 5 * 10 * 2 + 10 * 10 + 0.01 * 10


def test_transport_annuity_perfect_foresight(
    parameters_transport_multi_horizon,
):
    p = parameters_transport_multi_horizon.copy(deep=True)

    p["transport_early_decommissioning"] = True
    p["transport_annuity_perfect_foresight"] = True

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_transport_costs.to_numpy(), np.array([[50, 0], [50, 0]])
    )

    np.testing.assert_array_equal(
        s.planning_transport_power_capacity.to_numpy(),
        np.array([[10, np.nan], [0, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_transport_power_capacity.to_numpy(), np.array([10, 0])
    )

    np.testing.assert_array_equal(
        s.operation_transport_power.to_numpy(), np.array([10, 0])
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.to_numpy(),
        np.array([[80, 90], [0, 90]]),
    )

    assert model.objective.value == 5 * 10 * 2 + 10 * 10 + 0.01 * 10


def test_transport_three_zones(parameters_transport_multi_horizon_3_zones):
    p = parameters_transport_multi_horizon_3_zones

    model = build_model(p)
    model.solve(solver_name="highs")
    d = model.dual

    np.testing.assert_array_equal(
        d.planning_transport_costs_def.to_numpy(),
        np.array([[[-1, -1]], [[-1, -1]], [[0, 0]]]),
    )

    assert model.objective.value == 300.1


def test_transport_hurdle_weighted_like_variable_costs(parameters_transport_single_horizon):
    """A hurdle cost counts once per MWh of the represented year, discounted, in the transport costs."""
    p = parameters_transport_single_horizon.copy(deep=True)
    p.update(
        dict(
            conversion_power_capacity_investment_max=p.conversion_power_capacity_investment_max.expand_dims(
                area=p.area,
                conversion_tech=p.conversion_tech,
            ).copy(),
        )
    )
    p.conversion_power_capacity_investment_max.loc[
        dict(area="area_2", conversion_tech="wind_onshore")
    ] = 0

    objective = {}
    for hurdle in (0.0, 2.0):
        p["transport_hurdle_costs"] = hurdle
        model = build_model(p)
        model.solve(solver_name="highs")
        s = model.solution.squeeze(drop=True)
        objective[hurdle] = model.objective.value
        flow = float(s.operation_transport_power)
    normalization = float(p.operation_year_duration.squeeze()) / float(p.time_step_duration.sum())
    # one modelled hour stands for 8760: 10 MW carried cost 2 EUR/MWh x 10 MWh x 8760
    assert flow == pytest.approx(10)
    assert objective[2.0] - objective[0.0] == pytest.approx(2.0 * flow * normalization)
    # and the cost sits in the areas' transport costs (half at each end), hence in annualised_totex
    assert float(s.operation_transport_costs.sum()) == pytest.approx(
        float((p.transport_fixed_cost * s.operation_transport_power_capacity).sum()) + 2.0 * flow * normalization
    )


def _one_way_three_hours(parameters_transport, wind_area_1):
    """Area 2 cannot generate; area 1's wind availability over hours 0, 1, 2 is `wind_area_1`."""
    p = parameters_transport.sel(
        conversion_tech=["wind_onshore"],
        transport_tech=["power_line"],
        link=["link_1"],
        hour=[0, 1, 2],
        resource=["electricity"],
        year_dec=[2040, 2050],
        year_inv=[2030],
        year_op=[2030],
    ).copy(deep=True)
    p["conversion_power_capacity_investment_max"] = p.conversion_power_capacity_investment_max.expand_dims(
        area=p.area, conversion_tech=p.conversion_tech).copy()
    p.conversion_power_capacity_investment_max.loc[dict(area="area_2", conversion_tech="wind_onshore")] = 0
    p["conversion_availability"] = p.conversion_availability.expand_dims(area=p.area).copy()
    p.conversion_availability.loc[dict(area="area_1", conversion_tech="wind_onshore")] = wind_area_1
    return p


def test_transport_delay_shifts_delivery(parameters_transport):
    """What arrives at hour h left at hour h - delay, cyclic over the year."""
    p = _one_way_three_hours(parameters_transport, [1.0, 1.0, 1.0])
    p["transport_delay"] = 1.0

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True).sum("year_dec")

    injected = s.operation_transport_power_in.transpose("hour").to_numpy()
    delivered = s.operation_transport_power.transpose("hour").to_numpy()
    assert_equal_epsilon(delivered, np.roll(injected, 1))
    assert delivered.sum() == pytest.approx(injected.sum())


def test_transport_delay_moves_the_shortage(parameters_transport):
    """Area 1 cannot produce at hour 1. Without delay, area 2 is short at hour 1; with a
    one-hour delay, hour 1 is served by what left at hour 0 and the shortage moves to hour 2."""
    shortage = {}
    for label, delay in (("none", 0.0), ("1h", 1.0)):
        p = _one_way_three_hours(parameters_transport, [1.0, 0.0, 1.0])
        p["transport_delay"] = delay
        p["load_shedding_cost"] = xr.DataArray(1000.0)
        p["load_shedding_max_capacity"] = xr.DataArray(np.nan)
        model = build_model(p)
        model.solve(solver_name="highs")
        s = model.solution.squeeze(drop=True).sum("year_dec")
        shed = s.operation_load_shedding_power.sel(area="area_2").transpose("hour").to_numpy()
        shortage[label] = [bool(x > 1e-6) for x in shed]
    assert shortage["none"] == [False, True, False]
    assert shortage["1h"] == [False, False, True]


def _one_way_setup(p):
    """Area 2 cannot generate: its demand is met through link_1 from area 1."""
    p = p.copy(deep=True)
    p.update(
        dict(
            conversion_power_capacity_investment_max=p.conversion_power_capacity_investment_max.expand_dims(
                area=p.area,
                conversion_tech=p.conversion_tech,
            ).copy(),
        )
    )
    p.conversion_power_capacity_investment_max.loc[
        dict(area="area_2", conversion_tech="wind_onshore")
    ] = 0
    return p


def test_transport_efficiency(parameters_transport_single_horizon):
    p = _one_way_setup(parameters_transport_single_horizon)
    p["transport_efficiency"] = 0.8

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True).sum("year_dec")

    # 10 delivered to area 2 needs 12.5 injected at area 1
    assert_equal_epsilon(s.operation_transport_power.to_numpy(), np.array(10))
    assert_equal_epsilon(s.operation_transport_power_in.to_numpy(), np.array(12.5))
    assert_equal_epsilon(
        s.operation_conversion_power.to_numpy(), np.array([22.5, 0])
    )


def test_transport_default_efficiency_is_lossless(
    parameters_transport_single_horizon,
):
    # transport_efficiency at its default (1), as check_inputs fills it in
    p = _one_way_setup(parameters_transport_single_horizon)

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True).sum("year_dec")

    assert_equal_epsilon(
        s.operation_transport_power_in.to_numpy(),
        s.operation_transport_power.to_numpy(),
    )


def test_transport_fuel_consumption(parameters_transport_single_horizon):
    p = _one_way_setup(parameters_transport_single_horizon)
    p["transport_fuel_consumption"] = 0.1
    p["transport_fuel_resource"] = xr.DataArray("electricity")

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True).sum("year_dec")

    # area 1 covers its own 10, the 10 shipped and 1 of fuel for the link
    assert_equal_epsilon(s.operation_transport_power.to_numpy(), np.array(10))
    assert_equal_epsilon(
        s.operation_conversion_power.to_numpy(), np.array([21, 0])
    )


def test_transport_fuel_drawn_at_another_area(parameters_transport_single_horizon):
    """The fuel comes from the area named by transport_fuel_area (here the destination, standing for a bunker
    market), not from the origin."""
    p = _one_way_setup(parameters_transport_single_horizon)
    p["transport_fuel_consumption"] = 0.1
    p["transport_fuel_resource"] = xr.DataArray("electricity")
    p["transport_fuel_area"] = xr.DataArray("area_2")

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True).sum("year_dec")

    # area 2 burns 0.1 of what is injected out of what arrives: 10 / 0.9 shipped, all made in area 1
    assert_equal_epsilon(s.operation_transport_power.to_numpy(), np.array(10 / 0.9))
    assert_equal_epsilon(
        s.operation_conversion_power.to_numpy(), np.array([10 + 10 / 0.9, 0])
    )


def test_transport_fuel_drawn_from_another_resource(parameters_transport):
    """The fuel is a different resource from the one transported."""
    p = parameters_transport.sel(
        conversion_tech=["wind_onshore"],
        transport_tech=["power_line"],
        link=["link_1"],
        hour=[0],
        resource=["electricity", "methane"],
        year_dec=[2040, 2050],
        year_inv=[2030],
        year_op=[2030],
    ).copy(deep=True)
    p = _one_way_setup(p)
    p["demand"] = p.demand.where(p.resource == "electricity", 0)
    p["load_shedding_cost"] = p.load_shedding_cost.where(p.resource == "electricity", 1000)
    p["transport_fuel_consumption"] = 0.1
    p["transport_fuel_resource"] = xr.DataArray("methane")

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True).sum("year_dec")

    # no methane is produced anywhere: the only way to get it is load shedding
    # at area 1, which must equal the fuel drawn by the link (1)
    shed = s.operation_load_shedding_power.sel(resource="methane")
    assert_equal_epsilon(shed.sel(area="area_1").to_numpy(), np.array(1))
    assert_equal_epsilon(shed.sel(area="area_2").to_numpy(), np.array(0))
