import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal as assert_equal_epsilon
from numpy.testing import assert_array_equal as assert_equal

from pommes.model.build_model import build_model


@pytest.fixture()
def parameters_transport(parameters_dispatch_invest, transport):
    return xr.merge([parameters_dispatch_invest, transport])


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

    p = p.update(
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

    p = p.update(
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

    assert model.objective.value == 450.0 + 0.01 * 14


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
