import numpy as np
import pytest
from numpy.testing import assert_array_equal as assert_equal

from pommes.model.build_model import build_model


@pytest.fixture()
def parameters_conversion_single_horizon(parameters_dispatch_invest):
    p = parameters_dispatch_invest.sel(
        area=["area_1"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    return p


@pytest.fixture()
def parameters_conversion_multi_horizon(parameters_dispatch_invest):
    p = parameters_dispatch_invest.sel(
        conversion_tech=["wind_onshore"],
        area=["area_1"],
        hour=[2],
        resource=["electricity"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030],
    ).copy(deep=True)

    p["demand"].loc[dict(resource="electricity", year_op=2030)] = 0
    p["conversion_fixed_cost"].loc[dict(conversion_tech="wind_onshore")] = 10

    return p


def test_conversion_single_hour(parameters_conversion_single_horizon):
    p = parameters_conversion_single_horizon
    p = p.sel(
        conversion_tech=["ocgt", "wind_onshore"],
        hour=[0],
        resource=["electricity"],
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(
        s.planning_conversion_power_capacity.to_numpy(), np.array([0, 20])
    )
    assert model.objective.value == 200


def test_conversion_multi_hour(parameters_conversion_single_horizon):
    p = parameters_conversion_single_horizon
    p = p.sel(
        conversion_tech=["ocgt", "wind_onshore"],
        resource=["electricity"],
    )
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(
        s.planning_conversion_power_capacity.to_numpy(), np.array([10, 20])
    )

    assert_equal(
        s.operation_conversion_power.sel(conversion_tech="ocgt").to_numpy(),
        np.array([0.0, 0.0, 8.0, 10.0, 6.0, 2.0, 0.0, 0.0, 0.0, 4.0]),
    )

    assert_equal(
        s.operation_conversion_power.sel(
            conversion_tech="wind_onshore"
        ).to_numpy(),
        np.array([10.0, 14.0, 2.0, 0.0, 4.0, 8.0, 16.0, 18.0, 12.0, 6.0]),
    )

    assert_equal(
        s.operation_spillage_power.to_numpy(),
        np.array([0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 8.0, 2.0, 0.0]),
    )
    assert model.objective.value == 740


def test_conversion_availability(parameters_conversion_single_horizon):
    p = parameters_conversion_single_horizon.copy(deep=True)
    p = p.sel(
        conversion_tech=["ocgt", "wind_onshore"],
        resource=["electricity"],
    )
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    p.conversion_must_run[:] = [0, 0]
    p.conversion_availability.loc[
        dict(conversion_tech="wind_onshore", hour=0)
    ] = np.nan
    p["spillage_max_capacity"] = 0
    p["conversion_power_capacity_max"] = 20
    p["conversion_power_capacity_min"] = 20
    p["conversion_fixed_cost"] = 0
    p["conversion_annuity_cost"] = 0

    model = build_model(p)

    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(
        s.operation_conversion_power.sel(conversion_tech="ocgt").to_numpy(),
        np.array([0.0, 0.0, 8.0, 10.0, 6.0, 2.0, 0.0, 0.0, 0.0, 4.0]),
    )

    assert_equal(
        s.operation_conversion_power.sel(
            conversion_tech="wind_onshore"
        ).to_numpy(),
        np.array([10.0, 10.0, 2.0, 0.0, 4.0, 8.0, 10.0, 10.0, 10.0, 6.0]),
    )

    assert model.objective.value == 240


def test_conversion_availability_must_run_nan(
    parameters_conversion_single_horizon,
):
    p = parameters_conversion_single_horizon.copy(deep=True)
    p = p.sel(
        conversion_tech=["ocgt", "wind_onshore"],
        resource=["electricity"],
    )
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    p.conversion_must_run.loc["wind_onshore"] = np.nan
    p.conversion_availability.loc[
        dict(conversion_tech="wind_onshore", hour=0)
    ] = np.nan
    p["spillage_max_capacity"] = 0
    p["conversion_power_capacity_max"] = 20
    p["conversion_power_capacity_min"] = 20
    p["conversion_fixed_cost"] = 0
    p["conversion_annuity_cost"] = 0

    model = build_model(p)

    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(
        s.operation_conversion_power.sel(conversion_tech="ocgt").to_numpy(),
        np.array([0.0, 0.0, 8.0, 10.0, 6.0, 2.0, 0.0, 0.0, 0.0, 4.0]),
    )

    assert_equal(
        s.operation_conversion_power.sel(
            conversion_tech="wind_onshore"
        ).to_numpy(),
        np.array([10.0, 10.0, 2.0, 0.0, 4.0, 8.0, 10.0, 10.0, 10.0, 6.0]),
    )

    assert model.objective.value == 240


def test_conversion_multi_hour_energy(parameters_conversion_single_horizon):
    p = parameters_conversion_single_horizon
    p = p.sel(
        conversion_tech=["electrolysis", "ocgt", "wind_onshore"],
        resource=["electricity", "hydrogen"],
    )
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(
        s.planning_conversion_power_capacity.to_numpy(),
        np.array([3, 14.5, 29]),
    )
    # objective value = (300 + 290 + 290) + (0 + 145 + 0)
    # + 8 * sum([max(14.5 - 29 * 0.1 * k, 0) for k in range(10)])
    assert model.objective.value == 1373


def test_load_shedding(parameters_conversion_single_horizon):
    p = parameters_conversion_single_horizon
    p = p.sel(
        conversion_tech=["ocgt"],
        hour=[0],
        resource=["electricity", "methane"],
    )
    # Values here corresponds built for a single hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(
        s.planning_conversion_power_capacity.to_numpy(), np.array([10])
    )
    assert_equal(s.operation_conversion_power.to_numpy(), np.array([10.0]))
    assert_equal(
        s.operation_load_shedding_power.sel(resource="methane").to_numpy(),
        np.array([15]),
    )

    assert model.objective.value == 15000 + 200 + 100 + 80


def test_conversion_must_run(parameters_conversion_single_horizon):
    p = parameters_conversion_single_horizon
    p = p.sel(
        conversion_tech=["ocgt", "wind_onshore"],
        hour=[4, 5],
        resource=["electricity"],
    )
    # Values here corresponds built for a 2 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    p = p.copy(deep=True)

    p = p.update(
        dict(
            conversion_power_capacity_investment_min=(
                p.conversion_power_capacity_investment_min.expand_dims(
                    conversion_tech=p.conversion_tech,
                ).copy()
            ),
            conversion_power_capacity_investment_max=(
                p.conversion_power_capacity_investment_max.expand_dims(
                    conversion_tech=p.conversion_tech,
                ).copy()
            ),
        )
    )

    p.conversion_must_run.loc[dict(conversion_tech="ocgt")] = 0.5
    p.conversion_power_capacity_investment_min.loc[
        dict(conversion_tech="ocgt")
    ] = 2
    p.conversion_power_capacity_investment_max.loc[
        dict(conversion_tech="ocgt")
    ] = 2

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(s.planning_conversion_power_capacity, np.array([2, 40]))
    assert_equal(s.operation_conversion_power, np.array([[2, 1], [8, 16]]))
    assert_equal(s.operation_spillage_power, np.array([0, 7]))
    assert model.objective.value == 20 * 2 + 10 * 40 + 10 * 2 + 3 * 8


def test_conversion_multi_horizon(parameters_conversion_multi_horizon):
    p = parameters_conversion_multi_horizon

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_conversion_costs.to_numpy(), np.array([1000, 1000])
    )

    np.testing.assert_array_equal(
        s.planning_conversion_power_capacity.to_numpy(),
        np.array([[np.nan, np.nan], [100, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_conversion_power_capacity.to_numpy(), np.array([100, 100])
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.to_numpy(),
        np.array([0, 10]),
    )

    np.testing.assert_array_equal(
        s.operation_conversion_power.to_numpy(),
        np.array([10, 10]),
    )

    assert model.objective.value == 4000


def test_conversion_early_decommissioning(parameters_conversion_multi_horizon):
    p = parameters_conversion_multi_horizon.copy(deep=True)

    p["conversion_early_decommissioning"] = True

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_conversion_costs.to_numpy(), np.array([1000, 1000])
    )

    np.testing.assert_array_equal(
        s.planning_conversion_power_capacity.to_numpy(),
        np.array([[100, np.nan], [0, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_conversion_power_capacity.to_numpy(), np.array([100, 0])
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.to_numpy(),
        np.array([0, 0]),
    )

    np.testing.assert_array_equal(
        s.operation_conversion_power.to_numpy(),
        np.array([10, 0]),
    )

    assert model.objective.value == 3000


def test_conversion_annuity_perfect_foresight(
    parameters_conversion_multi_horizon,
):
    p = parameters_conversion_multi_horizon.copy(deep=True)

    p["conversion_early_decommissioning"] = True
    p["conversion_annuity_perfect_foresight"] = True

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_conversion_costs.to_numpy(), np.array([2000, 0])
    )

    np.testing.assert_array_equal(
        s.planning_conversion_power_capacity.to_numpy(),
        np.array([[100, np.nan], [0, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_conversion_power_capacity.to_numpy(), np.array([100, 0])
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.to_numpy(),
        np.array([0, 0]),
    )

    np.testing.assert_array_equal(
        s.operation_conversion_power.to_numpy(),
        np.array([10, 0]),
    )

    assert model.objective.value == 3000


def test_conversion_power_capacity_investment_min_early_decommissioning(
    parameters_dispatch_invest,
):
    p = parameters_dispatch_invest
    p = p.sel(
        area=["area_1"],
        conversion_tech=["smr"],
        hour=[0],
        resource=["hydrogen"],
    )
    p = p.copy(deep=True)

    # Values here corresponds built for a single hour operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values

    p["conversion_early_decommissioning"] = True
    p["conversion_annuity_perfect_foresight"] = True

    p = p.update(
        dict(
            conversion_power_capacity_investment_min=(
                p.conversion_power_capacity_investment_min.expand_dims(
                    conversion_tech=p.conversion_tech,
                    year_inv=p.year_inv,
                ).copy()
            ),
            conversion_power_capacity_investment_max=(
                p.conversion_power_capacity_investment_max.expand_dims(
                    conversion_tech=p.conversion_tech,
                    year_inv=p.year_inv,
                ).copy()
            ),
        )
    )

    p.conversion_power_capacity_investment_min.broadcast_like(p.year_inv).loc[
        dict(conversion_tech="smr", year_inv=2010)
    ] = 10
    p.conversion_power_capacity_investment_max.loc[
        dict(conversion_tech="smr", year_inv=2010)
    ] = 10

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(s.operation_conversion_power_capacity, [3, 2, 4, 0])

    assert_equal(
        s.planning_conversion_power_capacity,
        np.array(
            [
                [7, np.nan, np.nan, np.nan, np.nan],
                [3, 0, np.nan, np.nan, np.nan],
                [np.nan, 0, 0, np.nan, np.nan],
                [np.nan, np.nan, 2, 2, np.nan],
                [np.nan, np.nan, np.nan, 0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, 0],
            ]
        ),
    )

    assert_equal(
        s.operation_conversion_costs, np.array([93.0, 62.0, 124.0, -0.0])
    )
    assert_equal(
        s.planning_conversion_costs, np.array([150.0, 100.0, 300.0, 0])
    )

    assert model.objective.value == 829.0


def test_conversion_power_capacity_max_early_decommissioning(
    parameters_dispatch_invest,
):
    p = parameters_dispatch_invest
    p = p.sel(
        area=["area_1"],
        conversion_tech=["smr"],
        hour=[0],
        resource=["hydrogen"],
    )
    p = p.copy(deep=True)

    # Values here corresponds built for a single hour operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values

    p = p.update(
        dict(
            conversion_power_capacity_max=p.conversion_power_capacity_max.expand_dims(
                conversion_tech=p.conversion_tech,
                year_op=p.year_op,
            ).copy(),
        )
    )

    p["conversion_early_decommissioning"] = True

    p.conversion_power_capacity_max.loc[
        dict(conversion_tech="smr", year_op=2040)
    ] = 3

    p.load_shedding_max_capacity.loc[dict(resource="hydrogen")] = np.nan
    p.load_shedding_cost.loc[dict(resource="hydrogen")] = 200

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(s.operation_conversion_power_capacity, [3, 2, 3, 0])

    assert_equal(
        s.planning_conversion_power_capacity,
        np.array(
            [
                [0, np.nan, np.nan, np.nan, np.nan],
                [3, 0, np.nan, np.nan, np.nan],
                [np.nan, 0, 0, np.nan, np.nan],
                [np.nan, np.nan, 2, 1, np.nan],
                [np.nan, np.nan, np.nan, 0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, 0],
            ]
        ),
    )

    assert_equal(s.operation_load_shedding_power, np.array([0, 0, 1, 0]))

    assert model.objective.value == 898.0
