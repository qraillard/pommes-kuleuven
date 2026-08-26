import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal as assert_equal

from pommes.model.build_model import build_model


@pytest.fixture()
def parameters_process(parameters_dispatch_invest, process):
    p = xr.merge([parameters_dispatch_invest, process])
    p = p.sel(
        area=["area_1"],
        conversion_tech=["wind_onshore"],
        resource=["electricity", "heat", "methane"],
    ).copy(deep=True)

    p["load_shedding_cost"].loc[dict(resource="methane")] = 5
    p["demand"].loc[dict(resource="electricity")] = np.array([0, 0, 0, 0])

    p["conversion_power_capacity_investment_max"] = (
        10  # production of 2 MW of electricity at hour 2
    )
    p["conversion_power_capacity_investment_min"] = 10
    p["conversion_annuity_cost"] = 0
    p["conversion_fixed_cost"] = 0

    return p


@pytest.fixture()
def parameters_process_single_horizon(parameters_process):
    p = parameters_process.sel(
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    return p


@pytest.fixture()
def parameters_process_multi_horizon(parameters_process):
    p = parameters_process.sel(
        hour=[2],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030],
    ).copy(deep=True)

    p["demand"].loc[dict(resource="heat", year_op=2030)] = 0
    return p


def test_process_single_hour(parameters_process_single_horizon):
    p = parameters_process_single_horizon
    p = p.sel(hour=[2])
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(s.operation_conversion_power.to_numpy(), 1)
    assert_equal(s.planning_process_power_capacity.to_numpy(), 2)
    assert_equal(s.operation_process_power.to_numpy(), [1, 1])
    assert_equal(s.operation_process_net_generation.to_numpy(), [-1, 2, -1.5])
    assert_equal(s.operation_load_shedding_power, [np.nan, np.nan, 1.5])
    assert_equal(s.operation_spillage_power, [0, 0, 0])

    assert model.objective.value == 10 * 2 + 100 * 2 + 1 * 2 + 1.5 * 5


def test_process_multi_hour(parameters_process_single_horizon):
    p = parameters_process_single_horizon
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(s.planning_process_power_capacity.to_numpy(), np.array(2))

    assert_equal(
        s.operation_conversion_power.to_numpy(),
        np.array([5, 7, 1, 0, 2, 4, 8, 9, 6, 3]),
    )

    assert_equal(
        s.operation_spillage_power.transpose("resource", "hour"),
        np.array(
            [[3, 5, 0, 0, 0, 2, 6, 7, 4, 1]]
            + [[0] * len(p.hour)] * (len(p.resource) - 1)
        ),
    )

    assert_equal(
        s.operation_process_power.to_numpy(),
        np.array(
            [[2, 2, 1, 0, 2, 2, 2, 2, 2, 2], [0, 0, 1, 2, 0, 0, 0, 0, 0, 0]]
        ),
    )

    assert model.objective.value == 10 * 2 + 100 * 2 + 1 * 2 * 10 + 1.5 * 5 * 3





def test_process_multi_horizon(parameters_process_multi_horizon):
    p = parameters_process_multi_horizon
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_process_costs.to_numpy(), np.array([200, 200])
    )

    np.testing.assert_array_equal(
        s.planning_process_power_capacity.to_numpy(),
        np.array([[np.nan, np.nan], [2, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_process_power_capacity.to_numpy(), np.array([2, 2])
    )

    np.testing.assert_array_equal(
        s.operation_process_power.transpose("year_op", "mode").to_numpy(),
        np.array([[1, 1], [0, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.transpose("year_op", "resource"),
        [[0, 0, 0], [2, 0, 0]],
    )

    assert model.objective.value == 10 * 2 * 2 + 100 * 2 * 2 + 1 * 2 + 1.5 * 5


def test_process_early_decommissioning(parameters_process_multi_horizon):
    p = parameters_process_multi_horizon.copy(deep=True)
    p["process_early_decommissioning"] = True
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_process_costs.to_numpy(), np.array([200, 200])
    )

    np.testing.assert_array_equal(
        s.planning_process_power_capacity.to_numpy(),
        np.array([[2, np.nan], [0, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_process_power_capacity.to_numpy(), np.array([2, 0])
    )

    np.testing.assert_array_equal(
        s.operation_process_power.transpose("year_op", "mode").to_numpy(),
        np.array([[1, 1], [0, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.transpose("year_op", "resource"),
        [[0, 0, 0], [2, 0, 0]],
    )

    assert model.objective.value == 10 * 2 * 1 + 100 * 2 * 2 + 1 * 2 + 1.5 * 5


def test_process_annuity_perfect_foresight(parameters_process_multi_horizon):
    p = parameters_process_multi_horizon.copy(deep=True)
    p["process_early_decommissioning"] = True
    p["process_annuity_perfect_foresight"] = True
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_process_costs.to_numpy(), np.array([400, 0])
    )

    np.testing.assert_array_equal(
        s.planning_process_power_capacity.to_numpy(),
        np.array([[2, np.nan], [0, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_process_power_capacity.to_numpy(), np.array([2, 0])
    )

    np.testing.assert_array_equal(
        s.operation_process_power.transpose("year_op", "mode").to_numpy(),
        np.array([[1, 1], [0, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.transpose("year_op", "resource"),
        [[0, 0, 0], [2, 0, 0]],
    )

    assert model.objective.value == 10 * 2 * 1 + 100 * 2 * 2 + 1 * 2 + 1.5 * 5
