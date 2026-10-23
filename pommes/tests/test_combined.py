import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal as assert_equal

from pommes.model.build_model import build_model


@pytest.fixture()
def parameters_combined(parameters_dispatch_invest, combined):
    p = xr.merge([parameters_dispatch_invest, combined])
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
def parameters_combined_single_horizon(parameters_combined):
    p = parameters_combined.sel(
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    return p


@pytest.fixture()
def parameters_combined_multi_horizon(parameters_combined):
    p = parameters_combined.sel(
        hour=[2],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030],
    ).copy(deep=True)

    p["demand"].loc[dict(resource="heat", year_op=2030)] = 0
    return p


def test_combined_single_hour(parameters_combined_single_horizon):
    p = parameters_combined_single_horizon
    p = p.sel(hour=[2])
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(s.operation_conversion_power.to_numpy(), 1)
    assert_equal(s.planning_combined_power_capacity.to_numpy(), 2)
    assert_equal(s.operation_combined_power.to_numpy(), [1, 1])
    assert_equal(s.operation_combined_net_generation.to_numpy(), [-1, 2, -1.5])
    assert_equal(s.operation_load_shedding_power, [np.nan, np.nan, 1.5])
    assert_equal(s.operation_spillage_power, [0, 0, 0])

    assert model.objective.value == 10 * 2 + 100 * 2 + 1 * 2 + 1.5 * 5


def test_combined_multi_hour(parameters_combined_single_horizon):
    p = parameters_combined_single_horizon
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(s.planning_combined_power_capacity.to_numpy(), np.array(2))

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
        s.operation_combined_power.to_numpy(),
        np.array(
            [[2, 2, 1, 0, 2, 2, 2, 2, 2, 2], [0, 0, 1, 2, 0, 0, 0, 0, 0, 0]]
        ),
    )

    assert model.objective.value == 10 * 2 + 100 * 2 + 1 * 2 * 10 + 1.5 * 5 * 3


# def test_combined_must_run(parameters_combined_single_horizon):
#     p = parameters_combined_single_horizon
#     p = p.sel(
#         conversion_tech=["ocgt", "wind_onshore"],
#         hour=[4, 5],
#         resource=["electricity"],
#     )
#     p = p.copy(deep=True)
#
#     p = p.update(
#         dict(
#             conversion_power_capacity_investment_min=
#             p.conversion_power_capacity_investment_min.expand_dims(
#                 conversion_tech=p.conversion_tech,
#             ).copy(),
#             conversion_power_capacity_investment_max=
#             p.conversion_power_capacity_investment_max.expand_dims(
#                 conversion_tech=p.conversion_tech,
#             ).copy(),
#         )
#     )
#
#     p.conversion_must_run.loc[dict(conversion_tech="ocgt")] = 0.5
#     p.conversion_power_capacity_investment_min.loc[
#     dict(conversion_tech="ocgt")] = 2
#     p.conversion_power_capacity_investment_max.loc[
#     dict(conversion_tech="ocgt")] = 2
#
#     model = build_model(p)
#     model.solve(solver_name="highs")
#     s = model.solution.squeeze()
#
#     assert_equal(s.planning_combined_power_capacity, np.array([2, 40]))
#     assert_equal(s.operation_combined_power, np.array([[2, 1], [8, 16]]))
#     assert_equal(s.operation_spillage_power, np.array([0, 7]))
#     assert model.objective.value == 20 * 2 + 10 * 40 + 10 * 2 + 3 * 8


def test_combined_multi_horizon(parameters_combined_multi_horizon):
    p = parameters_combined_multi_horizon
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_combined_costs.to_numpy(), np.array([200, 200])
    )

    np.testing.assert_array_equal(
        s.planning_combined_power_capacity.to_numpy(),
        np.array([[np.nan, np.nan], [2, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_combined_power_capacity.to_numpy(), np.array([2, 2])
    )

    np.testing.assert_array_equal(
        s.operation_combined_power.transpose("year_op", "mode").to_numpy(),
        np.array([[1, 1], [0, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.transpose("year_op", "resource"),
        [[0, 0, 0], [2, 0, 0]],
    )

    assert model.objective.value == 10 * 2 * 2 + 100 * 2 * 2 + 1 * 2 + 1.5 * 5


def test_combined_early_decommissioning(parameters_combined_multi_horizon):
    p = parameters_combined_multi_horizon.copy(deep=True)
    p["combined_early_decommissioning"] = True
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_combined_costs.to_numpy(), np.array([200, 200])
    )

    np.testing.assert_array_equal(
        s.planning_combined_power_capacity.to_numpy(),
        np.array([[2, np.nan], [0, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_combined_power_capacity.to_numpy(), np.array([2, 0])
    )

    np.testing.assert_array_equal(
        s.operation_combined_power.transpose("year_op", "mode").to_numpy(),
        np.array([[1, 1], [0, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.transpose("year_op", "resource"),
        [[0, 0, 0], [2, 0, 0]],
    )

    assert model.objective.value == 10 * 2 * 1 + 100 * 2 * 2 + 1 * 2 + 1.5 * 5


def test_combined_annuity_perfect_foresight(parameters_combined_multi_horizon):
    p = parameters_combined_multi_horizon.copy(deep=True)
    p["combined_early_decommissioning"] = True
    p["combined_annuity_perfect_foresight"] = True
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_combined_costs.to_numpy(), np.array([400, 0])
    )

    np.testing.assert_array_equal(
        s.planning_combined_power_capacity.to_numpy(),
        np.array([[2, np.nan], [0, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_combined_power_capacity.to_numpy(), np.array([2, 0])
    )

    np.testing.assert_array_equal(
        s.operation_combined_power.transpose("year_op", "mode").to_numpy(),
        np.array([[1, 1], [0, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.transpose("year_op", "resource"),
        [[0, 0, 0], [2, 0, 0]],
    )

    assert model.objective.value == 10 * 2 * 1 + 100 * 2 * 2 + 1 * 2 + 1.5 * 5
