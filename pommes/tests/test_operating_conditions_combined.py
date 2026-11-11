"""Operating conditions in the combined module.

One area, four hours, one combined technology (`chp_gas`: mode m2 makes 1 MWh of electricity, spilled, and
0.6 MWh of heat from 2.2 MWh of methane, bought by shedding), heat demand only: the heat is covered by the
combined unit or shed at 1000 per MWh, so the optimum is checkable by hand.
"""
import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import (
    assert_condition_dims_consistent,
    with_conditions,
)
from pommes.tests.dataset_nnodes import build_nnode_parameters
from pommes.tests.test_flexdem import _with_defaults

N_HOURS = 4


def _base():
    ds = build_nnode_parameters(n_areas=1, n_hours=N_HOURS)
    ds = ds.sel(combined_tech=["chp_gas"])
    for module in ("conversion", "storage", "carbon"):
        ds[module] = False
    return _with_defaults(ds)


def _heat(ds, hourly):
    """Demand with the heat demand per hour given (other resources: no demand)."""
    demand = xr.zeros_like(ds.demand)
    demand.loc[dict(resource="heat")] = np.array(hourly, dtype=float).reshape(
        1, N_HOURS, 1
    )
    return demand


def _solve(ds):
    m = build_model(ds)
    if "operating_condition" in ds.coords:
        assert_condition_dims_consistent(m)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


def _cost(hourly):
    ds = _base()
    ds["demand"] = _heat(ds, hourly)
    return _solve(ds).objective.value


def _conditions(weights, a, b, labels=None):
    ds = _base()
    return with_conditions(
        ds, weights, labels, demand=[_heat(ds, a), _heat(ds, b)]
    )


def _power(m):
    return m.solution["operation_combined_power"].sum("mode")


def test_combined_dispatch_differs_by_condition():
    m = _solve(_conditions([0.5, 0.5], [6] * 4, [18] * 4))
    capacity = m.solution["planning_combined_power_capacity"]
    assert "operating_condition" not in capacity.dims
    assert float(capacity.sum()) == pytest.approx(18 / 0.6)
    power = _power(m)
    np.testing.assert_allclose(
        power.sel(operating_condition="a").values, 6 / 0.6, atol=1e-9
    )
    np.testing.assert_allclose(
        power.sel(operating_condition="b").values, 18 / 0.6, atol=1e-9
    )


def test_combined_cost_is_the_weighted_mean():
    # same peak in both conditions: the capacity, hence its fixed cost, is the same as alone
    a, b = [18, 6, 6, 6], [18, 18, 6, 6]
    m = _solve(_conditions([0.6, 0.4], a, b))
    expected = 0.6 * _cost(a) + 0.4 * _cost(b)
    assert m.objective.value == pytest.approx(expected, rel=1e-9)


def test_an_absent_pair_has_no_combined_row():
    ds = _base()
    ds["combined_ramp_up"] = xr.full_like(ds.combined_ramp_up, 0.5)
    ds["combined_ramp_down"] = xr.full_like(ds.combined_ramp_down, 0.5)
    ds["combined_mode_ramp_up"] = xr.full_like(ds.combined_mode_ramp_up, 0.5)
    ds["combined_mode_ramp_down"] = xr.full_like(ds.combined_mode_ramp_down, 0.5)
    ds = with_conditions(
        ds, [1.0, np.nan], demand=[_heat(ds, [6] * 4), _heat(ds, [6] * 4)]
    )
    m = build_model(ds)
    assert_condition_dims_consistent(m)
    checked = []
    for container in (m.variables, m.constraints):
        for name in container:
            if "combined" not in name:
                continue
            labels = container[name].labels
            if "operating_condition" not in labels.dims:
                continue
            checked.append(name)
            assert (labels.sel(operating_condition="b") == -1).all(), name
            if name in MUST_HAVE_ROWS_IN_A:
                assert (labels.sel(operating_condition="a") != -1).any(), name
    assert set(MUST_HAVE_ROWS_IN_A) <= set(checked)


def test_weight_zero_keeps_its_rows_and_costs_nothing():
    m = _solve(_conditions([1.0, 0.0], [6] * 4, [18] * 4))
    assert m.objective.value == pytest.approx(_cost([6] * 4), rel=1e-9)
    for name in (
        "operation_combined_power_max_constraint",
        "operation_combined_costs_def",
    ):
        labels = m.constraints[name].labels
        assert (labels.sel(operating_condition="b") != -1).all(), name


MUST_HAVE_ROWS_IN_A = [
    "operation_combined_power",
    "operation_combined_net_generation",
    "operation_combined_costs",
    "operation_combined_power_max_constraint",
    "operation_combined_power_min_constraint",
    "operation_combined_ramp_up_constraint",
    "operation_combined_ramp_down_constraint",
    "operation_combined_net_generation_def",
    "operation_combined_costs_def",
]


def _zero_factor(ds):
    ds["combined_factor"] = ds.combined_factor.where(ds.mode != "m1", 0.0)


def _zero_aggregate_ramps(ds):
    ds["combined_ramp_up"] = xr.zeros_like(ds.combined_ramp_up)
    ds["combined_ramp_down"] = xr.zeros_like(ds.combined_ramp_down)
    ds["combined_mode_ramp_up"] = xr.full_like(ds.combined_mode_ramp_up, 0.5)
    ds["combined_mode_ramp_down"] = xr.full_like(ds.combined_mode_ramp_down, 0.5)


def _zero_all_ramps(ds):
    _zero_aggregate_ramps(ds)
    ds["combined_mode_ramp_up"] = xr.zeros_like(ds.combined_mode_ramp_up)
    ds["combined_mode_ramp_down"] = xr.zeros_like(ds.combined_mode_ramp_down)


@pytest.mark.parametrize(
    "tweak, name",
    [
        (_zero_factor, "operation_combined_power_undefined_mode"),
        (_zero_aggregate_ramps, "operation_combined_constant_operation_constraint"),
        (_zero_all_ramps, "operation_combined_mode_constant_operation_constraint"),
    ],
)
def test_an_absent_pair_has_no_row_in_the_rarely_built_constraints(tweak, name):
    ds = _base()
    tweak(ds)
    ds = with_conditions(
        ds, [1.0, np.nan], demand=[_heat(ds, [6] * 4), _heat(ds, [6] * 4)]
    )
    m = build_model(ds)
    assert name in m.constraints
    labels = m.constraints[name].labels
    assert (labels.sel(operating_condition="b") == -1).all()
    assert (labels.sel(operating_condition="a") != -1).any()
