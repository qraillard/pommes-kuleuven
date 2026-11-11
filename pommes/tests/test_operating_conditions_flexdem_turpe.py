"""Operating conditions in the flexible demand and TURPE modules.

One area, four hours, electricity only: wind is switched off, so the demand is covered by imports and every optimum
is checkable by hand.
"""
import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import (
    assert_condition_dims_consistent,
    with_conditions,
)

HOURS = [0, 1, 2, 3]
FLEXDEM = [
    "operation_flexdem_demand",
    "operation_flexdem_demand_max",
    "operation_flexdem_demand_min",
    "operation_flexdem_demand_conservation",
    "operation_flexdem_demand_conservation_2h",
    "operation_flexdem_ramp_up_constraint",
    "operation_flexdem_ramp_down_constraint",
]
TURPE = [
    "operation_turpe_variable_costs",
    "operation_turpe_contract_power_max_constraint",
    "operation_turpe_variable_costs_def",
]


@pytest.fixture()
def p_base(parameters_dispatch_invest, net_import):
    p = xr.merge([parameters_dispatch_invest, net_import])
    p = p.sel(
        area=["area_1"],
        conversion_tech=["wind_onshore"],
        hour=HOURS,
        resource=["electricity"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    ).copy(deep=True)
    p["operation_year_duration"][:] = float(p.time_step_duration.sum())
    p["conversion_power_capacity_max"] = p.conversion_power_capacity_max * 0
    p["demand"] = p.demand * 0
    p["load_shedding_cost"] = xr.DataArray(1000.0)
    p["load_shedding_max_capacity"] = xr.DataArray(np.nan)
    p["net_import_import_price"] = p.net_import_import_price * 0
    p["net_import_export_price"] = p.net_import_export_price * 0
    p["net_import_max_yearly_energy_import"] = p.net_import_max_yearly_energy_import * 0 + 1e3
    p["net_import_max_yearly_energy_export"] = p.net_import_max_yearly_energy_export * 0 + 1e3
    p["carbon"] = False
    return p


def _hourly(p, values):
    """An (hour, resource) array from one value per hour."""
    return xr.DataArray(
        np.array(values, dtype=float)[:, None],
        dims=("hour", "resource"),
        coords=dict(hour=p.hour, resource=p.resource),
    )


def _solve(p):
    m = build_model(p)
    if "operating_condition" in p.coords:
        assert_condition_dims_consistent(m)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


# ---------------------------------------------------------------- flexible demand


@pytest.fixture()
def p_flex(p_base):
    p = p_base
    dims = ("area", "resource", "year_op")
    coords = dict(area=p.area, resource=p.resource, year_op=p.year_op)

    def per(value):
        return xr.DataArray(np.full((1, 1, 1), value), dims=dims, coords=coords)

    p["flexdem"] = True
    p["flexdem_conservation_hrs"] = per(2).astype("int64")
    p["flexdem_maxload_ratio"] = per(3.0)
    p["flexdem_minload_ratio"] = per(0.0)
    # finite and slack ramps build the ramp rows without changing the solution
    p["flexdem_ramp_up"] = per(1e3)
    p["flexdem_ramp_down"] = per(1e3)
    return p


def _flexdem_demand(p, value):
    return xr.DataArray(
        np.full((1, 4, 1, 1), float(value)),
        dims=("area", "hour", "resource", "year_op"),
        coords=dict(area=p.area, hour=p.hour, resource=p.resource, year_op=p.year_op),
    )


def test_flexdem_conserves_in_each_condition(p_flex):
    # a: 10 MWh per hour, cheap at the window edges; b: 20 MWh per hour, cheap inside the windows
    p = with_conditions(
        p_flex,
        [0.5, 0.5],
        flexdem_demand=[_flexdem_demand(p_flex, 10.0), _flexdem_demand(p_flex, 20.0)],
        net_import_import_price=[
            _hourly(p_flex, [1, 10, 10, 1]),
            _hourly(p_flex, [10, 1, 1, 10]),
        ],
    )
    m = _solve(p)
    x = m.solution["operation_flexdem_demand"].sel(area="area_1", resource="electricity", year_op=2020)
    a, b = (x.sel(operating_condition=c).values for c in "ab")
    # each window of 2 h takes its whole energy at its cheapest hour, in each condition
    np.testing.assert_allclose(a, [20, 0, 0, 20], atol=1e-6)
    np.testing.assert_allclose(b, [0, 40, 40, 0], atol=1e-6)
    # the shifted energy sums to 0 in each condition, over each window and over the horizon
    for values, demand in ((a, 10.0), (b, 20.0)):
        assert values[:2].sum() == pytest.approx(2 * demand)
        assert values[2:].sum() == pytest.approx(2 * demand)
    assert m.objective.value == pytest.approx(0.5 * 40 + 0.5 * 80)


def test_flexdem_an_absent_pair_has_no_row(p_flex):
    p_flex["flexdem_minload_ratio"] = p_flex.flexdem_minload_ratio * 0 + 0.1
    p = with_conditions(p_flex, [1.0, np.nan], flexdem_demand=[_flexdem_demand(p_flex, 10.0)] * 2)
    m = build_model(p)
    assert_condition_dims_consistent(m)
    for name in FLEXDEM:
        container = m.variables if name in m.variables else m.constraints
        assert name in container, name
        labels = container[name].labels
        assert (labels.sel(operating_condition="b") == -1).all(), name
        assert (labels.sel(operating_condition="a") != -1).any(), name


def test_flexdem_weight_zero_condition_keeps_its_rows(p_flex):
    p = with_conditions(
        p_flex,
        [1.0, 0.0],
        flexdem_demand=[_flexdem_demand(p_flex, 10.0)] * 2,
        net_import_import_price=[_hourly(p_flex, [1, 10, 10, 1]), _hourly(p_flex, [1e4] * 4)],
    )
    m = _solve(p)
    assert m.objective.value == pytest.approx(40.0)
    for name in FLEXDEM:
        if name == "operation_flexdem_demand_min":
            continue  # minload 0: not built
        container = m.variables if name in m.variables else m.constraints
        assert (container[name].labels.sel(operating_condition="b") != -1).any(), name


# ------------------------------------------------------------------------- TURPE

CALENDAR = ["HCE", "HCE", "P", "P"]
WITHDRAWAL = {"a": [4, 4, 10, 10], "b": [8, 8, 6, 6]}


@pytest.fixture()
def p_turpe(p_base, turpe):
    p = xr.merge([p_base, turpe.sel(hour=HOURS)], join="outer")
    p["turpe_calendar"] = xr.DataArray(CALENDAR, dims="hour", coords=dict(hour=p.hour))
    return p


def _demand(p, values):
    return _hourly(p, values).expand_dims(area=p.area, year_op=p.year_op)


@pytest.mark.parametrize("order", ["ab", "ba"])
def test_turpe_contract_covers_the_larger_condition(p_turpe, order):
    # the larger P withdrawal (10) is in condition 'a' for "ab" and in 'b' for "ba": a contract ignoring one fails
    p = with_conditions(
        p_turpe,
        [0.5, 0.5],
        demand=[_demand(p_turpe, WITHDRAWAL[c]) for c in order],
    )
    m = _solve(p)
    contract = m.solution["operation_turpe_contract_power"]
    fixed = m.solution["operation_turpe_fixed_costs"]
    variable = m.solution["operation_turpe_variable_costs"]
    # one contract and one fixed cost per year, whatever the conditions
    assert "operating_condition" not in contract.dims
    assert "operating_condition" not in fixed.dims
    assert "operating_condition" in variable.dims
    # P covers the larger hourly withdrawal of its hours, whichever condition holds it
    sel = contract.sel(area="area_1", year_op=2020)
    assert float(sel.sel(hour_type="P")) == pytest.approx(10.0)
    # the contract never decreases along the hour types: the cheapest way is 10 at HCE (fixed cost 5), counted once
    assert float(fixed.sum()) == pytest.approx(10 * 5)
    # variable cost per condition (0.01 at HCE, 0.5 at P) and in expectation
    expected = {"a": 8 * 0.01 + 20 * 0.5, "b": 16 * 0.01 + 12 * 0.5}
    for k, c in enumerate("ab"):
        assert float(variable.sel(operating_condition=c).sum()) == pytest.approx(expected[order[k]])
    assert m.objective.value == pytest.approx(50 + 0.5 * 10.08 + 0.5 * 6.16)


def test_turpe_an_absent_pair_has_no_row(p_turpe):
    p = with_conditions(
        p_turpe, [1.0, np.nan], demand=[_demand(p_turpe, WITHDRAWAL[c]) for c in "ab"]
    )
    m = build_model(p)
    assert_condition_dims_consistent(m)
    for name in TURPE:
        container = m.variables if name in m.variables else m.constraints
        assert name in container, name
        labels = container[name].labels
        assert (labels.sel(operating_condition="b") == -1).all(), name
        assert (labels.sel(operating_condition="a") != -1).any(), name
    for name in ("operation_turpe_contract_power", "operation_turpe_fixed_costs"):
        assert "operating_condition" not in m.variables[name].labels.dims
