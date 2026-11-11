"""Operating conditions in the carbon module.

One area, four hours, electricity only: `ocgt` (45 per MWh, 0.4 t per MWh) covers the demand and
shedding (1000 per MWh) is the alternative that the carbon goals force, so every optimum is
checkable by hand.
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

# per hour: 40 MWh, 16 t over the year, one-year duration = sum of the steps (normalization 1)
DEMAND = 10.0
GOAL = 8.0
EMISSIONS = 0.4 * 4 * DEMAND  # 16 t
GOALS = ["carbon_goal", "carbon_goal_area", "carbon_goal_group"]


def _base(demand=DEMAND):
    ds = build_nnode_parameters(n_areas=1, n_hours=4)
    ds = ds.sel(conversion_tech=["ocgt"], resource=["electricity"])
    for module in ("storage", "combined"):
        ds[module] = False
    ds = _with_defaults(ds)
    ds["operation_year_duration"][:] = float(ds.time_step_duration.sum())
    ds["demand"] = ds.demand * 0 + demand
    ds["conversion_power_capacity_max"] = ds.conversion_power_capacity_max * 0 + 200.0
    # no planning cost: the capacity is shared by the conditions, the plain runs below must not be
    # cheaper for it
    ds["conversion_fixed_cost"] = ds.conversion_fixed_cost * 0
    ds["conversion_annuity_cost"] = ds.conversion_annuity_cost * 0
    ds["load_shedding_cost"] = ds.load_shedding_cost * 0 + 1000.0
    ds["load_shedding_max_capacity"] = ("resource", [100.0])
    return ds


def _goal(ds, kind, value=GOAL):
    """Set one of the three goals (the group goal on a group holding the area); goals never carry a
    condition.
    """
    if kind == "carbon_goal_group":
        ds = ds.assign_coords(carbon_goal_group=["g"])
        ds["carbon_goal_group_areas"] = xr.DataArray(
            [[True]], coords=[ds.area, ds.carbon_goal_group]
        )
        ds["carbon_goal_group_max"] = xr.DataArray(
            [[value]], coords=[ds.carbon_goal_group, ds.year_op]
        )
    else:
        ds[kind] = xr.DataArray(value)
    return ds


def _tax_group(ds, rate=80.0):
    """A carbon tax group holding the area: its tax replaces the per-area one."""
    ds = ds.assign_coords(carbon_tax_group=["t"])
    ds["carbon_tax_group_areas"] = xr.DataArray([[True]], coords=[ds.area, ds.carbon_tax_group])
    ds["carbon_tax_group_rate"] = xr.DataArray([[rate]], coords=[ds.carbon_tax_group, ds.year_op])
    return ds


def _solve(ds):
    m = build_model(ds)
    if "operating_condition" in ds.coords:
        assert_condition_dims_consistent(m)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


def _emissions(m, condition):
    emissions = m.solution["operation_total_carbon_emissions"]
    return float(emissions.sel(operating_condition=condition).sum())


@pytest.mark.parametrize("kind", GOALS)
def test_carbon_goal_per_condition_and_on_expectation(kind):
    ds = _goal(_base(), kind)
    flag = f"{kind}_expected"
    # shedding is cheaper in a than in b
    shedding = [ds.load_shedding_cost * 0 + c for c in (1000.0, 2000.0)]

    # flag False: each condition is bound by the goal, both bind (the rest is shed)
    p = with_conditions(ds, [0.5, 0.5], load_shedding_cost=shedding)
    p[flag] = False
    m = _solve(p)
    assert _emissions(m, "a") == pytest.approx(GOAL)
    assert _emissions(m, "b") == pytest.approx(GOAL)

    # flag True: one goal on the weighted emissions; a gives up its emissions (cheaper shedding), b
    # stays above the goal
    p = with_conditions(ds, [0.5, 0.5], load_shedding_cost=shedding)
    p[flag] = True
    m = _solve(p)
    a, b = _emissions(m, "a"), _emissions(m, "b")
    assert 0.5 * a + 0.5 * b == pytest.approx(GOAL)
    assert a == pytest.approx(0.0, abs=1e-6)
    assert b == pytest.approx(EMISSIONS)
    assert b > GOAL


def _plain_objective(demand, group=False):
    ds = _base(demand)
    return _solve(_tax_group(ds) if group else ds).objective.value


@pytest.mark.parametrize("group", [False, True])
def test_carbon_tax_is_the_weighted_mean(group):
    # emissions differ per condition (16 t and 32 t): the tax, per area or per group, is paid on
    # the weighted mean
    ds = _base()
    p = with_conditions(ds, [0.25, 0.75], demand=[ds.demand, ds.demand * 2])
    if group:
        p = _tax_group(p)
    m = _solve(p)
    expected = 0.25 * _plain_objective(DEMAND, group) + 0.75 * _plain_objective(2 * DEMAND, group)
    assert m.objective.value == pytest.approx(expected, rel=1e-9)
    # one carbon cost per condition: the rate times that condition's emissions
    if group:
        name, rate = "operation_carbon_tax_group_costs", 80.0
    else:
        name, rate = "operation_carbon_costs", 50.0
    costs = m.solution[name].sum([d for d in m.solution[name].dims if d != "operating_condition"])
    assert float(costs.sel(operating_condition="a")) == pytest.approx(rate * EMISSIONS)
    assert float(costs.sel(operating_condition="b")) == pytest.approx(rate * 2 * EMISSIONS)


WITH_CONDITION = [
    "operation_carbon_emissions",
    "operation_total_carbon_emissions",
    "operation_carbon_costs",
    "operation_carbon_tax_group_costs",
    "operation_carbon_emissions_def",
    "operation_total_carbon_emissions_def",
    "operation_carbon_costs_def",
    "operation_carbon_tax_group_costs_def",
    "operation_carbon_goal_constraint",
    "operation_carbon_goal_area_constraint",
    "operation_carbon_goal_group_constraint",
]


def _everything(ds):
    """Every goal and a tax group, so that every carbon variable and constraint is built."""
    for kind in GOALS:
        ds = _goal(ds, kind)
    return _tax_group(ds)


def test_an_absent_pair_has_no_carbon_row():
    # weights [1, nan]: b is absent
    p = with_conditions(_everything(_base()), [1.0, np.nan])
    for kind in GOALS:
        p[f"{kind}_expected"] = False
    m = build_model(p)
    assert_condition_dims_consistent(m)
    checked = []
    for container in (m.variables, m.constraints):
        for name in container:
            if "carbon" not in name:
                continue
            labels = container[name].labels
            if "operating_condition" not in labels.dims:
                continue
            checked.append(name)
            assert (labels.sel(operating_condition="b") == -1).all(), name
            assert (labels.sel(operating_condition="a") != -1).any(), name
    assert set(WITH_CONDITION) <= set(checked)

    # flag True: the goals are on the expectation, with no condition dimension
    for kind in GOALS:
        p[f"{kind}_expected"] = True
    m = build_model(p)
    assert_condition_dims_consistent(m)
    for name in WITH_CONDITION[-3:]:
        labels = m.constraints[name].labels
        assert "operating_condition" not in labels.dims
        assert int((labels != -1).sum()) == 1


def test_a_weight_zero_condition_keeps_its_rows_and_costs_nothing():
    ds = _base()
    p = with_conditions(_everything(ds), [1.0, 0.0], demand=[ds.demand, ds.demand * 3])
    for kind in GOALS:
        p[f"{kind}_expected"] = False
    m = _solve(p)
    reference = _solve(_everything(_base()))
    assert m.objective.value == pytest.approx(reference.objective.value, rel=1e-9)
    for name in WITH_CONDITION:
        container = m.variables if name in m.variables else m.constraints
        assert (container[name].labels.sel(operating_condition="b") != -1).any(), name
