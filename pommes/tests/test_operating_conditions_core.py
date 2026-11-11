"""Operating conditions in the core: load shedding, spillage, adequacy, objective.

Only load shedding is active (one area, one year, every module off): the optimum is plain shedding of the demand,
unbounded, so the objective is linear in the demand and the weighted mean of two conditions is checkable by hand.
"""
import numpy as np
import pytest

from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import (
    assert_condition_dims_consistent,
    with_conditions,
)
from pommes.tests.dataset_nnodes import build_nnode_parameters
from pommes.tests.test_flexdem import _with_defaults

N_HOURS = 24


def _base():
    ds = build_nnode_parameters(n_areas=1, n_hours=N_HOURS)
    for module in ("conversion", "storage", "combined", "carbon"):
        ds[module] = False
    return _with_defaults(ds)


def _solve(ds):
    m = build_model(ds)
    if "operating_condition" in ds.coords:
        assert_condition_dims_consistent(m)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


def _cost(demand_scale):
    """Objective of the dataset without conditions, demand scaled."""
    ds = _base()
    ds["demand"] = ds.demand * demand_scale
    return _solve(ds).objective.value


def _two_conditions(weights, scales=(1.0, 2.0)):
    ds = _base()
    return with_conditions(ds, weights, demand=[ds.demand * s for s in scales])


def test_shedding_cost_is_the_weighted_mean():
    m = _solve(_two_conditions([0.6, 0.4]))
    expected = 0.6 * _cost(1.0) + 0.4 * _cost(2.0)
    assert m.objective.value == pytest.approx(expected, rel=1e-9)


def test_one_condition_of_weight_one_is_neutral():
    plain = _solve(_base())
    ds = _base()
    cond = _solve(with_conditions(ds, [1.0], demand=[ds.demand]))
    assert cond.objective.value == pytest.approx(plain.objective.value, rel=1e-9)
    a = cond.solution["operation_load_shedding_power"].sel(operating_condition="a", drop=True)
    b = plain.solution["operation_load_shedding_power"]
    np.testing.assert_allclose(a.values, b.values, rtol=1e-9, atol=1e-9)


def test_an_absent_pair_has_no_row():
    m = _solve(_two_conditions([1.0, np.nan]))
    labels = m.constraints["operation_adequacy_constraint"].labels
    assert (labels.sel(operating_condition="b") == -1).all()
    assert (labels.sel(operating_condition="a") != -1).all()
    assert m.objective.value == pytest.approx(_cost(1.0), rel=1e-9)


def test_weight_zero_costs_nothing():
    m = _solve(_two_conditions([1.0, 0.0], scales=(1.0, 3.0)))
    assert m.objective.value == pytest.approx(_cost(1.0), rel=1e-9)
    # kept with its constraints (weight 0 is not absent): rows exist and shed the whole demand at zero cost
    labels = m.constraints["operation_adequacy_constraint"].labels
    assert (labels.sel(operating_condition="b") != -1).all()
    shed = m.solution["operation_load_shedding_power"]
    shed_a = shed.sel(operating_condition="a").sum()
    shed_b = shed.sel(operating_condition="b").sum()
    assert float(shed_a) > 0
    assert float(shed_b) == pytest.approx(3 * float(shed_a), rel=1e-9)
