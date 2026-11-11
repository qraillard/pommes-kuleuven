"""Operating conditions in the net import module.

One area, three hours, electricity only: wind is switched off, so the demand is covered by imports (or shed at a high
cost) and every optimum is checkable by hand.
"""
import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import (
    assert_condition_dims_consistent,
    with_conditions,
)

HOURS = [3, 4, 5]
DEMAND = 10.0  # per hour: 30 MWh over the year


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
    p["demand"] = p.demand * 0 + DEMAND
    p["load_shedding_cost"] = xr.DataArray(1000.0)
    p["load_shedding_max_capacity"] = xr.DataArray(np.nan)
    p["net_import_import_price"] = p.net_import_import_price * 0 + 10.0
    p["net_import_export_price"] = p.net_import_export_price * 0
    p["net_import_max_yearly_energy_import"] = p.net_import_max_yearly_energy_import * 0 + 1e3
    p["net_import_max_yearly_energy_export"] = p.net_import_max_yearly_energy_export * 0 + 1e3
    p["carbon"] = False
    return p


def _per_resource(p, value):
    return xr.DataArray([float(value)], dims="resource", coords=dict(resource=p.resource))


def _solve(p):
    m = build_model(p)
    if "operating_condition" in p.coords:
        assert_condition_dims_consistent(m)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


def _energy(m, name, condition):
    return float(m.solution[name].sel(operating_condition=condition).sum())


def test_import_price_per_condition(p_base):
    prices = [_per_resource(p_base, 10.0), _per_resource(p_base, 30.0)]
    p = with_conditions(p_base, [0.25, 0.75], net_import_import_price=prices)
    m = _solve(p)
    # the whole demand is imported in both conditions: 30 MWh at the weighted mean price
    assert _energy(m, "operation_net_import_import", "a") == pytest.approx(30.0)
    assert _energy(m, "operation_net_import_import", "b") == pytest.approx(30.0)
    assert m.objective.value == pytest.approx(30.0 * (0.25 * 10 + 0.75 * 30))
    # one import cost per condition
    costs = m.solution["operation_net_import_costs"].sum(["area", "resource", "year_op"])
    assert float(costs.sel(operating_condition="a")) == pytest.approx(300.0)
    assert float(costs.sel(operating_condition="b")) == pytest.approx(900.0)


def test_yearly_import_ceiling_per_condition_and_on_expectation(p_base):
    name = "operation_net_import_import"
    # flag False: each condition has its own ceiling, both bind (the rest is shed)
    ceilings = [_per_resource(p_base, 10.0), _per_resource(p_base, 20.0)]
    p = with_conditions(p_base, [0.5, 0.5], net_import_max_yearly_energy_import=ceilings)
    p["net_import_max_yearly_energy_import_expected"] = False
    m = _solve(p)
    assert _energy(m, name, "a") == pytest.approx(10.0)
    assert _energy(m, name, "b") == pytest.approx(20.0)

    # flag True: one ceiling of 15 on the expectation; importing is cheaper in a, so a takes it all
    prices = [_per_resource(p_base, 10.0), _per_resource(p_base, 30.0)]
    p = with_conditions(p_base, [0.5, 0.5], net_import_import_price=prices)
    p["net_import_max_yearly_energy_import"] = _per_resource(p, 15.0)
    p["net_import_max_yearly_energy_import_expected"] = True
    m = _solve(p)
    a, b = (_energy(m, name, c) for c in "ab")
    assert 0.5 * a + 0.5 * b == pytest.approx(15.0)
    assert a == pytest.approx(30.0)
    assert b == pytest.approx(0.0)


def test_yearly_export_ceiling_per_condition_and_on_expectation(p_base):
    name = "operation_net_import_export"
    # exporting pays more than importing costs: the export ceilings bind
    p_base["net_import_export_price"] = p_base.net_import_export_price + 20.0
    ceilings = [_per_resource(p_base, 6.0), _per_resource(p_base, 9.0)]
    p = with_conditions(p_base, [0.5, 0.5], net_import_max_yearly_energy_export=ceilings)
    p["net_import_max_yearly_energy_export_expected"] = False
    m = _solve(p)
    assert _energy(m, name, "a") == pytest.approx(6.0)
    assert _energy(m, name, "b") == pytest.approx(9.0)

    # flag True: one ceiling of 8 on the expectation; b pays more, so b exports it all
    prices = [_per_resource(p_base, 20.0), _per_resource(p_base, 40.0)]
    p = with_conditions(p_base, [0.5, 0.5], net_import_export_price=prices)
    p["net_import_max_yearly_energy_export"] = _per_resource(p, 8.0)
    p["net_import_max_yearly_energy_export_expected"] = True
    m = _solve(p)
    a, b = (_energy(m, name, c) for c in "ab")
    assert 0.5 * a + 0.5 * b == pytest.approx(8.0)
    assert a == pytest.approx(0.0)
    assert b == pytest.approx(16.0)


WITH_CONDITION = [
    "operation_net_import_import",
    "operation_net_import_export",
    "operation_net_import_abs",
    "operation_net_import_net_generation",
    "operation_net_import_costs",
    "operation_net_import_abs_def",
    "operation_net_import_net_generation_def",
    "operation_net_import_costs_def",
    "operation_net_import_import_yearly_max_constraint",
    "operation_net_import_export_yearly_max_constraint",
]


def test_an_absent_pair_has_no_net_import_row(p_base):
    # weights [1, nan]: b is absent
    p = with_conditions(p_base, [1.0, np.nan])
    p["net_import_export_price"] = p.net_import_export_price + 20.0
    p["net_import_max_yearly_energy_import_expected"] = False
    p["net_import_max_yearly_energy_export_expected"] = False
    m = build_model(p)
    assert_condition_dims_consistent(m)
    checked = []
    for container in (m.variables, m.constraints):
        for name in container:
            if "net_import" not in name:
                continue
            labels = container[name].labels
            if "operating_condition" not in labels.dims:
                continue
            checked.append(name)
            assert (labels.sel(operating_condition="b") == -1).all(), name
            assert (labels.sel(operating_condition="a") != -1).any(), name
    assert set(WITH_CONDITION) <= set(checked)

    # flag True: the ceilings are on the expectation, with no condition dimension
    for flag in ("import", "export"):
        p[f"net_import_max_yearly_energy_{flag}_expected"] = True
    m = build_model(p)
    for flag in ("import", "export"):
        labels = m.constraints[f"operation_net_import_{flag}_yearly_max_constraint"].labels
        assert "operating_condition" not in labels.dims
        assert int((labels != -1).sum()) == 1


def test_a_weight_zero_condition_keeps_its_rows_and_costs_nothing(p_base):
    prices = [_per_resource(p_base, 10.0), _per_resource(p_base, 1e4)]
    p = with_conditions(p_base, [1.0, 0.0], net_import_import_price=prices)
    m = _solve(p)
    assert m.objective.value == pytest.approx(30.0 * 10.0)
    for name in WITH_CONDITION:
        container = m.variables if name in m.variables else m.constraints
        assert (container[name].labels.sel(operating_condition="b") != -1).any(), name
