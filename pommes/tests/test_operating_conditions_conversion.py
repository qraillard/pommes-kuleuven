"""Operating conditions in the conversion module.

One area, four hours, electricity only, no shedding allowed unless stated: the demand is covered by `ocgt`
(45 per MWh) and `hydro_ror` (free, nearly free capacity) so every optimum is checkable by hand.
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
DEMAND = 10.0  # per hour: 40 MWh over the year


def _base(years=1, hydro_variable_cost=0.0):
    ds = build_nnode_parameters(n_areas=1, n_hours=N_HOURS)
    ds = ds.sel(conversion_tech=["ocgt", "hydro_ror"], resource=["electricity"])
    for module in ("storage", "combined", "carbon"):
        ds[module] = False
    ds = _with_defaults(ds)
    if years == 2:
        ds = ds.reindex(year_op=[2030, 2040], method="ffill")
    ds["demand"] = ds.demand * 0 + DEMAND
    ds["load_shedding_max_capacity"] = ("resource", [0.0])  # no shedding
    ds["conversion_power_capacity_max"] = ds.conversion_power_capacity_max * 0 + 200.0
    ds["conversion_variable_cost"] = ("conversion_tech", [45.0, hydro_variable_cost])
    ds["conversion_fixed_cost"] = ("conversion_tech", [0.0, 0.0])
    ds["conversion_annuity_cost"] = ds.conversion_annuity_cost * xr.DataArray(
        [1.0, 1e-3], coords=dict(conversion_tech=ds.conversion_tech)
    )
    return ds


def _ceiling(ds, ocgt=np.nan, hydro=np.nan):
    return xr.DataArray(
        [[[ocgt] * ds.sizes["year_op"], [hydro] * ds.sizes["year_op"]]],
        dims=("area", "conversion_tech", "year_op"),
        coords=dict(
            area=ds.area, conversion_tech=ds.conversion_tech, year_op=ds.year_op
        ),
    )


def _solve(ds):
    m = build_model(ds)
    if "operating_condition" in ds.coords:
        assert_condition_dims_consistent(m)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


def _of(da, tech):
    """The part of a solution variable of one technology (dense: a dimension; sparse: units)."""
    if "conversion_tech_unit" in da.dims:
        unit_tech = da["_conversion_tech_unit_conversion_tech"]
        return da.where(unit_tech == tech, drop=True).sum("conversion_tech_unit")
    return da.sel(conversion_tech=tech)


def _power(m, tech):
    return _of(m.solution["operation_conversion_power"], tech)


def _capacity(m, tech):
    return float(_of(m.solution["planning_conversion_power_capacity"], tech).sum())


def _energy(m, tech, **sel):
    return _power(m, tech).sel(**sel).sum("hour").squeeze()


def test_capacity_is_shared_and_covers_the_larger_demand():
    ds = _base().sel(conversion_tech=["ocgt"])
    ds["conversion_availability"] = ds.conversion_availability.fillna(0.5)
    m = _solve(
        with_conditions(
            ds, [0.5, 0.5], demand=[ds.demand * 1.0, ds.demand * 3.0]
        )
    )
    capacity = m.solution["planning_conversion_power_capacity"]
    assert "operating_condition" not in capacity.dims
    assert _capacity(m, "ocgt") == pytest.approx(3 * DEMAND / 0.5)


def test_yearly_ceiling_per_condition_and_on_expectation():
    # flag False: each condition has its own ceiling, both bind (hydro is free, ocgt dear)
    ds = _base()
    ceilings = [_ceiling(ds, hydro=12.0), _ceiling(ds, hydro=28.0)]
    ds = with_conditions(ds, [0.5, 0.5], conversion_max_yearly_production=ceilings)
    ds["conversion_max_yearly_production_expected"] = False
    m = _solve(ds)
    dry = float(_energy(m, "hydro_ror", operating_condition="a"))
    wet = float(_energy(m, "hydro_ror", operating_condition="b"))
    assert dry == pytest.approx(12.0)
    assert wet == pytest.approx(28.0)

    # flag True: one ceiling on the expectation; ocgt is unavailable in b, so hydro must
    # produce 40 there and 0 in a: the weighted production is 20, one condition is above it
    ds = _base()
    availability = ds.conversion_availability.fillna(1.0)
    off = availability.where(availability.conversion_tech != "ocgt", 0.0)
    ds = with_conditions(
        ds, [0.5, 0.5], conversion_availability=[availability, off]
    )
    ds["conversion_max_yearly_production"] = _ceiling(ds, hydro=20.0)
    ds["conversion_max_yearly_production_expected"] = True
    m = _solve(ds)
    dry = float(_energy(m, "hydro_ror", operating_condition="a"))
    wet = float(_energy(m, "hydro_ror", operating_condition="b"))
    assert 0.5 * dry + 0.5 * wet == pytest.approx(20.0)
    assert wet == pytest.approx(40.0)
    assert dry == pytest.approx(0.0)


def test_cumulative_production_uses_the_expectation():
    ds = _base(years=2)
    ds["operation_year_weight"] = ("year_op", [1.0, 1.0])
    ds["conversion_max_cumulative_production"] = _ceiling(ds, hydro=50.0).isel(
        year_op=0, drop=True
    )
    demand = ds.demand
    ds = with_conditions(ds, [0.5, 0.5], demand=[demand * 1.0, demand * 0.5])
    m = _solve(ds)
    hydro = _power(m, "hydro_ror")
    per_year = (hydro * xr.DataArray([0.5, 0.5], dims="operating_condition")).sum(
        "operating_condition"
    )
    assert float(per_year.sum()) == pytest.approx(50.0)


def test_weight_zero_without_shedding_still_sets_the_capacity():
    ds = _base().sel(conversion_tech=["ocgt"])
    m = _solve(
        with_conditions(ds, [1.0, 0.0], demand=[ds.demand * 1.0, ds.demand * 3.0])
    )
    assert _capacity(m, "ocgt") == pytest.approx(3 * DEMAND)
    # the objective counts a's operation only: the cost of the capacity for b's demand plus a's fuel
    reference_b = _solve(_with_scaled_demand(ds, 3.0))
    fuel_b = 45.0 * 3 * DEMAND * N_HOURS
    fuel_a = 45.0 * DEMAND * N_HOURS
    assert m.objective.value == pytest.approx(
        reference_b.objective.value - fuel_b + fuel_a, rel=1e-9
    )
    assert _energy(m, "ocgt", operating_condition="a") == pytest.approx(
        DEMAND * N_HOURS
    )


def _with_scaled_demand(ds, scale):
    ds = ds.copy()
    ds["demand"] = ds.demand * scale
    return ds


def test_sparse_mask_keeps_a_pair_pinned_in_one_condition_only():
    ds = _base(hydro_variable_cost=60.0)
    ceilings = [_ceiling(ds, ocgt=0.0), _ceiling(ds)]
    ds = with_conditions(ds, [0.5, 0.5], conversion_max_yearly_production=ceilings)
    ds["conversion_max_yearly_production_expected"] = False
    m = _solve(ds)
    assert float(_energy(m, "ocgt", operating_condition="a")) == pytest.approx(0.0)
    assert float(_energy(m, "ocgt", operating_condition="b")) > 0
    assert _capacity(m, "ocgt") > 0


def test_an_absent_pair_has_no_conversion_row():
    # weights [1, nan]: b is absent, so nothing conversion-related may exist for it
    ds = _base()
    for name, value in (
        ("conversion_max_capacity_factor", 0.8),
        ("conversion_ramp_up", 0.5),
        ("conversion_ramp_down", 0.5),
        ("conversion_must_run", 0.1),
        ("conversion_max_daily_production", 100.0),
    ):
        ds[name] = xr.full_like(ds.conversion_power_capacity_max, value)
    ds["conversion_max_yearly_production"] = _ceiling(ds, hydro=30.0)
    ds["conversion_max_yearly_production_expected"] = False
    ds = with_conditions(ds, [1.0, np.nan])
    m = build_model(ds)
    assert_condition_dims_consistent(m)
    checked = []
    for container in (m.variables, m.constraints):
        for name in container:
            if "conversion" not in name:
                continue
            labels = container[name].labels
            if "operating_condition" not in labels.dims:
                continue
            checked.append(name)
            assert (labels.sel(operating_condition="b") == -1).all(), name
            if name in MUST_HAVE_ROWS_IN_A:
                assert (labels.sel(operating_condition="a") != -1).any(), name
    assert set(MUST_HAVE_ROWS_IN_A) <= set(checked)

    # flag True: the ceiling is on the expectation, one row per year, with no condition dimension
    ds["conversion_max_yearly_production_expected"] = True
    m = build_model(ds)
    labels = m.constraints["operation_conversion_yearly_production_max_constraint"].labels
    assert "operating_condition" not in labels.dims
    assert int((labels != -1).sum()) == 1  # hydro, the one pair with a ceiling, in the one year


MUST_HAVE_ROWS_IN_A = [
    "operation_conversion_power",
    "operation_conversion_net_generation",
    "operation_conversion_costs",
    "operation_conversion_power_max_constraint",
    "operation_conversion_must_run_constraint_inequality",
    "operation_conversion_ramp_up_constraint",
    "operation_conversion_ramp_down_constraint",
    "operation_conversion_yearly_production_max_constraint",
    "operation_conversion_capacity_factor_max_constraint",
    "operation_conversion_daily_production_max_constraint",
    "operation_conversion_net_generation_def",
    "operation_conversion_costs_def",
]
