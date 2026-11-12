"""Carbon on block groups and coarse imports (multi-resolution, sections 6.7, 15.1, 16.2)."""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.model.combined import combined_groups
from pommes.model.conversion import conversion_groups
from pommes.tests.time_blocks_helper import (
    CARBON_BOILERS,
    CARBON_HEATER,
    CARBON_KILNS,
    KILNS,
    assert_twin_equal,
    coarse_step_dataset,
    coarse_step_reference,
    combined_all_block_carbon_dataset,
    combined_carbon_dataset,
    combined_toy,
    conversion_carbon_dataset,
    conversion_toy,
    flat_twin,
    net_import_toy,
    solve_model,
    solved,
    taxed,
    toy_n,
    with_blocks,
    with_carbon,
)
from pommes.tests.conditions_helper import with_conditions
from pommes.utils import time_blocks

EMISSIONS = "operation_carbon_emissions"
# the builder of each case and the block lengths its flat twin repeats
CASES = {
    "conversion": (conversion_carbon_dataset, {"conversion": {t: 4 for t in CARBON_KILNS if t.startswith("kiln")}}),
    "combined": (combined_carbon_dataset, {"combined": {"boiler_block": 2}}),
}


def goaled(p, goal):
    return p.assign(carbon_goal=np.array(goal))


def hourly_emissions(m):
    return solved(m, EMISSIONS)


def assert_emissions_equal(m, twin):
    a, b = hourly_emissions(m), hourly_emissions(twin)
    np.testing.assert_allclose(a.values, b.transpose(*a.dims).values, atol=1e-6)


def yearly(m, p):
    return float(toy_n(p) * hourly_emissions(m).sum())


@pytest.mark.parametrize("case", CASES)
def test_block_tech_hourly_emissions_equal_twin(case):
    make, lengths = CASES[case]
    p = taxed(make())
    m = solve_model(build_model(p))
    twin = solve_model(flat_twin(p, **lengths))
    assert_twin_equal(m, twin, time_blocks(p), [])
    assert_emissions_equal(m, twin)
    assert hourly_emissions(m).sum() > 0.1  # something is emitted


@pytest.mark.parametrize("case", CASES)
def test_carbon_tax_and_goal_with_blocks_equal_twin(case):
    make, lengths = CASES[case]
    free_p = make()
    free = yearly(solve_model(build_model(free_p)), free_p)
    # binding goal: 80 % of what the cost-optimal dispatch emits
    p = goaled(free_p, 0.8 * free)
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, **lengths))
    assert abs(m.constraints["operation_carbon_goal_constraint"].dual.values).max() > 1e-9
    assert_twin_equal(m, twin, time_blocks(p), [])
    assert_emissions_equal(m, twin)
    assert yearly(m, p) == pytest.approx(0.8 * free, rel=1e-7)
    # binding tax: the emissions fall below the free ones
    p = taxed(free_p)
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, **lengths))
    assert yearly(m, p) < free - 1e-6
    assert_twin_equal(m, twin, time_blocks(p), [])
    assert_emissions_equal(m, twin)


def test_carbon_with_an_all_block_module():
    """No hourly ``operation_combined_power``: carbon skips it, builds, and equals the twin."""
    p = combined_all_block_carbon_dataset()
    lengths = {t: 2 for t in CARBON_BOILERS}
    m = solve_model(build_model(p))
    assert "operation_combined_power" not in m.variables
    twin = solve_model(flat_twin(p, combined=lengths))
    assert_twin_equal(m, twin, time_blocks(p), ["operation_combined_block_2h_power"])
    assert_emissions_equal(m, twin)
    assert hourly_emissions(m).sum() > 0.1


def test_block_net_import_emissions_hand_toy():
    """
    Section 16.2 D: the block factor is the step-weighted mean of the hourly ones.

    Steps (1, 3) h, 2 MWh imported in the block of 4 h. Emission factor (0, 1): block factor 0.75, rows 0.375 and 1.125 t,
    yearly total 1.5 t. Total emission factor (0.5, 2): block factor 1.625, rows 0.8125 and 2.4375 t. The two factors differ,
    so that a row reading the wrong one fails.
    """
    p = with_blocks(
        net_import_toy([1.0, 3.0], [2, 0], [5, 5], factor=[0.0, 1.0], total_factor=[0.5, 2.0]).assign(
            carbon=np.array(True)
        ),
        resource={"steel": 4},
    )
    m = solve_model(build_model(p))
    rows = hourly_emissions(m).values.ravel()
    np.testing.assert_allclose(rows, [0.375, 1.125], atol=1e-9)
    assert rows.sum() == pytest.approx(1.5)
    total = solved(m, "operation_total_carbon_emissions").values.ravel()
    np.testing.assert_allclose(total, [0.8125, 2.4375], atol=1e-9)


UNEVEN = [1.0, 2.0] * 4  # 12 h: with blocks of 3 h each block holds a step of 1 h and a step of 2 h


@pytest.mark.parametrize("module", ["conversion", "combined"])
def test_block_emissions_are_apportioned_by_step_duration(module):
    """
    Uneven steps inside a block: its tCO2 is shared among the steps in proportion to their duration.

    The twin (flat power in the block) emits ``power x factor x step`` in each step, so an equal share per step
    would differ from its rows.
    """
    if module == "conversion":
        p = with_carbon(conversion_toy(UNEVEN), conversion=CARBON_KILNS)
        lengths = {"conversion": {k: 3 for k in KILNS}}
    else:
        p = with_carbon(combined_toy(UNEVEN), conversion=CARBON_HEATER, combined=CARBON_BOILERS)
        lengths = {"combined": {"boiler_block": 3}}
    p = taxed(with_blocks(p, **lengths))
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, **lengths))
    assert_twin_equal(m, twin, time_blocks(p, warn=False), [])
    assert_emissions_equal(m, twin)
    assert hourly_emissions(m).sum() > 0.1


def test_carbon_tax_group_and_conditions_with_a_block_technology_equal_twin():
    """A tax group (the per-area tax of its members is masked) and two operating conditions on a block technology."""
    base = with_carbon(combined_toy([1.0] * 10), conversion=CARBON_HEATER, combined=CARBON_BOILERS)
    p = with_conditions(base, [0.6, 0.4], demand=[base.demand, base.demand * 1.5])
    p = p.assign_coords(carbon_tax_group=["t"])
    p["carbon_tax_group_areas"] = xr.DataArray([[True], [True]], coords=[p.area, p.carbon_tax_group])
    p["carbon_tax_group_rate"] = xr.DataArray([[20.0]], coords=[p.carbon_tax_group, p.year_op])
    lengths = {"combined": {"boiler_block": 2}}
    p = taxed(with_blocks(p, **lengths))
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, **lengths))
    assert_twin_equal(m, twin, time_blocks(p, warn=False), [])
    assert_emissions_equal(m, twin)
    group = "operation_carbon_tax_group_costs"
    got, want = solved(m, group), solved(twin, group)
    assert float(got.sum()) > 1e-6  # the group pays
    np.testing.assert_allclose(got.values, want.transpose(*got.dims).values, atol=1e-6)
    # the two conditions emit differently: the conditions are not mixed
    per_condition = hourly_emissions(m).sum("hour")
    assert abs(float(per_condition.isel(operating_condition=0).sum() - per_condition.isel(operating_condition=1).sum())) > 1e-6


@pytest.mark.parametrize("L, horizon", [(4, 24), (6, 24)])
def test_coarse_net_import_carbon_tax_equals_reference(L, horizon):
    """Constant-in-block factors, a binding tax: the yearly totals and the objective equal steps of L hours."""
    p = coarse_step_dataset(L, horizon, net_import=True, carbon=True)
    m = solve_model(build_model(p))
    ref = solve_model(build_model(coarse_step_reference(p, L)))
    assert m.objective.value == pytest.approx(ref.objective.value, rel=1e-7)
    for name in (EMISSIONS, "operation_total_carbon_emissions"):
        got, want = solved(m, name).sum("hour"), solved(ref, name).sum("hour")
        np.testing.assert_allclose(got.values, want.transpose(*got.dims).values, atol=1e-6, err_msg=name)
    assert float(solved(m, "operation_carbon_costs").sum()) > 1e-3  # the tax is paid
    free = coarse_step_dataset(L, horizon, net_import=True, carbon=True).assign(carbon_tax=np.array(0.0))
    assert yearly(m, p) < yearly(solve_model(build_model(free)), free) - 1e-6  # and it binds


def test_t0_carbon_gathers_the_same_emission_factor(parameters):
    """``keep`` changes the sparse T0 path: the emission factor is today's gather (dense: ``p`` itself)."""
    p = parameters
    # the unread conversion parameters that ``keep`` drops before the gather include per-hour ones (C3)
    assert any("hour" in p[n].dims for n in p.data_vars if n.startswith("conversion_") and n != "conversion_emission_factor")
    tb = time_blocks(p)
    [(L, p_g, idx_g)] = conversion_groups(p, tb, keep=["conversion_emission_factor"])
    [(Lc, pc_g, idxc_g)] = combined_groups(p, tb, keep=["combined_emission_factor"])
    assert L is None and Lc is None
    if idx_g is None:  # dense
        assert p_g is p and pc_g is p
        return
    from pommes.model.combined import combined_active_mask
    from pommes.model.conversion import conversion_active_mask
    from pommes.model.sparsity import to_sparse_units

    for module, got, got_idx, mask in (
        ("conversion", p_g, idx_g, conversion_active_mask(p)),
        ("combined", pc_g, idxc_g, combined_active_mask(p)),
    ):
        want, want_idx = to_sparse_units(p, "area", f"{module}_tech", mask, var_prefix=f"{module}_")
        name = f"{module}_emission_factor"
        xr.testing.assert_identical(got[name], want[name])
        assert got_idx.equals(want_idx), module  # the same units, in the same order
