"""Carbon on block groups and coarse imports (multi-resolution, sections 6.7, 15.1, 16.2)."""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.model.combined import combined_groups
from pommes.model.conversion import conversion_groups
from pommes.tests.time_blocks_helper import (
    assert_twin_equal,
    coarse_step_dataset,
    coarse_step_reference,
    combined_t2_dataset,
    combined_toy,
    conversion_t2_dataset,
    flat_twin,
    net_import_toy,
    solve_model,
    solved,
    toy_n,
    with_blocks,
    with_carbon,
)
from pommes.utils import time_blocks

EMISSIONS = "operation_carbon_emissions"
KILNS = {"well": 0.1, "backup": 0.9, "kiln_a": 0.3, "kiln_b": 0.5, "kiln_c": 0.2, "kiln_d": 0.4}
BOILERS = {"electric_boiler": [0.1, 0.1], "boiler_block": [0.5, 0.2], "steady_kiln": [0.3, 0.3]}
HEATER = {"well": 0.3, "heater": 0.8}
TAX = 15.0


def conversion_blocks(**kw):
    return with_carbon(conversion_t2_dataset(), conversion=KILNS, **kw), {"conversion": {t: 4 for t in KILNS if t.startswith("kiln")}}


def combined_blocks(**kw):
    return with_carbon(combined_t2_dataset(), conversion=HEATER, combined=BOILERS, **kw), {"combined": {"boiler_block": 2}}


CASES = {"conversion": conversion_blocks, "combined": combined_blocks}


def hourly_emissions(m):
    return solved(m, EMISSIONS)


def assert_emissions_equal(m, twin):
    a, b = hourly_emissions(m), hourly_emissions(twin)
    np.testing.assert_allclose(a.values, b.transpose(*a.dims).values, atol=1e-6)


def yearly(m, p):
    return float(toy_n(p) * hourly_emissions(m).sum())


@pytest.mark.parametrize("case", CASES)
def test_block_tech_hourly_emissions_equal_twin(case):
    p, lengths = CASES[case](tax=TAX)
    m = solve_model(build_model(p))
    twin = solve_model(flat_twin(p, **lengths))
    assert_twin_equal(m, twin, time_blocks(p), [])
    assert_emissions_equal(m, twin)
    assert hourly_emissions(m).sum() > 0.1  # something is emitted


@pytest.mark.parametrize("case", CASES)
def test_carbon_tax_and_goal_with_blocks_equal_twin(case):
    free_p, lengths = CASES[case]()
    free = yearly(solve_model(build_model(free_p)), free_p)
    # binding goal: 80 % of what the cost-optimal dispatch emits
    p, _ = CASES[case](goal=0.8 * free)
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, **lengths))
    assert abs(m.constraints["operation_carbon_goal_constraint"].dual.values).max() > 1e-9
    assert_twin_equal(m, twin, time_blocks(p), [])
    assert_emissions_equal(m, twin)
    assert yearly(m, p) == pytest.approx(0.8 * free, rel=1e-7)
    # binding tax: the emissions fall below the free ones
    p, _ = CASES[case](tax=TAX)
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, **lengths))
    assert yearly(m, p) < free - 1e-6
    assert_twin_equal(m, twin, time_blocks(p), [])
    assert_emissions_equal(m, twin)


def test_carbon_with_an_all_block_module():
    """No hourly ``operation_combined_power``: carbon skips it, builds, and equals the twin."""
    lengths = {t: 2 for t in BOILERS}
    p = with_blocks(with_carbon(combined_toy([1.0] * 10), conversion=HEATER, combined=BOILERS, tax=TAX), combined=lengths)
    m = solve_model(build_model(p))
    assert "operation_combined_power" not in m.variables
    twin = solve_model(flat_twin(p, combined=lengths))
    assert_twin_equal(m, twin, time_blocks(p), ["operation_combined_block_2h_power"])
    assert_emissions_equal(m, twin)
    assert hourly_emissions(m).sum() > 0.1


def test_block_net_import_emissions_hand_toy():
    """Section 16.2 D: block factor 0.75, hourly rows 0.375 and 1.125 t, yearly total 1.5 t."""
    p = with_blocks(
        net_import_toy([1.0, 3.0], [2, 0], [5, 5], factor=[0.0, 1.0]).assign(carbon=np.array(True)),
        resource={"steel": 4},
    )
    m = solve_model(build_model(p))
    rows = hourly_emissions(m).values.ravel()
    np.testing.assert_allclose(rows, [0.375, 1.125], atol=1e-9)
    assert rows.sum() == pytest.approx(1.5)
    total = solved(m, "operation_total_carbon_emissions").values.ravel()
    np.testing.assert_allclose(total, [0.375, 1.125], atol=1e-9)


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


def test_t0_carbon_gathers_the_same_emission_factor():
    """``keep`` changes the sparse T0 path: the emission factor is today's gather (dense: ``p`` itself)."""
    p = with_carbon(combined_toy([1.0] * 4), conversion=HEATER, combined=BOILERS)
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

    for module, got, mask in (("conversion", p_g, conversion_active_mask(p)), ("combined", pc_g, combined_active_mask(p))):
        want, _ = to_sparse_units(p, "area", f"{module}_tech", mask, var_prefix=f"{module}_")
        name = f"{module}_emission_factor"
        xr.testing.assert_identical(got[name], want[name])
