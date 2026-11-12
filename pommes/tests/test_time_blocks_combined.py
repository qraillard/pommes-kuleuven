"""Combined technologies on time blocks (multi-resolution, sections 5, 6.2, 6.3, 15)."""

import numpy as np
import pandas as pd
import pytest

from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import with_conditions
from pommes.tests.time_blocks_helper import (
    BLOCK_PARAMETERS,
    assert_twin_equal,
    combined_derived_coarse_dataset,
    combined_t1_dataset,
    combined_t2_dataset,
    combined_toy,
    flat_twin,
    pinned_module_dataset,
    solve_model,
    solved,
    to_hours,
    units_of,
    variables_in,
    with_blocks,
)
from pommes.model.combined import combined_groups
from pommes.utils import check_group_coord, check_partition, time_blocks

BLOCK = "operation_combined_block_2h_"
ALL = ["electric_boiler", "boiler_block", "steady_kiln"]


def tb_of(p):
    return time_blocks(p)


def test_t0_combined_group_is_the_input():
    p = combined_toy([1.0] * 4)
    [(L, p_g, idx_g)] = combined_groups(p, tb_of(p))
    assert L is None
    if idx_g is None:  # dense: the dataset itself
        assert p_g is p
    else:  # sparse: today's to_sparse_units result
        import xarray as xr

        from pommes.model.combined import combined_active_mask
        from pommes.model.sparsity import to_sparse_units

        expected, index = to_sparse_units(
            p, "area", "combined_tech", combined_active_mask(p), var_prefix="combined_"
        )
        xr.testing.assert_identical(p_g, expected)
        assert idx_g.equals(index)


def test_combined_block_names_and_dims():
    m = build_model(combined_t2_dataset())
    power = m.variables[f"{BLOCK}power"]
    assert set(power.labels.dims) - {"area"} == {
        next(d for d in power.labels.dims if d.startswith("combined_tech")),
        "mode",
        "time_block_2h",
        "year_op",
    }
    assert {u[1] for u in units_of(power)} == {"boiler_block"}
    assert {u[1] for u in units_of(m.variables["operation_combined_power"])} == {
        "electric_boiler",
        "steady_kiln",
    }
    for name in (
        f"{BLOCK}ramp_up_constraint",
        f"{BLOCK}ramp_down_constraint",
        f"{BLOCK}mode_ramp_up_constraint_m1_m2",
        f"{BLOCK}mode_ramp_down_constraint_m1_m2",
        f"{BLOCK}power_max_constraint",
        f"{BLOCK}net_generation_def",
        f"{BLOCK}costs_def",
    ):
        assert name in m.constraints, name
    assert "operation_combined_block_2h_net_generation" in m.variables


def _values(m, name, tb=None):
    da = solved(m, name)
    return to_hours(da, tb) if tb is not None else da


def test_combined_t1_block_equal_to_step():
    p = combined_t1_dataset()
    hourly = solve_model(build_model(p.drop_vars(BLOCK_PARAMETERS)))
    m = solve_model(build_model(p))
    assert "operation_combined_power" not in m.variables  # every combined tech is on 1 h blocks
    assert m.objective.value == pytest.approx(hourly.objective.value, rel=1e-9)
    block = _values(m, "operation_combined_block_1h_power", tb_of(p))
    np.testing.assert_allclose(
        block.values, solved(hourly, "operation_combined_power").transpose(*block.dims).values, atol=1e-6
    )
    for name in ("operation_combined_power_capacity", "planning_combined_power_capacity"):
        np.testing.assert_allclose(solved(m, name).values, solved(hourly, name).values, atol=1e-6)


@pytest.mark.parametrize(
    "dt,L",
    [
        ([1.0] * 10, 2),
        ([1.0, 1.0, 2.0] * 3, 4),
        ([2.0, 1.0, 1.0] * 3, 4),  # first step 2 h: not D / L
        ([1.0] * 10, 4),
    ],
    ids=["L2_on_1h", "L4_on_uneven_steps", "L4_first_step_2h", "short_last_block"],
)
def test_combined_t2_flat_twin(dt, L):
    p = (
        combined_t2_dataset()
        if (dt, L) == ([1.0] * 10, 2)
        else with_blocks(combined_toy(dt), combined={"boiler_block": L})
    )
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, combined={"boiler_block": L}))
    hourly = solve_model(build_model(p.drop_vars(BLOCK_PARAMETERS)))
    assert m.objective.value > hourly.objective.value + 1.0  # the flat constraint really binds
    assert_twin_equal(m, twin, tb_of(p), [f"operation_combined_block_{L}h_power"])


def test_combined_short_last_block():
    p = with_blocks(combined_toy([1.0] * 10), combined={"boiler_block": 4})
    tb = tb_of(p)
    assert tb.duration[4].values.tolist() == [4.0, 4.0, 2.0]
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, combined={"boiler_block": 4}))
    assert_twin_equal(m, twin, tb, ["operation_combined_block_4h_power"])


def test_combined_all_block_has_no_hourly_variable():
    lengths = {t: 2 for t in ALL}
    p = with_blocks(combined_toy([1.0] * 10), combined=lengths)
    m = solve_model(build_model(p))
    assert "operation_combined_power" not in m.variables
    assert "operation_combined_net_generation" not in m.variables
    assert "operation_combined_costs_def" not in m.constraints
    assert {u[1] for u in units_of(m.variables[f"{BLOCK}power"])} == set(ALL)
    # gas and heat keep an hourly contributor, so the twin applies
    assert_twin_equal(m, solve_model(flat_twin(p, combined=lengths)), tb_of(p), [f"{BLOCK}power"])


def test_costs_rows_cover_each_unit_once():
    m = build_model(combined_t2_dataset())
    rows = [
        pair
        for name in ("operation_combined_costs_def", f"{BLOCK}costs_def")
        for pair in units_of(m.constraints[name])
    ]
    assert sorted(rows) == sorted(units_of(m.variables["operation_combined_costs"]))
    assert len(rows) == len(set(rows))


def test_gates_read_the_group():
    # constant operation only on the hourly tech (steady_kiln): the block group has no such row
    m = build_model(combined_t2_dataset())
    assert "operation_combined_constant_operation_constraint" in m.constraints
    assert not [n for n in m.constraints if n.startswith(f"{BLOCK}constant_operation")]
    assert not [n for n in m.constraints if n.startswith(f"{BLOCK}mode_constant_operation")]
    # and the other way round: the constant tech on blocks, the hourly group has no such row
    p = with_blocks(combined_toy([1.0] * 10), combined={"steady_kiln": 2})
    m = build_model(p)
    assert f"{BLOCK}constant_operation_constraint" in m.constraints
    assert "operation_combined_constant_operation_constraint" not in m.constraints


def test_check_group_coord_fires():
    m = build_model(combined_t2_dataset())
    con = m.constraints["operation_combined_power_max_constraint"]  # two units in the hourly group
    dim = next(d for d in con.labels.dims if d.startswith("combined_tech"))
    check_group_coord(con, con.indexes[dim], dim)
    with pytest.raises(RuntimeError, match="coordinate"):
        check_group_coord(con, con.indexes[dim][::-1], dim)


def test_check_partition():
    full = pd.Index(["a", "b", "c", "d"])
    parts = {None: pd.Index(["a", "b"]), 2: pd.Index(["c", "d"]), 4: pd.Index([])}
    assert list(check_partition(parts, full)) == [None, 2]  # the empty part is left out
    with pytest.raises(AssertionError):
        check_partition({None: pd.Index(["a", "b"]), 2: pd.Index(["b", "c", "d"])}, full)
    with pytest.raises(AssertionError):
        check_partition({None: pd.Index(["a", "b"]), 2: pd.Index(["c"])}, full)


def test_derived_coarse_heat_equals_the_flat_twin():
    """A resource whose contributors are all block techs is coarse; blocks, shedding and spillage meet."""
    blocks = combined_derived_coarse_dataset()
    lengths = {t: 2 for t in ALL}
    assert time_blocks(blocks).resources(2) == ["heat"]
    m = solve_model(build_model(blocks))
    coarse = m.constraints["operation_adequacy_block_2h_constraint"]
    assert list(coarse.labels.indexes["resource"]) == ["heat"]
    # nor a dead (masked) term of an hourly variable: shedding, spillage and one term per block unit
    per_area = pd.Series(units_of(m.variables[f"{BLOCK}net_generation"])).str[0].value_counts()
    assert coarse.vars.sizes["_term"] == 2 + per_area.max()
    assert variables_in(m, coarse) <= {
        "operation_load_shedding_block_2h_power",
        "operation_spillage_block_2h_power",
        f"{BLOCK}net_generation",
    }
    assert float(solved(m, "operation_load_shedding_block_2h_power").sel(resource="heat").sum()) > 0
    twin = solve_model(flat_twin(blocks, combined=lengths))
    assert_twin_equal(m, twin, time_blocks(blocks), [f"{BLOCK}power"])


def test_pinned_module_datasets_solve_hourly():
    for module in ("combined", "conversion"):
        solve_model(build_model(pinned_module_dataset(module)))


def test_block_mode_constant_operation_flat_twin():
    """A block tech with every ramp 0 gets one equality per mode (the block mode_constant rows)."""
    p = combined_toy([1.0] * 10)
    p["combined_mode_ramp_up"].loc[{"combined_tech": "steady_kiln"}] = 0.0
    p["combined_mode_ramp_down"].loc[{"combined_tech": "steady_kiln"}] = 0.0
    p = with_blocks(p, combined={"steady_kiln": 2})
    m = solve_model(build_model(p))
    assert f"{BLOCK}mode_constant_operation_constraint" in m.constraints
    assert f"{BLOCK}constant_operation_constraint" not in m.constraints
    twin = solve_model(flat_twin(p, combined={"steady_kiln": 2}))
    assert_twin_equal(m, twin, tb_of(p), [f"{BLOCK}power"])


def test_block_two_years_two_conditions_flat_twin():
    """Two year_op and two operating conditions (weights 0.6 / 0.4) on a block group."""
    base = combined_toy([1.0] * 10)
    base = base.reindex(year_op=np.array([2030, 2031], dtype="int64"), method="ffill")
    low, high = base.demand, base.demand * 1.5
    p = with_conditions(base, [0.6, 0.4], demand=[low, high])
    p = with_blocks(p, combined={"boiler_block": 2})
    m = solve_model(build_model(p))
    twin = solve_model(flat_twin(p, combined={"boiler_block": 2}))
    assert_twin_equal(m, twin, tb_of(p), [f"{BLOCK}power"])
