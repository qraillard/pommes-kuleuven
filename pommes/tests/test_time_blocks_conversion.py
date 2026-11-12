"""Conversion technologies on time blocks (multi-resolution, sections 5, 6.1, 8.5, 15, 16)."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.model.conversion import conversion_groups
from pommes.model.data_validation.dataset_check import check_inputs
from pommes.tests.time_blocks_helper import (
    BLOCK_PARAMETERS,
    KILNS,
    assert_twin_equal,
    conversion_t1_dataset,
    conversion_t2_dataset,
    conversion_t2b_dataset,
    conversion_toy,
    flat_twin,
    hand_toy,
    hand_toy_c,
    hand_toy_d,
    must_run_relaxation_dataset,
    per_tech,
    pinned_module_dataset,
    solve_model,
    solved,
    toy_n,
    units_of,
    with_blocks,
    with_values,
)
from pommes.utils import check_block_dims, day_of_blocks, conversion_block_availability, time_blocks

ALL_KILNS = {t: 4 for t in KILNS}
POWER_4H = "operation_conversion_block_4h_power"


def tb_of(p):
    return time_blocks(p)


def without(p, name):
    """``p`` with the parameter ``name`` NaN everywhere (a default, not an absent variable)."""
    return p.assign({name: p[name] * np.nan})


def test_t0_conversion_group_is_the_input():
    p = conversion_toy([1.0] * 4)
    [(L, p_g, idx_g)] = conversion_groups(p, tb_of(p))
    assert L is None
    if idx_g is None:  # dense: the dataset itself
        assert p_g is p
    else:  # sparse: today's to_sparse_units result
        from pommes.model.conversion import conversion_active_mask
        from pommes.model.sparsity import to_sparse_units

        expected, index = to_sparse_units(
            p, "area", "conversion_tech", conversion_active_mask(p), var_prefix="conversion_"
        )
        xr.testing.assert_identical(p_g, expected)
        assert idx_g.equals(index)


def test_hand_toy_c_block_furnace():
    p = hand_toy_c()
    m = solve_model(build_model(p))
    a, v, n = 10.0, 1.0, toy_n(p)
    assert time_blocks(p).resources(2) == ["steel"]
    assert m.objective.value == pytest.approx(a + 4 * n * v, rel=1e-9)
    assert solved(m, "operation_conversion_power_capacity").item() == pytest.approx(1)
    assert solved(m, "operation_conversion_block_2h_power").values.ravel() == pytest.approx([1, 1])


def test_hand_toy_d_forced_to_the_step():
    p = hand_toy_d()
    m = solve_model(build_model(p))
    a, v, s, n = 10.0, 1.0, 3.0, toy_n(p)
    assert m.objective.value == pytest.approx(2 * a + 8 * n * v + 4 * n * s, rel=1e-9)
    assert solved(m, "operation_conversion_power_capacity").item() == pytest.approx(2)
    assert solved(m, "operation_conversion_block_2h_power").values.ravel() == pytest.approx([2, 2])
    spill = solved(m, "operation_spillage_block_1h_power").sel(resource="steel")
    assert spill.values.ravel() == pytest.approx([2, 0, 2, 0])


def steel_coarse_dataset():
    """Every steel contributor on 4 h blocks: steel is derived coarse, gas keeps the hourly well."""
    return with_blocks(conversion_toy([1.0] * 8), conversion={t: 4 for t in ["backup", *KILNS]})


def test_block_contributor_has_one_term_per_unit_in_a_coarse_row():
    """Section 8.2 point 3: a block of 4 h gives one term per unit in the coarse row, not one per hour."""
    m = build_model(steel_coarse_dataset())
    con = m.constraints["operation_adequacy_block_4h_constraint"]
    assert list(con.labels.indexes["resource"]) == ["steel"]
    var = m.variables["operation_conversion_block_4h_net_generation"]
    cell = var.labels.sel(resource="steel").isel(time_block_4h=0, year_op=0)
    live_units = pd.Series([u[0] for u, x in zip(units_of(var), cell.values.ravel()) if x != -1])
    assert len(live_units) > 2
    live = sorted(set(np.unique(var.labels.sel(resource="steel").values)) - {-1})
    counts = con.vars.isin(live).sum("_term")  # terms of the net generation per (area, block, resource, year)
    for area, expected in live_units.value_counts().items():
        assert (counts.sel(area=area) == expected).all(), area


def _terms(m, variable, constraint, resource):
    """Coefficients and the number of terms per row of ``variable`` in the rows of ``resource`` of ``constraint``."""
    live = sorted(set(np.unique(m.variables[variable].labels.sel(resource=resource).values)) - {-1})
    con = m.constraints[constraint]
    vars_, coeffs = con.vars.sel(resource=resource), con.coeffs.sel(resource=resource)
    mine = vars_.isin(live)
    return np.unique(coeffs.where(mine).values[mine.values]), mine.sum("_term")


def test_routing_of_a_block_group_to_a_finer_derived_and_a_coarser_forced_balance():
    """Checkpoint point 9: a block group reaches a finer balance by broadcast and a coarser one by sum."""
    # finer: steel is derived at 2 h (the smallest length), the kilns of 4 h are shared out by duration
    lengths = {"backup": 2, **{t: 4 for t in KILNS}}
    m = build_model(with_blocks(conversion_toy([1.0] * 8), conversion=lengths))
    con = "operation_adequacy_block_2h_constraint"
    assert list(m.constraints[con].labels.indexes["resource"]) == ["steel"]
    coeffs, per_row = _terms(m, "operation_conversion_block_4h_net_generation", con, "steel")
    assert coeffs.tolist() == [0.5]  # 2 h of a 4 h block
    coeffs, _ = _terms(m, "operation_conversion_block_2h_net_generation", con, "steel")
    assert coeffs.tolist() == [1.0]  # identity
    # coarser: steel is forced to 4 h, the blocks of 2 h are summed: two terms per unit and row
    p = with_blocks(conversion_toy([1.0] * 8), conversion={t: 2 for t in ["backup", *KILNS]}, resource={"steel": 4})
    m = build_model(p)
    con = "operation_adequacy_block_4h_constraint"
    coeffs, per_row = _terms(m, "operation_conversion_block_2h_net_generation", con, "steel")
    assert coeffs.tolist() == [1.0]
    live = (m.variables["operation_conversion_block_2h_net_generation"].labels.sel(resource="steel") != -1)
    per_block = live.isel(time_block_2h=0, year_op=0).values.ravel()
    areas = pd.Series([u[0] for u in units_of(m.variables["operation_conversion_block_2h_net_generation"])])
    for area, units in areas[per_block].value_counts().items():
        assert (per_row.sel(area=area) == 2 * units).all(), area
    solve_model(m)


def test_block_model_is_smaller():
    """M7: the RAM aim. Block variables have no hour dimension, the coarse balance has B_L rows."""
    p = hand_toy_c()
    m = build_model(p)
    power = m.variables["operation_conversion_block_2h_power"]
    assert "hour" not in power.labels.dims
    assert power.labels.size == len(units_of(power)) * tb_of(p).duration[2].size
    assert "operation_conversion_power" not in m.variables  # every tech is on blocks
    con = m.constraints["operation_adequacy_block_2h_constraint"]
    assert con.labels.sizes["time_block_2h"] == tb_of(p).duration[2].size
    live = (con.labels != -1).sum("time_block_2h")
    assert (live == tb_of(p).duration[2].size).all()


def test_conversion_gates_read_the_group():
    """Constant operation, a daily cap and a cumulative maximum on the hourly tech only: no block row."""
    p = conversion_toy([1.0] * 10)
    p = with_values(p, "conversion_ramp_up", {"backup": 0.0})
    p = with_values(p, "conversion_ramp_down", {"backup": 0.0})
    p = with_values(p, "conversion_max_daily_production", {"backup": 1e6})
    p = with_values(p, "conversion_max_cumulative_production", {"backup": 1e9})
    p = p.assign(operation_year_weight=("year_op", [10.0]))
    m = build_model(with_blocks(p, conversion=ALL_KILNS))
    for stem in ("constant_operation_constraint", "daily_production_max_constraint", "cumulative_production_max_constraint"):
        assert f"operation_conversion_{stem}" in m.constraints, stem
        assert f"operation_conversion_block_4h_{stem}" not in m.constraints, stem
    # the other way round: the gates sit on the block technology only
    p = conversion_toy([1.0] * 10)
    p = with_values(p, "conversion_ramp_up", {"kiln_d": 0.0})
    p = with_values(p, "conversion_ramp_down", {"kiln_d": 0.0})
    p = with_values(p, "conversion_max_daily_production", {"kiln_d": 1e6})
    p = with_values(p, "conversion_max_cumulative_production", {"kiln_d": 1e9})
    m = build_model(with_blocks(p.assign(operation_year_weight=("year_op", [10.0])), conversion=ALL_KILNS))
    for stem in ("constant_operation_constraint", "daily_production_max_constraint", "cumulative_production_max_constraint"):
        assert f"operation_conversion_block_4h_{stem}" in m.constraints, stem
        assert f"operation_conversion_{stem}" not in m.constraints, stem
    # the weight is read from the group's dataset: a block group with a cumulative maximum needs it
    with pytest.raises(ValueError, match="operation_year_weight"):
        build_model(with_blocks(p.assign(operation_year_weight=("year_op", [np.nan])), conversion=ALL_KILNS))


def test_pinned_module_and_empty_group_conversion():
    """Section 16.9: every conversion pair pinned to zero with a block length; a length on a tech without unit."""
    base = pinned_module_dataset("conversion")
    hourly = solve_model(build_model(base))
    p = with_blocks(base, conversion={"well": 2})
    m = solve_model(build_model(p))
    assert np.isfinite(m.objective.value)
    assert m.objective.value == pytest.approx(hourly.objective.value, rel=1e-9)
    # a length on a technology with no unit leaves the other variables unchanged
    q = conversion_toy([1.0] * 10)
    q = q.assign(conversion_power_capacity_max=q.conversion_power_capacity_max.where(q.conversion_tech != "kiln_d", 0.0))
    ref = solve_model(build_model(q))
    withlen = solve_model(build_model(with_blocks(q, conversion={"kiln_d": 2})))
    assert withlen.objective.value == pytest.approx(ref.objective.value, rel=1e-9)
    for name in ("operation_conversion_power_capacity", "operation_conversion_power", "operation_conversion_costs"):
        others = sorted(t for t in q.conversion_tech.values if t != "kiln_d")
        a = solved(ref, name).sel(conversion_tech=others)
        b = solved(withlen, name).sel(conversion_tech=others).transpose(*a.dims)
        np.testing.assert_allclose(a.values, b.values, atol=1e-9, err_msg=name)


def test_retrofit_hourly_source_block_target_smoke(parameters_retrofit_conversion):
    """M13: the retrofit rows read no operation variable, so a block target builds and solves."""
    p = parameters_retrofit_conversion.sel(
        conversion_tech=["smr", "smr_ccs"],
        area=["area_1"],
        hour=[0, 1, 2, 3],
        resource=["electricity", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030],
        year_ret=[2030, 2040],
    ).copy(deep=True)
    p = p.assign(**{k: np.array(False) for k in ("carbon", "net_import", "turpe")})
    # every resource may be shed at a price, so that the hourly source and the block target solve
    p = p.assign(
        load_shedding_max_capacity=p.load_shedding_max_capacity * np.nan,
        load_shedding_cost=p.load_shedding_cost * 0 + 1000.0,
    )
    p = with_blocks(p, conversion={"smr_ccs": 2})
    m = build_model(p)
    assert "operation_conversion_block_2h_power" in m.variables
    assert "operation_conversion_power" in m.variables  # smr, hourly
    retrofit = [n for n in [*m.variables, *m.constraints] if "retrofit" in n]
    assert retrofit
    for name in retrofit:
        obj = m.variables[name] if name in m.variables else m.constraints[name]
        dims = set(obj.labels.dims) | (set(obj.vars.dims) if hasattr(obj, "vars") else set())
        assert not [d for d in dims if d == "hour" or d.startswith("time_block_")], name
    assert m.solve(solver_name="highs")[:2] == ("ok", "optimal")  # a break of the group code shows here


def test_block_availability_values():
    p = conversion_toy([1.0, 1.0, 2.0, 1.0, 1.0])
    a = np.array([1.0, 0.5, np.nan, 0.8, np.nan])
    p = per_tech(p, "conversion_availability", {t: a for t in KILNS})
    p = with_blocks(p, conversion={t: 4 for t in KILNS}, rule={"kiln_b": "mean"})
    tb = tb_of(p)  # blocks of 4 h: steps (1, 1, 2), then (1, 1)
    assert tb.duration[4].values.tolist() == [4.0, 2.0]
    got = conversion_block_availability(p, tb, 4, KILNS)
    # min: the NaN hour reads 1; mean: it reads 1 as well, weighted by its duration (2 h in block 0)
    assert got.sel(conversion_tech="kiln_a").values.tolist() == pytest.approx([0.5, 0.8])
    assert got.sel(conversion_tech="kiln_b").values.tolist() == pytest.approx(
        [(1 * 1 + 1 * 0.5 + 2 * 1) / 4, (0.8 + 1) / 2]
    )
    # an all-NaN block is 1
    p2 = per_tech(p, "conversion_availability", {t: [np.nan] * 5 for t in KILNS})
    assert (conversion_block_availability(p2, tb, 4, KILNS) == 1).all()
    # no hour: constant in time, the filled availability itself; absent: 1
    p3 = p.assign(conversion_availability=p.conversion_availability.isel(hour=0, drop=True))
    assert "hour" not in conversion_block_availability(p3, tb, 4, KILNS).dims
    assert conversion_block_availability(p.drop_vars("conversion_availability"), tb, 4, KILNS).item() == 1


def test_block_availability_per_condition_not_mixed():
    p = conversion_toy([1.0] * 4)
    a = xr.DataArray(
        [[[1.0, 0.6, 0.9, 0.8], [0.5, 1.0, 0.7, 0.4]]] * p.sizes["conversion_tech"],
        dims=("conversion_tech", "operating_condition", "hour"),
        coords={
            "conversion_tech": p.conversion_tech.values,
            "operating_condition": ["c1", "c2"],
            "hour": p.hour.values,
        },
    )
    p = with_blocks(p.assign(conversion_availability=a), conversion={t: 2 for t in KILNS})
    got = conversion_block_availability(p, tb_of(p), 2, ["kiln_a"]).sel(conversion_tech="kiln_a")
    assert got.sel(operating_condition="c1").values.tolist() == pytest.approx([0.6, 0.8])
    assert got.sel(operating_condition="c2").values.tolist() == pytest.approx([0.5, 0.4])


def test_must_run_with_a_nan_hour_reads_it_as_one():
    """The block equality reads the filled availability (D9'), never a raw NaN."""
    p = hand_toy([1.0] * 4, [2.0] * 4)
    p = p.assign(conversion_must_run=("conversion_tech", [1.0]))
    p = per_tech(p, "conversion_availability", {"furnace": [np.nan, np.nan, 0.7, np.nan]})
    p = with_blocks(p, conversion={"furnace": 2})
    m = solve_model(build_model(p))
    cap = solved(m, "operation_conversion_power_capacity").item()
    power = solved(m, "operation_conversion_block_2h_power").values.ravel()
    # blocks (nan, nan) and (0.7, nan) read (1, 1) and (0.7, 1): block values 1 and 0.7
    assert cap > 0.1
    assert power == pytest.approx(cap * np.array([1.0, 0.7]))


def test_a_block_object_with_an_hour_dimension_is_refused():
    """H5: a block costs row built with ``time_step_duration`` instead of ``D`` keeps ``hour``."""
    p = hand_toy_c()
    m = build_model(p)
    check_block_dims(m)  # a clean block model passes
    m.add_variables(coords=[p.hour], name="operation_conversion_block_2h_leak")
    with pytest.raises(RuntimeError, match="hour"):
        check_block_dims(m)


def test_conversion_t1_block_equal_to_step():
    p = conversion_t1_dataset()
    tb = tb_of(p)
    m = solve_model(build_model(p))
    hourly = solve_model(build_model(p.drop_vars(BLOCK_PARAMETERS)))
    assert "operation_conversion_power" not in m.variables  # every tech is on 3 h blocks
    assert m.objective.value > 0
    assert_twin_equal(m, hourly, tb, ["operation_conversion_block_3h_power"])


def _t2_pair(p, lengths=None):
    lengths = lengths or ALL_KILNS
    return solve_model(build_model(p)), solve_model(flat_twin(p, conversion=lengths))


def test_conversion_t2_flat_twin():
    p = conversion_t2_dataset()
    m, twin = _t2_pair(p)
    hourly = solve_model(build_model(p.drop_vars(BLOCK_PARAMETERS)))
    assert tb_of(p).duration[4].values.tolist() == [4.0, 4.0, 2.0]  # a short last block
    assert m.objective.value > hourly.objective.value + 1.0  # the flat constraint really binds
    assert_twin_equal(m, twin, tb_of(p), [POWER_4H])
    cap = solved(m, "operation_conversion_power_capacity")
    assert float(cap.sel(conversion_tech="kiln_b").sum()) > 0.1  # must-run 1 is built
    assert float(cap.sel(conversion_tech="kiln_c").sum()) > 0.1  # must-run 0.5 is built
    # the yearly maximum and the capacity factor really bind
    for name in ("conversion_max_yearly_production", "conversion_max_capacity_factor"):
        free = solve_model(build_model(without(p, name)))
        assert free.objective.value < m.objective.value - 1e-3, name


@pytest.mark.parametrize("dt", [[1.0] * 10, [2.0, 1.0, 1.0] * 3], ids=["unit_steps", "first_step_2h"])
def test_conversion_t2_steps(dt):
    """Unit steps and a first step of 2 h (so Δ0 is not D / L), the last block short."""
    p = with_blocks(conversion_toy(dt), conversion={"kiln_a": 4, "kiln_d": 4})
    p = with_values(p, "conversion_ramp_up", {"kiln_a": 0.5})
    p = with_values(p, "conversion_ramp_down", {"kiln_a": 0.5})
    m, twin = _t2_pair(p, {"kiln_a": 4, "kiln_d": 4})
    assert_twin_equal(m, twin, tb_of(p), [POWER_4H])


def _daily_dataset(n, L, cap_fraction):
    """``n`` hourly steps, the kilns on blocks of ``L`` h, a daily cap on ``kiln_b`` (the cheap kiln) that binds."""
    p = with_blocks(conversion_toy([1.0] * n), conversion={t: L for t in KILNS})
    norm = 8760.0 / n
    return with_values(p, "conversion_max_daily_production", {"kiln_b": cap_fraction * norm})


def test_daily_cap_blocks_inside_days():
    """L = 6 on 48 h: blocks lie inside days (section 8.5)."""
    p = _daily_dataset(48, 6, 30.0)  # 30 MWh a day, against about 70 uncapped
    m, twin = _t2_pair(p, {t: 6 for t in KILNS})
    assert "operation_conversion_block_6h_daily_production_max_constraint" in m.constraints
    free = solve_model(build_model(without(p, "conversion_max_daily_production")))
    assert free.objective.value < m.objective.value - 1e-3  # the cap binds
    assert_twin_equal(m, twin, tb_of(p), ["operation_conversion_block_6h_power"])


def test_daily_cap_days_inside_blocks():
    """L = 48 on 120 h (the last block lasts 24 h): days lie inside blocks (section 8.5)."""
    p = _daily_dataset(120, 48, 30.0)
    assert tb_of(p).duration[48].values.tolist() == [48.0, 48.0, 24.0]
    m, twin = _t2_pair(p, {t: 48 for t in KILNS})
    free = solve_model(build_model(without(p, "conversion_max_daily_production")))
    assert free.objective.value < m.objective.value - 1e-3
    assert_twin_equal(m, twin, tb_of(p), ["operation_conversion_block_48h_power"])


def test_must_run_on_varying_availability_is_a_relaxation():
    """C16: the block objective never exceeds the twin's; capacity > 0 where the twin pins it to 0 only on the designed case."""
    for make in (conversion_t2_dataset, must_run_relaxation_dataset):
        p = make()
        lengths = {t: int(L) for t, L in zip(p.conversion_tech.values, p.conversion_time_block_hrs.values) if np.isfinite(L)}
        m, twin = _t2_pair(p, lengths)
        assert m.objective.value <= twin.objective.value + 1e-7 * abs(twin.objective.value), make.__name__
    # Spec 15.2 says the capacity is "possibly built": strict only where the dataset is designed for it.
    # The furnace is must-run 1 on (1, 0.5) inside each block: the twin's equalities P_h = cap * a_h
    # with a flat P force cap = 0; the block row P = cap * min is feasible and the furnace is cheap.
    p = must_run_relaxation_dataset()
    m, twin = _t2_pair(p, {"furnace": 2})
    assert solved(twin, "operation_conversion_power_capacity").sel(conversion_tech="furnace").item() == pytest.approx(0, abs=1e-9)
    assert solved(m, "operation_conversion_power_capacity").sel(conversion_tech="furnace").item() > 0.1
    assert m.objective.value < twin.objective.value - 1.0


def test_t2b_mean_rule():
    p_min, p_mean = conversion_t2_dataset(), conversion_t2b_dataset()
    m_min, m_mean = solve_model(build_model(p_min)), solve_model(build_model(p_mean))
    assert m_mean.objective.value <= m_min.objective.value + 1e-9
    # a plant at full load (must-run 1) produces over the year what the hourly plant could
    p = with_blocks(must_run_relaxation_dataset(), conversion={"furnace": 2}, rule={"furnace": "mean"})
    m = solve_model(build_model(p))
    tb = tb_of(p)
    cap = solved(m, "operation_conversion_power_capacity").sel(conversion_tech="furnace").item()
    assert cap > 0.1
    power = solved(m, "operation_conversion_block_2h_power").sel(conversion_tech="furnace").values.ravel()
    produced = float((power * tb.duration[2].values).sum())
    a = p.conversion_availability.sel(conversion_tech="furnace").values
    assert produced == pytest.approx(float((tb.step.values * a).sum()) * cap)


def test_r6_raised_at_build_without_check_inputs():
    p = conversion_toy([1.0] * 72)
    p = with_values(p, "conversion_max_daily_production", {"kiln_a": 100.0})
    p = with_blocks(p, conversion={"kiln_a": 36})
    with pytest.raises(ValueError, match="operation_day_duration"):
        build_model(p)


def _t2_base_dataset():
    """The t2 dataset's inputs without block lengths: must-run 0.5 on ``kiln_c``, ``kiln_a`` varies inside blocks."""
    return conversion_t2_dataset().drop_vars(BLOCK_PARAMETERS)


def test_must_run_never_warns():
    p = with_blocks(_t2_base_dataset(), conversion={t: 4 for t in ["well", "backup", *KILNS]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = build_model(p)
    assert "operation_conversion_block_4h_must_run_constraint_inequality" in m.constraints


def test_length_longer_than_horizon_is_one_block():
    p = with_blocks(conversion_toy([1.0] * 10), conversion={t: 1000 for t in KILNS})
    assert tb_of(p).duration[1000].values.tolist() == [10.0]
    m, twin = _t2_pair(p, {t: 1000 for t in KILNS})
    assert_twin_equal(m, twin, tb_of(p), ["operation_conversion_block_1000h_power"])


@pytest.mark.parametrize("checked", [False, True], ids=["build_only", "check_inputs"])
def test_block_group_with_default_availability(checked):
    p = hand_toy_c().drop_vars("conversion_availability")
    assert "conversion_availability" not in p
    if checked:
        p = check_inputs(p)
        assert "conversion_availability" in p  # the 0-d default
    m = solve_model(build_model(p))
    # Ā = 1: capacity 1 carries the 1 MW of the 2 h block
    assert solved(m, "operation_conversion_power_capacity").item() == pytest.approx(1)
    assert solved(m, "operation_conversion_block_2h_power").values.ravel() == pytest.approx([1, 1])


def test_block_sums_per_year_op():
    base = conversion_toy([1.0] * 10)
    base = base.reindex(year_op=np.array([2030, 2031], dtype="int64"), method="ffill")
    p = base.assign(operation_year_weight=("year_op", [10.0, 10.0]))
    yearly = xr.DataArray(
        [[np.nan, np.nan]] * p.sizes["conversion_tech"],
        dims=("conversion_tech", "year_op"),
        coords={"conversion_tech": p.conversion_tech.values, "year_op": p.year_op.values},
    )
    yearly.loc[{"conversion_tech": "kiln_b", "year_op": 2030}] = 8000.0  # binds in 2030 only
    # kiln_b is dearer in 2031: moving its energy between the years is not free, the optimum is unique
    cost = p.conversion_variable_cost.expand_dims(year_op=p.year_op.values).copy()
    cost.loc[{"conversion_tech": "kiln_b", "year_op": 2031}] = 1.2
    p = p.assign(conversion_max_yearly_production=yearly, conversion_variable_cost=cost)
    p = with_values(p, "conversion_max_cumulative_production", {"kiln_b": 1.5e5})
    p = with_blocks(p, conversion={"kiln_b": 4, "kiln_d": 4})
    m, twin = _t2_pair(p, {"kiln_b": 4, "kiln_d": 4})
    # a binding annual sum leaves the split of its energy between blocks free (equal costs): the
    # objective, the capacities and the energy of each year are compared, not the power profile
    assert_twin_equal(m, twin, tb_of(p), [])
    tb = tb_of(p)
    yearly, cumulative = "conversion_max_yearly_production", "conversion_max_cumulative_production"
    free = solve_model(build_model(without(without(p, yearly), cumulative))).objective.value
    for kept, dropped in ((yearly, cumulative), (cumulative, yearly)):  # each binds on its own
        assert solve_model(build_model(without(p, dropped))).objective.value > free + 1e-3, kept
    block = solved(m, POWER_4H).sel(conversion_tech="kiln_b")
    produced = (block * tb.duration[4].values[None, :, None]).sum("time_block_4h")
    hourly = solved(twin, "operation_conversion_power").sel(conversion_tech="kiln_b")
    twin_produced = (hourly * tb.step.values[None, :, None]).sum("hour")
    np.testing.assert_allclose(produced.values, twin_produced.values, atol=1e-6)
    # the yearly maximum holds in 2030, where it is set: 8000 / n MWh in each area
    assert (produced.sel(year_op=2030).values <= 8000.0 / (8760.0 / 10) + 1e-6).all()


def test_daily_cap_non_dyadic_steps():
    """20-min steps, L = 6 on 48 h: a block starting an ulp below a day boundary stays in its day (R5 tolerance).

    Block path only: the hourly daily row keeps 0.6.1+kul's tolerance-free floor (R12), so no twin comparison.
    """
    p = with_blocks(conversion_toy([1 / 3] * 144), conversion={t: 6 for t in KILNS})
    p = with_values(p, "conversion_max_daily_production", {"kiln_b": 30.0 * 8760.0 / 144 * 3})
    tb = tb_of(p)
    assert day_of_blocks(tb, 6, 24).values.tolist() == [0] * 4 + [1] * 4
    m = build_model(p)
    row = m.constraints["operation_conversion_block_6h_daily_production_max_constraint"]
    assert row.coords["day"].values.tolist() == [0, 1]
