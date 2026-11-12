"""Flexible demand on the blocks of a coarse balance (multi-resolution, sections 6.4, 9, 15.2 T3(c), 16.4)."""

import numpy as np
import pytest
import xarray as xr
from linopy import Model

from pommes.model.build_model import build_model
from pommes.model.flexdem import add_flexdem
from pommes.tests.time_blocks_helper import (
    BLOCK_PARAMETERS,
    check_coarse_step_contract,
    coarse_step_dataset,
    coarse_step_reference,
    flexdem_block_dataset,
    flexdem_ratio_dataset,
    flexdem_short_block_dataset,
    flexdem_toy,
    solve_model,
    solved,
    with_blocks,
)
from pommes.utils import block_dim, check_block_dims, conversion_block_availability, time_blocks

BLOCK = "operation_flexdem_block_4h_"


def test_flexdem_does_not_count():
    """A flexible demand is not a contributor: steel with a 4 h furnace is derived at 4 h."""
    p = with_blocks(flexdem_toy(8), conversion={"furnace": 4})
    tb = time_blocks(p)
    assert tb.resource_length.sel(resource="steel") == 4
    assert tb.resources(None) == ["idle"]
    off = p.assign(flexdem=np.array(False))
    assert time_blocks(off).resource_length.equals(tb.resource_length)


def test_coarse_flexdem_names_and_dims():
    m = build_model(flexdem_block_dataset())
    var = m.variables[f"{BLOCK}demand"]
    assert var.labels.dims == ("area", "resource", "time_block_4h", "year_op")
    assert list(var.labels.indexes["resource"]) == ["gas", "heat"]
    assert var.labels.sizes["time_block_4h"] == 6
    hourly = m.variables["operation_flexdem_demand"]
    assert hourly.labels.dims == ("area", "resource", "hour", "year_op")
    assert list(hourly.labels.indexes["resource"]) == ["idle"]
    assert (hourly.labels == -1).all()  # nothing flexible on the hourly balances
    rows = {
        "demand_max": {"area", "resource", "time_block_4h", "year_op"},
        "demand_min": {"area", "resource", "time_block_4h", "year_op"},
        "demand_conservation": {"area", "resource", "year_op"},
        "demand_conservation_8h": {"area", "resource", "window", "year_op"},
        "ramp_up_constraint": {"area", "resource", "time_block_4h", "year_op"},
        "ramp_down_constraint": {"area", "resource", "time_block_4h", "year_op"},
    }
    for name, dims in rows.items():
        assert set(m.constraints[f"{BLOCK}{name}"].labels.dims) == dims, name
    assert m.constraints[f"{BLOCK}demand_conservation_8h"].labels.sizes["window"] == 3
    for name in [*m.variables, *m.constraints]:
        if name.startswith("operation_flexdem_block_"):
            obj = m.variables[name] if name in m.variables else m.constraints[name]
            assert "hour" not in obj.labels.dims, name
    check_block_dims(m)


@pytest.mark.parametrize("L, horizon", [(4, 24), (6, 12)])
def test_coarse_step_dataset_is_uniform(L, horizon):
    p = coarse_step_dataset(L, horizon)
    tb = time_blocks(p)
    length = tb.resource_length
    assert np.isnan(length.sel(resource="idle"))
    assert (length.drop_sel(resource="idle") == L).all()
    for module in ("conversion", "combined"):
        assert (tb.tech_length[module] == L).all()
    assert p.sizes["hour"] == horizon


def test_coarse_step_contract():
    """C17: a ramp or a daily cap set after the fact, and a partly-NaN availability, are refused."""
    p = coarse_step_dataset(4, 24)
    check_coarse_step_contract(p)
    for name in (
        "conversion_ramp_up",
        "conversion_ramp_down",
        "combined_ramp_up",
        "combined_ramp_down",
        "combined_mode_ramp_up",
        "combined_mode_ramp_down",
        "conversion_max_daily_production",
    ):
        assert np.isnan(p[name]).all(), name
        bad = p.assign({name: xr.full_like(p[name], 0.5)})
        with pytest.raises(AssertionError, match=name):
            check_coarse_step_contract(bad)
        with pytest.raises(AssertionError, match=name):
            coarse_step_reference(bad, 4)
    partly = p["conversion_availability"].copy(deep=True)
    partly.loc[dict(conversion_tech="heater", hour=3)] = np.nan
    p = p.assign(conversion_availability=partly)
    with pytest.raises(AssertionError, match="partly-NaN"):
        check_coarse_step_contract(p)
    # the reference reads a NaN as 1 before the mean, as the block model does
    tb = time_blocks(p)
    ref = coarse_step_reference(p, 4)["conversion_availability"]
    by_hand = np.where(np.isnan(partly.sel(conversion_tech="heater").values), 1.0, partly.sel(conversion_tech="heater").values)
    np.testing.assert_allclose(ref.sel(conversion_tech="heater").values, by_hand.reshape(6, 4).mean(1))
    assert ref.sel(conversion_tech="heater").values[0] == pytest.approx(
        (1 + 0.6 + 0.9 + 1.0) / 4  # hour 3 (availability 0.8) is NaN: read as 1
    )
    block = conversion_block_availability(p, tb, 4, list(p.conversion_tech.values))
    np.testing.assert_allclose(ref.rename(hour="time_block_4h").transpose(*block.dims).values, block.values)


def _binds(m, name, equality=False):
    """Some live row of ``name`` has a non-zero dual or, for an inequality, a zero slack."""
    con = m.constraints[name]
    live = (con.vars != -1).any("_term")  # a row on a flexible variable (a masked variable leaves 0 <= 0)
    dual = con.dual.where(live)
    hit = abs(dual) > 1e-9
    if not equality:  # the slack of an equality is always zero: only its dual says it matters
        hit = hit | (abs((con.rhs - con.lhs.solution).where(live)) < 1e-9)
    return bool(hit.any())


@pytest.mark.parametrize(
    "case, rows",
    [
        (flexdem_block_dataset, ("demand_conservation_8h", "ramp_up_constraint", "ramp_down_constraint")),
        (flexdem_ratio_dataset, ("demand_max", "demand_min")),
    ],
    ids=["window_and_ramps", "ratios"],
)
def test_coarse_flexdem_equals_uniform_coarse_step(case, rows):
    """T3(c): the block model equals today's code run with steps of 4 h; each row family it covers binds."""
    p = case()
    m = solve_model(build_model(p))
    reference_dataset = coarse_step_reference(p, 4)
    reference = solve_model(build_model(reference_dataset))
    assert m.objective.value == pytest.approx(reference.objective.value, rel=1e-7)
    block = solved(m, f"{BLOCK}demand").transpose("area", "resource", "time_block_4h", "year_op")
    step = solved(reference, "operation_flexdem_demand").transpose("area", "resource", "hour", "year_op")
    np.testing.assert_allclose(block.values, step.sel(resource=block.resource).values, atol=1e-6)
    heat_demand = reference_dataset.flexdem_demand.sel(resource="heat").transpose("area", "hour", "year_op").values
    assert np.abs(block.sel(resource="heat").values - heat_demand[:, :, None, :].squeeze(2)).max() > 0.1  # demand shifts
    for name in rows:
        assert _binds(m, f"{BLOCK}{name}", equality=name.startswith("demand_conservation")), name


def test_window_of_four_blocks():
    p = coarse_step_dataset(4, 32)
    p = p.assign(flexdem_conservation_hrs=p.flexdem_conservation_hrs * 0 + 16)
    m = solve_model(build_model(p))
    assert m.constraints[f"{BLOCK}demand_conservation_16h"].labels.sizes["window"] == 2
    F = solved(m, f"{BLOCK}demand").sel(resource="heat", year_op=2030).transpose("area", "time_block_4h").values
    FD = p.flexdem_demand.sel(resource="heat", year_op=2030).transpose("area", "hour").values.reshape(2, 8, 4).sum(2)
    shifted = F - FD
    np.testing.assert_allclose(shifted.reshape(2, 2, 4).sum(2), 0.0, atol=1e-6)
    assert np.abs(shifted).max() > 0.1  # demand is shifted
    assert np.abs(shifted.reshape(2, 4, 2).sum(2)).max() > 1e-3  # and not conserved over each 8 h: the window is 16 h


def test_window_not_multiple_raised_at_build():
    """R9 without ``check_inputs``: a window or an offset inside a block is refused by the build."""
    p = flexdem_block_dataset()
    with pytest.raises(ValueError, match="flexdem_conservation_hrs"):
        build_model(p.assign(flexdem_conservation_hrs=p.flexdem_conservation_hrs * 0 + 6))
    with pytest.raises(ValueError, match="flexdem_window_offset_hrs"):
        build_model(p.assign(flexdem_window_offset_hrs=np.array(2.0)))


def test_short_last_block_shifts_nothing():
    """10 h in blocks of 4 (the last lasts 2 h), constant demand: the mean-power ramps shift nothing."""
    m = solve_model(build_model(flexdem_short_block_dataset()))
    F = solved(m, f"{BLOCK}demand").sel(resource="steel").values.ravel()
    np.testing.assert_allclose(F, [4.0, 4.0, 2.0], atol=1e-6)


@pytest.mark.slow
def test_short_last_block_8760h():
    """8760 h in weeks (the last block lasts 24 h), constant demand: no energy is shifted."""
    p = with_blocks(flexdem_toy(8760), conversion={"furnace": 168})
    m = solve_model(build_model(p))
    F = solved(m, "operation_flexdem_block_168h_demand").sel(resource="steel").values.ravel()
    FD = np.r_[np.full(52, 168.0), 24.0]
    np.testing.assert_allclose(F, FD, atol=1e-6)


# --- rolling horizon: the toy of test_flexdem_rolling.py on a coarse heat balance ---

L_ROLL = 4


def _rolling_model(n, demand, carry=None, offset=None):
    """
    ``add_flexdem`` alone on a forced coarse heat balance (blocks of 4 h) and an hourly steel balance.

    Heat is bought at a price that falls from block to block, so the demand goes to the end of each 8 h window.
    ``demand`` is the heat demand of each block (per step), ``carry`` and ``offset`` the rolling-horizon inputs.
    """
    per = ("area", "resource", "year_op")
    one = lambda v: (per, np.full((1, 2, 1), v))  # noqa: E731
    fd = np.zeros((1, 2, n, 1))
    fd[0, 0, :, 0] = np.repeat(demand, L_ROLL)
    p = xr.Dataset(
        dict(
            flexdem_demand=(("area", "resource", "hour", "year_op"), fd),
            flexdem_conservation_hrs=(per, np.array([[[8], [0]]], dtype="int64")),
            flexdem_maxload_ratio=one(1.5),
            flexdem_minload_ratio=one(0.0),
            flexdem_ramp_up=one(np.nan),
            flexdem_ramp_down=one(np.nan),
            time_step_duration=("hour", np.ones(n)),
            resource_time_block_hrs=("resource", [float(L_ROLL), np.nan]),
        ),
        coords=dict(area=["a"], resource=["heat", "steel"], year_op=[2030], hour=np.arange(n)),
    )
    if carry is not None:
        p["flexdem_shifted_energy_initial"] = (per, np.array([[[carry], [0.0]]]))
    if offset is not None:
        p["flexdem_window_offset_hrs"] = np.array(float(offset))
    tb = time_blocks(p)
    m = Model()
    time = tb.duration[L_ROLL][block_dim(L_ROLL)]
    steel = m.add_variables(lower=0, coords=[p.area, p.resource.sel(resource=["steel"]), p.hour, p.year_op], name="steel_supply")
    heat = m.add_variables(lower=0, coords=[p.area, p.resource.sel(resource=["heat"]), time, p.year_op], name="heat_supply")
    adequacy = m.add_constraints(steel == 0, name="operation_adequacy_constraint")
    m.add_constraints(heat == 0, name=f"operation_adequacy_block_{L_ROLL}h_constraint")
    add_flexdem(m, p, None, adequacy, tb=tb)
    price = xr.DataArray(np.arange(time.size, 0, -1.0), dims=block_dim(L_ROLL), coords={block_dim(L_ROLL): time.values})
    m.objective = (price * heat).sum()
    return m


def _heat(m):
    return m.solution[f"operation_flexdem_block_{L_ROLL}h_demand"].sel(resource="heat").values.ravel()


def test_rolling_horizon_on_blocks():
    demand = np.array([4.0, 6.0, 8.0, 4.0])  # per step: 4 h blocks of 16, 24, 32 and 16
    stand_alone = _rolling_model(16, demand)
    assert stand_alone.solve(solver_name="highs")[:2] == ("ok", "optimal")
    whole = _heat(stand_alone)
    shifted = whole - demand * L_ROLL
    assert np.abs(shifted).max() > 1.0  # demand is shifted: the test is not vacuous
    # The run on hours 4..15 starts in the middle of the first 8 h window: 4 h in (offset, a multiple of the
    # block), with the energy block 0 shifted carried in. It gives the stand-alone run's blocks 1 to 3.
    carry = float(shifted[0])
    rolling = _rolling_model(12, demand[1:], carry=carry, offset=4)
    assert rolling.solve(solver_name="highs")[:2] == ("ok", "optimal")
    np.testing.assert_allclose(_heat(rolling), whole[1:], atol=1e-6)
    # without the carry-in the first window closes on its own: another answer
    alone = _rolling_model(12, demand[1:], offset=4)
    assert alone.solve(solver_name="highs")[:2] == ("ok", "optimal")
    assert np.abs(_heat(alone) - whole[1:]).max() > 1.0
