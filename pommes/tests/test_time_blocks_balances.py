"""Coarse adequacy, block shedding and spillage, the H5 checks (multi-resolution, sections 6.6 and 12)."""

import numpy as np
import pytest
import xarray as xr
from linopy import Model

import pommes.model.build_model as build_model_module
from pommes.model.build_model import build_model
from pommes.tests.time_blocks_helper import (
    STEEL_DEMAND as DEMAND,
    assert_t0,
    hand_toy,
    hand_toy_a,
    hand_toy_b,
    methane_forced,
    methane_hourly,
    process_forced,
    process_hourly,
    solved,
    toy_n,
    with_blocks,
)
from pommes.utils import check_block_dims

BLOCK = "operation_adequacy_block_2h_constraint"
# block technology names appear from Task 5 on: only these three families are checked
BLOCK_PREFIXES = (
    "operation_adequacy_block_",
    "operation_load_shedding_block_",
    "operation_spillage_block_",
)


def solve(p):
    m = build_model(p)
    status, condition = m.solve(solver_name="highs")
    assert (status, condition) == ("ok", "optimal")
    return m


def test_hand_toy_a_today():
    p = hand_toy_a()
    m = solve(p)
    a, v, n = 10.0, 1.0, toy_n(p)
    assert m.objective.value == pytest.approx(2 * a + 4 * n * v, rel=1e-9)
    assert solved(m, "operation_conversion_power_capacity").item() == pytest.approx(2)
    assert solved(m, "operation_conversion_power").values.ravel() == pytest.approx(
        [0, 2, 0, 2]
    )


def test_hand_toy_b_coarse_balance():
    p = hand_toy_b()
    m = solve(p)
    a, v, n = 10.0, 1.0, toy_n(p)
    assert m.objective.value == pytest.approx(a + 4 * n * v, rel=1e-9)
    assert solved(m, "operation_conversion_power_capacity").item() == pytest.approx(1)
    assert solved(m, "operation_conversion_power").values.ravel() == pytest.approx(
        [1, 1, 1, 1]
    )


def test_hourly_adequacy_holds_only_hourly_resources():
    m = solve(hand_toy_b())
    hourly = m.constraints["operation_adequacy_constraint"]
    assert list(hourly.labels.indexes["resource"]) == ["idle"]
    block = m.constraints[BLOCK]
    assert set(block.labels.dims) == {"area", "time_block_2h", "resource", "year_op"}
    assert list(block.labels.indexes["resource"]) == ["steel"]
    for name in ("operation_load_shedding_power", "operation_spillage_power"):
        assert list(m.variables[name].labels.indexes["resource"]) == ["idle"]
    spill = solved(m, "operation_spillage_block_2h_power").sel(resource="steel")
    assert spill.values.ravel() == pytest.approx([0, 0])


def test_coarse_demand_is_summed_per_block():
    p = hand_toy([1.0] * 5, [0] * 5)
    p["demand"] = xr.DataArray(
        np.array([[[0.0], [1.0]]]),
        dims=("area", "resource", "year_op"),
        coords={"area": p.area, "resource": p.resource, "year_op": p.year_op},
    )  # no hour: the same energy in every step (H12)
    m = build_model(with_blocks(p, resource={"steel": 2}))
    rhs = m.constraints[BLOCK].rhs.sel(resource="steel")
    assert rhs.values.ravel() == pytest.approx([2, 2, 1])


def test_coarse_demand_reads_nan_as_zero():
    """R8: a NaN step does not drop the other demand of its block."""
    p = hand_toy([1.0] * 4, [2, np.nan, 1, 1])
    m = build_model(with_blocks(p, resource={"steel": 2}))
    rhs = m.constraints[BLOCK].rhs.sel(resource="steel")
    assert rhs.values.ravel() == pytest.approx([2, 2])


def _shedding_toy(dt, rates, years=1):
    """Steel demand ``rate * dt``, shedding at cost 7 up to 0.5 MW, the furnace covers the rest."""
    base = hand_toy(dt, [0] * len(dt), a=1e5)
    base["load_shedding_max_capacity"] = ("resource", [np.nan, 0.5])
    base["load_shedding_cost"] = ("resource", [0.0, 7.0])
    if years == 2:
        base = base.reindex(year_op=[2030, 2035])
        for name, value in (
            ("operation_year_duration", 8760.0),
            ("discount_rate", 0.0),
            ("discount_factor", 1.0),
        ):
            base[name] = ("year_op", [value, value])
        base["conversion"] = False  # the shedding alone: no invest dims over two years
        rates = np.array([rates, 0.5 * np.asarray(rates)])
    else:
        rates = np.array([rates])
    demand = np.zeros((1, len(dt), 2, years))
    demand[0, :, 1, :] = (rates * np.asarray(dt)).T.reshape(len(dt), years)
    base["demand"] = (("area", "hour", "resource", "year_op"), demand)
    return base


@pytest.mark.parametrize(
    "case",
    [
        ("even", [1.0] * 4, [2, 2, 1, 1], 1),
        ("uneven", [0.5, 1.5, 1.0, 1.0, 0.5], [2, 2, 1, 1, 3], 1),  # short last block
        ("two_years", [1.0] * 4, [0.4, 0.4, 0.3, 0.3], 2),
    ],
    ids=lambda c: c[0],
)
def test_block_shedding_and_spillage_are_exact(case):
    _, dt, rates, years = case
    base = _shedding_toy(dt, rates, years)
    hourly, coarse = solve(base), solve(with_blocks(base, resource={"steel": 2}))
    shed = solved(hourly, "operation_load_shedding_power").sel(resource="steel")
    assert float(shed.sum()) > 0  # shedding really occurs
    assert coarse.objective.value == pytest.approx(hourly.objective.value, rel=1e-9)
    name = "operation_load_shedding_costs_def"
    assert coarse.constraints[name].labels.dims == hourly.constraints[name].labels.dims
    assert "operation_load_shedding_block_2h_max_constraint" in coarse.constraints
    assert "operation_spillage_block_2h_max_constraint" in coarse.constraints


@pytest.mark.parametrize("case", ["methane", "process"])
def test_forced_coarse_balance_sums_hourly_contributors(
    case, parameters, parameters_dispatch_invest, process
):
    if case == "methane":
        flat, forced = methane_hourly(parameters), methane_forced(parameters)
    else:
        flat = process_hourly(parameters_dispatch_invest, process)
        forced = process_forced(parameters_dispatch_invest, process)
    hourly, coarse = solve(flat), solve(forced)
    block = coarse.constraints["operation_adequacy_block_2h_constraint"]
    assert "methane" in block.labels.indexes["resource"]
    assert coarse.objective.value <= hourly.objective.value * (1 + 1e-6) + 1e-6


def test_check_block_dims_fires():
    m = Model()
    cap = m.add_variables(coords=[xr.DataArray([0], dims="u", name="u")], name="cap")
    p2h = m.add_variables(
        coords=[xr.DataArray(np.arange(2), dims="time_block_2h", name="time_block_2h")],
        name="P",
    )
    dt = xr.DataArray(np.ones(4), dims="hour", coords={"hour": np.arange(4)})
    m.add_constraints(p2h * dt - cap <= 0, name="bad_second")
    with pytest.raises(RuntimeError, match="bad_second"):
        check_block_dims(m)

    m = Model()
    x = m.add_variables(coords=[xr.DataArray(np.arange(3), dims="i", name="i")], name="x")
    y = m.add_variables(coords=[xr.DataArray(np.arange(2), dims="k", name="k")], name="y")
    c = m.add_constraints(x == 1, name="widened")
    c.lhs = c.lhs + y
    with pytest.raises(RuntimeError, match="widened"):
        check_block_dims(m)

    check_block_dims(build_model(hand_toy_b()))  # a clean block model passes


@pytest.mark.filterwarnings("ignore:Time blocks give no gain")
def test_no_sel_without_a_coarse_resource(parameters, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("resource subset taken without a coarse balance")

    # C5: no coarse resource, no resource subset: build_model works on p itself
    monkeypatch.setattr(build_model_module, "_select_resources", forbidden)
    p = methane_hourly(parameters)
    blocked = with_blocks(p, conversion={"wind_onshore": 2})
    m = build_model(blocked)
    for name in [*m.variables, *m.constraints]:
        assert not name.startswith(BLOCK_PREFIXES), name
    hourly = m.constraints["operation_adequacy_constraint"]
    assert list(hourly.labels.indexes["resource"]) == list(p.resource.values)
    shed = m.variables["operation_load_shedding_power"]
    assert list(shed.labels.indexes["resource"]) == list(p.resource.values)
    assert_t0(p)
