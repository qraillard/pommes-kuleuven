"""change_resolution and add_to_balances (multi-resolution, sections 7-8)."""

import dataclasses
import subprocess
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from linopy import Model

from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs
from pommes.tests.time_blocks_helper import solved, with_blocks
from pommes.utils import add_to_balances, block_dim, change_resolution, time_blocks


def tb_of(hours=8, steps=(1.0,), **resource):
    """TimeBlocks of a minimal dataset: ``resource`` forces balance lengths."""
    names = ["electricity", "steel", "ore"]
    step = np.resize(np.array(steps, dtype="float64"), hours)
    p = xr.Dataset(
        {
            "time_step_duration": ("hour", step),
            "resource_time_block_hrs": (
                "resource",
                np.array([resource.get(r, np.nan) for r in names], dtype=float),
            ),
        },
        coords={"hour": np.arange(hours), "resource": names},
    )
    return time_blocks(p)


def hourly_expr(tb):
    """``1 * x`` over the hours; returns (model, variable, expression)."""
    m = Model()
    hours = tb.block_of_hour[next(iter(tb.block_of_hour))]["hour"]
    x = m.add_variables(coords=[hours], name="x")
    return m, x, 1 * x


def block_expr(tb, L):
    m = Model()
    x = m.add_variables(coords=[tb.duration[L][block_dim(L)]], name="x")
    return m, x, 1 * x


def rows(expr):
    """Per row: {variable label: coefficient}, padding removed."""
    v, c = expr.vars.values, expr.coeffs.values
    v, c = v.reshape(-1, v.shape[-1]), c.reshape(-1, c.shape[-1])
    return [
        {int(a): float(b) for a, b in zip(rv, rc) if a != -1}
        for rv, rc in zip(v, c)
    ]


def test_identity_returns_the_same_expression():
    tb = tb_of(steel=4)
    _, _, e = hourly_expr(tb)
    assert change_resolution(e, tb, None, None) is e
    _, _, e = block_expr(tb, 4)
    assert change_resolution(e, tb, 4, 4) is e


def test_hourly_to_block_sums():
    tb = tb_of(hours=6, steel=2)
    _, _, e = hourly_expr(tb)
    out = change_resolution(e, tb, None, 2)
    assert out.dims == ("time_block_2h", "_term")
    assert rows(out) == [{0: 1, 1: 1}, {2: 1, 3: 1}, {4: 1, 5: 1}]


def test_block_to_hourly_broadcasts_the_duration_share():
    tb = tb_of(hours=6, steps=(1, 1, 2), steel=4)
    _, x, e = block_expr(tb, 4)
    out = change_resolution(e, tb, 4, None)
    assert out.dims == ("hour", "_term")
    r = rows(out)
    assert [list(d.values()) for d in r] == [
        [0.25], [0.25], [0.5], [0.25], [0.25], [0.5]
    ]
    assert [next(iter(d)) for d in r] == [0, 0, 0, 1, 1, 1]


def test_short_last_block():
    tb = tb_of(hours=10, steel=4)
    _, _, e = block_expr(tb, 4)
    r = rows(change_resolution(e, tb, 4, None))
    assert [next(iter(d.values())) for d in r] == [0.25] * 8 + [0.5, 0.5]


def test_block_to_finer_block():
    tb = tb_of(hours=8, steel=4, ore=2)
    _, _, e = block_expr(tb, 4)
    out = change_resolution(e, tb, 4, 2)
    assert out.dims == ("time_block_2h", "_term")
    assert rows(out) == [{0: 0.5}, {0: 0.5}, {1: 0.5}, {1: 0.5}]


def test_block_to_coarser_block():
    tb = tb_of(hours=8, steel=4, ore=2)
    _, _, e = block_expr(tb, 2)
    out = change_resolution(e, tb, 2, 4)
    assert out.dims == ("time_block_4h", "_term")
    assert rows(out) == [{0: 1, 1: 1}, {2: 1, 3: 1}]


@pytest.mark.parametrize(
    "src,dst", [(None, 2), (4, None), (4, 2), (2, 4)]
)
def test_no_leftover_coordinate(src, dst):
    tb = tb_of(hours=8, steel=4, ore=2)
    _, _, e = hourly_expr(tb) if src is None else block_expr(tb, src)
    out = change_resolution(e, tb, src, dst)
    assert set(out.coords) <= set(out.dims)


def test_two_d_group_is_refused():
    tb = tb_of(hours=4, steel=2)
    g = xr.DataArray(
        np.zeros((4, 2), dtype="int64"),
        dims=("hour", "operating_condition"),
        coords={"hour": np.arange(4)},
        name=block_dim(2),
    )
    bad = dataclasses.replace(tb, block_of_hour={2: g})
    _, _, e = hourly_expr(tb)
    with pytest.raises(AssertionError):
        change_resolution(e, bad, None, 2)


@pytest.mark.parametrize("src,dst", [(None, 2), (4, None), (4, 2), (2, 4)])
def test_two_d_group_is_refused_on_every_branch(src, dst):
    tb = tb_of(hours=8, steel=4, ore=2)
    key = dst if src is None else (dst if dst is not None and dst > src else src)
    g = tb.block_of_hour[key]
    bad = dataclasses.replace(
        tb,
        block_of_hour={
            **tb.block_of_hour,
            key: xr.concat([g, g], "operating_condition")
            .transpose("hour", "operating_condition")
            .rename(g.name),
        },
    )
    _, _, e = hourly_expr(tb) if src is None else block_expr(tb, src)
    with pytest.raises(AssertionError):
        change_resolution(e, bad, src, dst)


def test_identity_also_refuses_a_leftover_coordinate():
    tb = tb_of(hours=4, steel=2)
    _, _, e = hourly_expr(tb)
    e = e.assign_coords(label=("hour", list("abcd")))
    with pytest.raises(AssertionError):
        change_resolution(e, tb, None, None)


def test_leftover_coordinate_is_refused():
    tb = tb_of(hours=4, steel=2)
    _, _, e = block_expr(tb, 2)
    e = e.assign_coords(label=(block_dim(2), ["a", "b"]))
    with pytest.raises(AssertionError):
        change_resolution(e, tb, 2, None)


def pd_index(dim, labels):
    return pd.Index(list(labels), name=dim)


def constraint(m, name, **coords):
    """A hand-made constraint ``x == 0`` over ``coords`` (dim -> labels)."""
    x = m.add_variables(
        coords=[pd_index(k, v) for k, v in coords.items()], name=f"v_{name}"
    )
    return m.add_constraints(1 * x == 0, name=name)


def test_inactive_is_todays_operator():
    tb = tb_of()
    assert not tb.active
    m = Model()
    hour = pd_index("hour", range(3))
    w = m.add_variables(coords=[hour], name="w")
    z = m.add_variables(coords=[hour], name="z")
    f = m.add_variables(coords=[hour], name="f")
    c1 = m.add_constraints(1 * w == 0, name="c1")
    c2 = m.add_constraints(1 * w == 0, name="c2")
    add_to_balances(c1, 2 * z, tb)
    c2.lhs += 2 * z
    add_to_balances(c1, -(3 * f), tb)
    c2.lhs -= 3 * f
    for attr in ("vars", "coeffs", "const"):
        xr.testing.assert_equal(getattr(c1.lhs, attr), getattr(c2.lhs, attr))


def test_routes_each_resource_to_its_balance():
    tb = tb_of(hours=4, steel=2)
    m = Model()
    y = m.add_variables(
        coords=[
            pd_index("hour", range(4)),
            pd_index("resource", ["electricity", "steel"]),
        ],
        name="y",
    )
    hourly = constraint(
        m, "operation_adequacy_constraint", hour=range(4), resource=["electricity"]
    )
    coarse = constraint(
        m,
        "operation_adequacy_block_2h_constraint",
        **{block_dim(2): range(2), "resource": ["steel"]},
    )
    add_to_balances(hourly, 1 * y, tb)
    mine = {int(v) for v in y.labels.values.ravel()}
    elec = y.labels.sel(resource="electricity").values
    steel = y.labels.sel(resource="steel").values
    for h, row in enumerate(rows(hourly.lhs)):
        assert {k for k in row if k in mine} == {int(elec[h])}
    for b, row in enumerate(rows(coarse.lhs)):
        assert {k for k in row if k in mine} == {int(steel[2 * b]), int(steel[2 * b + 1])}


def test_join_is_by_label():
    tb = tb_of(hours=2, steel=2)
    m = Model()
    hour = pd_index("hour", range(2))
    res = pd_index("resource", ["electricity"])
    ab = pd_index("area", ["a", "b"])
    ba = pd_index("area", ["b", "a"])
    con = m.add_constraints(
        1 * m.add_variables(coords=[ab, hour, res], name="base") == 0,
        name="operation_adequacy_constraint",
    )
    y = m.add_variables(coords=[ba, hour, res], name="y")
    add_to_balances(con, 1 * y, tb)
    for area in ["a", "b"]:
        sel = con.lhs.sel(area=area)
        own = set(y.labels.sel(area=area).values.ravel())
        found = {int(v) for v in sel.vars.values.ravel() if v in own}
        assert found == own, (area, found, own)


def test_routing_with_another_tb_raises():
    other = tb_of(hours=4, ore=2)  # steel is hourly in this tb
    m = Model()
    hour = pd_index("hour", range(4))
    res = pd_index("resource", ["electricity", "steel"])
    y = m.add_variables(coords=[hour, res], name="y")
    con = constraint(m, "operation_adequacy_constraint", hour=range(4), resource=["electricity"])
    with pytest.raises(AssertionError):
        add_to_balances(con, 1 * y, other)


def test_unrouted_resource_raises():
    tb = tb_of(hours=4, steel=2)
    m = Model()
    hour = pd_index("hour", range(4))
    res = pd_index("resource", ["electricity", "unknown"])
    y = m.add_variables(coords=[hour, res], name="y")
    con = constraint(m, "operation_adequacy_constraint", hour=range(4), resource=["electricity"])
    with pytest.raises(AssertionError):
        add_to_balances(con, 1 * y, tb)


def test_build_model_raises_time_blocks_refusals(parameters):
    # datasets that skipped check_inputs still meet R1 and R4 in build_model
    with pytest.raises(ValueError, match="ocgt"):
        build_model(with_blocks(parameters, conversion={"ocgt": 1.5}))
    with pytest.raises(ValueError):
        build_model(
            with_blocks(
                parameters, conversion={"ocgt": 4}, resource={"electricity": 6}
            )
        )


def test_build_model_warns_once_and_check_inputs_stays_silent(parameters):
    # carbon off: with a block technology it is refused until Task 9
    p = with_blocks(parameters.assign(carbon=np.array(False)), conversion={"ocgt": 4})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = check_inputs(p)
    with pytest.warns(UserWarning, match="no gain") as rec:
        build_model(p)
    assert len([w for w in rec if "no gain" in str(w.message)]) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_model(with_blocks(parameters))


def test_solved_reads_area_and_tech_first(parameters):
    m = build_model(parameters)
    m.solve(solver_name="highs")
    da = solved(m, "operation_conversion_power")
    assert da.dims[:2] == ("area", "conversion_tech")


def test_solved_keeps_values_and_drops_all_nan_techs():
    da = xr.DataArray(
        np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, np.nan]]),
        dims=("conversion_tech", "hour"),
        coords={"conversion_tech": ["a", "b", "c"], "hour": [0, 1]},
    ).expand_dims(area=["x"])
    m = SimpleNamespace(solution={"v": da.transpose("hour", "conversion_tech", "area")})
    out = solved(m, "v")
    assert out.dims[:2] == ("area", "conversion_tech")
    assert list(out.conversion_tech.values) == ["a", "c"]  # b: all NaN
    np.testing.assert_array_equal(out.values[0], [[1.0, 2.0], [3.0, np.nan]])


def _block_source_case(steps, src, dst, resource):
    """A block variable on ``src``-hour blocks routed by ``add_to_balances`` (uneven steps)."""
    lengths = {"steel": dst} if src == dst else {"steel": dst, "ore": src}
    tb = tb_of(hours=8, steps=steps, **lengths)
    m = Model()
    y = m.add_variables(
        coords=[
            tb.duration[src][block_dim(src)].to_index(),
            pd_index("resource", resource),
        ],
        name="y",
    )
    return tb, m, y


def _y_rows(con, y):
    """Rows of ``con`` restricted to the labels of ``y``."""
    mine = {int(v) for v in y.labels.values.ravel()}
    return [{k: v for k, v in row.items() if k in mine} for row in rows(con.lhs)]


STEPS = (1, 1, 2)  # 8 steps: blocks of 4 h hold 3, 3 and 2 steps (the last lasts 2 h)


def test_block_source_routes_to_both_balances():
    """M2: a 4 h block source reaches an hourly balance by share and a 4 h balance by identity."""
    tb, m, y = _block_source_case(STEPS, 4, 4, ["electricity", "steel"])
    hourly = constraint(m, "operation_adequacy_constraint", hour=range(8), resource=["electricity"])
    coarse = constraint(
        m,
        "operation_adequacy_block_4h_constraint",
        **{block_dim(4): range(3), "resource": ["steel"]},
    )
    add_to_balances(hourly, 1 * y, tb, src=4)
    share = [row[next(iter(row))] for row in _y_rows(hourly, y)]
    assert share == pytest.approx([1 / 4, 1 / 4, 2 / 4, 1 / 4, 1 / 4, 2 / 4, 1 / 2, 1 / 2])
    labels = y.labels.sel(resource="electricity").values
    assert [list(r) for r in _y_rows(hourly, y)] == [[int(labels[b])] for b in [0, 0, 0, 1, 1, 1, 2, 2]]
    steel = y.labels.sel(resource="steel").values
    assert _y_rows(coarse, y) == [{int(steel[b]): 1.0} for b in range(3)]


def test_block_source_reaches_a_finer_balance_by_share():
    """M2: a 4 h source into a 2 h balance gives each 2 h block its share of the 4 h block."""
    tb, m, y = _block_source_case(STEPS, 4, 2, ["steel"])
    coarse = constraint(
        m,
        "operation_adequacy_block_2h_constraint",
        **{block_dim(2): range(5), "resource": ["steel"]},
    )
    add_to_balances(coarse, 1 * y, tb, src=4)
    lab = y.labels.sel(resource="steel").values
    assert _y_rows(coarse, y) == [
        {int(lab[0]): 0.5},
        {int(lab[0]): 0.5},
        {int(lab[1]): 0.5},
        {int(lab[1]): 0.5},
        {int(lab[2]): 1.0},
    ]


def test_block_source_reaches_a_coarser_balance_by_sum():
    """M2: a 2 h source into a 4 h balance sums the 2 h blocks of each 4 h block."""
    tb, m, y = _block_source_case(STEPS, 2, 4, ["steel"])
    coarse = constraint(
        m,
        "operation_adequacy_block_4h_constraint",
        **{block_dim(4): range(3), "resource": ["steel"]},
    )
    add_to_balances(coarse, 1 * y, tb, src=2)
    lab = y.labels.sel(resource="steel").values
    assert _y_rows(coarse, y) == [
        {int(lab[0]): 1.0, int(lab[1]): 1.0},
        {int(lab[2]): 1.0, int(lab[3]): 1.0},
        {int(lab[4]): 1.0},
    ]


@pytest.mark.parametrize("src,dst", [(None, 2), (4, None), (4, 2), (2, 4)])
def test_a_reordered_axis_is_refused(src, dst):
    """M1: linopy's groupby and isel are positional, so a reversed axis must not pass."""
    tb = tb_of(hours=8, steel=4, ore=2)
    m = Model()
    axis = (
        tb.block_of_hour[dst]["hour"].to_index()
        if src is None
        else tb.duration[src][block_dim(src)].to_index()
    )
    x = m.add_variables(coords=[axis[::-1]], name="x")
    with pytest.raises(AssertionError, match="axis"):
        change_resolution(1 * x, tb, src, dst)


def test_dead_slices_are_not_summed_into_a_coarse_row():
    """I1: a resource that no technology touches gets no dead terms from an hourly source."""
    tb = tb_of(hours=4, steel=2)
    m = Model()
    hour = pd_index("hour", range(4))
    res = pd_index("resource", ["electricity", "steel"])
    y = m.add_variables(coords=[hour, res], name="y", mask=xr.DataArray(
        [[True, False]] * 4, dims=("hour", "resource"), coords={"hour": range(4), "resource": ["electricity", "steel"]}
    ))
    hourly = constraint(m, "operation_adequacy_constraint", hour=range(4), resource=["electricity"])
    coarse = constraint(
        m, "operation_adequacy_block_2h_constraint", **{block_dim(2): range(2), "resource": ["steel"]}
    )
    before = coarse.lhs.vars.sizes["_term"]
    add_to_balances(hourly, 1 * y, tb)
    assert coarse.lhs.vars.sizes["_term"] == before  # steel is all -1 in y: nothing added
    assert hourly.lhs.vars.sizes["_term"] == 2  # the electricity slice is routed


def test_a_constant_on_a_dead_slice_is_kept():
    """I1: an all-masked slice with a nonzero constant still moves the right-hand side."""
    tb = tb_of(hours=4, steel=2)
    m = Model()
    hour = pd_index("hour", range(4))
    res = pd_index("resource", ["electricity", "steel"])
    y = m.add_variables(coords=[hour, res], name="y", mask=xr.DataArray(
        [[True, False]] * 4, dims=("hour", "resource"), coords={"hour": range(4), "resource": ["electricity", "steel"]}
    ))
    hourly = constraint(m, "operation_adequacy_constraint", hour=range(4), resource=["electricity"])
    coarse = constraint(
        m, "operation_adequacy_block_2h_constraint", **{block_dim(2): range(2), "resource": ["steel"]}
    )
    add_to_balances(hourly, 1 * y + 3, tb)
    assert coarse.rhs.values.ravel().tolist() == [-6.0, -6.0]  # 3 per hour, 2 hours per block


def test_carbon_with_a_block_technology_is_refused(parameters):
    """M3: until the carbon rows read the block groups, carbon on is refused."""
    for lengths in ({"combined": {"electric_boiler": 2}}, {"conversion": {"ocgt": 2}}):
        with pytest.raises(ValueError, match="carbon"):
            build_model(with_blocks(parameters, **lengths))


def test_the_time_map_guards_survive_python_O():
    """Minor 4: ``check_group`` and ``check_axis`` raise explicitly, so ``python -O`` keeps them."""
    code = (
        "import numpy as np, pandas as pd, xarray as xr\n"
        "from pommes.utils import check_axis, check_group\n"
        "g = xr.DataArray(np.zeros((2, 2)), dims=('hour', 'x'))\n"
        "for call in (lambda: check_group(g, None),\n"
        "             lambda: check_axis(xr.DataArray([0, 1], dims='hour', coords={'hour': [1, 0]}), 'hour', pd.Index([0, 1]))):\n"
        "    try:\n"
        "        call()\n"
        "    except AssertionError:\n"
        "        continue\n"
        "    raise SystemExit('a guard was dropped')\n"
    )
    run = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr or run.stdout
