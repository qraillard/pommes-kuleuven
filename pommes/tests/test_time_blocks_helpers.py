"""change_resolution and add_to_balances (multi-resolution, sections 7-8)."""

import dataclasses
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
    p = with_blocks(parameters, conversion={"ocgt": 4})
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
