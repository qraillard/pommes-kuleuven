"""Transport technologies on time blocks: the parameter, the refusals R11 and R12, the pair mask, the groups."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pommes.model import transport
from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import (
    check_inputs,
    check_time_blocks,
)
from pommes.model.transport import transport_groups
from pommes.tests.time_blocks_helper import (
    BLOCK_PARAMETERS,
    assert_same,
    assert_twin_equal,
    binds,
    flat_twin,
    solve_model,
    solved,
    transport_t2_availability_dataset,
    transport_t2_efficiency_dataset,
    transport_t2_hurdle_dataset,
    transport_t2_interior_dataset,
    transport_t2_mixed_group_costs_dataset,
    transport_t2_ramps_uneven_steps_dataset,
    transport_t2_short_last_block_dataset,
    transport_t2_two_lengths_dataset,
    transport_toy,
    with_blocks,
)
from pommes.utils import (
    TimeBlocks,
    check_block_dims,
    check_block_transport,
    expand_to_hours,
    time_blocks,
    transport_block_availability,
    transport_pair_exists,
)

LENGTH = "transport_time_block_hrs"
pytestmark = pytest.mark.filterwarnings("ignore:Time blocks give no gain")


def lengths(p, **values):
    """The parameter on ``transport_tech``: NaN except for the named techs."""
    da = xr.DataArray(
        np.full(p.transport_tech.size, np.nan),
        dims="transport_tech",
        coords={"transport_tech": p.transport_tech.values},
    )
    for tech, value in values.items():
        da.loc[{"transport_tech": tech}] = value
    return da


def ship(p, L=5.0, tech="power_line"):
    return p.assign({LENGTH: lengths(p, **{tech: L})})


def on_pairs(p, name, cells, fill, extra=None):
    """``name`` on (link, transport_tech[, year_op]) set to ``cells = {(link, tech): value}``."""
    dims = ["link", "transport_tech"] + (["year_op"] if extra else [])
    shape = [p.sizes[d] for d in dims]
    da = xr.DataArray(np.full(shape, fill), dims=dims, coords={d: p[d].values for d in dims})
    for (link, tech), value in cells.items():
        da.loc[{"link": link, "transport_tech": tech}] = value
    return p.assign({name: da})


def without_pair(p, link, tech):
    """``p`` where ``(link, tech)`` carries no resource (its ``transport_resource`` is NaN)."""
    grid = xr.DataArray(np.zeros((p.sizes["link"], p.sizes["transport_tech"])), coords=[p.link, p.transport_tech])
    resource = p.transport_resource.broadcast_like(grid).transpose("link", "transport_tech").copy().astype(object)
    resource.loc[{"link": link, "transport_tech": tech}] = "nan"
    return p.assign(transport_resource=resource.astype(str))


def with_delay(p, cells):
    return on_pairs(p, "transport_delay", cells, 0.0)


def with_linepack(p, cells):
    return on_pairs(p, "transport_linepack_capacity", cells, np.nan, extra=True)


# --- the parameter ---


def test_parameter_is_defaulted(parameters):
    p = check_inputs(parameters.drop_vars(LENGTH))
    assert p[LENGTH].dtype == np.float64 and bool(p[LENGTH].isnull().all())
    off = parameters.drop_vars(LENGTH).assign(transport=np.array(False))
    assert LENGTH not in check_inputs(off)
    # a length given while the module is off stays, and is ignored
    given = ship(parameters).assign(transport=np.array(False))
    assert LENGTH in check_inputs(given)
    assert time_blocks(given).active is False
    tb = time_blocks(ship(parameters, 24))
    assert tb.tech_length["transport"].dims == ("transport_tech",)
    assert tb.techs("transport", 24) == ["power_line"]


@pytest.mark.parametrize("dim", ["link", "area", "year_op", "hour"])
def test_parameter_refuses_other_coordinates(parameters, dim):
    p = parameters.copy()
    p[LENGTH] = lengths(p, power_line=24).expand_dims({dim: p[dim].values})
    with pytest.raises(ValueError, match=LENGTH):
        check_inputs(p)
    with pytest.raises(
        ValueError,
        match="transport_time_block_hrs: only the transport_tech coordinate is allowed",
    ):
        time_blocks(p)


def test_parameter_refuses_operating_condition(parameters):
    p = parameters.assign_coords(operating_condition=["a", "b"]).assign(
        operating_condition_weight=xr.DataArray([0.5, 0.5], dims="operating_condition")
    )
    p[LENGTH] = lengths(p, power_line=24).expand_dims(operating_condition=["a", "b"])
    with pytest.raises(ValueError, match=LENGTH):
        check_inputs(p)
    with pytest.raises(ValueError, match="only the transport_tech coordinate is allowed"):
        time_blocks(p)


def test_r1_r2_r3_with_a_ship_length(parameters):
    for bad in (0.5, -1):
        with pytest.raises(ValueError, match=rf"transport_time_block_hrs\[power_line\] = {bad}"):
            time_blocks(ship(parameters, bad))
    steps = parameters.assign(time_step_duration=xr.full_like(parameters.time_step_duration, 3.0))
    with pytest.raises(ValueError, match="crosses a block boundary") as e:
        time_blocks(ship(steps, 5))
    assert "power_line" not in str(e.value)
    zero = ship(parameters, 5).copy(deep=True)
    zero["time_step_duration"][3] = 0.0
    with pytest.raises(ValueError, match="time_step_duration") as e:
        time_blocks(zero)
    assert "power_line" not in str(e.value)


# --- R11 and R12 ---


def r11(link, value, tech="power_line", L=5):
    return (
        f"{tech}: transport_delay = {value} h on link {link}: a block transport technology "
        f"({L} h) takes no delay."
    )


def test_r11_delay(parameters):
    base = ship(parameters)
    p = with_delay(base, {("link_2", "power_line"): 3.0, ("link_1", "power_line"): 2.0})
    with pytest.raises(ValueError) as e:
        check_time_blocks(p)
    assert str(e.value) == r11("link_1", 2.0)
    for value in (-1.0, 0.4):
        q = with_delay(base, {("link_2", "power_line"): value})
        with pytest.raises(ValueError) as e:
            check_time_blocks(q)
        assert str(e.value) == r11("link_2", value)
    # a NaN delay reads as 0
    check_time_blocks(with_delay(base, {("link_1", "power_line"): np.nan}))
    # a pair that does not exist: its resource is NaN
    gone = with_delay(base, {("link_2", "power_line"): 2.0})
    check_time_blocks(without_pair(gone, "link_2", "power_line"))
    # a delay on an hourly tech
    check_time_blocks(with_delay(base, {("link_1", "methane_pipe"): 2.0}))


def test_r11_uneven_steps_give_r11(parameters):
    p = with_delay(ship(parameters, 3), {("link_1", "power_line"): 2.0})
    steps = np.resize([1.0, 2.0], p.hour.size)
    p["time_step_duration"] = xr.DataArray(steps, dims="hour", coords={"hour": p.hour.values})
    with pytest.raises(ValueError) as e:
        build_model(p)
    assert str(e.value) == r11("link_1", 2.0, L=3)
    assert "uniform" not in str(e.value)


def r12(link, tech="power_line", L=5):
    return (
        f"{tech}: transport_linepack_capacity is finite on link {link}: a block transport "
        f"technology ({L} h) takes no linepack."
    )


def test_r12_linepack(parameters):
    base = ship(parameters)
    p = with_linepack(base, {("link_1", "power_line"): np.nan})
    p["transport_linepack_capacity"].loc[{"link": "link_1", "transport_tech": "power_line", "year_op": p.year_op.values[-1]}] = 5.0
    assert p.year_op.size > 1
    with pytest.raises(ValueError) as e:
        check_time_blocks(p)
    assert str(e.value) == r12("link_1")
    # finite on a pair that does not exist, and on an hourly tech
    gone = with_linepack(base, {("link_2", "power_line"): 5.0})
    check_time_blocks(without_pair(gone, "link_2", "power_line"))
    check_time_blocks(with_linepack(base, {("link_1", "methane_pipe"): 5.0}))
    # a length on a tech with no existing pair, or longer than the horizon
    lone = ship(parameters, 5, "big_methane_pipe")
    for link in lone.link.values:
        lone = without_pair(lone, link, "big_methane_pipe")
    lone = with_linepack(with_delay(lone, {("link_1", "big_methane_pipe"): 2.0}), {("link_1", "big_methane_pipe"): 5.0})
    check_time_blocks(lone)
    check_time_blocks(ship(parameters, 1000))


def test_r11_comes_before_r12(parameters):
    p = with_linepack(with_delay(ship(parameters), {("link_2", "power_line"): 2.0}), {("link_1", "power_line"): 5.0})
    with pytest.raises(ValueError, match="transport_delay"):
        check_time_blocks(p)


def test_r11_r12_without_check_inputs(parameters):
    delayed = with_delay(ship(parameters), {("link_1", "power_line"): 2.0})
    packed = with_linepack(ship(parameters), {("link_1", "power_line"): 5.0})
    for p, text in ((delayed, r11("link_1", 2.0)), (packed, r12("link_1"))):
        with pytest.raises(ValueError) as e:
            build_model(p)
        assert str(e.value) == text
        with pytest.raises(ValueError) as e:
            check_inputs(p)
        assert str(e.value) == text


def test_check_block_transport_returns_without_inputs(parameters):
    bad = with_delay(parameters, {("link_1", "power_line"): 2.0})
    check_block_transport(bad, time_blocks(bad))  # inactive
    p = ship(bad)
    tb = time_blocks(p)
    assert tb.active
    check_block_transport(p.drop_vars("transport_resource"), tb)
    check_block_transport(p.assign(transport=np.array(False)), time_blocks(p.assign(transport=np.array(False))))


# --- the pair mask ---


def _expected_mask(p):
    resources = {str(r) for r in p.resource.values}
    res = p.transport_resource.broadcast_like(
        xr.DataArray(np.zeros((p.sizes["link"], p.sizes["transport_tech"])), coords=[p.link, p.transport_tech])
    )
    return np.array(
        [[str(res.sel(link=l, transport_tech=t).item()) in resources for t in p.transport_tech.values] for l in p.link.values]
    )


def test_pair_exists_is_the_pair_mask(parameters):
    mask = transport_pair_exists(parameters)
    assert mask.dims == ("link", "transport_tech") and mask.dtype == bool
    assert mask.values.tolist() == _expected_mask(parameters).tolist() and mask.all()
    other = parameters.copy(deep=True)
    resource = other.transport_resource.copy().astype(object)
    resource.loc[{"transport_tech": "power_line"}] = "coal"
    other["transport_resource"] = resource.astype(str)
    mask = transport_pair_exists(other)
    assert mask.values.tolist() == _expected_mask(other).tolist()
    assert not mask.sel(transport_tech="power_line").any() and mask.sel(transport_tech="methane_pipe").all()


def test_r11_r12_two_offending_techs_on_a_reversed_axis(parameters):
    """R11 is tested over all techs before R12; among equal offences the first tech of the axis is named."""
    p = parameters.isel(transport_tech=slice(None, None, -1))
    assert list(p.transport_tech.values) == ["power_line", "methane_pipe", "big_methane_pipe"]
    base = p.assign({LENGTH: lengths(p, power_line=5, big_methane_pipe=5)})
    # a linepack on the first tech of the axis, a delay on the last: R11 comes first
    q = with_linepack(with_delay(base, {("link_1", "big_methane_pipe"): 2.0}), {("link_1", "power_line"): 5.0})
    with pytest.raises(ValueError) as e:
        check_time_blocks(q)
    assert str(e.value) == r11("link_1", 2.0, tech="big_methane_pipe")
    # two delays: the first tech in axis order, not in alphabetical order
    q = with_delay(base, {("link_1", "big_methane_pipe"): 2.0, ("link_1", "power_line"): 3.0})
    with pytest.raises(ValueError) as e:
        check_time_blocks(q)
    assert str(e.value) == r11("link_1", 3.0, tech="power_line")
    # two linepacks
    q = with_linepack(base, {("link_1", "big_methane_pipe"): 5.0, ("link_1", "power_line"): 5.0})
    with pytest.raises(ValueError) as e:
        check_time_blocks(q)
    assert str(e.value) == r12("link_1", tech="power_line")


# --- Task 2: groups, group variables, net generation, costs, routing ---

NG24 = "operation_transport_block_24h_net_generation"
HOURLY_NAMES = (
    "operation_transport_power",
    "operation_transport_power_in",
    "operation_transport_linepack_level",
    "operation_transport_net_generation",
)


def toy_blocks(n=48, techs=None, conversion=None, resource=None, transport=None, **kw):
    """``transport_toy`` on ``n`` steps of 1 h with the given block lengths (a dict per module)."""
    p = transport_toy([1.0] * n, **({"techs": techs} if techs else {}), **kw)
    return with_blocks(p, conversion=conversion, resource=resource, transport=transport)


def pairless(p, tech):
    """``p`` where ``tech`` carries no resource on any link."""
    for link in p.link.values:
        p = without_pair(p, link, tech)
    return p


def _coords_of(labels):
    """{label: frozenset of (dim, coordinate)} of the live labels of a variable or a constraint."""
    series = labels.to_series()
    out = {}
    for idx, label in series.items():
        if label >= 0:
            out[int(label)] = frozenset(zip(labels.dims, idx if isinstance(idx, tuple) else (idx,)))
    return out


def terms(m, name):
    """{row coordinates: {(variable, its coordinates): coefficient}} of the constraint ``name`` (live terms, summed)."""
    var_at = {}
    for vname in m.variables:
        for label, coords in _coords_of(m.variables[vname].labels).items():
            var_at[label] = (vname, coords)
    con = m.constraints[name]
    row_at = _coords_of(con.labels)
    flat = con.flat
    flat = flat[(flat.labels >= 0) & (flat.vars >= 0)]
    out = {}
    for label, var, coeff in zip(flat.labels, flat.vars, flat.coeffs):
        row = out.setdefault(row_at[int(label)], {})
        key = var_at[int(var)]
        row[key] = row.get(key, 0.0) + float(coeff)
    return out


def terms_of(m, name, variable):
    """[(row, variable coordinates, coefficient)] of the terms of ``variable`` in the constraint ``name``."""
    return [
        (dict(row), dict(coords), coeff)
        for row, items in terms(m, name).items()
        for (vname, coords), coeff in items.items()
        if vname == variable
    ]


def test_groups_unsplit_is_the_input():
    p = transport_toy([1.0] * 4)
    for q in (p, with_blocks(p, conversion={"mill": 2}), with_blocks(p)):
        groups = transport_groups(q, time_blocks(q))
        assert len(groups) == 1 and groups[0][0] is None and groups[0][1] is q and groups[0][2] is None
    # the module is off: a length is ignored
    off = with_blocks(p, transport={"S": 2}).assign(transport=np.array(False))
    assert transport_groups(off, time_blocks(off))[0][1] is off


def test_groups_split_by_length():
    p = toy_blocks(48, techs=["Z", "S2", "S3", "H"], transport={"Z": 24, "S2": 12, "S3": 24})
    groups = transport_groups(p, time_blocks(p))
    assert [(L, list(g.indexes["transport_tech"]), idx) for L, g, idx in groups] == [
        (None, ["H"], None),
        (12, ["S2"], None),
        (24, ["Z", "S3"], None),
    ]
    assert "transport_availability" in groups[0][1] and "transport_availability" not in groups[2][1]
    assert set(groups[2][1].indexes["transport_tech"]) <= set(p.indexes["transport_tech"])


def test_block_names_and_dims():
    p = toy_blocks(48, transport={"S": 24})
    m = build_model(p)
    check_block_dims(m)
    prefix = "operation_transport_block_24h_"
    assert m.variables[prefix + "power"].labels.dims == ("link", "transport_tech", "time_block_24h", "year_op")
    assert m.variables[prefix + "power_in"].labels.dims == ("link", "transport_tech", "time_block_24h", "year_op")
    assert m.variables[prefix + "net_generation"].labels.dims == (
        "area",
        "time_block_24h",
        "transport_tech",
        "resource",
        "year_op",
    )
    for name in ("power", "power_in", "net_generation"):
        assert list(m.variables[prefix + name].labels.indexes["transport_tech"]) == ["S"]
        assert m.variables[prefix + name].labels.sizes["time_block_24h"] == 2
    for name in HOURLY_NAMES[:2] + HOURLY_NAMES[3:]:
        assert list(m.variables[name].labels.indexes["transport_tech"]) == ["H"]
        assert "time_block_24h" not in m.variables[name].labels.dims
    assert "operation_transport_linepack_level" in m.variables
    assert prefix + "linepack_level" not in m.variables
    for name in ("operation_transport_power_capacity", "planning_transport_power_capacity", "operation_transport_costs"):
        assert list(m.variables[name].labels.indexes["transport_tech"]) == ["H", "S"]
    (L, p_g, _), = [g for g in transport_groups(p, time_blocks(p)) if g[0] == 24]
    assert set(p_g.indexes["transport_tech"]) <= set(p.indexes["transport_tech"])
    assert "transport_availability" not in p_g


def test_example_1_routing():
    p = toy_blocks(48, conversion={"mill": 24, "bunker": 24}, resource={"cargo": 24}, transport={"S": 24})
    m = build_model(p)
    block = m.constraints["operation_adequacy_block_24h_constraint"]
    assert "cargo" in block.labels.indexes["resource"] and "idle" not in block.labels.indexes["resource"]
    assert block.labels.sizes["time_block_24h"] == 2
    hourly = m.constraints["operation_adequacy_constraint"]
    assert "idle" in hourly.labels.indexes["resource"] and "cargo" not in hourly.labels.indexes["resource"]
    got = terms_of(m, "operation_adequacy_block_24h_constraint", NG24)
    assert len(got) == 4  # two areas, two blocks
    for row, var, coeff in got:
        assert coeff == 1.0 and row["resource"] == "cargo" == var["resource"]
        assert var["time_block_24h"] == row["time_block_24h"] and var["area"] == row["area"]
    # the net generation row: -NG + D_b (sum of P over the links arriving at the area - sum of Pin over those leaving it)
    year = int(p.year_op[0])
    rows = terms(m, "operation_transport_block_24h_net_generation_def")
    assert len(rows) == 4  # two areas, two blocks
    for row, items in rows.items():
        row = dict(row)
        area, block = row["area"], row["time_block_24h"]
        at = {("transport_tech", "S"), ("year_op", year), ("time_block_24h", block)}
        expected = {("operation_transport_block_24h_net_generation", frozenset({("area", area), ("resource", "cargo")} | at)): -1.0}
        for link in ("ab", "ba"):
            ends = p.transport_area_to.sel(link=link, transport_tech="S").item(), p.transport_area_from.sel(link=link, transport_tech="S").item()
            if ends[0] == area:
                expected[("operation_transport_block_24h_power", frozenset({("link", link)} | at))] = 24.0
            if ends[1] == area:
                expected[("operation_transport_block_24h_power_in", frozenset({("link", link)} | at))] = -24.0
        assert items == pytest.approx(expected), row
    # S runs b to a on ab and a to b on ba, H the other way: the ends are read on the group's own columns
    assert p.transport_area_to.sel(link="ab", transport_tech="S").item() == "a"
    assert p.transport_area_to.sel(link="ab", transport_tech="H").item() == "b"


def test_example_2_spread_flat():
    p = toy_blocks(48, conversion={"mill": 24, "bunker": 24}, transport={"S": 24}, silo=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        m = build_model(p)
    gain = [str(w.message) for w in record if "Time blocks give no gain" in str(w.message)]
    assert len(gain) == 1 and "cargo: hourly because of storage tech silo" in gain[0]
    assert "cargo" in m.constraints["operation_adequacy_constraint"].labels.indexes["resource"]
    got = terms_of(m, "operation_adequacy_constraint", NG24)
    assert len(got) == 2 * 48  # one ship term per (area, hour) row
    assert len({(r["area"], r["hour"]) for r, _, _ in got}) == len(got)
    for row, var, coeff in got:
        assert var["time_block_24h"] == row["hour"] // 24 and var["area"] == row["area"]
        assert coeff == pytest.approx(1 / 24)


def test_example_3_forced():
    for forced in (24, 168, 12):
        p = toy_blocks(
            48, conversion={"mill": 24, "bunker": 24}, resource={"cargo": forced}, transport={"S": 24}, silo=True
        )
        m = build_model(p)
        got = terms_of(m, f"operation_adequacy_block_{forced}h_constraint", NG24)
        key = f"time_block_{forced}h"
        if forced == 24:  # identity
            assert len(got) == 4
            assert all(c == 1.0 and v["time_block_24h"] == r[key] for r, v, c in got)
        elif forced == 168:  # the two blocks are summed into the one row
            assert len(got) == 4 and {r[key] for r, _, _ in got} == {0}
            assert sorted(v["time_block_24h"] for r, v, _ in got if r["area"] == "a") == [0, 1]
            assert all(c == 1.0 for _, _, c in got)
        else:  # a block broadcast to its two rows
            assert len(got) == 2 * 4
            assert all(c == 0.5 and v["time_block_24h"] == r[key] // 2 for r, v, c in got)


def test_block_costs_def_coefficients():
    p = with_blocks(transport_toy([1.0] * 6), transport={"S": 3})
    m = build_model(p)
    norm = 8760.0 / 6
    rows = {(dict(r)["area"], dict(r)["transport_tech"]): items for r, items in terms(m, "operation_transport_block_3h_costs_def").items()}
    assert set(rows) == {("a", "S"), ("b", "S")}
    year = int(p.year_op[0])
    for area in ("a", "b"):
        expected = {("operation_transport_costs", frozenset({("area", area), ("transport_tech", "S"), ("year_op", year)})): -1.0}
        for link in ("ab", "ba"):  # a link counts once at each of its two areas
            fixed = float(p.transport_fixed_cost.sel(link=link, transport_tech="S", year_op=year))
            hurdle = float(p.transport_hurdle_costs.sel(link=link, transport_tech="S", year_op=year))
            at = {("link", link), ("transport_tech", "S"), ("year_op", year)}
            expected[("operation_transport_power_capacity", frozenset(at))] = 0.5 * fixed
            for block in (0, 1):
                key = ("operation_transport_block_3h_power", frozenset(at | {("time_block_3h", block)}))
                expected[key] = 0.5 * hurdle * norm * 3.0
        got = rows[(area, "S")]
        assert got.keys() == expected.keys()
        for key, value in expected.items():
            assert got[key] == pytest.approx(value), key


def test_each_cost_cell_defined_once_mixed_group():
    p = with_blocks(transport_toy([1.0] * 6), transport={"S": 3})
    m = build_model(p)
    cells = []
    for name in ("operation_transport_costs_def", "operation_transport_block_3h_costs_def"):
        labels = m.constraints[name].labels
        live = labels.to_series()
        cells += [dict(zip(labels.dims, idx)) for idx, v in live.items() if v >= 0]
    keys = [(c["area"], c["transport_tech"], c["year_op"]) for c in cells]
    assert len(keys) == len(set(keys)) == 4
    assert {k[1] for k in keys} == {"H", "S"}


def test_block_group_without_pair_builds_and_defines_its_costs():
    p = pairless(toy_blocks(24, techs=["Z", "A", "H"], transport={"Z": 12, "A": 12}), "A")
    assert not transport_pair_exists(p).sel(transport_tech="A").any()
    m = build_model(p)  # no RuntimeError from the group guards
    rows = {
        (dict(r)["area"], dict(r)["transport_tech"]): items
        for r, items in terms(m, "operation_transport_block_12h_costs_def").items()
    }
    assert {t for _, t in rows} == {"Z", "A"}
    for area in ("a", "b"):
        only = {("operation_transport_costs", frozenset({("area", area), ("transport_tech", "A"), ("year_op", 2030)})): -1.0}
        assert rows[(area, "A")] == only  # pinned to 0
        assert len(rows[(area, "Z")]) > 1


def test_all_block_has_no_hourly_variable():
    p = toy_blocks(6, transport={"H": 3, "S": 3})
    m = build_model(p)
    check_block_dims(m)
    assert not set(HOURLY_NAMES) & set(m.variables)
    hourly_rows = [
        n
        for n in m.constraints
        if n.startswith(("operation_transport_power_in_def", "operation_transport_linepack", "operation_transport_ramp"))
        or n
        in (
            "operation_transport_power_max_constraint",
            "operation_transport_power_in_max_constraint",
            "operation_transport_net_generation_def",
            "operation_transport_costs_def",
        )
    ]
    assert hourly_rows == []
    for name in ("operation_transport_power_capacity", "planning_transport_power_capacity", "operation_transport_costs"):
        assert name in m.variables
    assert "operation_transport_block_3h_power" in m.variables
    assert "operation_transport_block_3h_costs_def" in m.constraints


def test_delay_loop_reads_the_hourly_group():
    """N4: a delay held by a block tech on a pair that does not exist builds no empty delay set."""
    p = pairless(toy_blocks(6, transport={"S": 3}, delay={"S": 2}), "S")
    m = build_model(p)
    assert [n for n in m.constraints if "power_in_def_delay" in n] == []


def test_net_generation_mask_reads_the_group():
    """N2: the mask of the block net generation is the one computed on the group's own techs."""
    p = pairless(toy_blocks(24, techs=["Z", "A", "H"], transport={"Z": 12, "A": 12}), "A")
    m = build_model(p)
    (L, p_g, _), = [g for g in transport_groups(p, time_blocks(p)) if g[0] == 12]
    expected = np.equal(p_g.transport_resource, p_g.resource).any("link").transpose("transport_tech", "resource")
    live = (m.variables["operation_transport_block_12h_net_generation"].labels >= 0).any(
        ["area", "time_block_12h", "year_op"]
    )
    assert list(live.indexes["transport_tech"]) == list(expected.indexes["transport_tech"]) == ["Z", "A"]
    assert live.transpose("transport_tech", "resource").values.tolist() == expected.values.tolist()
    assert expected.values.tolist() == [[False, True, False], [False, False, False]]


# --- guards, each broken once (H1, H2, H5, H13, H10) ---


def break_block_rows(monkeypatch, tweak=None, after=None):
    """``_TransportRows`` of the block groups built with ``tweak(args)`` applied to its positional arguments."""
    original = transport._TransportRows

    class Broken(original):
        def __init__(self, *args, **kwargs):
            args = list(args)
            block = args[3] is not None  # (m, p_g, tb, L, variables, capacity, costs, norm, pair, ...)
            if block and tweak:
                tweak(args)
            super().__init__(*args, **kwargs)
            if block and after:
                after(self)

    monkeypatch.setattr(transport, "_TransportRows", Broken)


def test_partner_in_another_order_fires(monkeypatch):
    """H1: the capacity restricted in another order is refused in ``costs_def`` (the group guard, or xarray's alignment)."""

    def reverse(args):
        capacity = args[5]
        args[5] = capacity.sel(transport_tech=capacity.indexes["transport_tech"][::-1])

    p = toy_blocks(6, transport={"H": 3, "S": 3})
    build_model(p)
    break_block_rows(monkeypatch, tweak=reverse)
    with pytest.raises((RuntimeError, ValueError), match="transport_tech"):
        build_model(p)


def test_costs_unrestricted_fires(monkeypatch):
    """H2: the costs variable not restricted to the group grows the row of ``costs_def``."""

    def unrestricted(args):
        args[6] = args[0].variables["operation_transport_costs"]

    p = toy_blocks(6, transport={"S": 3})
    break_block_rows(monkeypatch, tweak=unrestricted)
    with pytest.raises(RuntimeError, match="transport_tech"):
        build_model(p)


def test_time_step_duration_in_a_block_row_fires(monkeypatch):
    """H5: a block row built with the step duration keeps the hour dimension."""

    def hourly_step(rows):
        rows.D = rows.p.time_step_duration

    p = toy_blocks(6, transport={"S": 3})
    break_block_rows(monkeypatch, after=hourly_step)
    with pytest.raises(RuntimeError, match="hour"):
        build_model(p)


def test_block_term_without_src_fires(monkeypatch):
    """H13: a block term routed as an hourly one is refused."""
    routed = transport.add_to_balances
    monkeypatch.setattr(
        transport, "add_to_balances", lambda adequacy, expr, tb, src=None: routed(adequacy, expr, tb, src=None)
    )
    with pytest.raises(RuntimeError):
        build_model(toy_blocks(6, transport={"S": 3}))


def test_partition_missing_a_tech_fires(monkeypatch):
    """H10: groups that do not cover the transport techs are refused."""
    p = toy_blocks(6, transport={"S": 3})
    tb = time_blocks(p)
    techs = TimeBlocks.techs
    monkeypatch.setattr(
        TimeBlocks,
        "techs",
        lambda self, module, L: techs(self, module, L)[1:] if (module, L) == ("transport", None) else techs(self, module, L),
    )
    with pytest.raises(AssertionError, match="cover"):
        transport_groups(p, tb)


# --- interim guards (Tasks 2 to 5), deleted by the task that builds the rows ---


def test_interim_guard_fuel():
    with pytest.raises(NotImplementedError, match="fuel"):
        build_model(toy_blocks(6, transport={"S": 3}, fuel={"S": 0.1}))
    build_model(toy_blocks(6, transport={"S": 3}, fuel={"H": 0.1}))


def test_interim_guard_emissions():
    def factors(p, tech):
        da = xr.zeros_like(p.transport_fixed_cost)
        da.loc[{"transport_tech": tech}] = 0.3
        return p.assign(transport_fuel_emission_factor=da, transport_fuel_total_emission_factor=da)

    p = toy_blocks(6, transport={"S": 3}, fuel={"S": 0.1})
    with pytest.raises(NotImplementedError, match="emission"):
        build_model(factors(p, "S"))
    build_model(factors(toy_blocks(6, transport={"S": 3}, fuel={"H": 0.1}), "H"))
    # a factor without fuel on the block ship is no emission
    build_model(factors(toy_blocks(6, transport={"S": 3}), "S"))


# --- Task 3: block availability, power maximum, power_in with efficiency ---

DAY = [1.0] * 20 + [0.5, 0.5, np.nan, np.nan]
POWER24, POWER_IN24 = "operation_transport_block_24h_power", "operation_transport_block_24h_power_in"


def test_transport_block_availability():
    # a block with a 0.5, a block of NaN, a block of 1.2 and NaN (NaN reads as 1 before the minimum)
    p = toy_blocks(72, transport={"S": 24}, availability={"S": DAY + [np.nan] * 24 + [1.2] * 12 + [np.nan] * 12})
    tb = time_blocks(p)
    got = transport_block_availability(p, tb, 24, ["S", "H"])
    assert list(got.indexes["transport_tech"]) == ["S", "H"]
    np.testing.assert_allclose(got.sel(transport_tech="S", link="ab", year_op=2030).values, [0.5, 1.0, 1.0])
    np.testing.assert_allclose(got.sel(transport_tech="H", link="ab", year_op=2030).values, [1.0, 1.0, 1.0])  # none given
    assert got.sizes["time_block_24h"] == 3 and "hour" not in got.dims
    # two operating conditions are aggregated each alone
    b = xr.DataArray(np.ones(72), dims="hour", coords={"hour": p.hour})
    b[3] = 0.7
    a = p.transport_availability.sel(transport_tech=["S"]).squeeze(drop=True)
    both = xr.concat([a, a * 0 + b], dim=pd.Index(["x", "y"], name="operating_condition"))
    q = p.assign(transport_availability=both.broadcast_like(p.transport_availability))
    got = transport_block_availability(q, tb, 24, ["S"])
    assert {"operating_condition", "time_block_24h"} <= set(got.dims)
    s = got.sel(transport_tech="S", link="ab", year_op=2030)
    np.testing.assert_allclose(s.sel(operating_condition="x").values, [0.5, 1.0, 1.0])
    np.testing.assert_allclose(s.sel(operating_condition="y").values, [0.7, 1.0, 1.0])
    # no hour: the value itself, NaN read as 1
    flat = p.assign(transport_availability=xr.where(p.transport_availability.isel(hour=0, drop=True).isnull(), np.nan, 0.4))
    out = transport_block_availability(flat, tb, 24, ["S", "H"])
    assert "hour" not in out.dims and "time_block_24h" not in out.dims
    np.testing.assert_allclose(out.sel(transport_tech="S").values, 0.4)
    np.testing.assert_allclose(out.sel(transport_tech="H").values, 1.0)
    # absent parameter: 1.0
    assert float(transport_block_availability(p.drop_vars("transport_availability"), tb, 24, ["S"])) == 1.0


def test_block_group_has_no_hourly_availability():
    p = toy_blocks(48, transport={"S": 24}, availability={"S": DAY * 2, "H": DAY * 2})
    groups = dict((L, g) for L, g, _ in transport_groups(p, time_blocks(p)))
    assert "transport_block_availability" in groups[24] and "transport_availability" not in groups[24]
    assert "transport_availability" in groups[None] and "transport_block_availability" not in groups[None]
    assert list(groups[24].transport_block_availability.indexes["transport_tech"]) == ["S"]


def _pair_cells(p, tech):
    pair = transport_pair_exists(p).sel(transport_tech=tech)
    return {l for l in pair.link.values if bool(pair.sel(link=l))}


def test_block_power_rows_structure():
    prefix = "operation_transport_block_24h_"
    for p in (transport_t2_availability_dataset(), without_pair(transport_t2_availability_dataset(), "ab", "S")):
        p = p.assign(transport_efficiency=transport_t2_efficiency_dataset().transport_efficiency)
        m = build_model(p)
        check_block_dims(m)
        assert not [n for n in m.constraints if n.startswith(prefix) and ("delay" in n or "linepack" in n)]
        a = p.transport_availability.sel(transport_tech="S", link="ba", year_op=2030).fillna(1.0).values.reshape(2, 24).min(1)
        assert a.tolist() == [0.5, 0.8]
        links = _pair_cells(p, "S")
        rows = terms(m, prefix + "power_max_constraint")
        assert {dict(r)["link"] for r in rows} == links and len(rows) == 2 * len(links)
        for row, items in rows.items():
            row = dict(row)
            at = {("link", row["link"]), ("transport_tech", "S"), ("year_op", 2030)}
            cap = ("operation_transport_power_capacity", frozenset(at))
            power = (prefix + "power", frozenset(at | {("time_block_24h", row["time_block_24h"])}))
            assert items == pytest.approx({power: 1.0, cap: -float(a[row["time_block_24h"]])})
        rows = terms(m, prefix + "power_in_def")
        assert {dict(r)["link"] for r in rows} == links and len(rows) == 2 * len(links)
        for row, items in rows.items():
            row = dict(row)
            at = {("link", row["link"]), ("transport_tech", "S"), ("year_op", 2030), ("time_block_24h", row["time_block_24h"])}
            eff = float(p.transport_efficiency.sel(link=row["link"], transport_tech="S", year_op=2030))
            assert eff == (0.8 if row["link"] == "ba" else 1.0)
            assert items == pytest.approx({(prefix + "power_in", frozenset(at)): eff, (prefix + "power", frozenset(at)): -1.0})
        for name in ("power_max_constraint", "power_in_def"):  # the mask is the pair mask
            live = m.constraints[prefix + name].labels.sel(transport_tech="S") >= 0
            assert {l for l in live.link.values if bool(live.sel(link=l).all())} == links
    # the hourly group's rows keep the hourly availability on the hour axis
    m = build_model(transport_t2_availability_dataset().assign(transport_availability=lambda d: d.transport_availability.fillna(0.9)))
    assert "hour" in m.constraints["operation_transport_power_max_constraint"].labels.dims


def test_transport_block_equals_step_power_rows():
    base = transport_t2_availability_dataset().assign(transport_efficiency=transport_t2_efficiency_dataset().transport_efficiency)
    p = with_blocks(base.drop_vars(BLOCK_PARAMETERS, errors="ignore"), transport={"S": 1})
    m, hourly = solve_model(build_model(p)), solve_model(build_model(p.drop_vars(BLOCK_PARAMETERS, errors="ignore")))
    assert m.objective.value > 0
    assert_twin_equal(m, hourly, time_blocks(p), ["operation_transport_block_1h_power", "operation_transport_block_1h_power_in"])


# --- T2: the flat twin ---


def _compare_with_twin(p, lengths):
    """Block model against its flat twin (T2): objective, capacities, flows, net generation, adequacy duals."""
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, transport=lengths))
    tb = time_blocks(p)
    check_block_dims(m)
    powers = [f"operation_transport_block_{L}h_{k}" for L in sorted(set(lengths.values())) for k in ("power", "power_in")]
    assert_twin_equal(m, twin, tb, powers)
    hourly = [t for t in p.indexes["transport_tech"] if t not in lengths]
    for name in ("operation_transport_power", "operation_transport_power_in"):
        if hourly:
            assert_same(solved(m, name).sel(transport_tech=hourly), solved(twin, name).sel(transport_tech=hourly), name)
    for L in sorted(set(lengths.values())):
        techs = [t for t, v in lengths.items() if v == L]
        ng = expand_to_hours(solved(m, f"operation_transport_block_{L}h_net_generation"), tb, "energy")
        assert_same(ng, solved(twin, "operation_transport_net_generation").sel(transport_tech=techs), f"NG {L}")
    a, b = m.constraints["operation_adequacy_constraint"].dual, twin.constraints["operation_adequacy_constraint"].dual
    a, b = xr.align(a, b, join="exact")
    np.testing.assert_allclose(a.values, b.transpose(*a.dims).values, rtol=1e-6, atol=1e-6 * float(abs(b).max()))
    return m, twin


def test_t2_efficiency():
    m, twin = _compare_with_twin(transport_t2_efficiency_dataset(), {"S": 24})
    assert binds(m, "operation_transport_block_24h_power_max_constraint")
    pin, power = solved(m, POWER_IN24), solved(m, POWER24)
    assert float(abs(pin - power).max()) > 0.1  # the losses are real
    # a dropped efficiency changes the optimum
    q = transport_t2_efficiency_dataset()
    lossless = solve_model(build_model(q.assign(transport_efficiency=q.transport_efficiency.where(q.transport_tech != "S", 1.0))))
    assert abs(lossless.objective.value - m.objective.value) > 1.0


def test_t2_availability_inside_blocks():
    p = transport_t2_availability_dataset()
    m, twin = _compare_with_twin(p, {"S": 24})
    assert binds(m, "operation_transport_block_24h_power_max_constraint")
    cap = solved(m, "operation_transport_power_capacity").sel(transport_tech="S", link="ba").item()
    np.testing.assert_allclose(solved(m, POWER24).sel(link="ba").values.ravel(), [0.5 * cap, 0.8 * cap])


def test_t2_short_last_block():
    p = transport_t2_short_last_block_dataset()
    assert time_blocks(p).duration[24].values.tolist() == [24.0, 24.0, 12.0]
    m, _ = _compare_with_twin(p, {"S": 24})
    assert binds(m, "operation_transport_block_24h_power_max_constraint")


def test_t2_two_lengths():
    m, _ = _compare_with_twin(transport_t2_two_lengths_dataset(), {"S": 12, "T": 24})
    for name in ("operation_transport_costs_def", "operation_transport_block_12h_costs_def", "operation_transport_block_24h_costs_def"):
        assert name in m.constraints


def test_t2_mixed_group_costs():
    p = transport_t2_mixed_group_costs_dataset()
    m, twin = _compare_with_twin(p, {"S": 24})
    assert (solved(m, "operation_transport_power").sum() > 0) and (solved(m, POWER24).sum() > 0)  # both ships carry
    cost = solved(m, "operation_transport_costs")
    assert float(cost.sel(transport_tech="H").sum()) > 0 and float(cost.sel(transport_tech="S").sum()) > 0
    assert_same(cost, solved(twin, "operation_transport_costs"), "costs")


def test_t2_hurdle_per_link_and_year():
    p = transport_t2_hurdle_dataset()
    m, twin = _compare_with_twin(p, {"S": 24})
    assert list(m.variables[POWER24].labels.indexes["year_op"]) == [2030, 2035]
    assert_same(solved(m, "operation_transport_costs"), solved(twin, "operation_transport_costs"), "costs")


# --- Task 4: ramps on blocks ---

RAMP4 = "operation_transport_block_4h_"


def _ramp_terms(m, p, name):
    """{(link, block): (P_b coefficient, P_{b-1} coefficient, capacity coefficient)} of the live rows of ``name``."""
    out = {}
    for row, items in terms(m, RAMP4 + name).items():
        row = dict(row)
        at = {("link", row["link"]), ("transport_tech", "S"), ("year_op", 2030)}
        b = row["time_block_4h"]
        get = lambda var, extra: items.get((var, frozenset(at | extra)), 0.0)  # noqa: E731
        out[(row["link"], b)] = (
            get(RAMP4 + "power", {("time_block_4h", b)}),
            get(RAMP4 + "power", {("time_block_4h", b - 1)}),
            get("operation_transport_power_capacity", set()),
        )
        assert len(items) == 3 - (b == 0)
    return out


def _ramps_on(p, up, down):
    """``p`` with ramps ``up`` / ``down`` on ``S``: {link: value}, NaN elsewhere."""
    for name, cells in (("transport_ramp_up", up), ("transport_ramp_down", down)):
        p = on_pairs(p, name, {(l, "S"): v for l, v in cells.items()}, np.nan, extra=True)
    return p


def test_ramp_rows_structure():
    p = transport_t2_ramps_uneven_steps_dataset()
    tb = time_blocks(p)
    assert tb.first_step_duration[4].values.tolist() == [2.0] * 4  # the divisor is the first step, not D_b or D_b / n_b
    assert tb.duration[4].values.tolist() == [4.0] * 4
    d0 = 2.0
    m = build_model(p)
    check_block_dims(m)
    for name, sign in (("ramp_up_constraint", 1.0), ("ramp_down_constraint", -1.0)):
        rows = _ramp_terms(m, p, name)
        assert set(rows) == {(l, b) for l in ("ab", "ba") for b in (1, 2, 3)}  # none on block 0
        for cur, prev, cap in rows.values():
            assert (cur, prev, cap) == pytest.approx((sign / d0, -sign / d0, -0.1))
    # a ramp finite on one pair only, and the pair mask
    q = _ramps_on(p, {"ab": 0.1, "ba": 0.1}, {"ba": 0.1})
    m = build_model(q)
    assert {l for l, _ in _ramp_terms(m, q, "ramp_up_constraint")} == {"ab", "ba"}
    assert {l for l, _ in _ramp_terms(m, q, "ramp_down_constraint")} == {"ba"}
    q = without_pair(_ramps_on(p, {"ab": 0.1, "ba": 0.1}, {"ab": 0.1, "ba": 0.1}), "ab", "S")
    m = build_model(q)
    for name in ("ramp_up_constraint", "ramp_down_constraint"):
        assert {l for l, _ in _ramp_terms(m, q, name)} == {"ba"}
    # no ramp, no row
    m = build_model(_ramps_on(p, {}, {}))
    assert not [n for n in m.constraints if n.startswith(RAMP4) and "ramp" in n and (m.constraints[n].labels >= 0).any()]


def test_t2_ramps_uneven_steps():
    p = transport_t2_ramps_uneven_steps_dataset()
    m, twin = _compare_with_twin(p, {"S": 4})
    for name in ("ramp_up_constraint", "ramp_down_constraint"):
        row = m.constraints[RAMP4 + name]
        assert int((row.labels >= 0).sum()) == 6
        assert binds(m, RAMP4 + name)
    assert float(abs(solved(m, RAMP4 + "power_in") - solved(m, RAMP4 + "power")).max()) > 0.1  # Pin is not P
    # the ramps hold on P with the first step as divisor, and each is tight somewhere (the twin runs the same code, so it
    # cannot see a swapped row or a ramp written on Pin: the solution does)
    power = solved(m, RAMP4 + "power").sel(transport_tech="S")
    cap = solved(m, "operation_transport_power_capacity").sel(transport_tech="S")
    step = (power - power.shift(time_block_4h=1)) / 2.0
    for rate in (step, -step):
        assert float((rate - 0.1 * cap).where(cap > 1e-6).max()) == pytest.approx(0.0, abs=1e-6)  # a link with no ship has none
    free = solve_model(build_model(_ramps_on(p, {}, {})))
    assert abs(free.objective.value - m.objective.value) > 1.0  # the ramps matter


def test_t1_with_ramps():
    avail = [1.0, 0.4] * 8
    p = _t2_steps3(avail)
    m = solve_model(build_model(p))
    hourly = solve_model(build_model(p.drop_vars(BLOCK_PARAMETERS, errors="ignore")))
    assert binds(m, "operation_transport_block_3h_ramp_up_constraint") and binds(m, "operation_transport_block_3h_ramp_down_constraint")
    assert_twin_equal(m, hourly, time_blocks(p), ["operation_transport_block_3h_power", "operation_transport_block_3h_power_in"])


def _t2_steps3(avail):
    from pommes.tests.time_blocks_helper import _transport_t2

    p = _transport_t2([3.0] * 16, {"H": 1.5, "S": 3.0}, eff={"S": 0.8}, ramp={"S": (0.1, 0.1)}, availability={"S": avail})
    return with_blocks(p, transport={"S": 3})


def test_t2_interior():
    p = transport_t2_interior_dataset()
    m, _ = _compare_with_twin(p, {"S": 24, "T": 12})
    power = solved(m, "operation_transport_block_24h_power")
    assert len(np.unique(power.round(6).values)) > 2  # the flows vary from block to block
    con = m.constraints["operation_transport_block_12h_power_max_constraint"]  # T: slack in some block
    live = (con.vars != -1).any("_term")
    assert bool(((con.rhs - con.lhs.solution).where(live) > 1e-6).any())
    hourly = solve_model(build_model(p.drop_vars(BLOCK_PARAMETERS, errors="ignore")))
    assert m.objective.value > hourly.objective.value + 1e-3  # holding the flow over the block costs something
