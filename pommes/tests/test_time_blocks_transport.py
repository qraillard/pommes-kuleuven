"""Transport technologies on time blocks: the parameter, the refusals R11 and R12, the pair mask, the groups."""

import warnings

import numpy as np
import pytest
import xarray as xr

from pommes.model import transport
from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import (
    check_inputs,
    check_time_blocks,
)
from pommes.model.transport import transport_groups
from pommes.tests.time_blocks_helper import transport_toy, with_blocks
from pommes.utils import (
    TimeBlocks,
    check_block_dims,
    check_block_transport,
    time_blocks,
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
    """H1: the capacity restricted in another order is refused in ``costs_def`` (the group guard, xarray's alignment, or the order assert of the sparse fold)."""

    def reverse(args):
        capacity = args[5]
        args[5] = capacity.sel(transport_tech=capacity.indexes["transport_tech"][::-1])

    p = toy_blocks(6, transport={"H": 3, "S": 3})
    build_model(p)
    break_block_rows(monkeypatch, tweak=reverse)
    with pytest.raises((RuntimeError, ValueError, AssertionError), match="transport_tech"):
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


def test_interim_guard_ramp():
    for ramp in ({"S": (0.5, np.nan)}, {"S": (np.nan, 0.5)}):
        with pytest.raises(NotImplementedError, match="ramp"):
            build_model(toy_blocks(6, transport={"S": 3}, ramp=ramp))
    build_model(toy_blocks(6, transport={"S": 3}, ramp={"H": (0.5, 0.5)}))


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
