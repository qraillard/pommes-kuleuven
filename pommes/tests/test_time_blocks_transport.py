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
from pommes.tests.conditions_helper import assert_condition_dims_consistent
from pommes.tests.test_transport_fuel_emission import _balance_terms
from pommes.tests.time_blocks_helper import (
    BLOCK_PARAMETERS,
    assert_same,
    assert_twin_equal,
    binds,
    coarse_step_reference,
    flat_twin,
    solve_model,
    solved,
    to_hours,
    transport_hand_toy,
    transport_t1_dataset,
    transport_t2_conditions_dataset,
    transport_t2_multi_horizon_dataset,
    transport_t2_ramps_short_last_block_dataset,
    transport_t2_retrofit_dataset,
    transport_t3b_derived_finer_dataset,
    transport_t3b_forced_dataset,
    transport_t3c_uniform_dataset,
    variables_in,
    transport_t2_availability_dataset,
    transport_t2_efficiency_dataset,
    transport_t2_fuel_area_bunker_dataset,
    transport_t2_fuel_from_end_dataset,
    transport_t2_fuel_other_resource_dataset,
    transport_t2_hurdle_dataset,
    transport_t2_interior_dataset,
    transport_t2_mixed_group_costs_dataset,
    transport_t2_ramps_uneven_steps_dataset,
    transport_t2_short_last_block_dataset,
    transport_t2_two_lengths_dataset,
    transport_t2_fuel_emissions_dataset,
    transport_t2_fuel_emissions_efficiency_dataset,
    transport_t2_fuel_short_last_block_dataset,
    transport_t2_fuel_supplier_emissions_dataset,
    transport_t2_fuel_uneven_steps_dataset,
    transport_toy,
    with_blocks,
    with_transport_emissions,
    EMISSION_TAXED,
    EMISSION_TOTAL,
    lp_text,
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


# --- Task 6: the two fuel emission terms ---

TAXED_ROW, TOTAL_ROW = "operation_carbon_emissions_def", "operation_total_carbon_emissions_def"
POWER_IN3 = "operation_transport_block_3h_power_in"
_ROWS = (TAXED_ROW, TOTAL_ROW)


def _emission_nterm(m):
    return {r: m.constraints[r].lhs.nterm for r in _ROWS}


def _emissions_toy(dt, transport, fuel, taxed=EMISSION_TAXED, total=EMISSION_TOTAL):
    return with_blocks(with_transport_emissions(transport_toy(dt, fuel=fuel), taxed, total), transport=transport)


def test_block_emission_terms_structure():
    dt = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    phi = 0.1
    p = _emissions_toy(dt, {"S": 3}, {"S": phi})
    m = build_model(p)
    check_block_dims(m)
    for row, factor, other in ((TAXED_ROW, EMISSION_TAXED, EMISSION_TOTAL), (TOTAL_ROW, EMISSION_TOTAL, EMISSION_TAXED)):
        got = terms_of(m, row, POWER_IN3)
        assert len(got) == 2 * 6  # two links, six hours
        for r, var, coeff in got:
            # phi * f * (the step of the hour), at the arrival area of the link, in every hour of the block
            assert coeff == pytest.approx(phi * factor * dt[r["hour"]]), (row, coeff, other)
            # blocks of 3 h hold the steps (1, 2)
            assert var["time_block_3h"] == r["hour"] // 2 and var["transport_tech"] == "S"
            assert r["area"] == p.transport_area_to.sel(link=var["link"], transport_tech="S").item()
        assert {(var["link"], r["hour"]) for r, var, _ in got} == {(l, h) for l in ("ab", "ba") for h in range(6)}
    for row in _ROWS:  # no hourly term of the block ship
        assert not terms_of(m, row, "operation_transport_power_in")


@pytest.mark.parametrize("zero", [0.0, np.nan])
def test_zero_factor_adds_nothing_on_blocks(zero):
    p = _emissions_toy([1.0] * 6, {"S": 3}, {"S": 0.1}, zero, zero)
    base = p.drop_vars(["transport_fuel_emission_factor", "transport_fuel_total_emission_factor"])
    m, m0 = build_model(p), build_model(base)
    assert _emission_nterm(m) == _emission_nterm(m0)
    assert lp_text(m) == lp_text(m0)
    # one factor set: its row gets the term, the other row stays as it was
    one = build_model(_emissions_toy([1.0] * 6, {"S": 3}, {"S": 0.1}, EMISSION_TAXED, zero))
    assert one.constraints[TAXED_ROW].lhs.nterm > m0.constraints[TAXED_ROW].lhs.nterm
    assert one.constraints[TOTAL_ROW].lhs.nterm == m0.constraints[TOTAL_ROW].lhs.nterm


def test_factor_on_the_hourly_ship_only():
    fuel = {"H": 0.1, "S": 0.2}
    p = with_blocks(with_transport_emissions(transport_toy([1.0] * 6, fuel=fuel), {"H": 0.25}, {"H": 0.4}), transport={"S": 3})
    m = build_model(p)
    check_block_dims(m)
    for row in _ROWS:
        assert terms_of(m, row, "operation_transport_power_in")
        assert not terms_of(m, row, POWER_IN3)
    # the block ship's fuel does not change the emission rows
    only_h = build_model(with_blocks(with_transport_emissions(transport_toy([1.0] * 6, fuel={"H": 0.1}), {"H": 0.25}, {"H": 0.4}), transport={"S": 3}))
    assert _emission_nterm(m) == _emission_nterm(only_h)


def test_all_block_transport_with_emissions():
    p = _emissions_toy([1.0] * 6, {"H": 3, "S": 3}, {"H": 0.1, "S": 0.2})
    m = build_model(p)  # no hourly variable: reading it would raise a KeyError
    check_block_dims(m)
    assert "operation_transport_power_in" not in m.variables
    for row in _ROWS:
        assert {var["transport_tech"] for _, var, _ in terms_of(m, row, POWER_IN3)} == {"H", "S"}


def test_mixed_ships_one_arrival_area():
    p = _emissions_toy([1.0] * 6, {"S": 3}, {"H": 0.1, "S": 0.2})
    m = build_model(p)
    for row in _ROWS:
        # H runs a to b on ab and S a to b on ba: the area b receives both terms in each hour
        by_row = {}
        for name in ("operation_transport_power_in", POWER_IN3):
            for r, var, coeff in terms_of(m, row, name):
                by_row.setdefault((r["area"], r["hour"]), {})[(name, var["link"])] = coeff
        both = by_row[("b", 2)]
        assert set(both) == {("operation_transport_power_in", "ab"), (POWER_IN3, "ba")}
        factor = EMISSION_TAXED if row == TAXED_ROW else EMISSION_TOTAL
        assert both[("operation_transport_power_in", "ab")] == pytest.approx(0.1 * factor)
        assert both[(POWER_IN3, "ba")] == pytest.approx(0.2 * factor)


def _emissions_twin(p, L=24, idle=(("b", "oil"),)):
    m, twin = _compare_with_twin(p, {"S": L}, idle)
    for name in ("operation_carbon_emissions", "operation_total_carbon_emissions"):
        assert_same(solved(m, name), solved(twin, name), name)
    # the oil the ships draw is made by the bunker: the fuel drawn is the same hour by hour
    # the conversion power is fixed by the optimum only as a sum over the hours (probed): compare that sum
    name = "operation_conversion_power"
    assert_same(solved(m, name).sum("hour"), solved(twin, name).sum("hour"), "conversion")
    assert float(solved(m, POWER_IN24 if L == 24 else f"operation_transport_block_{L}h_power_in").sum()) > 0
    return m, twin


def test_t2_fuel_emissions():
    m, _ = _emissions_twin(transport_t2_fuel_emissions_dataset())
    taxed, total = (solved(m, n).sel(area="b") for n in ("operation_carbon_emissions", "operation_total_carbon_emissions"))
    assert float(taxed.sum()) > 0.1
    np.testing.assert_allclose(total.values, taxed.values * EMISSION_TOTAL / EMISSION_TAXED, atol=1e-9)  # not swapped
    for name in ("operation_carbon_emissions", "operation_total_carbon_emissions"):
        assert float(abs(solved(m, name).sel(area="a")).max()) == 0.0  # the arrival area only
    assert float(solved(m, "operation_carbon_costs").sum()) > 1e-3  # the tax is paid
    assert np.nanmax(abs(m.constraints["operation_carbon_goal_area_constraint"].dual.values)) > 1e-9  # the goal binds


def test_t2_fuel_emissions_with_efficiency():
    """Emissions follow the fuel drawn at the departure end (``power_in``), not the power delivered: objective and emissions equal the flat twin's."""
    p = transport_t2_fuel_emissions_efficiency_dataset()
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, transport={"S": 24}))
    assert m.objective.value == pytest.approx(twin.objective.value, rel=1e-7)
    for name in ("operation_carbon_emissions", "operation_total_carbon_emissions"):
        assert_same(solved(m, name), solved(twin, name), name)
    # the twin shares the emission code with the block rows: also check by hand, fuel drawn x factor x consumption
    drawn = float(solved(m, POWER_IN24).sum()) * 24 * 0.2 + float(solved(m, "operation_transport_power_in").sel(transport_tech="H").sum()) * 0.1
    assert float(solved(m, "operation_carbon_emissions").sum()) == pytest.approx(drawn * EMISSION_TAXED, rel=1e-6)
    assert drawn > 0.1


def test_t2_fuel_supplier_emissions():
    m, _ = _emissions_twin(transport_t2_fuel_supplier_emissions_dataset())
    # the bunker's own factor counts in the supplier's area: area a, where it is built
    assert float(solved(m, "operation_carbon_emissions").sel(area="a").sum()) > 0.1


def test_t2_fuel_emissions_short_last_block():
    _emissions_twin(transport_t2_fuel_short_last_block_dataset())


def test_t2_fuel_emissions_uneven_steps():
    _emissions_twin(transport_t2_fuel_uneven_steps_dataset(), L=4)


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


def _compare_with_twin(p, lengths, idle=(), skip=()):
    """
    Block model against its flat twin (T2): objective, capacities, flows, net generation, adequacy duals.

    ``idle``: the (area, resource) rows that nothing touches at the optimum (no flow, no supply): their dual is any value
    of an interval, which the optimum does not fix, so it is left out of the comparison. ``skip``: variables the optimum
    leaves open (``assert_twin_equal``).
    """
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, transport=lengths))
    tb = time_blocks(p)
    check_block_dims(m)
    powers = [f"operation_transport_block_{L}h_{k}" for L in sorted(set(lengths.values())) for k in ("power", "power_in")]
    assert_twin_equal(m, twin, tb, powers, skip=skip)
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
    for area, resource in idle:
        a, b = (x.where(~((x.area == area) & (x.resource == resource))) for x in (a, b))
    np.testing.assert_allclose(a.fillna(0).values, b.transpose(*a.dims).fillna(0).values, rtol=1e-6, atol=1e-6 * float(abs(b).max()))
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
    assert p.transport_ramp_up.sel(transport_tech="S").values.tolist() == [[0.1]] * 2  # the two rates differ
    assert p.transport_ramp_down.sel(transport_tech="S").values.tolist() == [[0.12]] * 2
    for name, sign, rate in (("ramp_up_constraint", 1.0, 0.1), ("ramp_down_constraint", -1.0, 0.12)):
        rows = _ramp_terms(m, p, name)
        assert set(rows) == {(l, b) for l in ("ab", "ba") for b in (1, 2, 3)}  # none on block 0
        for cur, prev, cap in rows.values():
            assert (cur, prev, cap) == pytest.approx((sign / d0, -sign / d0, -rate))
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
    for rate, bound in ((step, 0.1), (-step, 0.12)):  # up, then down
        assert float((rate - bound * cap).where(cap > 1e-6).max()) == pytest.approx(0.0, abs=1e-6)  # a link with no ship has none
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


# --- Task 5: fuel drawn at transport_fuel_area ---

POWER_IN12 = "operation_transport_block_12h_power_in"


def _adequacy_nterm(m):
    """{balance: number of terms of its left-hand side}: dead terms count (linopy drops them from the LP text)."""
    return {n: m.constraints[n].lhs.nterm for n in m.constraints if n.startswith("operation_adequacy")}


def test_fuel_gate_reads_the_group():
    # H and T name oil as their fuel but burn none: only S (24 h) draws fuel. A gate read on another group's
    # consumption would build a term of zero coefficients for them (they still have a fuel resource)
    p = toy_blocks(48, techs=["H", "S", "T"], conversion={"bunker": 24}, transport={"S": 24, "T": 12}, fuel={"S": 0.1})
    named = p.assign(transport_fuel_resource=p.transport_fuel_resource.copy(data=np.full(p.transport_fuel_resource.shape, "oil")))
    m, m_named = build_model(p), build_model(named)
    check_block_dims(m_named)
    assert _adequacy_nterm(m_named) == _adequacy_nterm(m)
    adequacy = [n for n in m_named.constraints if n.startswith("operation_adequacy")]
    got = [t for n in adequacy for t in terms_of(m_named, n, POWER_IN24)]
    assert got and all(v["transport_tech"] == "S" and c == pytest.approx(-0.1 * 24) for _, v, c in got)  # S draws, and it alone
    assert list(m_named.constraints["operation_adequacy_block_24h_constraint"].labels.indexes["resource"]) == ["oil"]  # a daily balance
    for name in adequacy:
        assert not terms_of(m_named, name, POWER_IN12) and not terms_of(m_named, name, "operation_transport_power_in")


def test_example_1_fuel_into_a_daily_balance():
    phi, D = 0.1, 24.0
    p = toy_blocks(48, conversion={"mill": 24, "bunker": 24}, transport={"S": 24}, fuel={"S": phi})
    m = build_model(p)
    check_block_dims(m)
    name = "operation_adequacy_block_24h_constraint"
    # oil is daily (bunker and S are 24 h); the cargo, which the hourly demand and spillage pin, is not
    assert list(m.constraints[name].labels.indexes["resource"]) == ["oil"]
    assert "oil" not in m.constraints["operation_adequacy_constraint"].labels.indexes["resource"]
    got = terms_of(m, name, POWER_IN24)
    assert len(got) == 2 * 2  # two links, two blocks
    year = int(p.year_op[0])
    for row, var, coeff in got:
        # -phi * D_b, identity (block b into block b), at the "from" end of the link (S runs b to a on ab, a to b on ba)
        assert coeff == pytest.approx(-phi * D)
        assert row["resource"] == "oil" and var["time_block_24h"] == row["time_block_24h"]
        assert var["transport_tech"] == "S" and var["year_op"] == year
        assert row["area"] == p.transport_area_from.sel(link=var["link"], transport_tech="S").item()
    assert {(var["link"], row["time_block_24h"]) for row, var, _ in got} == {(l, b) for l in ("ab", "ba") for b in (0, 1)}


def _burnt(m, p):
    """Fuel (MWh) the ships draw: the hourly ``H`` and the 24 h ``S``, from the solution."""
    phi = p.transport_fuel_consumption
    hourly = (solved(m, "operation_transport_power_in").sel(transport_tech="H") * float(phi.sel(transport_tech="H").max()))
    block = solved(m, POWER_IN24).sel(transport_tech="S") * 24.0 * float(phi.sel(transport_tech="S").max())
    return float(hourly.sum()), float(block.sum())


def _fuel_twin(p, idle):
    m, twin = _compare_with_twin(p, {"S": 24}, idle)
    assert float(solved(m, POWER_IN24).sum()) > 0
    # the fuel matters: the same ships burning nothing give another optimum
    free = solve_model(build_model(p.assign(transport_fuel_consumption=xr.full_like(p.transport_fuel_consumption, np.nan))))
    assert abs(free.objective.value - m.objective.value) > 1e-3
    return m, twin


def test_t2_fuel_from_end():
    p = transport_t2_fuel_from_end_dataset()
    m, _ = _fuel_twin(p, [("a", "oil"), ("b", "oil")])  # the cargo is burnt, no bunker runs
    assert float(solved(m, "operation_transport_power_in").sel(transport_tech="H").sum()) > 0
    # the cargo burnt leaves the mill's output at a: it makes what the ships inject and burn (nothing is demanded there)
    mill = float(solved(m, "operation_conversion_power").sel(conversion_tech="mill", area="a").sum())
    hourly, block = _burnt(m, p)
    injected = float(solved(m, "operation_transport_power_in").sel(transport_tech="H").sum()) + 24.0 * float(
        solved(m, POWER_IN24).sel(transport_tech="S").sum()
    )
    assert hourly > 0.1 and block > 0.1
    assert mill == pytest.approx(injected + hourly + block, rel=1e-6)


def test_t2_fuel_area_bunker():
    p = transport_t2_fuel_area_bunker_dataset()
    m, _ = _fuel_twin(p, [("a", "oil")])
    _, block = _burnt(m, p)
    oil = solved(m, "operation_conversion_power").sel(conversion_tech="bunker")
    assert block > 0.1 and float(oil.sel(area="a").sum()) == 0.0  # the bunker at a has no capacity, b makes all of it
    assert float(oil.sel(area="b").sum()) == pytest.approx(block, rel=1e-6)


def test_t2_fuel_other_resource():
    p = transport_t2_fuel_other_resource_dataset()
    m, _ = _fuel_twin(p, [("b", "oil")])  # the ships burn at a (the "from" end of the cargo's way)
    hourly, block = _burnt(m, p)
    assert hourly > 0.1 and block > 0.1
    oil = solved(m, "operation_conversion_power").sel(conversion_tech="bunker")
    assert float(oil.sum()) == pytest.approx(hourly + block, rel=1e-6)


def test_fuel_that_is_not_a_resource_draws_nothing_on_blocks():
    """A ship whose fuel is no resource of the model adds no term to the balances: they are those of a ship that burns nothing."""
    p = toy_blocks(48, conversion={"mill": 24, "bunker": 24}, transport={"S": 24}, fuel={"S": 0.1})
    named = p.assign(transport_fuel_resource=xr.where(p.transport_fuel_resource == "oil", "bunker_oil", p.transport_fuel_resource))
    assert "bunker_oil" not in named.resource
    m, silent = build_model(named), build_model(p.assign(transport_fuel_consumption=xr.full_like(p.transport_fuel_consumption, np.nan)))
    # live terms: a block term that routes onto its own length may leave dead ones (label -1), as the hourly ``+=`` does
    assert _balance_terms(m) == _balance_terms(silent)
    for name in m.constraints:
        if name.startswith("operation_adequacy_block"):
            assert terms(m, name) == terms(silent, name)
    # and a fuel that is one does draw: the check is not vacuous
    assert terms(build_model(p), "operation_adequacy_block_24h_constraint") != terms(silent, "operation_adequacy_block_24h_constraint")


def test_t1_with_fuel():
    from pommes.tests.time_blocks_helper import _t2_fuel

    p = with_blocks(_t2_fuel([3.0] * 16, {"H": 1.5, "S": 3.0}, {"H": 0.1, "S": 0.2}), transport={"S": 3})
    m = solve_model(build_model(p))
    hourly = solve_model(build_model(p.drop_vars(BLOCK_PARAMETERS, errors="ignore")))
    assert float(solved(m, "operation_transport_block_3h_power_in").sum()) > 0
    assert_twin_equal(m, hourly, time_blocks(p), ["operation_transport_block_3h_power", "operation_transport_block_3h_power_in"])
    a, b = m.constraints["operation_adequacy_constraint"].dual, hourly.constraints["operation_adequacy_constraint"].dual
    a, b = xr.align(a, b, join="exact")
    a, b = (x.where(~((x.area == "b") & (x.resource == "oil"))) for x in (a, b))  # no bunker at b, nothing burns there: no fixed dual
    np.testing.assert_allclose(a.fillna(0).values, b.transpose(*a.dims).fillna(0).values, rtol=1e-6, atol=1e-6 * float(abs(b).max()))


# --- Task 7: whole-model tests (T1, the rest of T2, T3) ---


def test_transport_t1_block_equals_step():
    """Uniform steps of 3 h and L = 3: efficiency, fuel, ramps, hurdle and availability, every value equals the hourly model."""
    p = transport_t1_dataset()
    tb = time_blocks(p)
    assert tb.duration[3].values.tolist() == [3.0] * 16 and tb.first_step_duration[3].values.tolist() == [3.0] * 16
    hourly_p = p.drop_vars(BLOCK_PARAMETERS, errors="ignore")
    m, hourly = solve_model(build_model(p)), solve_model(build_model(hourly_p))
    block = "operation_transport_block_3h_"
    for row in ("power_max_constraint", "ramp_up_constraint", "ramp_down_constraint"):
        assert binds(m, block + row), row
    assert binds(m, block + "power_in_def", equality=True)
    assert float(abs(solved(m, block + "power_in") - solved(m, block + "power")).max()) > 0.1  # the losses are real
    assert float(solved(m, block + "power_in").sum()) > 0.1 and float(solved(m, "operation_transport_power").sum()) > 0.1
    assert m.objective.value == pytest.approx(hourly.objective.value, rel=1e-7)
    assert_twin_equal(m, hourly, tb, [block + "power", block + "power_in"])
    ng = to_hours(solved(m, block + "net_generation"), tb)  # a block is a step
    assert_same(ng, solved(hourly, "operation_transport_net_generation").sel(transport_tech=["S"]), "NG")
    for name in ("operation_transport_power", "operation_transport_power_in", "operation_transport_net_generation"):
        assert_same(solved(m, name), solved(hourly, name).sel(transport_tech=["H"]), name)
    a, b = m.constraints["operation_adequacy_constraint"].dual, hourly.constraints["operation_adequacy_constraint"].dual
    a, b = xr.align(a, b, join="exact")
    a, b = (x.where(~((x.area == "b") & (x.resource == "oil"))) for x in (a, b))  # no bunker at b, nothing burns there
    np.testing.assert_allclose(a.fillna(0).values, b.transpose(*a.dims).fillna(0).values, rtol=1e-6, atol=1e-6 * float(abs(b).max()))
    # the ramps, the efficiency and the fuel matter: dropping each changes the optimum
    for names in (("transport_ramp_up", "transport_ramp_down"), ("transport_fuel_consumption",)):
        free = solve_model(build_model(p.assign({name: xr.full_like(p[name], np.nan) for name in names})))
        assert abs(free.objective.value - m.objective.value) > 1e-3, names
    lossless = solve_model(build_model(p.assign(transport_efficiency=xr.ones_like(p.transport_efficiency))))
    assert abs(lossless.objective.value - m.objective.value) > 1e-3


def test_t2_conditions():
    """Two operating conditions (weights 0.4 and 0.6) with different demand and availability: equal to the flat twin."""
    p = transport_t2_conditions_dataset()
    assert list(p.operating_condition.values) == ["a", "b"]
    m, twin = _compare_with_twin(p, {"S": 24})
    assert_condition_dims_consistent(m)
    assert "operating_condition" in m.variables[POWER24].dims and "operating_condition" not in m.variables["operation_transport_power_capacity"].dims
    assert binds(m, "operation_transport_block_24h_power_max_constraint")
    # each condition follows its own block minimum of the availability (day 1 and day 2): a 0.5, 0.8 and b 0.7, 0.6
    cap = solved(m, "operation_transport_power_capacity").sel(transport_tech="S", link="ba").item()
    power = solved(m, POWER24).sel(link="ba")
    for condition, minima in (("a", [0.5, 0.8]), ("b", [0.7, 0.6])):
        np.testing.assert_allclose(power.sel(operating_condition=condition).values.ravel(), np.array(minima) * cap)


# the (year_dec, year_inv) split of the capacities is left out (the brief): the transport one is compared by its sum over
# year_dec below; the conversion one is fixed by cost gaps of the order of 0.01, which a solver tolerance may move
T2_SPLIT = ("planning_transport_power_capacity", "planning_conversion_power_capacity")


def test_t2_multi_horizon():
    """Investment, early decommissioning and two operation years: the unique quantities equal the flat twin's."""
    p = transport_t2_multi_horizon_dataset()
    assert p.year_op.values.tolist() == [2030, 2040] and p.transport_early_decommissioning.item()
    m, twin = _compare_with_twin(p, {"S": 24}, skip=T2_SPLIT)
    planning = {name: solved(x, "planning_transport_power_capacity").sum("year_dec") for name, x in (("block", m), ("twin", twin))}
    assert_same(planning["block"], planning["twin"], "planning capacity per vintage")
    capacity = solved(m, "operation_transport_power_capacity")
    assert_same(capacity, solved(twin, "operation_transport_power_capacity"), "capacity")
    s = capacity.sel(transport_tech="S", link="ba")
    assert float(s.sel(year_op=2030)) > 0.1 and float(s.sel(year_op=2040)) > 0.1 and float(abs(s.sel(year_op=2030) - s.sel(year_op=2040))) > 0.1
    # early decommissioning is used: a vintage leaves before the end of its life
    split = solved(m, "planning_transport_power_capacity").sel(transport_tech="S")
    early = split.sel(year_dec=2040, year_inv=2030)
    assert float(early.sum()) > 0.1
    for year in (2030, 2040):
        assert float(solved(m, POWER24).sel(year_op=year).sum()) > 0.1


def test_t2_retrofit():
    """A retrofit from the hourly ``H`` to the block ship ``S``: objective, capacities and transfers equal the flat twin's."""
    p = transport_t2_retrofit_dataset()
    m, twin = _compare_with_twin(p, {"S": 24})
    transfer = solved(m, "planning_retrofit_transport_power_capacity")
    assert float(transfer.sum()) > 1e-3  # a transfer really happens
    assert_same(transfer, solved(twin, "planning_retrofit_transport_power_capacity"), "transfer")
    # the retrofit reads the year-level capacity by name: the one row of both techs, written once, holds the transfer
    row = m.constraints["operation_transport_power_capacity_def"]
    assert list(row.labels.indexes["transport_tech"]) == ["H", "S"]
    assert "planning_retrofit_transport_power_capacity" in variables_in(m, row)
    read = set()
    for name in [c for c in m.constraints if c.startswith(("retrofit_", "planning_retrofit_"))]:
        for variable in variables_in(m, m.constraints[name]):
            read.add(variable)
            dims = m.variables[variable].dims
            assert not any(d == "hour" or d.startswith("time_block_") for d in dims), (name, variable, dims)
    assert "planning_transport_power_capacity" in read
    # the transferred capacity is what the block ship runs on in the second year
    assert float(solved(m, POWER24).sel(year_op=2040).sum()) > 0.1
    assert float(solved(m, "operation_transport_power_capacity").sel(transport_tech="S", year_op=2040).sum()) > 0.1


def test_t2_ramps_short_last_block():
    """T2: short last block, steps of 2, 1 and 1 h; both ramp rows bind, also into and out of the short block."""
    p = transport_t2_ramps_short_last_block_dataset()
    tb = time_blocks(p)
    assert tb.duration[8].values.tolist() == [8.0, 8.0, 4.0] and tb.first_step_duration[8].values.tolist() == [2.0] * 3
    m, _ = _compare_with_twin(p, {"S": 8})
    block = "operation_transport_block_8h_"
    for name in ("ramp_up_constraint", "ramp_down_constraint"):
        assert int((m.constraints[block + name].labels >= 0).sum()) == 4  # two links, blocks 1 and 2
        assert binds(m, block + name)
    assert float(abs(solved(m, block + "power_in") - solved(m, block + "power")).max()) > 0.1  # Pin is not P
    # the ramps hold on P with the first step as divisor, also on the short last block, and each is tight
    power = solved(m, block + "power").sel(transport_tech="S")
    cap = solved(m, "operation_transport_power_capacity").sel(transport_tech="S")
    step = (power - power.shift(time_block_8h=1)) / 2.0
    for rate, bound in ((step, 0.1), (-step, 0.12)):  # up, then down
        assert float((rate - bound * cap).where(cap > 1e-6).max()) == pytest.approx(0.0, abs=1e-6)
    free = solve_model(build_model(p.assign(transport_ramp_up=xr.full_like(p.transport_ramp_up, np.nan), transport_ramp_down=xr.full_like(p.transport_ramp_down, np.nan))))
    assert abs(free.objective.value - m.objective.value) > 1.0  # the ramps matter


# --- T3: the coarse balance ---


def test_t3a_all_block_hand_toy():
    """An all-block model by hand: a daily balance gives 144, the flat twin (hourly balances) gives 288."""
    p = transport_hand_toy()
    tb = time_blocks(p)
    assert tb.balance_lengths == [24] and tb.resources(24) == ["cargo"] and tb.duration[24].values.tolist() == [24.0, 24.0]
    m = solve_model(build_model(p))
    check_block_dims(m)
    assert not set(HOURLY_NAMES) & set(m.variables)
    # per day: 24 delivered (1 MW for 24 h), 30 injected (1 / 0.8), 30 produced at 2 each, 0.5 * 24 hurdle: 2 * 30 + 12 = 72
    assert m.objective.value == pytest.approx(144.0, rel=1e-9)
    power, power_in = solved(m, POWER24).sel(link="ab"), solved(m, POWER_IN24).sel(link="ab")
    np.testing.assert_allclose(power.values.ravel(), [1.0, 1.0])
    np.testing.assert_allclose(power_in.values.ravel(), [1.25, 1.25])
    mill = solved(m, "operation_conversion_block_24h_power").sel(conversion_tech="mill", area="a")
    np.testing.assert_allclose(mill.values.ravel(), [1.25, 1.25])
    twin = solve_model(flat_twin(p, conversion={"mill": 24}, transport={"S": 24}))
    assert twin.objective.value == pytest.approx(288.0, rel=1e-9)
    # the twin delivers 2 MW all day and spills what the 12 idle hours do not use
    assert float(solved(twin, "operation_transport_power").sel(link="ab").max()) == pytest.approx(2.0)
    assert float(solved(twin, "operation_spillage_power").sel(resource="cargo", area="b").sum()) == pytest.approx(2 * 24.0)


@pytest.mark.parametrize("forced", [24, 168, 12])
def test_t3b_forced_balances_are_relaxations(forced):
    """A forced balance sums the hourly storage into its row: objective below the twin's, the block's energy conserved."""
    base = transport_t3b_forced_dataset()
    p = with_blocks(base, resource={"cargo": forced}, transport={"S": 24})
    tb = time_blocks(p)
    assert tb.balance_lengths == [forced] and tb.resources(forced) == ["cargo"]
    m = solve_model(build_model(p))
    twin = solve_model(flat_twin(base, transport={"S": 24}))
    assert m.objective.value < twin.objective.value - 1e-3  # a relaxation, and the storage's cost is saved
    # energy conserved per block: each (area, block) term of the ship enters the balance rows with a total weight of one
    got = terms_of(m, f"operation_adequacy_block_{forced}h_constraint", NG24)
    weight = {}
    for row, var, coeff in got:
        weight[(var["area"], var["time_block_24h"])] = weight.get((var["area"], var["time_block_24h"]), 0.0) + coeff
    assert weight == pytest.approx({(a, b): 1.0 for a in ("a", "b") for b in (0, 1)})
    assert float(abs(solved(m, NG24)).max()) > 0.1  # and the ship works


def test_t3b_derived_finer_than_the_ship():
    """A ship of 24 h and a conversion of 12 h: the derived balance is 12 h, the ship is spread by halves."""
    p = transport_t3b_derived_finer_dataset()
    tb = time_blocks(p)
    assert tb.balance_lengths == [12] and tb.resources(12) == ["cargo"]
    m = solve_model(build_model(p))
    got = terms_of(m, "operation_adequacy_block_12h_constraint", NG24)
    assert len(got) == 2 * 4  # two areas, four rows of 12 h
    assert all(c == 0.5 and v["time_block_24h"] == r["time_block_12h"] // 2 and v["area"] == r["area"] for r, v, c in got)
    weight = {}
    for row, var, coeff in got:
        weight[(var["area"], var["time_block_24h"])] = weight.get((var["area"], var["time_block_24h"]), 0.0) + coeff
    assert weight == pytest.approx({(a, b): 1.0 for a in ("a", "b") for b in (0, 1)})  # the block's energy is conserved
    twin = solve_model(flat_twin(p, conversion={"mill": 12}, transport={"S": 24}))
    assert m.objective.value < twin.objective.value - 1e-3


def test_t3c_uniform_coarse_step():
    """Ship, mill and bunker all of length L: equal to today's model with steps of L hours."""
    L = 4
    p = transport_t3c_uniform_dataset()
    tb = time_blocks(p)
    assert tb.balance_lengths == [L] and set(tb.resources(L)) == {"cargo", "oil"}
    assert (tb.tech_length["transport"] == L).all() and (tb.tech_length["conversion"] == L).all()
    m = solve_model(build_model(p))
    reference = solve_model(build_model(coarse_step_reference(p, L)))
    assert m.objective.value == pytest.approx(reference.objective.value, rel=1e-7)
    block = f"operation_transport_block_{L}h_"
    for kind in ("power", "power_in", "net_generation"):
        got = solved(m, block + kind).rename(time_block_4h="hour")
        want = solved(reference, f"operation_transport_{kind}")
        assert_same(got, want.assign_coords(hour=got.hour), kind)
    assert_same(solved(m, "operation_transport_power_capacity"), solved(reference, "operation_transport_power_capacity"), "capacity")
    assert_same(solved(m, "operation_transport_costs"), solved(reference, "operation_transport_costs"), "costs")
    conversion = solved(m, f"operation_conversion_block_{L}h_power").rename(time_block_4h="hour")
    want = solved(reference, "operation_conversion_power")
    assert_same(conversion, want.assign_coords(hour=conversion.hour), "conversion")
    # the row families it claims bind
    for row in ("power_max_constraint",):
        assert binds(m, block + row), row
    for row in ("power_in_def", "net_generation_def", "costs_def"):
        assert binds(m, block + row, equality=True), row
    for resource in ("cargo", "oil"):
        dual = m.constraints[f"operation_adequacy_block_{L}h_constraint"].dual.sel(resource=resource)
        assert float(abs(dual).max()) > 1e-6, resource
    assert float(solved(m, block + "power_in").sum()) > 0.1 and float(solved(m, "operation_conversion_block_4h_power").sel(conversion_tech="bunker").sum()) > 0.1
    # the minimum of the availability, not its mean: the block holds an availability of 0.6 next to 1.0
    assert float(p.transport_availability.min()) < 1.0 and bool(p.transport_availability.isnull().any())
