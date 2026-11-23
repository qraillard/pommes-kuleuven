"""Transport technologies with modes: inputs, ``transport_modal``, refusals R13-R18, the warning W1 and the contributors."""

import re
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pommes import test_case_path
from pommes.io.build_input_dataset import build_input_parameters, load_inputs_as_dict_of_df, read_config_file
from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs
from pommes.tests.conditions_helper import assert_condition_dims_consistent
from pommes.tests.time_blocks_helper import (
    BLEND,
    LEAN,
    PURE,
    as_modal,
    assert_same,
    densify_any,
    solve_model,
    solved,
    transport_modes_block_elsewhere_dataset,
    transport_modes_carbon_dataset,
    transport_modes_conditions_dataset,
    transport_modes_multi_horizon_dataset,
    transport_modes_pairless_first_dataset,
    transport_modes_retrofit_dataset,
    transport_modes_toy,
    transport_t2_fuel_emissions_dataset,
    transport_t2_multi_horizon_dataset,
    _transport_t2,
    transport_toy,
    variables_in,
    with_blocks,
    with_carbon,
    with_modes,
)
from pommes.utils import (
    OPERATING_CONDITION,
    OPERATING_CONDITION_WEIGHT,
    TransportModes,
    _contributors,
    time_blocks,
    transport_modal,
    transport_pair_exists,
)

NAMES = ["transport_mode_factor", "transport_mode_ramp_up", "transport_mode_ramp_down"]
FACTOR_DIMS = ("transport_tech", "transport_mode", "resource")
LINK_TECH = ("link", "transport_tech")


def _refused(p, message):
    """``p`` is refused by ``check_inputs`` and by ``build_model`` (which does not call it), with ``message``."""
    with pytest.raises(ValueError, match=message):
        check_inputs(p.copy(deep=True))
    with pytest.raises(ValueError, match=message):
        build_model(p)


def _with_links(p, order=("ba", "ab")):
    """``p`` with a second link ``ba`` that copies ``ab``; the links in the order ``order``."""
    other = p.assign_coords(link=["ba"])
    both = xr.concat([p, other], dim="link", data_vars="minimal", coords="minimal", compat="override")
    return both.sel(link=list(order))


def _two_years(p):
    """``p`` with a second operation year 2040 that copies 2030."""
    later = p.assign_coords(year_op=p.year_op.values + 10)
    return xr.concat([p, later], dim="year_op", data_vars="minimal", coords="minimal", compat="override")


def _per(p, name, fill, dims, cells):
    """``p`` with ``name`` on ``dims`` equal to ``fill`` except at the ``cells`` ({coordinate: label} -> value)."""
    da = xr.DataArray(
        np.full([p.sizes[d] for d in dims], fill, dtype="float64"), dims=dims, coords={d: p[d].values for d in dims}
    )
    for cell, value in cells:
        da.loc[cell] = value
    return p.assign({name: da})


def _resource(p, **cells):
    """``transport_resource`` of ``p`` (``ch4`` on every pair) with ``nan`` (no pair) at the cells ``link=..., transport_tech=...``."""
    da = xr.DataArray(
        np.full([p.sizes[d] for d in LINK_TECH], "ch4"), dims=LINK_TECH, coords={d: p[d].values for d in LINK_TECH}
    )
    for link, tech in cells.get("nan", []):
        da.loc[dict(link=link, transport_tech=tech)] = "nan"
    return da


def _factor(p, cell, value):
    q = p.copy(deep=True)
    q["transport_mode_factor"].loc[cell] = value
    return q


def test_modes_toy_plain_is_100():
    m = solve_model(build_model(transport_modes_toy()))
    assert m.objective.value == pytest.approx(100.0, rel=1e-7)


def test_transport_modal_tables():
    p = transport_modes_toy(modes=BLEND)
    modal = transport_modal(p)
    assert isinstance(modal, TransportModes)
    assert list(modal.techs) == ["pipe"] and modal.techs.name == "transport_tech"
    assert list(modal.modes) == ["blend", "methane"] and modal.modes.name == "transport_mode"
    assert modal.factor.dims == FACTOR_DIMS and modal.factor.dtype == np.float64
    assert modal.defined.dims == ("transport_tech", "transport_mode") and modal.defined.all()
    assert modal.carried.dims == ("transport_tech", "resource")
    assert set(modal.carried.resource.values[modal.carried.sel(transport_tech="pipe").values]) == {"ch4", "h2"}
    assert modal.modal.dims == ("transport_tech",) and modal.modal.all()
    # a NaN reads as 0; a mode whose only factor is NaN is not defined
    q = with_modes(transport_modes_toy(), {"pipe": {**BLEND, "ghost": {"h2": 1.0}}})
    q = _factor(_factor(q, dict(transport_mode="ghost", resource="h2"), np.nan), dict(transport_mode="blend", resource="h2"), np.nan)
    modal = transport_modal(q)
    assert modal.defined.sel(transport_tech="pipe").values.tolist() == [True, True, False]
    assert float(modal.factor.sel(transport_tech="pipe", transport_mode="blend", resource="h2")) == 0.0
    assert not np.isnan(modal.factor.values).any()
    assert set(modal.carried.resource.values[modal.carried.sel(transport_tech="pipe").values]) == {"ch4"}
    # nothing sorts the modes: the coordinate as the dataset gives it (the sense of up and down depends on it)
    ordered = transport_modal(transport_modes_toy(modes=BLEND, order=["methane", "blend"]))
    assert list(ordered.modes) == ["methane", "blend"]
    assert ordered.defined.transport_mode.values.tolist() == ["methane", "blend"]
    assert ordered.factor.transport_mode.values.tolist() == ["methane", "blend"]
    assert float(ordered.factor.sel(transport_tech="pipe", transport_mode="blend", resource="h2")) == 0.1
    # no modal technology, or none of the conditions of section 5.3
    assert transport_modal(p.assign(transport=np.array(False))) is None
    assert transport_modal(p.drop_vars(["transport_mode_factor", "transport_mode"])) is None
    assert transport_modal(p.drop_vars("transport_mode_factor")) is None
    zero = p.assign(transport_mode_factor=xr.zeros_like(p.transport_mode_factor))
    assert transport_modal(zero) is None
    assert transport_modal(zero.assign(transport_mode_factor=zero.transport_mode_factor + np.nan)) is None
    assert transport_modal(p.assign(transport_mode_factor=np.array(0.0))) is None


def test_defaults_added_when_transport_is_on():
    p = check_inputs(transport_modes_toy().drop_vars(NAMES))
    for name, value in zip(NAMES, (0.0, np.nan, np.nan)):
        assert p[name].dims == () and p[name].dtype == np.float64
        assert np.isnan(float(p[name])) if np.isnan(value) else float(p[name]) == value


def test_not_defaulted_when_transport_is_off():
    p = transport_modes_toy().drop_vars(NAMES).assign(transport=np.array(False))
    assert not [n for n in NAMES if n in check_inputs(p)]


def test_factor_refuses_another_dimension():
    p = transport_modes_toy(modes=BLEND)
    p = p.assign(transport_mode_factor=p.transport_mode_factor.expand_dims(year_op=p.year_op))
    _refused(p, "transport_mode_factor")
    with pytest.raises(ValueError, match="only the transport_tech, transport_mode and resource"):
        transport_modal(p)


def test_factor_missing_a_dimension_is_broadcast():
    p = transport_modes_toy(techs=("pipe", "pipe2"), modes={"pipe": BLEND, "pipe2": BLEND})
    table = p.transport_mode_factor.sel(transport_tech="pipe", drop=True)
    assert table.dims == ("transport_mode", "resource")
    modal = transport_modal(p.assign(transport_mode_factor=table))
    assert list(modal.techs) == ["pipe", "pipe2"]
    assert modal.factor.dims == FACTOR_DIMS
    for tech in ("pipe", "pipe2"):
        assert (modal.factor.sel(transport_tech=tech, drop=True) == table).all()
    assert modal.carried.sel(transport_tech="pipe2", resource="h2")


def test_r18_dimension_without_coordinate():
    message = re.escape(
        "transport_mode is a dimension without a coordinate: call assign_coords(transport_mode=[...]) first."
    )
    p = transport_modes_toy(modes=BLEND).drop_vars("transport_mode")
    assert "transport_mode" in p.dims and "transport_mode" not in p.coords
    _refused(p, message)
    # transport off: modal data are never checked
    check_inputs(p.assign(transport=np.array(False)))


def test_r16_negative_factor():
    p = transport_modes_toy(modes=BLEND)
    q = _factor(p, dict(transport_mode="blend", resource="h2"), -0.1)
    _refused(q, re.escape("pipe: transport_mode_factor[blend, h2] = -0.1: a factor is finite and >= 0 (NaN reads as 0)."))
    q = _factor(p, dict(transport_mode="methane", resource="ch4"), np.inf)
    _refused(q, re.escape("pipe: transport_mode_factor[methane, ch4] = inf: a factor is finite and >= 0 (NaN reads as 0)."))
    # the first one in the order technology, mode, resource
    two = transport_modes_toy(techs=("pipe", "pipe2"), modes={"pipe": BLEND, "pipe2": BLEND})
    two = _factor(two, dict(transport_tech="pipe2", transport_mode="blend", resource="h2"), -1.0)
    two = _factor(two, dict(transport_tech="pipe", transport_mode="methane", resource="h2"), np.inf)
    two = _factor(two, dict(transport_tech="pipe", transport_mode="blend", resource="ch4"), -2.0)
    two = _factor(two, dict(transport_tech="pipe", transport_mode="blend", resource="heat"), -3.0)
    _refused(two, re.escape("pipe: transport_mode_factor[blend, ch4] = -2.0:"))
    # a NaN is not refused
    check_inputs(_factor(p, dict(transport_mode="blend", resource="h2"), np.nan))
    # a modal technology of another dataset is not looked at when transport is off
    check_inputs(q.assign(transport=np.array(False)))


def test_r17_main_resource_without_factor():
    p = _with_links(transport_modes_toy(modes=PURE), order=("ba", "ab"))
    p = _factor(p, dict(transport_mode="methane", resource="ch4"), 0.0)  # no mode carries ch4 any more
    _refused(p, re.escape("pipe: transport_resource ch4 on link ba has no positive transport_mode_factor in any mode."))
    # a pair that does not exist is not looked at
    q = _with_links(transport_modes_toy(modes=BLEND), order=("ba", "ab"))
    check_inputs(q.assign(transport_resource=_resource(q, nan=[("ba", "pipe")])))


def test_r15_block_length_on_a_modal_technology():
    p = with_blocks(transport_modes_toy(modes=BLEND), transport={"pipe": 2})
    message = re.escape(
        "pipe: transport_time_block_hrs = 2 h: a transport technology with modes cannot be a block technology."
    )
    _refused(p, message)
    # with a delay on the same technology it is R15, not R11
    q = p.assign(transport_delay=_per(p, "d", 0.0, LINK_TECH, [(dict(link="ab"), 1.0)]).d)
    _refused(q, message)
    with pytest.raises(ValueError, match="block transport technology"):
        check_inputs(with_blocks(transport_modes_toy(), transport={"pipe": 2}).assign(transport_delay=q.transport_delay))
    # a block length on a technology that has no mode of a dataset that has a modal one
    two = transport_modes_toy(techs=("pipe", "ship"), modes={"pipe": BLEND})
    check_inputs(with_blocks(two, transport={"ship": 2}))


def test_r13_delay():
    p = _with_links(transport_modes_toy(techs=("pipe", "plain"), modes={"pipe": BLEND}), order=("ba", "ab"))
    delay = _per(p, "d", 0.0, LINK_TECH, [(dict(link="ab", transport_tech="pipe"), 1.0), (dict(link="ba", transport_tech="pipe"), 2.0)]).d
    refused = p.assign(transport_delay=delay)
    _refused(refused, re.escape("pipe: transport_delay = 2.0 h on link ba: a transport technology with modes takes no delay."))
    negative = p.assign(transport_delay=delay.where(delay != 2.0, -3.0))
    _refused(negative, re.escape("pipe: transport_delay = -3.0 h on link ba:"))
    # not refused: a NaN delay, a pair that does not exist, a technology that is not modal, factors that are all 0
    check_inputs(p.assign(transport_delay=xr.full_like(delay, np.nan)))
    gone = _resource(p, nan=[("ba", "pipe")])
    check_inputs(p.assign(transport_delay=delay.where(delay != 1.0, 0.0), transport_resource=gone))
    check_inputs(p.assign(transport_delay=_per(p, "d", 0.0, LINK_TECH, [(dict(transport_tech="plain"), 2.0)]).d))
    zero = p.assign(transport_mode_factor=xr.zeros_like(p.transport_mode_factor), transport_delay=delay)
    check_inputs(zero)


def test_r14_linepack():
    p = _two_years(_with_links(transport_modes_toy(modes=BLEND), order=("ba", "ab")))
    cell = dict(link="ab", transport_tech="pipe", year_op=2040)
    linepack = _per(p, "l", np.nan, (*LINK_TECH, "year_op"), [(cell, 5.0)]).l
    refused = p.assign(transport_linepack_capacity=linepack)
    _refused(refused, re.escape("pipe: transport_linepack_capacity is finite on link ab: a transport technology with modes takes no linepack."))
    # not refused: a pair that does not exist, no finite value
    check_inputs(refused.assign(transport_resource=_resource(p, nan=[("ab", "pipe")])))
    check_inputs(p.assign(transport_linepack_capacity=xr.full_like(linepack, np.nan)))


def _warning_texts(p):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_inputs(p.copy(deep=True))
    return [str(w.message) for w in caught if "Transport modes whose factors" in str(w.message)]


def test_w1_sum_warning():
    modes = {"lean": {"ch4": 0.90, "h2": 0.03}, "rich": {"ch4": 1.2}, "whole": {"ch4": 1.0}}
    p = transport_modes_toy(modes=modes)
    expected = (
        "Transport modes whose factors do not sum to 1: pipe lean: 0.93; pipe rich: 1.2. "
        "A flow of 1 MWh in such a mode carries that many MWh of resources."
    )
    assert _warning_texts(p) == [expected]
    with pytest.warns(UserWarning, match=re.escape("pipe lean: 0.93; pipe rich: 1.2.")):
        check_inputs(p.copy(deep=True))
    # the same warning, once, from build_model (which does not call check_inputs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_model(p)
    assert [str(w.message) for w in caught if "Transport modes whose factors" in str(w.message)] == [expected]
    # a sum of 1 + 1e-6 is listed (the tolerance is 1e-9)
    off = transport_modes_toy(modes={"m": {"ch4": 1.0, "h2": 1e-6}})
    assert [t.split(". A flow")[0] for t in _warning_texts(off)] == ["Transport modes whose factors do not sum to 1: pipe m: 1"]
    # every defined mode sums to 1 (a sum that is 1 up to a rounding error); an undefined mode is not listed
    rounding = {"third": {"h2": 0.7, "ch4": 0.2, "heat": 0.1}, "blend": BLEND["blend"], "unused": {}}
    assert np.array([0.0, 0.7, 0.2, 0.1]).sum() != 1.0
    assert _warning_texts(transport_modes_toy(modes=rounding)) == []
    assert _warning_texts(transport_modes_toy(modes=BLEND)) == []
    # a technology that is not modal is not listed; transport off: no warning
    other = transport_modes_toy(techs=("pipe", "pipe2"), modes={"pipe": BLEND})
    assert _warning_texts(other) == []
    assert _warning_texts(p.assign(transport=np.array(False))) == []


def test_modal_inputs_from_csv(tmp_path):
    case = tmp_path / "case"
    shutil.copytree(test_case_path, case)
    config = read_config_file(file_path=str(case / "config.yaml"))
    config["coords"]["transport_mode"] = {"type": "str", "values": ["methane", "blend"]}  # not alphabetical
    factor_file = "\n".join(
        [
            "transport_tech;transport_mode;resource;transport_mode_factor",
            "methane_pipe;blend;methane;0.9",
            "methane_pipe;blend;hydrogen;0.1",
            "methane_pipe;blend;ammonia;0.5",  # not a resource of the study: dropped by the reader
            "methane_pipe;methane;methane;1.0",
        ]
    )
    (case / "data" / "transport_mode.csv").write_text(factor_file + "\n")
    (case / "data" / "transport_mode_ramp.csv").write_text(
        "link;transport_tech;year_op;ramp_up;ramp_down\nlink_1;methane_pipe;2030;0.5;0.7\nlink_2;methane_pipe;2040;0.25;0.125\n"
    )
    config["input"]["parameters"]["transport_mode_factor"] = dict(
        file="transport_mode.csv",
        column="transport_mode_factor",
        type="float64",
        fill=0.0,
        index_input=["transport_tech", "transport_mode", "resource"],
    )
    for name, column in (("transport_mode_ramp_up", "ramp_up"), ("transport_mode_ramp_down", "ramp_down")):
        config["input"]["parameters"][name] = dict(
            file="transport_mode_ramp.csv",
            column=column,
            type="float64",
            fill=float("nan"),
            index_input=["link", "transport_tech", "year_op"],
        )
    p = check_inputs(build_input_parameters(config, load_inputs_as_dict_of_df(config)))
    assert list(p.transport_mode.values) == ["methane", "blend"]  # the order of the config: up and down depend on it
    factor = p.transport_mode_factor
    assert set(factor.dims) == set(FACTOR_DIMS) and factor.dtype == np.float64
    cell = dict(transport_tech="methane_pipe")
    assert float(factor.sel(**cell, transport_mode="blend", resource="methane")) == 0.9
    assert float(factor.sel(**cell, transport_mode="blend", resource="hydrogen")) == 0.1
    assert float(factor.sel(**cell, transport_mode="methane", resource="methane")) == 1.0
    assert float(factor.sum()) == pytest.approx(2.0)  # filled with 0; ammonia is dropped, silently
    assert "ammonia" not in factor.resource.values
    modal = transport_modal(p)
    assert list(modal.techs) == ["methane_pipe"]
    assert list(modal.modes) == ["methane", "blend"]
    up, down = p.transport_mode_ramp_up, p.transport_mode_ramp_down
    assert set(up.dims) == set(down.dims) == {"link", "transport_tech", "year_op"}
    assert float(up.sel(link="link_1", transport_tech="methane_pipe", year_op=2030)) == 0.5
    assert float(down.sel(link="link_2", transport_tech="methane_pipe", year_op=2040)) == 0.125
    assert int(up.notnull().sum()) == 2 and int(down.notnull().sum()) == 2
    # the coordinate must be declared: the reader raises, it is not R18
    del config["coords"]["transport_mode"]
    with pytest.raises(KeyError, match="transport_mode"):
        load_inputs_as_dict_of_df(config)


def _contributor_table(p, tb):
    """``_contributors`` with the NaN lengths of the hourly contributors as None (NaN is not equal to itself)."""
    found = _contributors(p, tb.tech_length)
    return {r: [(kind, name, None if np.isnan(L) else L) for kind, name, L in rows] for r, rows in found.items()}


def _contributor_dataset(transport=None, **kw):
    """The hand toy on 24 steps with the h2 producer and consumer on 24 h blocks; the rest is hourly."""
    p = transport_modes_toy(dt=(1.0,) * 24, **kw)
    return with_blocks(p, conversion={"electrolyser": 24, "boiler_h2": 24}, transport=transport)


def test_contributors_count_a_modal_technology():
    modal = _contributor_dataset(modes=BLEND)
    with pytest.warns(UserWarning, match=re.escape("h2: hourly because of transport tech pipe")):
        tb = time_blocks(modal)
    assert np.isnan(tb.resource_length.sel(resource="h2").item())
    hourly = [(kind, name) for kind, name, length in _contributors(modal, tb.tech_length)["h2"] if np.isnan(length)]
    assert ("transport tech", "pipe") in hourly
    # without the modal factor h2 follows its producer and its consumer
    plain = _contributor_dataset()
    assert time_blocks(plain, warn=False).resource_length.sel(resource="h2").item() == 24
    # a modal pipe with a block length keeps its 0.8.0 contributions: time_blocks is that of 0.8.0 and R15 comes after
    blocked = _contributor_dataset(transport={"pipe": 24}, modes=BLEND)
    twin = blocked.drop_vars("transport_mode_factor")
    tb, tb_twin = time_blocks(blocked, warn=False), time_blocks(twin, warn=False)
    assert tb.resource_length.equals(tb_twin.resource_length)
    assert tb.resource_length.sel(resource="h2").item() == 24
    assert _contributor_table(blocked, tb) == _contributor_table(twin, tb_twin)
    assert all(name != "pipe" for _, name, _ in _contributors(blocked, tb.tech_length)["h2"])
    _refused(blocked, re.escape("pipe: transport_time_block_hrs = 24 h: a transport technology with modes cannot be a block technology."))


# --- Task 2: the first offending technology, the order of the checks, transport_modal on a group ---


def test_first_offending_technology_in_model_order():
    two = lambda **kw: transport_modes_toy(techs=("pipe", "pipe2"), modes={"pipe": BLEND, "pipe2": BLEND}, **kw)  # noqa: E731
    # R13: a delay on both
    p = two()
    delay = _per(p, "d", 0.0, LINK_TECH, [(dict(link="ab"), 1.0)]).d
    _refused(p.assign(transport_delay=delay), re.escape("pipe: transport_delay = 1.0 h on link ab:"))
    # R14: a finite linepack on both
    linepack = _per(p, "l", np.nan, (*LINK_TECH, "year_op"), [(dict(link="ab"), 5.0)]).l
    _refused(p.assign(transport_linepack_capacity=linepack), re.escape("pipe: transport_linepack_capacity is finite on link ab:"))
    # R17: the main resource (ch4) has no factor on either
    q = _factor(_factor(p, dict(transport_tech="pipe", transport_mode="blend", resource="ch4"), 0.0), dict(transport_tech="pipe2", transport_mode="blend", resource="ch4"), 0.0)
    q = _factor(_factor(q, dict(transport_tech="pipe", transport_mode="methane", resource="ch4"), 0.0), dict(transport_tech="pipe2", transport_mode="methane", resource="ch4"), 0.0)
    _refused(q, re.escape("pipe: transport_resource ch4 on link ab has no positive transport_mode_factor in any mode."))


def test_order_of_the_checks_one_dataset_two_violations():
    toy = lambda **kw: transport_modes_toy(**kw)  # noqa: E731
    # two resources on one technology, and a negative factor: the single-resource message
    p = _with_links(toy(modes=BLEND), order=("ab", "ba"))
    p = p.assign(transport_resource=_resource(p).where(p.link == "ab", "h2"))
    p = _factor(p, dict(transport_mode="blend", resource="h2"), -0.1)
    _refused(p, "a transport tech must carry a single resource")
    # a negative factor, and a main resource without factor: R16
    q = _factor(_factor(toy(modes=PURE), dict(transport_mode="methane", resource="ch4"), 0.0), dict(transport_mode="hydrogen", resource="h2"), -0.5)
    _refused(q, "transport_mode_factor.hydrogen, h2. = -0.5")
    # R17 and a block length: R17
    r = with_blocks(_factor(toy(modes=PURE), dict(transport_mode="methane", resource="ch4"), 0.0), transport={"pipe": 2})
    _refused(r, "has no positive transport_mode_factor")
    # a delay and a finite linepack: R13
    s = toy(modes=BLEND)
    s = s.assign(
        transport_delay=_per(s, "d", 0.0, LINK_TECH, [(dict(link="ab"), 1.0)]).d,
        transport_linepack_capacity=_per(s, "l", np.nan, (*LINK_TECH, "year_op"), [(dict(link="ab"), 5.0)]).l,
    )
    _refused(s, "takes no delay")


def test_transport_modal_on_a_group_is_read_by_label():
    own = {"blend": {"ch4": 0.8, "h2": 0.2}, "methane": {"ch4": 1.0}}
    p = transport_modes_toy(techs=("pipe", "pipe2"), modes={"pipe": BLEND, "pipe2": own})
    full = transport_modal(p)
    group = transport_modal(p.sel(transport_tech=["pipe2", "pipe"]))
    assert list(group.techs) == ["pipe2", "pipe"]
    assert list(full.techs) == ["pipe", "pipe2"]
    for tech in ("pipe", "pipe2"):
        assert group.factor.sel(transport_tech=tech).equals(full.factor.sel(transport_tech=tech))
    assert float(group.factor.sel(transport_tech="pipe2", transport_mode="blend", resource="h2")) == 0.2
    assert float(group.factor.sel(transport_tech="pipe", transport_mode="blend", resource="h2")) == 0.1


# --- Task 2: the flow by mode, the tie, the modal net generation ---

MODE_FLOW = "operation_transport_mode_power_in"
NET = "operation_transport_net_generation"


def _solve(p):
    return solve_model(build_model(p))


def _flow(m, name=MODE_FLOW, tech="pipe"):
    """A solved flow of the technology ``tech`` on link ``ab`` and year 2030: dims (mode, hour) or (hour)."""
    da = solved(m, name).sel(link="ab", year_op=2030)
    return da.sel(transport_tech=tech) if "transport_tech" in da.dims else da


def _net(m, resource, tech="pipe"):
    """Net generation of ``resource`` of ``tech`` on the hand toy: dims (area, hour)."""
    return solved(m, NET).sel(transport_tech=tech, resource=resource, year_op=2030)


def _live(m, name):
    """Number of live cells of the variable or constraint ``name`` (0 when the model has no such name)."""
    for container in (m.variables, m.constraints):
        if name in container:
            return int((container[name].labels != -1).sum())
    return 0


def test_share_cap_of_the_richest_mode():
    m = _solve(transport_modes_toy(modes=BLEND))
    assert m.objective.value == pytest.approx(92.0, rel=1e-7)
    flow = _flow(m)
    np.testing.assert_allclose(flow.sel(transport_mode="blend").values, [10, 10], atol=1e-6)
    np.testing.assert_allclose(flow.sel(transport_mode="methane").values, [0, 0], atol=1e-6)
    np.testing.assert_allclose(_net(m, "h2").sel(area="b").values, [1, 1], atol=1e-6)
    np.testing.assert_allclose(_net(m, "h2").sel(area="a").values, [-1, -1], atol=1e-6)
    for demand in (5.0, 10.0, 15.0):
        m = _solve(transport_modes_toy(modes=BLEND, demand=demand))
        h2, ch4 = _net(m, "h2").sel(area="b").values, _net(m, "ch4").sel(area="b").values
        np.testing.assert_allclose(h2 / (h2 + ch4), [0.1, 0.1], atol=1e-6)  # the richest mode binds, never above it
        assert (h2 / (h2 + ch4) <= 0.10 + 1e-6).all()


def _twin(plain, modal, rows=None):
    """The objective and the six variables of section 6.1 of ``plain`` and of its modal twin ``modal`` are equal."""
    a, b = _solve(plain), _solve(modal)
    assert b.objective.value == pytest.approx(a.objective.value, rel=1e-7)
    for name in (
        "operation_transport_power",
        "operation_transport_power_in",
        "operation_transport_power_capacity",
        "operation_transport_costs",
        "planning_transport_costs",
        NET,
    ):
        assert_same(solved(a, name), solved(b, name), name)
    name = "planning_transport_power_capacity"  # the split between decommissioning years is not fixed
    assert_same(solved(a, name).sum("year_dec"), solved(b, name).sum("year_dec"), name)
    return a, b


def test_one_mode_factor_one_is_the_plain_technology():
    toy = transport_modes_toy()
    a, b = _twin(toy, as_modal(toy, ["pipe"]))
    assert a.objective.value == pytest.approx(100.0, rel=1e-7)
    cases = dict(
        eff=dict(eff={"H": 0.9, "S": 0.8}),
        ramp=dict(ramp={"H": (0.5, 0.5), "S": (0.4, 0.6)}),
        availability=dict(availability={"H": [1.0, 0.5, 1.0, 0.8, 1.0, 1.0], "S": [0.7, 1.0, 1.0, 1.0, 0.9, 1.0]}),
        fuel=dict(fuel={"H": 0.1, "S": 0.05}),
    )
    for name, kw in cases.items():
        plain = transport_toy([1.0] * 6, **kw)
        for techs in (["H", "S"], ["S"]):
            _twin(plain, as_modal(plain, techs))


def test_free_factors_the_capacity_counts_the_flow():
    m = _solve(transport_modes_toy(modes=LEAN))
    assert m.objective.value == pytest.approx(97.41935, abs=1e-4)
    np.testing.assert_allclose(_flow(m).sel(transport_mode="lean").values, [10.7527] * 2, atol=1e-4)
    m = _solve(transport_modes_toy(modes=LEAN, capacity=10.0, shed=True))
    assert m.objective.value == pytest.approx(230.6, abs=1e-4)
    np.testing.assert_allclose(_flow(m, "operation_transport_power").values, [10, 10], atol=1e-6)


def test_pure_modes_share_the_capacity_like_the_official_model():
    assert _solve(transport_modes_toy(modes=PURE)).objective.value == pytest.approx(20.0, rel=1e-7)
    m = _solve(transport_modes_toy(modes=PURE, demand=15.0, capacity=10.0, shed=True))
    flow = _flow(m)
    np.testing.assert_allclose(flow.sel(transport_mode="hydrogen").values, [10, 10], atol=1e-6)
    np.testing.assert_allclose(flow.sel(transport_mode="methane").values, [0, 0], atol=1e-6)
    shed = [n for n in m.variables if "shedding" in n and "power" in n]
    assert shed
    assert m.objective.value == pytest.approx(1020.0, rel=1e-7)  # 2 * (10 + 5 * 100): one capacity, not one per mode
    np.testing.assert_allclose(solved(m, shed[0]).sel(resource="heat").sum("area").values.ravel(), [5, 5], atol=1e-6)


def test_efficiency():
    m = _solve(transport_modes_toy(modes=BLEND, eta=0.9))
    assert m.objective.value == pytest.approx(102.22222, abs=1e-4)
    np.testing.assert_allclose(_net(m, "h2").sel(area="b").values, [1, 1], atol=1e-6)
    np.testing.assert_allclose(_net(m, "h2").sel(area="a").values, [-1.11111] * 2, atol=1e-4)
    # the hurdle cost is on the withdrawn flow P = 0.9 * 11.1111 = 10, not on the injected flow
    p = transport_modes_toy(modes=BLEND, eta=0.9)
    m = _solve(p.assign(transport_hurdle_costs=xr.full_like(p.transport_hurdle_costs, 0.5)))
    assert m.objective.value == pytest.approx(102.22222 + 0.5 * 10 * 2, abs=1e-4)


def test_step_duration_scales_the_net_generation():
    m = _solve(transport_modes_toy(dt=(2.0, 0.5), modes=BLEND))
    assert m.objective.value == pytest.approx(46.0 * 2.5, rel=1e-7)
    np.testing.assert_allclose(_flow(m).sel(transport_mode="blend").values, [10, 10], atol=1e-6)  # MW, not MWh
    np.testing.assert_allclose(_net(m, "h2").sel(area="b").values, [2.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(_net(m, "h2").sel(area="a").values, [-2.0, -0.5], atol=1e-6)


def _live_cells(var, dims):
    """Bool DataArray: the cells of ``var`` (a linopy variable) that have a label, reduced to ``dims``."""
    labels = var.labels != -1
    return labels.any([d for d in labels.dims if d not in dims])


def _modal_flow_dataset():
    """Techs pipe (blend, methane), pipe2 (methane, lean, ch4 only; no pair on link ba), plain (no mode); links ab and ba."""
    own = {"methane": {"ch4": 1.0}, "lean": {"ch4": 0.9}}  # pipe2 carries ch4 only
    p = transport_modes_toy(techs=("pipe", "pipe2", "plain"), modes={"pipe": BLEND, "pipe2": own}, order=["blend", "methane", "lean"])
    p = _with_links(p, order=("ab", "ba"))
    resource = p.transport_resource.copy()
    resource.loc[dict(link="ba", transport_tech="pipe2")] = "nan"
    return p.assign(transport_resource=resource)


def test_modal_flow_cells():
    p = _modal_flow_dataset()
    m = build_model(p)
    modal = transport_modal(p)
    flow = m.variables[MODE_FLOW]
    assert flow.labels.indexes["transport_tech"].equals(modal.techs)
    assert {"link", "transport_tech", "transport_mode", "hour", "year_op"} <= set(flow.labels.dims)
    expected = transport_pair_exists(p).sel(transport_tech=modal.techs) & modal.defined.sel(transport_tech=modal.techs)
    live = _live_cells(flow, ("link", "transport_tech", "transport_mode"))
    assert live.transpose("link", "transport_tech", "transport_mode").values.tolist() == expected.transpose("link", "transport_tech", "transport_mode").values.tolist()
    assert not live.sel(link="ba", transport_tech="pipe2").any() and not live.sel(transport_tech="pipe2", transport_mode="blend").any()
    # the net generation is live at (modal k, r) iff carried(k, r); a plain technology keeps its own rule
    ng = _live_cells(m.variables[NET], ("transport_tech", "resource")).reindex(transport_tech=p.transport_tech, resource=p.resource, fill_value=False)
    main = (p.transport_resource.isel(link=0) == p.resource)  # (transport_tech, resource)
    for tech in p.transport_tech.values:
        want = modal.carried.sel(transport_tech=tech) if tech in modal.techs else main.sel(transport_tech=tech)
        assert ng.sel(transport_tech=tech).values.tolist() == want.values.tolist(), tech
    assert _live(m, NET) == _live(m, "operation_transport_net_generation_def") + _live(m, "operation_transport_mode_net_generation_def")
    assert _live(m, "operation_transport_mode_net_generation_def") > 0 and _live(m, "operation_transport_net_generation_def") > 0
    # the tie is live where the pair exists, for the modal technologies
    assert _live(m, "operation_transport_mode_power_in_def") == int(expected.any("transport_mode").sum()) * p.sizes["hour"]


def test_not_refused():
    # factors summing to 0.93 and 1.2 build (with the warning W1); two modes with the same factors; one defined mode alone
    for modes in ({"m": {"ch4": 0.9, "h2": 0.03}}, {"m": {"ch4": 1.2}}, {"a": BLEND["blend"], "b": BLEND["blend"]}, {"a": {"ch4": 1.0}}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _solve(transport_modes_toy(modes=modes))
    # a mode defined for one technology only
    _solve(transport_modes_toy(techs=("pipe", "pipe2"), modes={"pipe": BLEND, "pipe2": {"methane": {"ch4": 1.0}}}, order=["blend", "methane"]))
    # modal data with transport off
    build_model(transport_modes_toy(modes=BLEND).assign(transport=np.array(False)))
    # a finite mode ramp on a technology that is not modal
    two = with_modes(transport_modes_toy(techs=("pipe", "ship")), {"pipe": BLEND}, ramp={"ship": (0.5, 0.5)})
    _solve(two)
    # a modal technology with no existing pair: its NG cells are pinned to 0, no label of X
    p = transport_modes_toy(techs=("pipe", "ghost"), modes={"pipe": BLEND, "ghost": BLEND})
    p = p.assign(transport_resource=_resource(p, nan=[("ab", "ghost")]))
    m = _solve(p)
    assert (m.variables[MODE_FLOW].labels.sel(transport_tech="ghost") == -1).all()
    assert (m.variables[MODE_FLOW].labels.sel(transport_tech="pipe") != -1).any()
    ghost = _net(m, "h2", tech="ghost")
    np.testing.assert_allclose(ghost.values, 0.0, atol=1e-9)
    assert not np.isnan(ghost.values).any()
    assert m.objective.value == pytest.approx(92.0, rel=1e-7)


def test_net_generation_unsorted_labels():
    own = {"blend": {"ch4": 0.8, "h2": 0.2}, "methane": {"ch4": 1.0}}

    def build(techs, order, links):
        p = transport_modes_toy(techs=techs, modes={"Z": BLEND, "A": own}, order=order)
        p = _with_links(p, order=links)
        hurdle = p.transport_hurdle_costs.copy()
        hurdle.loc[dict(transport_tech="Z")] = 0.02
        hurdle.loc[dict(transport_tech="A")] = 0.01
        hurdle.loc[dict(link="ba")] += 0.001
        return p.assign(transport_hurdle_costs=hurdle)

    sorted_case = _solve(build(("A", "Z"), ["blend", "methane"], ("ab", "ba")))
    unsorted = _solve(build(("Z", "A"), ["methane", "blend"], ("ba", "ab")))
    assert unsorted.objective.value == pytest.approx(sorted_case.objective.value, rel=1e-7)
    assert sorted_case.objective.value < 92.0  # A (20 % hydrogen) is used
    for name in (MODE_FLOW, NET, "operation_transport_power", "operation_transport_power_in"):
        assert_same(solved(sorted_case, name), solved(unsorted, name), name)
    flow = solved(unsorted, MODE_FLOW)
    assert float(flow.sel(transport_tech="A", transport_mode="blend").sum()) > 1.0


def test_two_modal_technologies_with_opposite_ends():
    plain = transport_toy([1.0] * 6, techs=["Z", "B"])
    ends = plain.transport_area_from.values
    assert (ends[:, 0] != ends[:, 1]).all()  # the tech of odd index runs the other way
    assert transport_pair_exists(plain).all()
    _twin(plain, as_modal(plain, ["Z", "B"]))


def test_split_module_modal_hourly_technology_and_a_block_ship():
    # the hourly group holds the modal H, at position 1 of the technology axis; the ship S is on 24 h blocks
    plain = with_blocks(
        _transport_t2([1.0] * 48, {"H": 1.5, "S": 3.0}, techs=("S", "H"), eff={"H": 0.9, "S": 0.8}), transport={"S": 24}
    )
    a, b = _twin(plain, as_modal(plain, ["H"]))
    assert a.variables["operation_transport_block_24h_power"].labels.max() >= 0  # the module is split
    assert b.variables[MODE_FLOW].labels.indexes["transport_tech"].tolist() == ["H"]


def test_modal_statements_come_after_the_hourly_ones_and_before_the_blocks():
    plain = with_blocks(_transport_t2([1.0] * 4, {"H": 1.5, "S": 3.0}), transport={"S": 2})
    m = build_model(as_modal(plain, ["H"]))
    for names, last_hourly, modal in (
        (list(m.variables), "planning_transport_costs", [MODE_FLOW]),
        (list(m.constraints), "planning_transport_costs_def", ["operation_transport_mode_power_in_def", "operation_transport_mode_net_generation_def"]),
    ):
        first_block = min(i for i, n in enumerate(names) if n.startswith("operation_transport_block_"))
        assert names.index(last_hourly) < names.index(modal[0]) and names.index(modal[-1]) < first_block
        assert names[names.index(modal[0]) : names.index(modal[-1]) + 1] == modal
    assert list(m.variables).index("operation_transport_power_in") < list(m.variables).index(MODE_FLOW)
    if "operation_transport_net_generation_def" in names:  # the dense engine builds it empty, the sparse one not at all
        assert names.index("operation_transport_net_generation_def") < names.index("operation_transport_mode_power_in_def")


def _obj(**kw):
    return _solve(transport_modes_toy(modes=BLEND, order=["blend", "methane"], **kw))


def test_mode_ramp_up():
    m = _obj(availability=[0, 1], ramp=(0.5, np.nan))
    assert m.objective.value == pytest.approx(98.0, rel=1e-7)
    flow = _flow(m)
    np.testing.assert_allclose(flow.sel(transport_mode="blend").values[1], 5.0, atol=1e-6)
    np.testing.assert_allclose(flow.sel(transport_mode="methane").values[1], 5.0, atol=1e-6)


def test_mode_ramp_reads_the_modal_technology_not_the_first_one():
    # a plain technology comes first on the axis, with capacity 0 and a finite ramp that must be ignored
    p = transport_modes_toy(
        techs=("plain", "pipe"), modes={"pipe": BLEND}, order=["blend", "methane"], ramp=(0.5, np.nan), availability=[0, 1]
    )
    for name in ("transport_power_capacity_min", "transport_power_capacity_max"):
        v = p[name].copy()
        v.loc[{"transport_tech": "plain"}] = 0.0
        p = p.assign({name: v})
    r = p.transport_mode_ramp_up.copy()
    r.loc[{"transport_tech": "plain"}] = 0.0
    m = _solve(p.assign(transport_mode_ramp_up=r))
    assert m.objective.value == pytest.approx(98.0, rel=1e-7)
    np.testing.assert_allclose(_flow(m).sel(transport_mode="blend").values[1], 5.0, atol=1e-6)


def test_mode_ramp_down():
    assert _obj(availability=[1, 0], ramp=(np.nan, 0.5)).objective.value == pytest.approx(98.0, rel=1e-7)


def test_mode_ramp_default_is_free():
    assert _obj(availability=[0, 1]).objective.value == pytest.approx(96.0, rel=1e-7)
    assert _obj(availability=[0, 1], ramp=(np.nan, np.nan)).objective.value == pytest.approx(96.0, rel=1e-7)


def test_mode_ramp_bounds_a_single_mode_too():
    # the rise of the second mode alone is bounded by down, not by up
    kw = dict(availability=[0, 0], demand=[0, 20], shed=True)
    m = _obj(ramp=(np.nan, 0.5), **kw)
    assert m.objective.value == pytest.approx(1050.0, rel=1e-7)
    np.testing.assert_allclose(_flow(m).sel(transport_mode="methane").values[1], 10.0, atol=1e-6)
    assert _obj(ramp=(0.5, np.nan), **kw).objective.value == pytest.approx(100.0, rel=1e-7)


def test_mode_ramp_divides_by_the_step():
    kw = dict(dt=(2.0, 2.0), availability=[0, 1])
    assert _obj(ramp=(0.5, np.nan), **kw).objective.value == pytest.approx(192.0, rel=1e-7)


def test_mode_ramp_pairs_of_defined_modes_only(tmp_path):
    for label in ("pure_h2", "h2/ch4"):
        order = ["blend", "methane", label]
        p = transport_modes_toy(
            techs=("pipe", "pipe2"),
            modes={"pipe": BLEND, "pipe2": {**BLEND, label: {"h2": 1.0}}},
            order=order,
            ramp=(0.5, 0.5),
        )
        m = build_model(p)
        ramps = [n for n in m.constraints if "mode_ramp" in n]
        pairs = ["0_1", "0_2", "1_2"]
        # the order: for each pair up then down
        assert ramps == [f"operation_transport_mode_ramp_{d}_constraint_{ij}" for ij in pairs for d in ("up", "down")]
        for d in ("up", "down"):
            for ij, techs in (("0_1", 2), ("0_2", 1), ("1_2", 1)):
                labels = m.constraints[f"operation_transport_mode_ramp_{d}_constraint_{ij}"].labels
                live = (labels != -1).any([x for x in labels.dims if x != "transport_tech"])
                assert int(live.sum()) == techs
                if techs == 1:
                    assert live.sel(transport_tech="pipe2").item() and not live.sel(transport_tech="pipe").item()
        if label == "h2/ch4":
            m = solve_model(m)
            from pommes.io.save_solution import save_solution

            save_solution(m, str(tmp_path / "out"), export_csv=True, save_input=False)


def test_mode_ramp_on_a_non_modal_technology_is_ignored():
    two = with_modes(transport_modes_toy(techs=("pipe", "ship")), {"pipe": BLEND}, ramp={"ship": (0.5, 0.5)})
    m = build_model(two)
    assert not [n for n in m.constraints if "mode_ramp" in n]


def test_mode_ramp_does_not_bound_the_first_step():
    # a flow of 20 MW from the first step on: the ramp is not cyclic and has no step before it
    m = _obj(availability=[0, 0], demand=[20, 0], shed=True, ramp=(np.nan, 0.5))
    assert m.objective.value == pytest.approx(100.0, rel=1e-7)


# --- Task 4 (step 0): the mode ramp, uneven steps, a ramp of 0, the capacity on the right-hand side, a pairless technology ---

RAMP_UP_0_1 = "operation_transport_mode_ramp_up_constraint_0_1"
PIN_MH = 321199.9439065916
RAMP_DOWN_0_1 = "operation_transport_mode_ramp_down_constraint_0_1"


def test_mode_ramp_with_uneven_steps():
    # the ramp divides the change by the duration of the current step (2.0, 0.5 and 1.0 h), not of the previous one
    m = _obj(dt=(2.0, 0.5, 1.0), availability=[0, 1, 1], ramp=(0.5, np.nan))
    assert m.objective.value == pytest.approx(171.5, rel=1e-7)


def test_mode_ramp_of_zero_freezes_the_difference():
    # a ramp of 0 is written (it is finite) and the difference of the two modes cannot change: 100, the free case is 96
    m = _obj(availability=[0, 1], ramp=(0.0, np.nan))
    assert m.objective.value == pytest.approx(100.0, rel=1e-7)
    assert _live(m, RAMP_UP_0_1) == 1


def test_mode_ramp_right_hand_side_follows_the_capacity():
    assert _obj(capacity=30.0, availability=[0, 1], ramp=(0.2, np.nan)).objective.value == pytest.approx(98.8, rel=1e-7)


def test_mode_ramp_cells_of_a_technology_without_a_pair_are_not_live():
    p = transport_modes_toy(techs=("ghost", "pipe"), modes={"ghost": BLEND, "pipe": BLEND}, order=["blend", "methane"], ramp=(0.5, 0.5))
    p = p.assign(transport_resource=_resource(p, nan=[("ab", "ghost")]))
    m = build_model(p)
    for way in ("up", "down"):
        labels = m.constraints[f"operation_transport_mode_ramp_{way}_constraint_0_1"].labels
        live = (labels != -1).any([d for d in labels.dims if d != "transport_tech"])
        assert live.sel(transport_tech="pipe").item() and not live.sel(transport_tech="ghost").item()
        assert int(live.sum()) == 1  # the technologies with a pair


def test_pairless_modal_technology_first_in_the_model_order():
    p = transport_modes_pairless_first_dataset()
    assert list(p.transport_tech.values) == ["ghost", "pipe"] and not transport_pair_exists(p).sel(transport_tech="ghost").any()
    m = _solve(p)
    assert m.objective.value == pytest.approx(92.0, rel=1e-7)
    np.testing.assert_allclose(_net(m, "h2", tech="ghost").values, 0.0, atol=1e-9)
    # the flows by mode of pipe are those of the plain blend toy, whose modes are ghost-free
    twin = _solve(transport_modes_toy(modes=BLEND, order=["blend", "methane", "hydrogen"]))
    flow = _flow(m)
    assert_same(flow.sel(transport_mode=["blend", "methane"]), _flow(twin).sel(transport_mode=["blend", "methane"]))
    np.testing.assert_allclose(flow.sel(transport_mode="hydrogen").fillna(0).values, 0.0, atol=1e-9)  # not defined for pipe: no cell
    for resource in ("h2", "ch4"):
        assert_same(_net(m, resource), _net(twin, resource), resource)


# --- Task 4: the whole model: conditions, years, retrofit, blocks elsewhere, outputs, carbon ---


def test_operating_conditions_with_modes():
    p = transport_modes_conditions_dataset()
    both = _solve(p)
    flow = both.variables[MODE_FLOW]
    assert OPERATING_CONDITION in flow.labels.dims
    assert_condition_dims_consistent(both)

    def only(weights):
        return p.assign({OPERATING_CONDITION_WEIGHT: p[OPERATING_CONDITION_WEIGHT].copy(data=weights)})

    a, b = _solve(only([1.0, np.nan])), _solve(only([np.nan, 1.0]))
    assert both.objective.value == pytest.approx(0.4 * a.objective.value + 0.6 * b.objective.value, rel=1e-7)
    assert a.objective.value == pytest.approx(98.0, rel=1e-7)
    assert b.objective.value == pytest.approx(118.0, rel=1e-7)
    assert both.objective.value == pytest.approx(110.0, rel=1e-7)
    # the mask follows condition_mask: a condition without a weight has no cell of the flow, the tie, the net generation, the ramps
    for m, absent in ((a, "b"), (b, "a")):
        for name in (MODE_FLOW, "operation_transport_mode_power_in_def", "operation_transport_mode_net_generation_def", RAMP_UP_0_1, RAMP_DOWN_0_1):
            labels = m.constraints[name].labels if name in m.constraints else m.variables[name].labels
            assert (labels.sel({OPERATING_CONDITION: absent}) == -1).all(), (name, absent)
            assert (labels.sel({OPERATING_CONDITION: "a" if absent == "b" else "b"}) != -1).any(), name
    # the mode ramp binds in both conditions: without it the objective is lower
    free = _solve(p.drop_vars(["transport_mode_ramp_up", "transport_mode_ramp_down"]))
    assert free.objective.value < both.objective.value - 1e-6
    # each condition has its own flows by mode
    x = solved(both, MODE_FLOW)
    assert not np.allclose(x.sel({OPERATING_CONDITION: "a"}).fillna(0), x.sel({OPERATING_CONDITION: "b"}).fillna(0))


def test_multi_horizon_with_modes():
    plain = transport_t2_multi_horizon_dataset()
    _twin(plain, as_modal(plain, ["H"]))  # one mode, factor 1: the plain technology, with investment and early decommissioning
    p = transport_modes_multi_horizon_dataset()
    m = _solve(p)
    flow = solved(m, MODE_FLOW).sel(transport_tech="H")
    power = solved(m, "operation_transport_power").sel(transport_tech="H")
    eta = p.transport_efficiency.sel(transport_tech="H")
    assert float(eta.sel(year_op=2030).min()) == pytest.approx(0.9) and float(eta.sel(year_op=2040).min()) == pytest.approx(0.85)
    for year in (2030, 2040):
        # both modes are used in each year; P = eta(year) * the injected flow
        assert float(flow.sel(year_op=year, transport_mode="mix").sum()) > 1e-3
        assert float(flow.sel(year_op=year, transport_mode="cargo").sum()) > 1e-3
        want = eta.sel(year_op=year) * flow.sel(year_op=year).sum("transport_mode")
        assert_same(power.sel(year_op=year), want)
        # the net generation of oil at b has the arrival side eta of the year and the factor 0.1
        net = solved(m, NET).sel(transport_tech="H", resource="oil", year_op=year)
        expected = 0.0
        for link in p.link.values:
            expected += (0.1 * eta.sel(link=link, year_op=year) * flow.sel(link=link, year_op=year, transport_mode="mix").sum()).item() * (
                p.transport_area_to.sel(link=link, transport_tech="H").item() == "b"
            )
        assert float(net.sel(area="b").sum()) == pytest.approx(expected, rel=1e-7)
    # the departure side has no eta: the net generation of oil at a is -0.1 * mix flow leaving a, +0.1 * eta * mix flow arriving
    for year in (2030, 2040):
        out = float(solved(m, NET).sel(transport_tech="H", resource="oil", year_op=year, area="a").sum())
        want = 0.0
        for link in p.link.values:
            mix = flow.sel(link=link, year_op=year, transport_mode="mix").sum().item()
            if p.transport_area_from.sel(link=link, transport_tech="H").item() == "a":
                want -= 0.1 * mix
            else:
                want += 0.1 * eta.sel(link=link, year_op=year).item() * mix
        assert out == pytest.approx(want, rel=1e-7)
        assert out < -1e-3
    # the ramps bind (the demand steps): each one alone raises the objective, the pinned value is with both
    assert m.objective.value == pytest.approx(PIN_MH, rel=1e-7)
    for dropped in ("transport_mode_ramp_up", "transport_mode_ramp_down"):
        assert _solve(p.drop_vars(dropped)).objective.value < m.objective.value - 1e-3, dropped
    # the mode ramps by operation year: up in 2030, down in 2040 only
    built = m.constraints
    for way, year, other in (("up", 2030, 2040), ("down", 2040, 2030)):
        labels = built[f"operation_transport_mode_ramp_{way}_constraint_0_1"].labels
        live = (labels != -1).any([d for d in labels.dims if d != "year_op"])
        assert live.sel(year_op=year).item() and not live.sel(year_op=other).item(), way


def test_retrofit_plain_to_modal():
    p = transport_modes_retrofit_dataset()
    m = _solve(p)
    transfer = solved(m, "planning_retrofit_transport_power_capacity")
    assert float(transfer.sum()) > 1e-3  # the transfer from the plain source H to the modal target S is live and positive
    host = m.constraints["operation_transport_power_capacity_def"]
    assert list(host.labels.indexes["transport_tech"]) == ["H", "S"] and (host.labels != -1).any()
    assert "planning_retrofit_transport_power_capacity" in variables_in(m, host)
    assert float(solved(m, MODE_FLOW).sel(transport_tech="S", year_op=2040).sum()) > 1e-3
    # one mode, factor 1 on the target: the plain-target twin
    plain = p.drop_vars(["transport_mode_factor", "transport_mode"], errors="ignore")
    a, b = _twin(plain, as_modal(plain, ["S"]))
    assert_same(solved(a, "planning_retrofit_transport_power_capacity"), solved(b, "planning_retrofit_transport_power_capacity"), "transfer")
    assert float(solved(a, "planning_retrofit_transport_power_capacity").sum()) > 1e-3
    assert b.objective.value == pytest.approx(a.objective.value, rel=1e-7)


def test_block_conversion_elsewhere():
    p = transport_modes_block_elsewhere_dataset()
    with pytest.warns(UserWarning, match=re.escape("h2: hourly because of transport tech pipe")):
        tb = time_blocks(p)
    assert np.isnan(tb.resource_length.sel(resource="h2").item())
    m = _solve(p)
    assert any("_block_" in name for name in m.variables)  # the conversion blocks are really built
    twin = _solve(with_blocks(p))  # the hourly twin: no block length at all
    assert not any("_block_" in name for name in twin.variables)
    assert m.objective.value == pytest.approx(twin.objective.value, rel=1e-7)
    assert m.objective.value == pytest.approx(1104.0, rel=1e-7)
    assert_same(solved(m, MODE_FLOW), solved(twin, MODE_FLOW), MODE_FLOW)


OUTER_JOIN = "Coordinates across variables not equal"


def test_outputs(tmp_path):
    from pommes.io.calculate_summaries import calculate_total_of_a_type, generate_summary_dataframes_from_results
    from pommes.io.read_solution import read_solution
    from pommes.io.save_solution import save_solution
    from pommes.post_process import get_capacities, get_net_generation

    # the toy with the blend: the flow by mode makes the round trip, NetCDF and CSV
    p = transport_modes_toy(modes=BLEND)
    m = _solve(p)
    solution = m.solution.copy(deep=True).map(densify_any)  # the sparse engine solves on units
    save_solution(m, str(tmp_path / "out"), export_csv=True, save_input=False)
    saved = xr.open_dataset(tmp_path / "out" / "solution.nc")
    assert saved[MODE_FLOW].dims == solution[MODE_FLOW].dims
    xr.testing.assert_allclose(saved[MODE_FLOW], solution[MODE_FLOW])
    read = read_solution(str(tmp_path / "out" / "variables"))
    assert "transport_mode" in read[MODE_FLOW].dims and list(read[MODE_FLOW].transport_mode.values) == ["blend", "methane"]
    got, ref = read[MODE_FLOW].to_series().dropna(), solution[MODE_FLOW].to_series().dropna()
    ref = ref.reorder_levels(got.index.names).sort_index()
    assert got.sort_index().index.equals(ref.index)
    np.testing.assert_allclose(got.sort_index().values, ref.values, rtol=1e-9, atol=1e-9)
    assert got.sum() == pytest.approx(20.0)
    # get_net_generation lists the modal technology under each carried resource
    net = get_net_generation(solution, p)
    pipe = net.sel(category="pipe")
    np.testing.assert_allclose(pipe.sel(resource="h2", area="b").values.ravel(), [1, 1], atol=1e-6)
    np.testing.assert_allclose(pipe.sel(resource="ch4", area="b").values.ravel(), [9, 9], atol=1e-6)
    # no summary has a column for the mode, and no new name looks like a 1:1 existing one
    carbon = with_carbon(p, conversion={"well": 0.2}, tax=7.0)  # carbon on: the emissions table exists
    s_carbon = _solve(carbon).solution.map(densify_any)
    for kind in ("power_capacity", "net_generation", "costs", "emissions"):
        table = calculate_total_of_a_type(kind, s_carbon, carbon, "operation", True, False)
        assert not table.empty and "transport_mode" not in table.index.names and "transport_mode" not in table.columns, kind
    for name, frame in generate_summary_dataframes_from_results(s_carbon, carbon).items():
        assert "transport_mode" not in list(getattr(frame, "index", pd.Index([])).names) + list(getattr(frame, "columns", [])), name
        assert "transport_mode" not in getattr(frame, "dims", ()), name
    plain = build_model(p.drop_vars(["transport_mode_factor", "transport_mode"]))
    assert set(m.variables) - set(plain.variables) == {MODE_FLOW}
    assert not any(n.endswith(("_power_capacity", "_net_generation", "_costs", "_emissions")) for n in set(m.variables) - set(plain.variables))
    # technologies ["Z", "A"]: the solution sorts transport_tech, the readers give the plain twin's tables
    base = _transport_t2([1.0] * 6, {"Z": 1.5, "A": 2.0}, techs=("Z", "A"))
    modal, twin = _solve(as_modal(base, ["Z"])), _solve(base)  # A stays plain: the flow by mode lives on Z only
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s_modal, s_twin = modal.variables.solution.map(densify_any), twin.variables.solution.map(densify_any)
    assert any(OUTER_JOIN in str(w.message) for w in caught)  # the warning is raised, and the readers do not fail on it
    assert list(s_twin["operation_transport_power"].transport_tech.values) == ["Z", "A"]
    assert list(s_modal["operation_transport_power"].transport_tech.values) == ["A", "Z"]  # outer-joined and sorted
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=OUTER_JOIN)
        for reader in (get_capacities, get_net_generation):
            a, b = reader(s_modal, as_modal(base, ["Z"])), reader(s_twin, base)
            assert float(abs(b).sum()) > 1e-3
            assert_same(a, b, reader.__name__)


def test_one_mode_factor_one_is_the_plain_technology_carbon():
    plain, modal = transport_t2_fuel_emissions_dataset(), transport_modes_carbon_dataset()
    a, b = _solve(plain), _solve(modal)
    assert b.objective.value == pytest.approx(a.objective.value, rel=1e-7)
    for name in ("operation_carbon_emissions", "operation_total_carbon_emissions"):
        assert float(abs(solved(a, name)).sum()) > 1e-3  # the fuel emits: the comparison is not vacuous
        assert_same(solved(a, name), solved(b, name), name)
    assert_same(solved(a, "operation_transport_power_in"), solved(b, "operation_transport_power_in"), "power_in")
