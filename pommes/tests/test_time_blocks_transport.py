"""Transport technologies on time blocks, Task 1: the parameter, the refusals R11 and R12, the pair mask."""

import warnings

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import (
    check_inputs,
    check_time_blocks,
)
from pommes.utils import (
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


# --- interim guard (Task 1 to 2), deleted by Task 2 ---


def test_block_ship_refused_until_task_2(parameters):
    with pytest.raises(NotImplementedError, match="block transport technologies are not built yet"):
        build_model(ship(parameters))
    build_model(parameters.assign({LENGTH: lengths(parameters)}))
    build_model(ship(parameters).assign(transport=np.array(False)))
