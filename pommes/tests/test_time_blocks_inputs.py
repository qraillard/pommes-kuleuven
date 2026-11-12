"""Inputs, block maps and refusals of the time blocks (multi-resolution)."""

import warnings

import numpy as np
import pytest
import xarray as xr

from pommes.model.data_validation.dataset_check import (
    check_inputs,
    check_time_blocks,
)
from pommes.utils import block_dim, conversion_block_rule, time_blocks

FOUR = [
    "conversion_time_block_hrs",
    "combined_time_block_hrs",
    "resource_time_block_hrs",
    "conversion_time_block_rule",
]


def per(p, dim, **values):
    """A float64 array over ``dim`` with NaN except for the named labels."""
    da = xr.DataArray(
        np.full(p[dim].size, np.nan), dims=dim, coords={dim: p[dim].values}
    )
    for label, value in values.items():
        da.loc[{dim: label}] = value
    return da


def blocks(p, conversion=None, combined=None, resource=None, steps=None):
    """``p`` with block lengths per tech or resource (dicts label -> hours)."""
    p = p.copy()
    if conversion is not None:
        p["conversion_time_block_hrs"] = per(
            p, "conversion_tech", **conversion
        )
    if combined is not None:
        p["combined_time_block_hrs"] = per(p, "combined_tech", **combined)
    if resource is not None:
        p["resource_time_block_hrs"] = per(p, "resource", **resource)
    if steps is not None:
        p["time_step_duration"] = xr.DataArray(
            np.resize(np.array(steps, dtype="float64"), p.hour.size),
            dims="hour",
            coords={"hour": p.hour.values},
        )
    return p


def test_four_parameters_are_defaulted(parameters):
    p = check_inputs(parameters.drop_vars(FOUR))
    for name in FOUR[:3]:
        assert p[name].dtype == np.float64 and bool(p[name].isnull().all())
    assert p["conversion_time_block_rule"].item() == "min"


def test_module_parameters_not_defaulted_when_the_module_is_off(parameters):
    p = parameters.drop_vars(FOUR).assign(conversion=np.array(False))
    p = check_inputs(p)
    assert "conversion_time_block_hrs" not in p
    assert "conversion_time_block_rule" not in p
    assert "resource_time_block_hrs" in p


@pytest.mark.parametrize("dim", ["area", "year_op", "hour"])
def test_new_parameters_refuse_other_coordinates(parameters, dim):
    p = parameters.copy()
    p["conversion_time_block_hrs"] = per(p, "conversion_tech", ocgt=24)
    p["conversion_time_block_hrs"] = p[
        "conversion_time_block_hrs"
    ].expand_dims({dim: p[dim].values})
    with pytest.raises(ValueError, match="conversion_time_block_hrs"):
        check_inputs(p)


def test_new_parameters_refuse_operating_condition(parameters):
    p = parameters.assign_coords(operating_condition=["a", "b"]).assign(
        operating_condition_weight=xr.DataArray(
            [0.5, 0.5], dims="operating_condition"
        )
    )
    p["conversion_time_block_hrs"] = per(
        p, "conversion_tech", ocgt=24
    ).expand_dims(operating_condition=["a", "b"])
    with pytest.raises(ValueError, match="conversion_time_block_hrs"):
        check_inputs(p)


def test_demand_refuses_a_block_coordinate(parameters):
    p = parameters.copy()
    p["demand"] = p["demand"].expand_dims(time_block_24h=np.arange(2))
    with pytest.raises(ValueError, match="demand"):
        check_inputs(p)


def test_rule_round_trips_through_netcdf(parameters, tmp_path):
    rule = xr.DataArray(
        np.array(["min", "mean", "min", "min", "min"]),
        dims="conversion_tech",
        coords={"conversion_tech": parameters.conversion_tech.values},
    )
    ds = xr.Dataset(
        {"conversion_time_block_rule": rule, "zero_d": np.array("min")}
    )
    ds.to_netcdf(tmp_path / "rule.nc", engine="h5netcdf")
    back = xr.load_dataset(tmp_path / "rule.nc", engine="h5netcdf")
    xr.testing.assert_equal(back["conversion_time_block_rule"], rule)
    assert back["zero_d"].item() == "min"


def test_block_maps(parameters):
    p = blocks(parameters, conversion={"ocgt": 4}, steps=[1])
    tb = time_blocks(p)
    name = block_dim(4)
    assert name == "time_block_4h"
    b = tb.block_of_hour[4]
    assert b.dims == ("hour",) and b.name == name
    assert b.dtype == np.int64
    assert b.values.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert tb.duration[4].values.tolist() == [4, 4, 2]
    assert tb.start[4].values.tolist() == [0, 4, 8]
    assert tb.first_hour[4].values.tolist() == [0, 4, 8]
    assert tb.first_step_duration[4].values.tolist() == [1, 1, 1]
    for da in (tb.duration[4], tb.start[4], tb.first_hour[4]):
        assert da.dims == (name,)
        assert da[name].dtype == np.int64
        assert da[name].values.tolist() == [0, 1, 2]
    assert tb.active
    assert tb.techs("conversion", 4) == ["ocgt"]
    assert tb.techs("conversion", None) == [
        "electrolysis",
        "smr",
        "smr_ccs",
        "wind_onshore",
    ]


def test_block_maps_uneven_steps(parameters):
    p = blocks(parameters, conversion={"ocgt": 4}, steps=[1, 1, 2])
    tb = time_blocks(p)
    assert tb.duration[4].values.tolist()[:2] == [4, 4]
    assert tb.duration[4].values.tolist() == [4, 4, 4, 1]
    assert tb.first_step_duration[4].values.tolist()[:2] == [1, 1]
    assert tb.block_of_hour[4].values.tolist()[:5] == [0, 0, 0, 1, 1]


def test_resource_lengths(parameters):
    tb = time_blocks(blocks(parameters, resource={"heat": 6}, steps=[1]))
    assert tb.resources(6) == ["heat"]
    assert tb.resources(None) == ["electricity", "hydrogen", "methane"]
    assert tb.balance_lengths == [6]
    assert tb.active


def test_tech_length_is_nan_when_the_module_is_off(parameters):
    p = blocks(parameters, conversion={"ocgt": 24}).assign(
        conversion=np.array(False)
    )
    tb = time_blocks(p)
    assert bool(tb.tech_length["conversion"].isnull().all())
    assert not tb.active
    tb = time_blocks(p.drop_vars("conversion"))
    assert bool(tb.tech_length["conversion"].isnull().all())
    assert not tb.active


def test_absent_parameters_read_as_nan(parameters):
    tb = time_blocks(parameters.drop_vars(FOUR))
    assert not tb.active
    assert tb.block_of_hour == {} and tb.balance_lengths == []
    assert bool(tb.resource_length.isnull().all())


@pytest.mark.parametrize("length", [0, -24, 1.5])
def test_r1_length_not_positive_or_not_whole(parameters, length):
    with pytest.raises(ValueError, match=rf"ocgt.*{length}"):
        time_blocks(blocks(parameters, conversion={"ocgt": length}))
    # checked on every present value, whatever the module flag
    off = blocks(parameters, conversion={"ocgt": length}).assign(
        conversion=np.array(False)
    )
    with pytest.raises(ValueError, match="ocgt"):
        time_blocks(off)
    with pytest.raises(ValueError, match="heat"):
        time_blocks(blocks(parameters, resource={"heat": length}))


def test_r2_step_crossing_a_block_boundary(parameters):
    cross = "crosses a block boundary"
    with pytest.raises(ValueError, match=cross):
        time_blocks(blocks(parameters, conversion={"ocgt": 2}, steps=[1, 2]))
    with pytest.raises(ValueError, match=cross):
        time_blocks(blocks(parameters, conversion={"ocgt": 1}, steps=[3]))
    # within the tolerance, and just beyond it
    steps = [1, 1 + 1e-10]
    time_blocks(blocks(parameters, conversion={"ocgt": 2}, steps=steps))
    steps = [1, 1 + 1e-8]
    with pytest.raises(ValueError, match=cross):
        time_blocks(blocks(parameters, conversion={"ocgt": 2}, steps=steps))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_r3_bad_time_step_duration(parameters, bad):
    p = blocks(parameters, conversion={"ocgt": 4}, steps=[1])
    p["time_step_duration"][3] = bad
    with pytest.raises(ValueError, match="time_step_duration"):
        time_blocks(p)
    with pytest.raises(ValueError, match="time_step_duration"):
        check_time_blocks(p)
    # without block inputs the same step is left alone
    q = parameters.copy(deep=True)
    q["time_step_duration"][3] = bad
    check_time_blocks(q)


def test_r8_rule_typo(parameters):
    p = parameters.copy()
    p["conversion_time_block_rule"] = xr.DataArray(
        np.array(["min", "mni", "min", "min", "min"]),
        dims="conversion_tech",
        coords={"conversion_tech": p.conversion_tech.values},
    )
    with pytest.raises(ValueError, match="ocgt.*mni"):
        time_blocks(p)
    with pytest.raises(ValueError, match="ocgt.*mni"):
        check_time_blocks(p)


def test_blank_rule_reads_as_min(parameters):
    p = parameters.copy()
    n = p.conversion_tech.size
    for blank in (np.array("nan"), np.array(np.nan), np.array("")):
        p["conversion_time_block_rule"] = xr.DataArray(blank)
        rule = conversion_block_rule(p)
        assert rule.dims == ("conversion_tech",)
        assert rule.values.tolist() == ["min"] * n
    p["conversion_time_block_rule"] = xr.DataArray(
        np.array(["mean", "nan", "", "min", "mean"]),
        dims="conversion_tech",
        coords={"conversion_tech": p.conversion_tech.values},
    )
    assert conversion_block_rule(p).values.tolist() == [
        "mean",
        "min",
        "min",
        "min",
        "mean",
    ]
    rule = conversion_block_rule(p.drop_vars(FOUR))
    assert rule.values.tolist() == ["min"] * n


def test_all_nan_raises_and_warns_nothing(parameters):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_inputs(parameters.copy(deep=True))
        check_time_blocks(parameters)


def test_blocks_anchor_at_the_first_step_of_a_slice(parameters):
    p = blocks(parameters, conversion={"ocgt": 4}, steps=[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tb = time_blocks(p.isel(hour=slice(3, 10)))
    assert tb.block_of_hour[4].values.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert tb.start[4].values.tolist() == [0, 4]
    assert tb.first_hour[4].values.tolist() == [3, 7]


def test_check_inputs_runs_the_time_block_checks(parameters):
    with pytest.raises(ValueError, match="ocgt"):
        check_inputs(blocks(parameters, conversion={"ocgt": 1.5}))
    p = blocks(parameters, conversion={"ocgt": 4}, steps=[1])
    p["time_step_duration"][3] = 0.0
    with pytest.raises(ValueError, match="time_step_duration"):
        check_inputs(p)
    p = parameters.copy()
    p["conversion_time_block_rule"] = xr.DataArray(
        np.array(["min", "mni", "min", "min", "min"]),
        dims="conversion_tech",
        coords={"conversion_tech": p.conversion_tech.values},
    )
    with pytest.raises(ValueError, match="ocgt.*mni"):
        check_inputs(p)


@pytest.mark.parametrize("step", [1 / 3, 0.1])
def test_non_dyadic_steps_are_not_refused(parameters, step):
    # R5: cumsum lands an ulp below a boundary, floor(start + 1e-9) absorbs it
    p = parameters.reindex(hour=np.arange(100))
    tb = time_blocks(blocks(p, conversion={"ocgt": 4}, steps=[step]))
    full = round(4 / step)
    nb = int(100 * step // 4)
    assert tb.duration[4].values[:nb] == pytest.approx(4)
    assert tb.start[4].values == pytest.approx(4 * np.arange(nb + 1))
    assert tb.block_of_hour[4].values[full - 1 : full + 1].tolist() == [0, 1]


def test_combined_block_length(parameters):
    p = blocks(parameters, combined={"electric_boiler": 24})
    tb = time_blocks(p)
    assert tb.active
    assert tb.techs("combined", 24) == ["electric_boiler"]
    assert tb.techs("combined", None) == []
    off = time_blocks(p.assign(combined=np.array(False)))
    assert bool(off.tech_length["combined"].isnull().all())
    assert not off.active
    with pytest.raises(ValueError, match="electric_boiler.*1.5"):
        time_blocks(blocks(parameters, combined={"electric_boiler": 1.5}))


def test_block_length_needs_time_step_duration(parameters):
    p = blocks(parameters, conversion={"ocgt": 4}).drop_vars(
        "time_step_duration"
    )
    with pytest.raises(ValueError, match="time_step_duration"):
        time_blocks(p)
