"""Inputs, block maps and refusals of the time blocks (multi-resolution)."""

import warnings

import numpy as np
import pytest
import xarray as xr

from pommes.model.data_validation.dataset_check import (
    check_inputs,
    check_time_blocks,
)
from pommes.utils import (
    block_dim,
    block_mean,
    check_block_daily_cap,
    check_block_flexdem,
    conversion_block_rule,
    time_blocks,
)

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
    # forced balances: no resource is derived hourly, so nothing to warn about
    p = blocks(
        parameters,
        conversion={"ocgt": 4},
        resource={"electricity": 4, "methane": 4},
        steps=[1],
    ).assign(turpe=np.array(False))
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


# --- Task 2: balance derivation, R4-R6, R9, R10, block_mean ---


def toy(
    conversion=None,
    combined=None,
    storage=None,
    transport=None,
    process=None,
    resources=("steel", "ore", "idle"),
    hours=8,
    **extra,
):
    """
    A minimal dataset with the inputs ``time_blocks`` reads.

    Each module argument maps a technology to ``{resource: factor}``
    (transport: ``(resource, fuel_resource or None)``).
    """
    res = list(resources)
    da = lambda dims, values, **c: xr.DataArray(values, dims=dims, coords=c)  # noqa: E731
    ds = xr.Dataset(
        {
            "time_step_duration": da(
                "hour", np.ones(hours), hour=np.arange(hours)
            ),
            "resource_time_block_hrs": da(
                "resource", np.full(len(res), np.nan), resource=res
            ),
        },
        coords={"resource": res, "hour": np.arange(hours)},
    )

    def factor(module, techs):
        dim = f"{module}_tech"
        names = list(techs)
        f = np.zeros((len(names), len(res)))
        for i, t in enumerate(names):
            for r, v in techs[t].items():
                f[i, res.index(r)] = v
        return names, dim, f

    for module, techs in (("conversion", conversion), ("combined", combined)):
        if techs is None:
            continue
        names, dim, f = factor(module, techs)
        ds[module] = np.array(True)
        ds[f"{module}_tech"] = names
        ds[f"{module}_factor"] = da(
            (dim, "resource"), f, **{dim: names}, resource=res
        )
        ds[f"{module}_time_block_hrs"] = da(
            dim, np.full(len(names), np.nan), **{dim: names}
        )
    if storage is not None:
        names, dim, f = factor("storage", storage)
        ds["storage"] = np.array(True)
        ds["storage_tech"] = names
        for k in ("in", "keep", "out"):
            ds[f"storage_factor_{k}"] = da(
                (dim, "resource"), f, storage_tech=names, resource=res
            )
    if process is not None:
        names, dim, f = factor("process", process)
        ds["process"] = np.array(True)
        ds["process_tech"] = names
        ds["process_factor"] = da(
            ("process_tech", "mode", "resource"),
            f[:, None, :],
            process_tech=names,
            mode=["m"],
            resource=res,
        )
    if transport is not None:
        names = list(transport)
        ds["transport"] = np.array(True)
        ds["transport_tech"] = names
        ds["link"] = ["l"]
        arr = lambda i: np.array(
            [[transport[t][i] for t in names]], dtype=object
        )  # noqa: E731
        ds["transport_resource"] = da(
            ("link", "transport_tech"),
            arr(0).astype(str),
            link=["l"],
            transport_tech=names,
        )
        ds["transport_fuel_resource"] = da(
            ("link", "transport_tech"),
            np.array([[transport[t][1] or "nan" for t in names]]),
            link=["l"],
            transport_tech=names,
        )
        ds["transport_fuel_consumption"] = da(
            ("link", "transport_tech"),
            np.array(
                [[np.nan if transport[t][1] is None else 0.1 for t in names]]
            ),
            link=["l"],
            transport_tech=names,
        )
    return ds.assign(**extra)


def setl(p, module, **lengths):
    dim = f"{module}_tech"
    return p.assign({f"{module}_time_block_hrs": per(p, dim, **lengths)})


def lengths_of(p):
    return dict(zip(p.resource.values, time_blocks(p).resource_length.values))


def hourly_hub(**kw):
    """steel's furnace is the only contributor of steel; ore and idle stay."""
    return toy(conversion={"hub": {"ore": 1.0}, **kw})


def test_derived_hourly_with_an_hourly_contributor():
    p = toy(
        conversion={"eaf": {"steel": 1.0}, "hub": {"steel": 1.0, "ore": -1}}
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = lengths_of(setl(p, "conversion", eaf=4))
    assert np.isnan(out["steel"])


def test_derived_smallest_block_length():
    p = toy(
        conversion={
            "a": {"steel": 1.0},
            "b": {"steel": 1.0},
            "hub": {"ore": 1},
        }
    )
    out = lengths_of(setl(p, "conversion", a=4, b=8))
    assert out["steel"] == 4 and np.isnan(out["ore"]) and np.isnan(out["idle"])


def test_derived_hourly_without_counted_contributor():
    p = toy(
        conversion={"a": {"ore": 1.0}},
        net_import=np.array(True),
        flexdem=np.array(True),
        demand=xr.DataArray(
            np.ones(8), dims="hour", coords={"hour": np.arange(8)}
        ),
    )
    out = lengths_of(setl(p, "conversion", a=4))
    assert np.isnan(out["steel"]) and out["ore"] == 4


def test_forced_length_overrides_derivation():
    p = toy(conversion={"a": {"steel": 1.0}, "hub": {"steel": 1.0}})
    p = setl(p, "conversion", a=4)
    p = p.assign(resource_time_block_hrs=per(p, "resource", steel=8))
    assert lengths_of(p)["steel"] == 8
    p = toy(conversion={"a": {"steel": 1.0}})
    p = p.assign(resource_time_block_hrs=per(p, "resource", steel=4))
    assert lengths_of(setl(p, "conversion", a=4))["steel"] == 4


def test_storage_makes_a_balance_hourly():
    p = toy(conversion={"a": {"steel": 1.0}}, storage={"tank": {"steel": 1.0}})
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert np.isnan(lengths_of(setl(p, "conversion", a=4))["steel"])


@pytest.mark.parametrize("factor", [1.0, -0.5])
def test_process_factor_counts_with_any_mode(factor):
    p = toy(
        conversion={"a": {"steel": 1.0}}, process={"pr": {"steel": factor}}
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert np.isnan(lengths_of(setl(p, "conversion", a=4))["steel"])
    p = toy(conversion={"a": {"steel": 1.0}}, process={"pr": {"ore": 1.0}})
    assert lengths_of(setl(p, "conversion", a=4))["steel"] == 4


def test_transport_fuel_counts():
    # carries ore, burns steel as fuel
    p = toy(
        conversion={"a": {"steel": 1.0}}, transport={"truck": ("ore", "steel")}
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = lengths_of(setl(p, "conversion", a=4))
    assert np.isnan(out["steel"])
    p = toy(
        conversion={"a": {"steel": 1.0}}, transport={"truck": ("ore", None)}
    )
    assert lengths_of(setl(p, "conversion", a=4))["steel"] == 4


def test_transport_resource_counts():
    # carries steel (transport_resource == r), no fuel
    p = toy(
        conversion={"a": {"steel": 1.0}}, transport={"truck": ("steel", None)}
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = lengths_of(setl(p, "conversion", a=4))
    assert np.isnan(out["steel"])
    p = toy(
        conversion={"a": {"steel": 1.0}}, transport={"truck": ("ore", None)}
    )
    assert lengths_of(setl(p, "conversion", a=4))["steel"] == 4


def test_factor_tested_on_the_full_grid():
    p = toy(
        conversion={"a": {"steel": 1.0}, "pinned": {"steel": 1.0}},
        conversion_power_capacity_investment_max=xr.DataArray(
            [np.inf, 0.0],
            dims="conversion_tech",
            coords={"conversion_tech": ["a", "pinned"]},
        ),
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert np.isnan(lengths_of(setl(p, "conversion", a=4))["steel"])


def test_why_hourly_warning():
    p = toy(
        conversion={"furnace": {"steel": 1.0}},
        storage={"steel_tank": {"steel": 1.0}},
        net_import=np.array(True),
    )
    p = setl(p, "conversion", furnace=4)
    with pytest.warns(UserWarning) as rec:
        time_blocks(p)
    assert len(rec) == 1
    text = str(rec[0].message)
    assert "steel: hourly because of storage tech steel_tank" in text
    assert "net_import" not in text
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        time_blocks(
            p.assign(conversion_time_block_hrs=per(p, "conversion_tech"))
        )
        check_time_blocks(p)  # silenced


def test_time_blocks_accessor_order():
    p = toy(
        conversion={
            "zeta": {"steel": 1.0},
            "alpha": {"ore": 1.0},
            "mid": {"idle": 1.0},
        },
        resources=("steel", "ore", "idle", "aaa", "zzz"),
    )
    p = setl(p, "conversion", zeta=4, mid=4)
    tb = time_blocks(p)
    assert tb.techs("conversion", 4) == ["zeta", "mid"]
    assert tb.techs("conversion", None) == ["alpha"]
    assert tb.resources(4) == ["steel", "idle"]
    assert tb.resources(None) == ["ore", "aaa", "zzz"]
    p = p.assign(resource_time_block_hrs=per(p, "resource", ore=8, aaa=12))
    tb = time_blocks(setl(p, "conversion", zeta=4, mid=4, alpha=8))
    assert tb.balance_lengths == [4, 8, 12]
    assert tb.resources(None) == ["zzz"]


def test_r4_nesting_derived():
    p = toy(
        conversion={
            "a": {"steel": 1.0},
            "b": {"steel": 1.0},
            "hub": {"ore": 1},
        }
    )
    with pytest.raises(ValueError, match=r"steel.*6 h.*4 h"):
        time_blocks(setl(p, "conversion", a=6, b=4))


def test_r4_nesting_forced():
    base = toy(conversion={"a": {"steel": 1.0}, "hub": {"ore": 1}})

    def run(forced, tech):
        p = setl(base, "conversion", a=tech)
        return time_blocks(
            p.assign(resource_time_block_hrs=per(p, "resource", steel=forced))
        )

    with pytest.raises(ValueError, match="steel"):
        run(6, 4)
    run(8, 4)
    run(2, 4)


def test_r5_turpe_forced():
    p = toy(
        conversion={"a": {"electricity": 1.0}, "hub": {"steel": 1}},
        resources=("electricity", "steel", "idle"),
        turpe=np.array(True),
    )
    p = p.assign(resource_time_block_hrs=per(p, "resource", electricity=4))
    with pytest.raises(ValueError, match="electricity"):
        time_blocks(setl(p, "conversion", a=4))
    time_blocks(setl(p.assign(turpe=np.array(False)), "conversion", a=4))


def test_r5_turpe_derived():
    p = toy(
        conversion={"a": {"electricity": 1.0}, "hub": {"steel": 1}},
        resources=("electricity", "steel", "idle"),
        turpe=np.array(True),
    )
    with pytest.raises(ValueError, match="electricity"):
        time_blocks(setl(p, "conversion", a=4))


def _capped(L, cap=100.0, day=24):
    p = toy(
        conversion={"a": {"steel": 1.0}, "hub": {"ore": 1}},
        operation_day_duration=np.array(day),
    )
    p["conversion_max_daily_production"] = xr.DataArray(
        [cap, np.nan],
        dims="conversion_tech",
        coords={"conversion_tech": ["a", "hub"]},
    )
    p = setl(p, "conversion", a=L)
    return p, time_blocks(p)


@pytest.mark.parametrize("L,ok", [(36, False), (6, True), (48, True)])
def test_r6_daily_cap(L, ok):
    p, tb = _capped(L)
    if ok:
        check_block_daily_cap(p, tb)
    else:
        with pytest.raises(ValueError, match="a.*36"):
            check_block_daily_cap(p, tb)
        with pytest.raises(ValueError, match="36"):
            check_time_blocks(p)
    p, tb = _capped(L, cap=np.inf)
    check_block_daily_cap(p, tb)


def _flex(hrs, offset=0.0, demand=1.0):
    p = toy(
        conversion={"a": {"steel": 1.0}, "hub": {"ore": 1}},
        hours=16,
        flexdem=np.array(True),
        flexdem_window_offset_hrs=np.array(offset),
    )
    p["flexdem_demand"] = xr.DataArray(
        np.full((16, 3), demand),
        dims=("hour", "resource"),
        coords={"hour": p.hour, "resource": p.resource},
    )
    p["flexdem_conservation_hrs"] = xr.DataArray(
        [hrs] * 3, dims="resource", coords={"resource": p.resource}
    )
    p = setl(p, "conversion", a=4)
    return p, time_blocks(p)


def test_r9_flexdem_windows():
    p, tb = _flex(10)
    with pytest.raises(ValueError, match="steel.*10"):
        check_block_flexdem(p, tb)
    with pytest.raises(ValueError, match="steel.*10"):
        check_time_blocks(p)
    check_block_flexdem(*_flex(8))
    with pytest.raises(ValueError, match="steel.*offset|offset.*steel"):
        check_block_flexdem(*_flex(8, offset=2.0))
    check_block_flexdem(*_flex(10, offset=2.0, demand=0.0))


def test_r10_every_balance_coarse():
    p = toy(conversion={"a": {"steel": 1.0, "ore": 1.0}})
    p = p.assign(
        resource_time_block_hrs=xr.full_like(p.resource_time_block_hrs, 4.0)
    )
    with pytest.raises(ValueError, match="hourly"):
        time_blocks(p)
    p = p.assign(resource_time_block_hrs=per(p, "resource", steel=4, ore=4))
    time_blocks(p)


def test_refusals_read_only_lengths_in_use():
    p = toy(
        conversion={
            "a": {"steel": 1.0},
            "b": {"steel": 1.0},
            "hub": {"ore": 1},
        }
    )
    p = setl(p, "conversion", a=6, b=4)
    with pytest.raises(ValueError):
        time_blocks(p)
    off = p.assign(conversion=np.array(False))
    time_blocks(off)  # conversion off: R4 does not read its lengths
    p, tb = _capped(36)
    check_block_daily_cap(
        p.assign(conversion=np.array(False)),
        time_blocks(p.assign(conversion=np.array(False))),
    )
    q, tb = _flex(10)
    check_block_flexdem(q.assign(flexdem=np.array(False)), tb)
    # R9: the coarse length belongs to a module that is off
    off = q.assign(conversion=np.array(False))
    check_block_flexdem(off, time_blocks(off))
    # R2: a step crossing a boundary of a length of a module that is off
    r2 = toy(conversion={"a": {"steel": 1.0}, "hub": {"ore": 1}}, hours=2)
    r2["time_step_duration"][:] = [1.0, 2.0]
    r2 = setl(r2, "conversion", a=2)
    with pytest.raises(ValueError, match="crosses a block boundary"):
        time_blocks(r2)
    time_blocks(r2.assign(conversion=np.array(False)))


def test_t0_path_reads_no_input():
    p = toy(conversion={"hub": {"ore": 1}}, hours=2)
    tb = time_blocks(p)
    assert not tb.active
    # inputs the checks would read are inconsistent, yet nothing is raised
    check_block_daily_cap(p.assign(conversion_max_daily_production=1), tb)
    check_block_flexdem(p.assign(flexdem=np.array(True)), tb)


def test_r9_nan_or_absent_offset_reads_as_zero():
    p, tb = _flex(8, offset=np.nan)
    check_block_flexdem(p, tb)
    check_block_flexdem(p.drop_vars("flexdem_window_offset_hrs"), tb)
    # the offset is checked only for a coarse balance with an active flexdem
    p, tb = _flex(8, offset=2.0, demand=0.0)
    check_block_flexdem(p, tb)


def test_block_mean():
    p = toy(conversion={"a": {"steel": 1.0}, "hub": {"ore": 1}}, hours=2)
    p["time_step_duration"][:] = [1.0, 3.0]
    tb = time_blocks(setl(p, "conversion", a=4))
    x = xr.DataArray([1.0, 10.0], dims="hour", coords={"hour": [0, 1]})
    assert block_mean(x, tb, 4).item() == pytest.approx(7.75)
    # unequal blocks: steps 1,3 | 2,2 | 1 (last block shorter), L = 4
    q = toy(conversion={"a": {"steel": 1.0}, "hub": {"ore": 1}}, hours=5)
    q["time_step_duration"][:] = [1.0, 3.0, 2.0, 2.0, 1.0]
    tq = time_blocks(setl(q, "conversion", a=4))
    y = xr.DataArray(
        [1.0, 10.0, 2.0, 6.0, 5.0], dims="hour", coords={"hour": range(5)}
    )
    got = block_mean(y, tq, 4)
    # (1*1+3*10)/4, (2*2+2*6)/4, 5
    assert got.values == pytest.approx([7.75, 4.0, 5.0])
    c = xr.DataArray(3.0)
    assert block_mean(c, tb, 4) is c
    nan = xr.DataArray([1.0, np.nan], dims="hour", coords={"hour": [0, 1]})
    assert np.isnan(block_mean(nan, tb, 4).item())
    two = xr.DataArray(
        [[1.0, 10.0], [2.0, 20.0]],
        dims=("operating_condition", "hour"),
        coords={"operating_condition": ["a", "b"], "hour": [0, 1]},
    )
    assert block_mean(two, tb, 4).sel(
        operating_condition="b"
    ).item() == pytest.approx((2 + 60) / 4)


def test_long_block_and_inactive_tech_are_not_refused():
    p = toy(
        conversion={"a": {"steel": 1.0}, "idle_tech": {}, "hub": {"ore": 1}},
        hours=8,
    )
    tb = time_blocks(setl(p, "conversion", a=100, idle_tech=4))
    assert tb.duration[100].values.tolist() == [8]
    assert tb.techs("conversion", 4) == ["idle_tech"]
