"""Shared helpers of the time-blocks tests (identical in both engines)."""

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from pommes.model.build_model import build_model

try:  # sparse engine: solved variables sit on a unit index
    from pommes.model.sparsity import densify_any
except ImportError:  # dense engine

    def densify_any(obj):
        return obj


_PARAMETERS = (
    ("conversion_time_block_hrs", "conversion_tech"),
    ("combined_time_block_hrs", "combined_tech"),
    ("resource_time_block_hrs", "resource"),
)


def _per(p, dim, values):
    da = xr.DataArray(
        np.full(p[dim].size, np.nan), dims=dim, coords={dim: p[dim].values}
    )
    for label, value in (values or {}).items():
        da.loc[{dim: label}] = value
    return da


def with_blocks(
    p: xr.Dataset,
    conversion: dict[str, float] | None = None,
    combined: dict[str, float] | None = None,
    resource: dict[str, float] | None = None,
    rule: dict[str, str] | None = None,
) -> xr.Dataset:
    """
    ``p`` with the four block parameters set (NaN / ``"min"`` elsewhere).

    A parameter whose dimension is not in ``p`` is left out.
    """
    p = p.copy()
    given = (conversion, combined, resource)
    for (name, dim), values in zip(_PARAMETERS, given):
        if dim in p.coords:
            p[name] = _per(p, dim, values)
    if "conversion_tech" in p.coords:
        labels = p["conversion_tech"].values
        r = np.array(["min"] * len(labels), dtype=object)
        for label, value in (rule or {}).items():
            r[list(labels).index(label)] = value
        p["conversion_time_block_rule"] = xr.DataArray(
            r.astype(str), dims="conversion_tech", coords={"conversion_tech": labels}
        )
    return p


def lp_text(m) -> str:
    """The LP file of ``m``, read back as text."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.lp"
        m.to_file(path, io_api="lp")
        return path.read_text()


def assert_t0(p: xr.Dataset) -> None:
    """
    Without block inputs the model has no trace of them (T0 plumbing pin).

    Builds ``p`` without the four parameters and with them all NaN: same LP
    text, no ``_block_`` name, no ``time_block_`` dimension, hourly adequacy
    over the full resource coordinate. Both models take the inactive path:
    this guards the NaN parameters and the ``tb`` plumbing, not drift against
    0.6.1+kul (``compare_lp.py`` does).
    """
    names = [n for n, _ in _PARAMETERS] + ["conversion_time_block_rule"]
    bare = build_model(p.drop_vars(names, errors="ignore"))
    nan = build_model(with_blocks(p))
    assert lp_text(bare) == lp_text(nan)
    for name in [*nan.variables, *nan.constraints]:
        assert "_block_" not in name, name
    for name in nan.variables:
        assert not any(d.startswith("time_block_") for d in nan.variables[name].dims)
    for name in nan.constraints:
        assert not any(d.startswith("time_block_") for d in nan.constraints[name].dims)
    adequacy = nan.constraints["operation_adequacy_constraint"]
    assert adequacy.labels.indexes["resource"].equals(p.indexes["resource"])


def solved(m, name: str) -> xr.DataArray:
    """
    Solution of variable ``name`` on ``[area, tech, ...]`` (engine-agnostic).

    Sparse: densified; the technologies of another group (all NaN) dropped.
    """
    da = densify_any(m.solution[name])
    techs = [d for d in da.dims if d.endswith("_tech") and d != "link"]
    for tech in techs:
        da = da.dropna(tech, how="all")
    lead = [d for d in ("area", *techs) if d in da.dims]
    return da.transpose(*lead, ...)


def hand_toy(
    dt: list[float],
    steel_demand: list[float],
    a: float = 10.0,
    v: float = 1.0,
    s: float = 3.0,
) -> xr.Dataset:
    """
    The hand toy of section 16.1: one furnace makes steel, nothing else.

    One area, resources ``idle`` and ``steel``, one ``year_op``, one
    conversion tech ``furnace`` (steel factor 1, no input), fixed cost ``a``
    per MW, variable cost ``v`` per MWh, spillage cost ``s``, steel shedding
    capped at 0, no invest cost, no discounting: the objective is exactly
    ``a * cap + n * (v * production + s * spill)`` with ``n = toy_n(p)``.
    """
    from pommes.model.data_validation import ref_inputs

    n = len(dt)
    year = np.array([2030], dtype="int64")
    year_inv, year_dec = np.array([2020]), np.array([2040])
    ds = xr.Dataset(
        coords=dict(
            area=["a"],
            hour=np.arange(n, dtype="int64"),
            resource=["idle", "steel"],
            conversion_tech=["furnace"],
            year_dec=year_dec,
            year_inv=year_inv,
            year_op=year,
        )
    )
    ds["conversion"] = True
    ds["time_step_duration"] = ("hour", np.array(dt, dtype="float64"))
    ds["operation_year_duration"] = ("year_op", [8760.0])
    ds["operation_day_duration"] = np.array(24, dtype="int64")
    ds["discount_rate"] = ("year_op", [0.0])
    ds["discount_factor"] = ("year_op", [1.0])
    demand = np.zeros((1, n, 2, 1))
    demand[0, :, 1, 0] = steel_demand
    ds["demand"] = (("area", "hour", "resource", "year_op"), demand)
    ds["load_shedding_max_capacity"] = ("resource", [np.nan, 0.0])
    ds["load_shedding_cost"] = ("resource", [0.0, 0.0])
    ds["spillage_max_capacity"] = ("resource", [np.nan, np.nan])
    ds["spillage_cost"] = ("resource", [0.0, s])
    ds["conversion_factor"] = (("conversion_tech", "resource"), [[0.0, 1.0]])
    ds["conversion_variable_cost"] = ("conversion_tech", [v])
    ds["conversion_fixed_cost"] = ("conversion_tech", [a])
    ds["conversion_invest_cost"] = ("conversion_tech", [0.0])
    ds["conversion_emission_factor"] = ("conversion_tech", [0.0])
    ds["conversion_power_capacity_max"] = (("area", "conversion_tech"), [[np.nan]])
    ds["conversion_power_capacity_min"] = (("area", "conversion_tech"), [[0.0]])
    ds["conversion_life_span"] = ("conversion_tech", [20.0])
    ds["conversion_end_of_life"] = (("conversion_tech", "year_inv"), [[2040.0]])
    ds["conversion_annuity_cost"] = (
        ("conversion_tech", "year_dec", "year_inv"),
        [[[0.0]]],
    )
    ds["conversion_early_decommissioning"] = np.array(False)
    ds["annuity_full_life"] = np.array(False)
    for name, spec in ref_inputs.items():
        if name not in ds.variables:
            ds[name] = np.array(spec["default"], dtype=spec["type"])
    return ds


def toy_n(p: xr.Dataset) -> float:
    """``operation_year_normalization`` of ``p`` (year duration over horizon)."""
    return float(p.operation_year_duration.item() / p.time_step_duration.sum())


STEEL_DEMAND = [0, 2, 0, 2]


def hand_toy_a(**kw) -> xr.Dataset:
    """The hand toy of section 16.1 on hourly balances (``kw``: ``a``, ``v``, ``s``)."""
    return hand_toy([1.0] * 4, STEEL_DEMAND, **kw)


def hand_toy_b(**kw) -> xr.Dataset:
    """``hand_toy_a`` with the steel balance forced to 2-hour blocks (section 16.1, case B)."""
    return with_blocks(hand_toy_a(**kw), resource={"steel": 2})


def priced_spillage(p: xr.Dataset) -> xr.Dataset:
    """``p`` with a cost on every spillage and distinct variable costs: a unique optimum."""
    n, k = p.sizes["resource"], p.sizes["conversion_tech"]
    return p.assign(
        spillage_cost=("resource", 1.0 + np.arange(n)),
        conversion_variable_cost=p.conversion_variable_cost + 0.013 * np.arange(1, k + 1),
    )


def methane_hourly(parameters: xr.Dataset) -> xr.Dataset:
    """The ``parameters`` fixture without carbon, net import and turpe, spillage priced."""
    quiet = parameters.assign(
        **{k: np.array(False) for k in ("carbon", "net_import", "turpe")}
    )
    return priced_spillage(quiet)


def methane_forced(parameters: xr.Dataset) -> xr.Dataset:
    """``methane_hourly`` with the methane balance forced to 2 h (storage, transport, conversion)."""
    return with_blocks(methane_hourly(parameters), resource={"methane": 2})


def process_hourly(parameters_dispatch_invest: xr.Dataset, process: xr.Dataset) -> xr.Dataset:
    """The process fixture dataset, as ``test_process.py::parameters_process`` builds it."""
    p = xr.merge([parameters_dispatch_invest, process])
    p = p.sel(
        area=["area_1"],
        conversion_tech=["wind_onshore"],
        resource=["electricity", "heat", "methane"],
    ).copy(deep=True)
    p["load_shedding_cost"].loc[dict(resource="methane")] = 5
    p["demand"].loc[dict(resource="electricity")] = np.array([0, 0, 0, 0])
    p["conversion_power_capacity_investment_max"] = 10
    p["conversion_power_capacity_investment_min"] = 10
    p["conversion_annuity_cost"] = 0
    p["conversion_fixed_cost"] = 0
    return p


def process_forced(parameters_dispatch_invest: xr.Dataset, process: xr.Dataset) -> xr.Dataset:
    """``process_hourly`` with the methane balance forced to 2 h."""
    return with_blocks(
        process_hourly(parameters_dispatch_invest, process), resource={"methane": 2}
    )
