"""Shared helpers of the time-blocks tests (identical in both engines)."""

import inspect
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import with_conditions
from pommes.utils import TimeBlocks, block_dim, block_mean, time_blocks

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

# every block parameter: dropping them all gives the hourly model
BLOCK_PARAMETERS = [n for n, _ in _PARAMETERS] + ["conversion_time_block_rule"]


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
    bare = build_model(p.drop_vars(BLOCK_PARAMETERS, errors="ignore"))
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


def _net_import_dataset(dt, resources, demand, price, factor=None, total_factor=None) -> xr.Dataset:
    """
    net_import only, no conversion technology: one area, one ``year_op``, ``idle`` first.

    ``demand``, ``price`` (import), ``factor`` and ``total_factor`` (the total emission factor, ``factor`` when
    not given) map a resource to its per-step list (a number is the same at every step); every other value is 0, and ``idle``
    imports at price 1 (it holds nothing, R10). Export price 0, shedding and spillage capped at 0.
    """
    from pommes.model.data_validation import ref_inputs

    n = len(dt)
    names = ["idle", *resources]
    ds = xr.Dataset(
        coords=dict(
            area=["a"],
            hour=np.arange(n, dtype="int64"),
            resource=names,
            year_op=np.array([2030], dtype="int64"),
        )
    )
    ds["time_step_duration"] = ("hour", np.array(dt, dtype="float64"))
    ds["operation_year_duration"] = ("year_op", [8760.0])
    ds["operation_day_duration"] = np.array(24, dtype="int64")
    ds["discount_rate"] = ("year_op", [0.0])
    ds["discount_factor"] = ("year_op", [1.0])
    per_step = ("area", "hour", "resource", "year_op")
    for name, values in (
        ("demand", demand),
        ("net_import_import_price", {"idle": 1.0, **price}),
        ("net_import_emission_factor", factor or {}),
        ("net_import_total_emission_factor", total_factor or factor or {}),
    ):
        x = np.zeros((1, n, len(names), 1))
        for resource, value in values.items():
            x[0, :, names.index(resource), 0] = value
        ds[name] = (per_step, x)
    ds["net_import"] = np.array(True)
    for name in ("load_shedding_max_capacity", "spillage_max_capacity"):
        ds[name] = ("resource", np.zeros(len(names)))
    for name in ("load_shedding_cost", "spillage_cost"):
        ds[name] = ("resource", np.zeros(len(names)))
    for name, spec in ref_inputs.items():
        if name not in ds.variables:
            ds[name] = np.array(spec["default"], dtype=spec["type"])
    return ds


def net_import_toy(
    dt: list[float],
    demand: list[float],
    price: list[float],
    factor: float | list[float] = 0.0,
    total_factor: float | list[float] | None = None,
) -> xr.Dataset:
    """
    The net_import toy of section 16.2: steel comes from the rest of the world only.

    One area, one ``year_op``, resources ``idle`` and ``steel``, **no conversion
    technology**. Steel demand ``demand`` (energy per step) over the steps ``dt``,
    import price ``price`` (per step, MWh), export price 0, emission factor
    ``factor`` (a number or one per step) and total emission factor ``total_factor`` (default: ``factor``; give
    another one to tell the two apart); shedding and spillage capped at 0.
    The objective is exactly ``n * sum(price_h * import_h)`` with ``n = toy_n(p)``.
    """
    return _net_import_dataset(
        dt,
        ["steel"],
        {"steel": demand},
        {"steel": price},
        {"steel": factor},
        None if total_factor is None else {"steel": total_factor},
    )


def net_import_pairs() -> xr.Dataset:
    """
    net_import on four resources, two hourly and two forced to 4 h blocks (H1, section 16.2).

    Steps (1, 3) h; resources ``idle`` and ``h1`` stay hourly, ``c1`` and ``c2`` are forced
    coarse (``resource_time_block_hrs`` 4), so the two subsets have the same size and a costs
    partner restricted in another order would be paired by position. Distinct demands and
    prices: ``h1`` (4, 6) for (2, 1) pays 14, ``c1`` (1, 10) for (2, 0) pays 15.5 (block mean
    price 7.75), ``c2`` (2, 6) for (0, 3) pays 15 (block mean price 5), each times ``n``.
    """
    p = _net_import_dataset(
        [1.0, 3.0],
        ["h1", "c1", "c2"],
        {"h1": [2, 1], "c1": [2, 0], "c2": [0, 3]},
        {"h1": [4, 6], "c1": [1, 10], "c2": [2, 6]},
        {"h1": [0.1, 0.2], "c1": [0.3, 0.4], "c2": [0.5, 0.6]},
    )
    return with_blocks(p, resource={"c1": 4, "c2": 4})


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


def hand_toy_c(**kw) -> xr.Dataset:
    """``hand_toy_a`` with the furnace on 2 h blocks, steel derived at 2 h (section 16.1, case C)."""
    return with_blocks(hand_toy_a(**kw), conversion={"furnace": 2})


def hand_toy_d(**kw) -> xr.Dataset:
    """``hand_toy_c`` with the steel balance forced to the step (section 16.1, case D)."""
    return with_blocks(hand_toy_a(**kw), conversion={"furnace": 2}, resource={"steel": 1})


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


HEAT_RATE = [2.0, 3.5, 1.0, 4.0, 2.5, 2.0, 3.0, 1.5, 4.5, 2.0]  # MW of heat per step


def combined_toy(dt: list[float], rate: list[float] | None = None) -> xr.Dataset:
    """
    Hourly combined-technology toy (two areas, one year, no carbon, no flexdem).

    Resources ``idle``, ``gas`` and ``heat``. Conversion: ``well`` (gas) and
    ``heater`` (expensive heat, the hourly contributor that keeps the heat
    balance hourly). Combined, two modes each, finite ramps and mode-pair
    ramps: ``electric_boiler`` (hourly), ``boiler_block`` (a copy with other
    costs, the one that gets block lengths) and ``steady_kiln`` (ramps 0:
    constant operation). Heat demand is ``rate`` (cycled) times the step.
    Costs are distinct and strictly positive and every spillage is priced, so
    that the optimum is unique.
    """
    from pommes.model.data_validation import ref_inputs

    dt = np.array(dt, dtype="float64")
    n = len(dt)
    rate = np.resize(np.array(HEAT_RATE if rate is None else rate), n)
    techs = ["electric_boiler", "boiler_block", "steady_kiln"]
    ds = xr.Dataset(
        coords=dict(
            area=["a", "b"],
            hour=np.arange(n, dtype="int64"),
            resource=["idle", "gas", "heat"],
            conversion_tech=["well", "heater"],
            combined_tech=techs,
            mode=["m1", "m2"],
            year_dec=np.array([2040]),
            year_inv=np.array([2020]),
            year_op=np.array([2030], dtype="int64"),
        )
    )
    for flag in ("conversion", "combined"):
        ds[flag] = True
    ds["time_step_duration"] = ("hour", dt)
    ds["operation_year_duration"] = ("year_op", [8760.0])
    ds["operation_day_duration"] = np.array(24, dtype="int64")
    ds["discount_rate"] = ("year_op", [0.0])
    ds["discount_factor"] = ("year_op", [1.0])
    demand = np.zeros((2, n, 3, 1))
    demand[0, :, 2, 0] = rate * dt
    demand[1, :, 2, 0] = 0.8 * rate * dt
    ds["demand"] = (("area", "hour", "resource", "year_op"), demand)
    ds["load_shedding_max_capacity"] = ("resource", [0.0, 0.0, 0.0])
    ds["load_shedding_cost"] = ("resource", [0.0, 0.0, 0.0])
    ds["spillage_max_capacity"] = ("resource", [np.nan, np.nan, np.nan])
    ds["spillage_cost"] = ("resource", [1.0, 2.0, 3.0])
    # conversion: well makes gas, heater makes heat
    ds["conversion_factor"] = (
        ("conversion_tech", "resource"),
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    ds["conversion_variable_cost"] = ("conversion_tech", [0.5, 40.0])
    ds["conversion_fixed_cost"] = ("conversion_tech", [0.7, 1.1])
    ds["conversion_invest_cost"] = ("conversion_tech", [0.0, 0.0])
    ds["conversion_emission_factor"] = ("conversion_tech", [0.0, 0.0])
    ds["conversion_power_capacity_max"] = (
        ("area", "conversion_tech"),
        [[np.nan, np.nan]] * 2,
    )
    ds["conversion_power_capacity_min"] = (
        ("area", "conversion_tech"),
        [[0.0, 0.0]] * 2,
    )
    ds["conversion_life_span"] = ("conversion_tech", [20.0, 20.0])
    ds["conversion_end_of_life"] = (
        ("conversion_tech", "year_inv"),
        [[2040.0], [2040.0]],
    )
    ds["conversion_annuity_cost"] = (
        ("conversion_tech", "year_dec", "year_inv"),
        np.zeros((2, 1, 1)),
    )
    ds["conversion_early_decommissioning"] = np.array(False)
    # combined: (heat, gas) per mode; m2 burns less gas per MW of heat
    factor = np.zeros((3, 2, 3))
    for i, (g1, h1, g2, h2) in enumerate(
        [(-1.25, 1.0, -0.6, 0.8), (-1.25, 1.0, -0.6, 0.8), (-1.5, 1.2, -0.7, 0.7)]
    ):
        factor[i, 0, 1:] = [g1, h1]
        factor[i, 1, 1:] = [g2, h2]
    ds["combined_factor"] = (("combined_tech", "mode", "resource"), factor)
    ds["combined_variable_cost"] = ("combined_tech", [2.1, 0.9, 1.3])
    ds["combined_fixed_cost"] = ("combined_tech", [3.1, 3.7, 6.3])
    ds["combined_invest_cost"] = ("combined_tech", [0.0, 0.0, 0.0])
    ds["combined_emission_factor"] = (("combined_tech", "mode"), np.zeros((3, 2)))
    ds["combined_power_capacity_max"] = (
        ("area", "combined_tech"),
        # electric_boiler pinned in a, steady_kiln in b: the unit index is not
        # rectangular and each group's units are interleaved in it
        [[0.0, 2.0, 1.5], [3.0, 2.0, 0.0]],
    )
    ds["combined_ramp_up"] = ("combined_tech", [1.0, 0.6, 0.0])
    ds["combined_ramp_down"] = ("combined_tech", [1.0, 0.6, 0.0])
    ds["combined_mode_ramp_up"] = ("combined_tech", [0.7, 0.4, np.nan])
    ds["combined_mode_ramp_down"] = ("combined_tech", [0.7, 0.4, np.nan])
    ds["combined_life_span"] = ("combined_tech", [20.0] * 3)
    ds["combined_end_of_life"] = (("combined_tech", "year_inv"), [[2040.0]] * 3)
    ds["combined_annuity_cost"] = (
        ("combined_tech", "year_dec", "year_inv"),
        np.zeros((3, 1, 1)),
    )
    ds["combined_early_decommissioning"] = np.array(False)
    ds["annuity_full_life"] = np.array(False)
    for name, spec in ref_inputs.items():
        if name not in ds.variables:
            ds[name] = np.array(spec["default"], dtype=spec["type"])
    return ds


def combined_t1_dataset() -> xr.Dataset:
    """``combined_toy`` on 10 hours with every combined length 1 (T1: block equals step)."""
    p = combined_toy([1.0] * 10)
    return with_blocks(p, combined={t: 1 for t in p.combined_tech.values})


def combined_t2_dataset() -> xr.Dataset:
    """``combined_toy`` on 10 hours, ``boiler_block`` on blocks of 2 h (the flat-twin case)."""
    return with_blocks(combined_toy([1.0] * 10), combined={"boiler_block": 2})


def combined_all_block_dataset() -> xr.Dataset:
    """``combined_toy`` on 10 h with every combined technology on 2 h blocks (no hourly combined variable), no carbon."""
    p = combined_toy([1.0] * 10)
    return with_blocks(p, combined={t: 2 for t in p.combined_tech.values})


def pinned_module_dataset(module: str) -> xr.Dataset:
    """
    ``combined_toy`` whose ``module`` is pinned to zero capacity (section 16.9).

    ``"combined"``: ``combined_power_capacity_max = 0`` everywhere, the heater
    serves the heat. ``"conversion"``: ``conversion_power_capacity_max = 0``
    everywhere, gas may be shed and the combined techs are free. No block length is set: tests add theirs.
    """
    p = combined_toy([1.0] * 4)
    if module == "combined":
        return p.assign(combined_power_capacity_max=p.combined_power_capacity_max * 0.0)
    assert module == "conversion"
    shed = p.load_shedding_max_capacity.copy()
    cost = p.load_shedding_cost.copy()
    shed.loc[dict(resource="gas")], cost.loc[dict(resource="gas")] = np.nan, 30.0
    return p.assign(
        conversion_power_capacity_max=p.conversion_power_capacity_max.fillna(0.0) * 0.0,
        combined_power_capacity_max=p.combined_power_capacity_max * np.nan,  # the boilers serve the heat
        load_shedding_max_capacity=shed,
        load_shedding_cost=cost,
    )


def with_carbon(p: xr.Dataset, conversion=None, combined=None, tax=None, goal=None) -> xr.Dataset:
    """
    ``p`` with carbon on and an emission factor on the given technologies.

    ``conversion`` and ``combined`` map a technology to its factor (``combined``: a number or one value per mode),
    every other factor keeps its value; ``tax`` and ``goal`` (per tCO2, tCO2 over the horizon's year) are scalars
    left NaN when None.
    """
    out = {"carbon": np.array(True)}
    for module, values in (("conversion", conversion), ("combined", combined)):
        name = f"{module}_emission_factor"
        ef = p[name].copy()
        for tech, value in (values or {}).items():
            ef.loc[{f"{module}_tech": tech}] = value
        out[name] = ef
    for name, value in (("carbon_tax", tax), ("carbon_goal", goal)):
        if value is not None:
            out[name] = np.array(value, dtype="float64")
    return p.assign(out)


CARBON_KILNS = {"well": 0.1, "backup": 0.9, "kiln_a": 0.3, "kiln_b": 0.5, "kiln_c": 0.2, "kiln_d": 0.4}
CARBON_BOILERS = {"electric_boiler": [0.1, 0.1], "boiler_block": [0.5, 0.2], "steady_kiln": [0.3, 0.3]}
CARBON_HEATER = {"well": 0.3, "heater": 0.8}


CARBON_TAX_15 = 15.0


def taxed(p: xr.Dataset) -> xr.Dataset:
    """``p`` with a carbon tax of ``CARBON_TAX_15`` per tCO2."""
    return p.assign(carbon_tax=np.array(CARBON_TAX_15))


def conversion_carbon_dataset() -> xr.Dataset:
    """``conversion_t2_dataset`` (kilns on 4 h blocks) with carbon on and an emission factor on every technology; no tax, no goal."""
    return with_carbon(conversion_t2_dataset(), conversion=CARBON_KILNS)


def combined_carbon_dataset() -> xr.Dataset:
    """``combined_t2_dataset`` (``boiler_block`` on 2 h blocks) with carbon on and an emission factor on every technology; no tax, no goal."""
    return with_carbon(combined_t2_dataset(), conversion=CARBON_HEATER, combined=CARBON_BOILERS)


def combined_all_block_carbon_dataset() -> xr.Dataset:
    """``combined_toy`` on 10 h with every combined technology on 2 h blocks (no hourly combined variable), carbon on, a tax of 15."""
    p = with_carbon(combined_toy([1.0] * 10), conversion=CARBON_HEATER, combined=CARBON_BOILERS, tax=15.0)
    return with_blocks(p, combined={t: 2 for t in CARBON_BOILERS})


def solve_model(m):
    """Solve ``m`` with HiGHS and assert it is optimal; returns ``m``."""
    assert m.solve(solver_name="highs")[:2] == ("ok", "optimal")
    return m


def to_hours(da: xr.DataArray, tb: TimeBlocks) -> xr.DataArray:
    """A block array on the hours: each block value repeated over its steps."""
    dim = next(d for d in da.dims if d.startswith("time_block_"))
    L = int(dim.removeprefix("time_block_").removesuffix("h"))
    b = tb.block_of_hour[L]
    return (
        da.isel({dim: b.values})
        .rename({dim: "hour"})
        .assign_coords(hour=b["hour"].values)
    )


def flat_twin(p: xr.Dataset, conversion=None, combined=None):
    """
    The flat twin (section 15.1): the hourly model of ``p`` without block
    inputs, plus ``power(h) = power(h-1)`` for every step that is not the
    first of its block, per technology (per unit and mode for combined) of
    ``conversion`` / ``combined`` (``{tech: length}``). Not solved.
    """
    clean = p.drop_vars(BLOCK_PARAMETERS, errors="ignore")
    m = build_model(clean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = time_blocks(with_blocks(clean, conversion=conversion, combined=combined))
    for module, lengths in (("conversion", conversion), ("combined", combined)):
        var = m.variables[f"operation_{module}_power"] if lengths else None
        for tech, L in (lengths or {}).items():
            dim = next(d for d in var.dims if d.endswith(("_tech", "_tech_unit")))
            index = var.indexes[dim]
            labels = index.get_level_values(-1) if isinstance(index, pd.MultiIndex) else index
            v = var.isel({dim: np.flatnonzero(labels == tech)})
            b = tb.block_of_hour[L].values
            same = xr.DataArray(
                np.r_[False, b[1:] == b[:-1]],
                dims="hour",
                coords={"hour": tb.block_of_hour[L]["hour"].values},
            )
            m.add_constraints(
                v - v.shift(hour=1) == 0, mask=same, name=f"twin_{module}_{tech}"
            )
    return m


def assert_same(a: xr.DataArray, b: xr.DataArray, name: str = "") -> None:
    """``a`` and ``b`` agree to 1e-6 on the union of their labels (NaN read as 0, dimensions in any order)."""
    a, b = xr.align(a, b, join="outer")
    np.testing.assert_allclose(a.fillna(0).values, b.transpose(*a.dims).fillna(0).values, atol=1e-6, err_msg=name)


def assert_twin_equal(m_block, m_twin, tb: TimeBlocks, powers: list[str], skip: tuple = ()) -> None:
    """
    A solved block model equals its solved flat twin (section 15.1).

    Objective to 1e-7 (relative); ``planning_*`` and capacity variables to
    1e-6; each block power of ``powers`` repeated over the steps of its
    blocks equals the twin's hourly power of the same technologies to 1e-6. ``skip`` names variables left out (a
    capacity split between vintages that the optimum leaves open).
    """
    assert m_block.objective.value == pytest.approx(m_twin.objective.value, rel=1e-7)

    for name in m_block.variables:
        if name not in skip and (name.startswith("planning_") or name.endswith("_capacity")):
            assert_same(solved(m_block, name), solved(m_twin, name), name)
    for name in powers:
        twin_name = re.sub(r"_block_\d+h_", "_", name)
        block = to_hours(solved(m_block, name), tb)
        tech = next(d for d in block.dims if d.endswith("_tech"))
        hourly = solved(m_twin, twin_name).sel({tech: block.indexes[tech]})
        assert_same(block, hourly, name)


def units_of(obj) -> list[tuple]:
    """``(area, tech)`` pairs along the technology or unit dimension of a variable or constraint."""
    dims = obj.labels.dims if hasattr(obj, "labels") else obj.dims
    dim = next(d for d in dims if d.endswith(("_tech", "_tech_unit")))
    if dim.endswith("_unit"):  # sparse: a MultiIndex of (area, tech)
        return [tuple(u) for u in obj.indexes[dim]]
    return [(a, t) for a in obj.indexes["area"] for t in obj.indexes[dim]]


def binds(m, name, equality=False):
    """Some live row of ``name`` has a non-zero dual or, for an inequality, a zero slack."""
    con = m.constraints[name]
    live = (con.vars != -1).any("_term")  # a row on a flexible variable (a masked variable leaves 0 <= 0)
    dual = con.dual.where(live)
    hit = abs(dual) > 1e-9
    if not equality:  # the slack of an equality is always zero: only its dual says it matters
        hit = hit | (abs((con.rhs - con.lhs.solution).where(live)) < 1e-9)
    return bool(hit.any())


def variables_in(m, con) -> set[str]:
    """Names of the variables with a live term (a label and a non-zero coefficient) in the left-hand side of ``con``."""
    labels, coeffs = con.lhs.vars.values, con.lhs.coeffs.values
    used = set(np.unique(labels[(labels != -1) & np.isfinite(coeffs) & (coeffs != 0)]))
    return {
        name
        for name in m.variables
        if used & set(np.unique(m.variables[name].labels.values))
    }


def combined_derived_coarse_dataset() -> xr.Dataset:
    """
    ``combined_toy`` whose heat has only block contributors, so it is coarse (2 h).

    Every combined tech is on 2 h blocks, the heater makes no heat, the heat
    demand is constant inside each block and heat may be shed (cap 0.6, cost
    0.5, cheaper than any plant) or spilled (priced).
    """
    p = combined_toy([1.0] * 10, rate=[2, 2, 3, 3, 1, 1, 4, 4, 2, 2])
    not_heat = xr.where(p.resource == "heat", 0.0, 1.0)
    return with_blocks(
        p.assign(
            conversion_factor=p.conversion_factor * not_heat,
            load_shedding_max_capacity=p.load_shedding_max_capacity.where(p.resource != "heat", 0.6),
            load_shedding_cost=p.load_shedding_cost.where(p.resource != "heat", 0.5),
        ),
        combined={t: 2 for t in p.combined_tech.values},
    )


def combined_unitless_dataset() -> xr.Dataset:
    """
    ``combined_toy`` with ``boiler_block`` pinned to zero capacity in every area.

    Without block lengths. In the sparse engine the technology has no unit.
    """
    p = combined_toy([1.0] * 10)
    return p.assign(
        combined_power_capacity_max=p.combined_power_capacity_max.where(p.combined_tech != "boiler_block", 0.0)
    )


KILNS = ["kiln_a", "kiln_b", "kiln_c", "kiln_d"]
STEP_10H = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0]  # 10 h: with L = 4 the last block is short


def conversion_toy(dt: list[float], rate: list[float] | None = None) -> xr.Dataset:
    """
    Hourly conversion toy (two areas, one year, no carbon, no flexdem, no net import).

    Resources ``idle``, ``gas`` and ``steel``. Technologies: ``well`` (gas) and
    ``backup`` (expensive steel, the hourly contributor that keeps steel
    hourly), then four kilns (steel from gas): ``kiln_a``, ``kiln_b``,
    ``kiln_c``, ``kiln_d``, whose availability, must-run, ramps and caps the
    builders below set. Every availability is NaN, every must-run NaN, every ramp
    NaN. Steel demand is ``rate`` (cycled) times the step. Costs are distinct
    and strictly positive and every spillage is priced, so that the optimum is
    unique. Capacity maxima pin ``kiln_c`` in area ``a`` (no unit there in the
    sparse engine) and bound ``kiln_a`` and ``kiln_b`` in one area each, so that the
    unit index is not rectangular.
    """
    from pommes.model.data_validation import ref_inputs

    dt = np.array(dt, dtype="float64")
    n = len(dt)
    rate = np.resize(np.array(HEAT_RATE if rate is None else rate), n)
    techs = ["well", "backup", *KILNS]
    k = len(techs)
    ds = xr.Dataset(
        coords=dict(
            area=["a", "b"],
            hour=np.arange(n, dtype="int64"),
            resource=["idle", "gas", "steel"],
            conversion_tech=techs,
            year_dec=np.array([2040]),
            year_inv=np.array([2020]),
            year_op=np.array([2030], dtype="int64"),
        )
    )
    ds["conversion"] = True
    ds["time_step_duration"] = ("hour", dt)
    ds["operation_year_duration"] = ("year_op", [8760.0])
    ds["operation_day_duration"] = np.array(24, dtype="int64")
    ds["discount_rate"] = ("year_op", [0.0])
    ds["discount_factor"] = ("year_op", [1.0])
    demand = np.zeros((2, n, 3, 1))
    demand[0, :, 2, 0] = rate * dt
    demand[1, :, 2, 0] = 0.8 * rate * dt
    ds["demand"] = (("area", "hour", "resource", "year_op"), demand)
    ds["load_shedding_max_capacity"] = ("resource", [0.0, 0.0, 0.0])
    ds["load_shedding_cost"] = ("resource", [0.0, 0.0, 0.0])
    ds["spillage_max_capacity"] = ("resource", [np.nan, np.nan, np.nan])
    ds["spillage_cost"] = ("resource", [1.0, 2.0, 3.0])
    # (gas, steel) per MWh of power: the kilns burn gas, the backup burns nothing
    factor = np.zeros((k, 3))
    factor[0, 1] = 1.0
    factor[1, 2] = 1.0
    for i, burn in enumerate([1.25, 1.5, 1.4, 1.6]):
        factor[2 + i, 1:] = [-burn, 1.0]
    ds["conversion_factor"] = (("conversion_tech", "resource"), factor)
    ds["conversion_variable_cost"] = (
        "conversion_tech",
        [0.5, 40.0, 2.1, 0.9, 1.3, 1.7],
    )
    ds["conversion_fixed_cost"] = (
        "conversion_tech",
        [0.7, 1.1, 3.1, 3.7, 4.3, 2.9],
    )
    ds["conversion_invest_cost"] = ("conversion_tech", np.zeros(k))
    ds["conversion_emission_factor"] = ("conversion_tech", np.zeros(k))
    cap_max = np.full((2, k), np.nan)
    cap_max[0, 2 + 2] = 0.0  # kiln_c pinned in area a
    cap_max[1, 2 + 0] = 2.0  # kiln_a bounded in area b
    cap_max[0, 2 + 1] = 3.0  # kiln_b bounded in area a
    ds["conversion_power_capacity_max"] = (("area", "conversion_tech"), cap_max)
    ds["conversion_power_capacity_min"] = (("area", "conversion_tech"), np.zeros((2, k)))
    ds["conversion_life_span"] = ("conversion_tech", np.full(k, 20.0))
    ds["conversion_end_of_life"] = (
        ("conversion_tech", "year_inv"),
        np.full((k, 1), 2040.0),
    )
    ds["conversion_annuity_cost"] = (
        ("conversion_tech", "year_dec", "year_inv"),
        np.zeros((k, 1, 1)),
    )
    ds["conversion_early_decommissioning"] = np.array(False)
    ds["annuity_full_life"] = np.array(False)
    for name, spec in ref_inputs.items():
        if name not in ds.variables:
            ds[name] = np.array(spec["default"], dtype=spec["type"])
    return ds


def per_tech(p: xr.Dataset, name: str, values: dict, dim="hour"):
    """``p`` with ``name`` on ``[conversion_tech, hour]`` (NaN elsewhere) from ``{tech: per-hour values}``."""
    da = xr.DataArray(
        np.full((p.sizes["conversion_tech"], p.sizes[dim]), np.nan),
        dims=("conversion_tech", dim),
        coords={"conversion_tech": p.conversion_tech.values, dim: p[dim].values},
    )
    for tech, v in values.items():
        da.loc[{"conversion_tech": tech}] = v
    return p.assign({name: da})


def with_values(p: xr.Dataset, name: str, values: dict) -> xr.Dataset:
    """``p`` with ``name`` on ``[conversion_tech]`` (NaN elsewhere) from ``{tech: value}``."""
    da = xr.DataArray(
        np.full(p.sizes["conversion_tech"], np.nan),
        dims="conversion_tech",
        coords={"conversion_tech": p.conversion_tech.values},
    )
    for tech, v in values.items():
        da.loc[{"conversion_tech": tech}] = v
    return p.assign({name: da})


def conversion_t1_dataset() -> xr.Dataset:
    """
    ``conversion_toy`` with steps of 3 h and every conversion length 3 (T1: block equals step).

    ``kiln_b`` is must-run 1 on a finite availability. Every tech is on blocks, so
    the gas and steel balances are derived at 3 h (one block per step); the idle
    resource stays hourly.
    """
    p = conversion_toy([3.0] * 6)
    p = per_tech(p, "conversion_availability", {"kiln_a": [1, 0.6, 0.9, 0.8, 0.5, 1], "kiln_b": [0.9, 0.7, 0.8, 0.9, 0.7, 0.8]})
    p = with_values(p, "conversion_must_run", {"kiln_b": 1.0, "kiln_c": 0.5})
    return with_blocks(p, conversion={t: 3 for t in p.conversion_tech.values})


def _t2_base(dt: list[float]) -> xr.Dataset:
    """Kilns with availability varying inside the blocks, must-run on constant-per-block ones, ramps, caps."""
    p = conversion_toy(dt)
    n = len(dt)
    inside = [1, 0.6, 0.9, 0.8, 0.5, 1, 0.7, 0.9][:n]  # kiln_a: varies inside the blocks of 4 h
    const = [0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.8, 0.8][:n]  # kiln_b, kiln_c: constant per block
    p = per_tech(p, "conversion_availability", {"kiln_a": inside, "kiln_b": const, "kiln_c": const})
    p = with_values(p, "conversion_must_run", {"kiln_b": 1.0, "kiln_c": 0.5})
    p = with_values(p, "conversion_ramp_up", {"kiln_a": 0.8, "kiln_d": 0.0})
    p = with_values(p, "conversion_ramp_down", {"kiln_a": 0.8, "kiln_d": 0.0})
    p = with_values(p, "conversion_max_yearly_production", {"kiln_a": 3000.0})
    p = with_values(p, "conversion_max_capacity_factor", {"kiln_d": 0.5})
    return p


def conversion_t2_dataset() -> xr.Dataset:
    """
    ``conversion_toy`` on 10 h (steps 1, 1, 2, ...), the four kilns on blocks of 4 h (short last block).

    Availability varies inside the blocks of ``kiln_a`` (rule min); ``kiln_b``
    (must-run 1) and ``kiln_c`` (must-run 0.5) have an availability constant inside
    each block; ramps of 0.8 on ``kiln_a``, none (constant operation) on ``kiln_d``;
    a yearly maximum on ``kiln_a`` and a capacity-factor maximum on ``kiln_d``.
    """
    return with_blocks(_t2_base(STEP_10H), conversion={t: 4 for t in KILNS})


def conversion_t2b_dataset() -> xr.Dataset:
    """``conversion_t2_dataset`` with the mean rule on ``kiln_a`` (the availability that varies)."""
    return with_blocks(
        _t2_base(STEP_10H), conversion={t: 4 for t in KILNS}, rule={"kiln_a": "mean"}
    )


def conversion_all_block_dataset() -> xr.Dataset:
    """
    ``conversion_t2_dataset`` with every technology on blocks of 4 h (the typical industrial case, M12).

    There is no hourly conversion variable; gas and steel are derived at 4 h, ``idle`` stays hourly. The steel
    demand is constant inside each block (the blocks hold the steps (0, 1, 2), (3, 4, 5) and (6, 7)), so that the
    hourly balance of the flat twin has the same optimum as the block rows.
    """
    p = _t2_base(STEP_10H)
    energy = np.repeat([2.0, 3.0, 4.0], [3, 3, 2]) * p.time_step_duration.values
    demand = np.zeros(p.demand.shape)
    steel = list(p.resource.values).index("steel")
    demand[0, :, steel, 0], demand[1, :, steel, 0] = energy, 0.8 * energy
    p = p.assign(demand=(p.demand.dims, demand))
    return with_blocks(p, conversion={t: 4 for t in p.conversion_tech.values})


def conversion_two_lengths_dataset() -> xr.Dataset:
    """``conversion_t2_dataset`` with ``kiln_a`` and ``kiln_b`` on blocks of 4 h, ``kiln_c`` and ``kiln_d`` on 2 h."""
    return with_blocks(_t2_base(STEP_10H), conversion={"kiln_a": 4, "kiln_b": 4, "kiln_c": 2, "kiln_d": 2})


def must_run_relaxation_dataset() -> xr.Dataset:
    """
    A must-run 1 furnace on an availability that varies inside its 2 h blocks (D9').

    Steel demand is flat, ``furnace`` is cheap and must-run 1 with availability
    (1, 0.5) in each block, ``backup`` is dear and hourly. The flat twin pins the
    furnace capacity to 0 (two different availabilities inside a block with a flat
    power); the block row ``P = cap * min`` is feasible and profitable.
    """
    p = hand_toy([1.0] * 4, [2.0] * 4)
    p = p.isel(conversion_tech=[0, 0]).assign_coords(conversion_tech=["furnace", "backup"])
    p["conversion_variable_cost"] = ("conversion_tech", [1.0, 30.0])
    p["conversion_fixed_cost"] = ("conversion_tech", [2.0, 2.0])
    p["conversion_factor"] = (("conversion_tech", "resource"), [[0.0, 1.0], [0.0, 1.0]])
    p["conversion_power_capacity_max"] = (("area", "conversion_tech"), [[np.nan, np.nan]])
    p["conversion_power_capacity_min"] = (("area", "conversion_tech"), [[0.0, 0.0]])
    p["conversion_must_run"] = ("conversion_tech", [1.0, np.nan])
    p = per_tech(p, "conversion_availability", {"furnace": [1.0, 0.5, 1.0, 0.5]})
    return with_blocks(p, conversion={"furnace": 2})


# --- the uniform coarse step (T3(c), section 15.2), Task 7 ---

# conversion and combined ramps and the daily cap are NaN on the datasets below: block ramps divide by the
# first step's duration and the block daily cap counts days, both on purpose (they differ from today's code)
_NO_RAMPS = (
    "conversion_ramp_up",
    "conversion_ramp_down",
    "combined_ramp_up",
    "combined_ramp_down",
    "combined_mode_ramp_up",
    "combined_mode_ramp_down",
    "conversion_max_daily_production",
)
_STEP_SUMS = ("demand", "flexdem_demand")  # energies per step: summed per block
_STEP_MEANS = (  # per-hour parameters: Δ-weighted mean per block, NaN read as 1 first (section 6.1)
    "conversion_availability",
    "net_import_import_price",
    "net_import_export_price",
    "net_import_emission_factor",
    "net_import_total_emission_factor",
)


def _check_no_ramps(p: xr.Dataset) -> None:
    for name in _NO_RAMPS:
        if name in p and np.isfinite(p[name]).any():
            raise AssertionError(f"{name} is finite: the uniform coarse step has no ramp and no daily cap (C17)")


def _check_availability_is_all_or_nothing(p: xr.Dataset) -> None:
    if "conversion_availability" not in p:
        return
    a = p["conversion_availability"]
    other = [d for d in a.dims if d != "conversion_tech"]
    some = np.isfinite(a).any(other) if other else np.isfinite(a)
    every = np.isfinite(a).all(other) if other else np.isfinite(a)
    if bool((some & ~every).any()):
        raise AssertionError("a technology has a partly-NaN availability: NaN reads as 1 in the block model (C17)")


def check_coarse_step_contract(p: xr.Dataset) -> None:
    """C17: every ramp, mode-pair ramp and daily cap NaN, each availability all-NaN or all-finite (raises ``AssertionError``)."""
    _check_no_ramps(p)
    _check_availability_is_all_or_nothing(p)


def coarse_step_reference(p: xr.Dataset, L: int) -> xr.Dataset:
    """
    Today's dataset with steps of ``L`` hours: the reference of the uniform coarse step (T3(c)).

    ``hour`` becomes one label per block of ``p``, ``time_step_duration`` the block durations, ``demand`` and
    ``flexdem_demand`` are summed per block, ``conversion_availability`` and the four net_import hourly parameters
    are the Δ-weighted mean per block (``block_mean``, NaN read as 1 first, as the block model does, section 6.1). The
    block inputs are removed, so that ``build_model`` runs today's code on it. Raises ``AssertionError`` on a finite
    ramp or daily cap (the block rows differ from today's there, on purpose) and ``ValueError`` on any other
    per-hour parameter.
    """
    _check_no_ramps(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = time_blocks(p)
    if L not in tb.block_of_hour:
        raise ValueError(f"p has no block of {L} h")
    block = block_dim(L)
    q = p.drop_vars(BLOCK_PARAMETERS, errors="ignore")
    new = {}
    for name in [n for n in q.data_vars if "hour" in q[n].dims]:
        da = q[name]
        if name == "time_step_duration":
            out = tb.duration[L]
        elif name in _STEP_SUMS:
            out = da.groupby(tb.block_of_hour[L]).sum()
        elif name in _STEP_MEANS:
            out = block_mean(xr.where(np.isfinite(da), da, 1.0), tb, L)
        else:
            raise ValueError(f"{name}: no rule to put a per-hour parameter on steps of {L} h")
        new[name] = out.rename({block: "hour"})
    q = q.drop_dims("hour").assign_coords(hour=np.arange(tb.duration[L].size, dtype="int64"))
    return q.assign(new)


FLEXDEM_RAMP = 0.03
# heat energy that is flexible in area ``a`` in each block (cycled; area ``b`` has 0.8 times): shifted by the
# window, the ratios and the ramps below (probed: the ratio rows do not bind, the window and both ramps do)
FLEXDEM_BLOCKS = [6.1, 4.8, 4.0, 5.9, 5.3, 6.7]


def coarse_step_dataset(L: int, horizon: int, net_import: bool = False, carbon: bool = False) -> xr.Dataset:
    """
    The dataset on which every contributor of every resource has the balance's length (T3(c), I5).

    ``combined_toy`` of ``horizon`` one-hour steps (a multiple of ``L``) with conversion and combined only (no
    storage, transport or process: their hourly contributors would keep the balances summed, a relaxation and not the
    reference), ``turpe`` off (R5), the idle resource kept (R10), every conversion and combined length ``L`` (so
    ``gas`` and ``heat`` are derived at ``L`` and ``idle`` stays hourly), the mean rule on every conversion
    technology with an availability that varies inside the blocks (``heater``; ``well`` has none), flexdem on heat
    (``FLEXDEM_BLOCKS``, constant inside each block, next to a fixed demand) with load ratios 0.75 and 1.4, a
    conservation window of ``2 L`` hours and ramps of 0.03. Every conversion and combined ramp, mode-pair ramp and
    ``conversion_max_daily_production`` is NaN, and each availability is all-NaN or all-finite (C17,
    ``check_coarse_step_contract``). ``net_import`` puts the rest of the world on ``heat`` (import, prices per
    block ``HEAT_IMPORT_PRICE``, below the heater's cost) and on ``gas`` (export, prices ``GAS_EXPORT_PRICE``, above
    the well's), prices and emission factors constant inside each block and different between blocks, with a
    binding yearly import maximum on heat and a binding yearly export maximum on gas (area ``a``); gas imports at 50
    and heat exports at 0, ``idle`` imports at 1. ``carbon`` puts a tax of ``CARBON_TAX`` and an emission factor on
    every conversion and combined technology (``COARSE_CARBON``) next to the net_import factors.
    ``coarse_step_reference(p, L)`` is the same system in today's code.
    """
    if horizon % L:
        raise ValueError(f"horizon {horizon} is not a multiple of L = {L}")
    p = combined_toy([1.0] * horizon)
    is_heat = p.resource == "heat"
    blocks = np.resize(FLEXDEM_BLOCKS, horizon // L)
    flexible = np.zeros((2, horizon, p.sizes["resource"], 1))
    for area, share in enumerate((1.0, 0.8)):
        flexible[area, :, list(p.resource.values).index("heat"), 0] = share * np.repeat(blocks, L) / L
    p = p.assign(
        turpe=np.array(False),
        demand=p.demand * xr.where(is_heat, 0.4, 1.0),
        flexdem=np.array(True),
        flexdem_demand=(("area", "hour", "resource", "year_op"), flexible),
    )
    per = ("area", "resource", "year_op")
    shape = tuple(p.sizes[d] for d in per)
    for name, value in (
        ("flexdem_maxload_ratio", 1.4),
        ("flexdem_minload_ratio", 0.75),
        ("flexdem_ramp_up", FLEXDEM_RAMP),
        ("flexdem_ramp_down", FLEXDEM_RAMP),
    ):
        p[name] = (per, np.full(shape, value))
    p["flexdem_conservation_hrs"] = (per, np.full(shape, 2 * L, dtype="int64"))
    p = per_tech(p, "conversion_availability", {"heater": np.resize([1.0, 0.6, 0.9, 0.8, 0.5, 1.0, 0.7, 0.9], horizon)})
    if net_import:
        p = _with_net_import(p, L, horizon)
    if carbon:
        p = with_carbon(p, tax=CARBON_TAX, **COARSE_CARBON)
    for name in _NO_RAMPS:
        p[name] = xr.full_like(p[name], np.nan) if name in p else xr.DataArray(np.nan)
    techs = {t: L for t in p.conversion_tech.values}
    p = with_blocks(
        p,
        conversion=techs,
        combined={t: L for t in p.combined_tech.values},
        rule={t: "mean" for t in techs},
    )
    check_coarse_step_contract(p)
    return p


CARBON_TAX = 12.0  # per tCO2, binding on ``coarse_step_dataset(..., carbon=True)`` (probed)
COARSE_CARBON = dict(
    conversion={"well": 0.3, "heater": 0.8},
    combined={"electric_boiler": [0.1, 0.1], "boiler_block": [0.5, 0.2], "steady_kiln": [0.3, 0.3]},
)

HEAT_IMPORT_PRICE = [0.2, 0.7, 0.4, 0.9, 0.3, 0.6]  # per block (cycled), below the combined technologies cost
GAS_EXPORT_PRICE = [2.0, 6.0, 3.0, 8.0, 5.0, 9.0]  # per block (cycled), above the well's 0.5
HEAT_IMPORT_MAX, GAS_EXPORT_MAX = 9.0, 4.0  # energy over the horizon, area a (probed: both bind)


def _with_net_import(p: xr.Dataset, L: int, horizon: int) -> xr.Dataset:
    """``p`` with the net_import of ``coarse_step_dataset``: prices and factors constant inside each block."""
    names = list(p.resource.values)
    per_step = ("area", "hour", "resource", "year_op")
    shape = (p.sizes["area"], horizon, p.sizes["resource"], 1)
    out = {"net_import": np.array(True)}
    for name, fill in (
        ("net_import_import_price", {"idle": 1.0, "gas": 50.0, "heat": np.repeat(np.resize(HEAT_IMPORT_PRICE, horizon // L), L)}),
        ("net_import_export_price", {"gas": np.repeat(np.resize(GAS_EXPORT_PRICE, horizon // L), L)}),
        ("net_import_emission_factor", {"heat": np.repeat(np.resize([0.2, 0.5, 0.1], horizon // L), L)}),
        ("net_import_total_emission_factor", {"heat": np.repeat(np.resize([0.3, 0.6, 0.2], horizon // L), L)}),
    ):
        x = np.zeros(shape)
        for resource, value in fill.items():
            x[:, :, names.index(resource), :] = np.reshape(value, (1, -1, 1)) if np.ndim(value) else value
        if name == "net_import_import_price":
            x[1, :, names.index("heat"), :] = 20.0  # area b never imports heat: the yearly maximum is shared, and a tie would not be unique
        out[name] = (per_step, x)
    n = p.operation_year_duration / horizon  # the yearly rows carry the year normalisation
    for name, resource, energy in (
        ("net_import_max_yearly_energy_import", "heat", HEAT_IMPORT_MAX),
        ("net_import_max_yearly_energy_export", "gas", GAS_EXPORT_MAX),
    ):
        x = np.full((p.sizes["area"], p.sizes["resource"], 1), np.nan)
        x[0, names.index(resource), :] = energy * float(n.item())
        out[name] = (("area", "resource", "year_op"), x)
    return p.assign(out)


def net_import_two_lengths() -> xr.Dataset:
    """
    ``coarse_step_dataset(4, 24, net_import=True)`` with gas on 2 h and heat on 4 h blocks (two coarse lengths).

    The well makes gas on 2 h blocks and the heater heat on 4 h; the combined technologies stay hourly (their
    contributions are summed into the coarse rows) and the idle resource stays hourly: three net_import groups.
    """
    p = coarse_step_dataset(4, 24, net_import=True)
    return with_blocks(
        p,
        conversion={"well": 2, "heater": 4},
        resource={"gas": 2, "heat": 4},
        rule={"well": "mean", "heater": "mean"},
    )


def flexdem_block_dataset() -> xr.Dataset:
    """``coarse_step_dataset(4, 24)``: flexdem on heat with ratios, a window of 8 h and ramps, blocks of 4 h."""
    return coarse_step_dataset(4, 24)


def flexdem_toy(n: int, level: float = 1.0, window: int = 0, ramp: float = 0.1) -> xr.Dataset:
    """
    The hand toy with a constant flexible steel demand (``level`` per step) and no fixed demand.

    Load ratios 0.5 and 1.5, ramps ``ramp``, a conservation window of ``window`` hours (0: none). No block length.
    """
    p = hand_toy([1.0] * n, [0.0] * n)
    per, shape = ("area", "resource", "year_op"), (1, 2, 1)
    demand = np.zeros((1, n, 2, 1))
    demand[0, :, 1, 0] = level
    return p.assign(
        flexdem=np.array(True),
        flexdem_demand=(("area", "hour", "resource", "year_op"), demand),
        flexdem_conservation_hrs=(per, np.full(shape, window, dtype="int64")),
        flexdem_maxload_ratio=(per, np.full(shape, 1.5)),
        flexdem_minload_ratio=(per, np.full(shape, 0.5)),
        flexdem_ramp_up=(per, np.full(shape, ramp)),
        flexdem_ramp_down=(per, np.full(shape, ramp)),
    )


def flexdem_short_block_dataset() -> xr.Dataset:
    """``flexdem_toy`` of 10 h with the furnace on blocks of 4 h: steel is derived at 4 h, the last block lasts 2 h."""
    return with_blocks(flexdem_toy(10), conversion={"furnace": 4})


def _ratio_cells(p: xr.Dataset) -> xr.Dataset:
    """
    ``p`` (a ``coarse_step_dataset`` of 4 h blocks) with ratios that bind and no flexdem ramp: area ``a`` meets the
    maximum, area ``b`` the minimum.
    """
    cells = {"a": ([2.3, 9.1, 5.7, 2.9, 8.6, 11.3], 1.43, 0.5), "b": ([4.1, 8.3, 5.0, 9.4, 9.2, 4.2], 3.0, 0.83)}
    heat = list(p.resource.values).index("heat")
    demand = p.flexdem_demand.copy(deep=True)
    for area, (blocks, maximum, minimum) in cells.items():
        i = list(p.area.values).index(area)
        demand.values[i, :, heat, 0] = np.repeat(blocks, 4) / 4
        p["flexdem_maxload_ratio"].values[i] = maximum
        p["flexdem_minload_ratio"].values[i] = minimum
    nan = p.flexdem_ramp_up * np.nan
    return p.assign(flexdem_demand=demand, flexdem_ramp_up=nan, flexdem_ramp_down=nan)


def flexdem_ratio_dataset() -> xr.Dataset:
    """
    ``flexdem_block_dataset`` with ratios that bind and no ramp: area ``a`` meets the maximum, area ``b`` the minimum.

    The areas are independent, so each one gets the flexible heat energy per block (probed) that pushes it against
    one of the two limits, with a unique optimum.
    """
    return _ratio_cells(flexdem_block_dataset())


# --- whole-model datasets (Task 12): ``parameters`` with every module on, an idle resource, block inputs ---


def _fixture(name: str, _cache: dict = {}):
    """
    The value of the conftest fixture ``name`` outside pytest (a zero-argument builder cannot request it).

    Calls the plain function behind the fixture, with its own arguments resolved the same way; the result is cached
    and handed out as a deep copy.
    """
    from pommes.tests import conftest

    if name not in _cache:
        fixture = getattr(conftest, name, None)
        # pytest assumption: a fixture function keeps the plain function it decorates in ``__wrapped__``
        function = getattr(fixture, "__wrapped__", None)
        if function is None:
            raise ValueError(f"{name!r} is not a fixture of pommes/tests/conftest.py: a builder can only use those")
        _cache[name] = function(*[_fixture(a) for a in inspect.signature(function).parameters])
    return _cache[name].copy(deep=True)


def with_idle(p: xr.Dataset) -> xr.Dataset:
    """``p`` with a first resource ``idle`` that holds nothing (R10): 0 in every factor, price, demand and cap."""
    names = ["idle", *p.resource.values]
    out = p.reindex(resource=names)
    for name in p.data_vars:
        if "resource" in p[name].dims:
            out[name] = p[name].reindex(resource=names, fill_value=0.0)
    return out


HEAT_IMPORT_PRICE_PER_STEP = [0.3, 0.3, 5.0, 5.0, 0.4, 0.4, 5.0, 5.0, 0.5, 0.5]  # cheap for two steps in four


def _whole_base() -> xr.Dataset:
    """
    The ``parameters`` fixture (storage, transport, turpe, net_import, carbon, conversion, combined) with an idle
    resource, priced spillage and a heat import that decides the optimum.

    One vintage and one year of operation (the fixture prices its vintages alike, so the capacities of several years
    can be split in several ways at the same cost, and the twin comparison of capacities needs one optimum). The
    fixture alone imports nothing and emits nothing: here heat may be imported at a price that is low for two steps in
    four and high otherwise (``HEAT_IMPORT_PRICE_PER_STEP``: a flat block boiler cannot follow it; no yearly maximum),
    with an emission factor taxed by the fixture's carbon tax (the carbon goal is NaN: a binding goal would tie the two
    areas). The electric boiler covers the rest of the heat. Every other price keeps the fixture's value, which does not
    depend on the hour.
    """
    p = _fixture("parameters").sel(year_dec=[2040], year_inv=[2020], year_op=[2020])
    p = priced_spillage(with_idle(p))
    price = p.net_import_import_price.expand_dims(hour=p.hour.values).transpose("hour", "resource").copy()
    price.loc[{"resource": "heat"}] = np.resize(HEAT_IMPORT_PRICE_PER_STEP, p.sizes["hour"])
    cap = p.net_import_max_yearly_energy_import.copy()
    cap.loc[{"resource": "heat"}] = np.nan
    factor, total = p.net_import_emission_factor.copy(), p.net_import_total_emission_factor.copy()
    factor.loc[{"resource": "heat"}], total.loc[{"resource": "heat"}] = 0.15, 0.25
    return p.assign(
        net_import_import_price=price,
        net_import_max_yearly_energy_import=cap,
        net_import_emission_factor=factor,
        net_import_total_emission_factor=total,
        carbon_goal=xr.full_like(p.carbon_goal, np.nan),
    )


def whole_t1_dataset() -> xr.Dataset:
    """``_whole_base`` with every conversion and combined length 1 h on steps of 1 h (T1: block equals step)."""
    p = _whole_base()
    return with_blocks(
        p,
        conversion={t: 1 for t in p.conversion_tech.values},
        combined={t: 1 for t in p.combined_tech.values},
    )


T2_ELECTROLYSIS_FACTOR = 0.3  # tCO2 per MWh of electrolysis power


def whole_t2_dataset() -> xr.Dataset:
    """
    ``_whole_base`` with the conversion technologies but ``wind_onshore`` and the electric boiler on blocks of 2 h.

    Every balance is hourly: electricity, hydrogen and methane have storage, and heat is forced to the step (1 h).
    ``wind_onshore`` stays hourly: it is must-run with an availability that varies inside a block, which is a
    relaxation of the twin and not an equality (D9').
    """
    p = _whole_base()
    # the electrolyser runs on blocks: give it an emission factor so that block technologies emit (0 in the fixture)
    p["conversion_emission_factor"] = p.conversion_emission_factor.where(
        p.conversion_tech != "electrolysis", T2_ELECTROLYSIS_FACTOR
    )
    return with_blocks(
        p,
        conversion={t: 2 for t in p.conversion_tech.values if t != "wind_onshore"},
        combined={"electric_boiler": 2},
        resource={"heat": 1},
    )


T3B_HOURS = 9  # with blocks of 2 h the last block holds one step
T3B_PROFILE = [1.2, 0.8, 1.0, 1.1, 0.9, 1.3, 0.7, 1.0, 1.2]  # demand shape of methane and hydrogen, per step


def whole_t3b_dataset() -> xr.Dataset:
    """
    ``_whole_base`` on 9 steps with ``methane`` and ``hydrogen`` forced to 2 h (the last block holds one step).

    Storage, transport and conversion contribute hourly to both balances (summed into the coarse rows). Methane and
    hydrogen demand vary from step to step (``T3B_PROFILE``) so that each coarse row holds a different sum. Their
    net_import prices and emission factors do not depend on the hour, so they are constant inside each block, and no
    flexdem ramp is set: the coarse model is a relaxation of the hourly one.
    """
    p = _whole_base().isel(hour=slice(0, T3B_HOURS))
    demand = p.demand.expand_dims(area=p.area.values, hour=p.hour.values).transpose("area", "hour", "resource", "year_op").copy()
    profile = xr.DataArray(T3B_PROFILE, dims="hour", coords={"hour": p.hour.values})
    demand.loc[{"resource": "methane"}] = xr.full_like(demand.sel(resource="methane"), 1.5) * profile
    demand.loc[{"resource": "hydrogen"}] = demand.sel(resource="hydrogen") * profile
    return with_blocks(p.assign(demand=demand), resource={"methane": 2, "hydrogen": 2})


# --- retrofit with blocks (section 16.3) ---

RETROFIT_YEARS = dict(year_inv=[2020, 2030, 2040], year_dec=[2040, 2050, 2060], year_op=[2030, 2040], year_ret=[2040])
RETROFIT_DEMAND_GROWTH = [1.0, 1.6]  # per year_op: the retrofitted capacity is needed in the second one


def _on_retrofit_years(p: xr.Dataset, base_cost: dict[str, float]) -> xr.Dataset:
    """
    ``p`` (one year, one vintage) on the retrofit years: operation in 2030 and 2040, vintages 2020, 2030, 2040.

    Every technology of the modules of ``base_cost`` lives 20 years (a vintage ends exactly 20 years later, early
    decommissioning off) at an annuity of the module's cost plus 0.1 per technology (distinct), and the demand grows in
    the second year.
    """
    years = {k: np.array(v, dtype="int64") for k, v in RETROFIT_YEARS.items() if k != "year_ret"}
    out = p.drop_dims(["year_inv", "year_dec", "year_op"]).assign_coords(years)
    growth = xr.DataArray(RETROFIT_DEMAND_GROWTH, dims="year_op", coords={"year_op": years["year_op"]})
    out["demand"] = p.demand.isel(year_op=0, drop=True) * growth
    for name, value in (("operation_year_duration", 8760.0), ("discount_rate", 0.0), ("discount_factor", 1.0)):
        out[name] = ("year_op", np.full(2, value))
    for module, first in base_cost.items():
        tech, k = f"{module}_tech", p.sizes[f"{module}_tech"]
        out[f"{module}_end_of_life"] = ((tech, "year_inv"), np.tile(years["year_inv"] + 20.0, (k, 1)))
        cost = (first + 0.1 * np.arange(k))[:, None, None] * np.ones((k, 3, 3))
        out[f"{module}_annuity_cost"] = ((tech, "year_dec", "year_inv"), cost)
    return out


def _retrofit_data(module: str, pair: tuple[str, str], year_ret: list[int]) -> dict:
    """The retrofit parameters of ``module`` for the one pair ``(from, to)``: a cheap transfer at ``year_ret``."""
    name, years = f"retrofit_{module}", np.array(year_ret, dtype="int64")
    return {
        name: np.array(True),
        f"{name}_tech_from": ([f"{name}_pair"], np.array([pair[0]])),
        f"{name}_tech_to": ([f"{name}_pair"], np.array([pair[1]])),
        f"{name}_life_span": np.array(20.0),
        f"{name}_finance_rate": np.array(0.0),
        f"{name}_early_decommissioning": np.array(False),
        f"{name}_annuity_perfect_foresight": np.array(False),
        f"{name}_invest_cost": np.array(1.0),
        f"{name}_ret_ratio": np.array(1.0),
        f"{name}_eligible_share": np.array(1.0),
        f"{name}_power_capacity_investment_min": np.array(np.nan),
        f"{name}_power_capacity_investment_max": np.array(np.nan),
        f"{name}_annuity_cost": (("year_dec", "year_ret"), np.full((3, years.size), 0.3)),
        f"{name}_end_of_life": (("year_ret",), (20 + years).astype("int64")),
    }


def _with_retrofit(p: xr.Dataset, module: str, pair: tuple[str, str]) -> xr.Dataset:
    pair_dim = f"retrofit_{module}_pair"
    year_ret = np.array(RETROFIT_YEARS["year_ret"], dtype="int64")
    return p.assign_coords({pair_dim: [f"{pair[0]}_to_{pair[1]}"], "year_ret": year_ret}).assign(
        _retrofit_data(module, pair, RETROFIT_YEARS["year_ret"])
    )


def retrofit_conversion_block_dataset() -> xr.Dataset:
    """
    A conversion retrofit ``old`` -> ``new`` that transfers capacity: ``old`` hourly, ``new`` on blocks of 2 h.

    One area, ``idle`` and ``steel``, steel demand (2, 3, 1, 3, 2, 2) growing by 1.6 in the second year (2030 and 2040).
    ``old`` (cheap, buildable in 2020 only, decommissioned in 2040) serves 2030; ``new`` (cheaper to run) cannot be
    invested in and gets its capacity only by retrofit of ``old`` in 2040, when the retrofit (annuity 0.3) is far
    cheaper than the hourly ``backup`` (variable cost 30), which keeps steel hourly. The reverse direction is the same
    dataset with ``with_blocks(p, conversion={"old": 2})``. Distinct costs and a priced spillage: a unique optimum.
    """
    p = hand_toy([1.0] * 6, [2.0, 3.0, 1.0, 3.0, 2.0, 2.0], s=3.0)
    p = p.isel(conversion_tech=[0, 0, 0]).assign_coords(conversion_tech=["old", "new", "backup"])
    p = _on_retrofit_years(p, {"conversion": 1.0})
    p["conversion_factor"] = (("conversion_tech", "resource"), [[0.0, 1.0]] * 3)
    p["conversion_variable_cost"] = ("conversion_tech", [1.0, 0.6, 30.0])
    p["conversion_fixed_cost"] = ("conversion_tech", [2.0, 2.5, 3.0])
    p["conversion_power_capacity_max"] = (("area", "conversion_tech"), [[np.nan] * 3])
    p["conversion_power_capacity_min"] = (("area", "conversion_tech"), [[0.0] * 3])
    invest = np.full((1, 3, 3), np.nan)
    invest[0, 0, 1:] = 0.0  # old: the 2020 vintage only
    invest[0, 1, :] = 0.0  # new: retrofit only
    p["conversion_power_capacity_investment_max"] = (("area", "conversion_tech", "year_inv"), invest)
    p["conversion_life_span"] = ("conversion_tech", np.full(3, 20.0))
    p = _with_retrofit(p, "conversion", ("old", "new"))
    return with_blocks(p, conversion={"new": 2})


def retrofit_combined_block_dataset() -> xr.Dataset:
    """
    A combined retrofit ``electric_boiler`` -> ``boiler_block`` that transfers capacity: source hourly, target on 2 h.

    ``combined_toy`` over 6 steps and the retrofit years: ``electric_boiler`` is buildable in 2020 only,
    ``boiler_block`` (cheaper to run) only by retrofit in 2040, ``steady_kiln`` is pinned to zero; the hourly ``heater``
    covers what the boilers cannot. The reverse direction: ``with_blocks(p, combined={"electric_boiler": 2})``.
    """
    p = _on_retrofit_years(combined_toy([1.0] * 6), {"combined": 3.0, "conversion": 0.0})
    p["combined_power_capacity_max"] = (("area", "combined_tech"), [[np.nan, np.nan, 0.0]] * 2)
    invest = np.full((2, 3, 3), np.nan)
    invest[:, 0, 1:] = 0.0  # electric_boiler: the 2020 vintage only
    invest[:, 1, :] = 0.0  # boiler_block: retrofit only
    p["combined_power_capacity_investment_max"] = (("area", "combined_tech", "year_inv"), invest)
    p = _with_retrofit(p, "combined", ("electric_boiler", "boiler_block"))
    return with_blocks(p, combined={"boiler_block": 2})


# the retrofit pair of each module: (builder, source, target)
RETROFIT = {
    "conversion": (retrofit_conversion_block_dataset, "old", "new"),
    "combined": (retrofit_combined_block_dataset, "electric_boiler", "boiler_block"),
}


def retrofit_source_block_dataset(module: str) -> xr.Dataset:
    """
    The retrofit dataset of ``module`` ("conversion" or "combined") with the block on the source, the target hourly.
    """
    builder, source, _ = RETROFIT[module]
    return with_blocks(builder(), **{module: {source: 2}})


# --- the uniform coarse step with net_import and carbon (T3(c)), operating conditions (section 16.5) ---


def t3c_dataset() -> xr.Dataset:
    """``coarse_step_dataset(4, 24, net_import=True, carbon=True)``: window, ramps, both yearly maxima, the tax bind."""
    return coarse_step_dataset(4, 24, net_import=True, carbon=True)


def t3c_ratio_dataset() -> xr.Dataset:
    """
    ``t3c_dataset`` without carbon, with the ratios of ``flexdem_ratio_dataset`` (the ratio rows bind, no ramp).

    Carbon is left out: with the ratios the optimum has two modes of a combined technology at the same cost (the
    objective is unique, the emissions are not), so the yearly totals would not compare.
    """
    return _ratio_cells(coarse_step_dataset(4, 24, net_import=True))


def conditions_block_dataset() -> xr.Dataset:
    """
    The hand toy on 8 steps with steel forced to 2 h, built so that the dual of each coarse row is unique.

    ``power`` is a second, hourly resource (see below). The furnace (on blocks of 2 h) is capped at 1 MW and steel may
    be shed at 7 per MWh: blocks that ask for more than 2 MWh (steps (3, 3) and (1.5, 1.5)) pay the shedding cost at the
    margin, the others (steps (0.5, 0.7) and (0.2, 0.4)) the furnace's variable cost, so the coarse prices are 7n, vn, 7n, vn (n = operation_year_normalization).
    """
    p = hand_toy([1.0] * 8, [3.0, 3.0, 0.5, 0.7, 1.5, 1.5, 0.2, 0.4])
    p["conversion_power_capacity_max"] = (("area", "conversion_tech"), [[1.0]])
    p["load_shedding_max_capacity"] = ("resource", [0.0, np.nan])
    p["load_shedding_cost"] = ("resource", [0.0, 7.0])
    # an hourly resource with a priced, binding demand (shed at 5 per MWh, nothing produces it): its hourly adequacy
    # duals are 5n, not zero
    names = [*p.resource.values, "power"]
    out = p.reindex(resource=names)
    for name in p.data_vars:
        if "resource" in p[name].dims:
            out[name] = p[name].reindex(resource=names, fill_value=0.0)
    out["demand"].loc[{"resource": "power"}] = 1.0
    out["load_shedding_max_capacity"].loc[{"resource": "power"}] = np.nan
    out["load_shedding_cost"].loc[{"resource": "power"}] = 5.0
    out["spillage_cost"].loc[{"resource": "power"}] = 1.0
    return with_blocks(out, conversion={"furnace": 2}, resource={"steel": 2})


def net_import_twin_dataset() -> xr.Dataset:
    """``coarse_step_dataset(4, 24, net_import=True)``: binding import and export maxima (the C6 net_import twin)."""
    return coarse_step_dataset(4, 24, net_import=True)


def net_import_expected_conditions_dataset() -> xr.Dataset:
    """
    ``coarse_step_dataset(4, 24, net_import=True)`` under two conditions (weights 0.4 and 0.6), both yearly maxima on
    the expectation.

    The second condition has import prices 1.5 times and export prices 0.8 times the first one's, so that the
    conditions import and export differently; ``net_import_max_yearly_energy_import_expected`` and
    ``..._export_expected`` are on.
    """
    p = net_import_twin_dataset()
    p = with_conditions(
        p,
        [0.4, 0.6],
        net_import_import_price=[p.net_import_import_price, 1.5 * p.net_import_import_price],
        net_import_export_price=[p.net_import_export_price, 0.8 * p.net_import_export_price],
    )
    return p.assign(
        net_import_max_yearly_energy_import_expected=np.array(True), net_import_max_yearly_energy_export_expected=np.array(True)
    )
