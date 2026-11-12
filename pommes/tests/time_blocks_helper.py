"""Shared helpers of the time-blocks tests (identical in both engines)."""

import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.utils import TimeBlocks, time_blocks

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


def assert_twin_equal(m_block, m_twin, tb: TimeBlocks, powers: list[str]) -> None:
    """
    A solved block model equals its solved flat twin (section 15.1).

    Objective to 1e-7 (relative); ``planning_*`` and capacity variables to
    1e-6; each block power of ``powers`` repeated over the steps of its
    blocks equals the twin's hourly power of the same technologies to 1e-6.
    """
    assert m_block.objective.value == pytest.approx(m_twin.objective.value, rel=1e-7)

    def same(a, b, name):
        a, b = xr.align(a, b, join="outer")
        np.testing.assert_allclose(
            a.fillna(0).values, b.transpose(*a.dims).fillna(0).values, atol=1e-6, err_msg=name
        )

    for name in m_block.variables:
        if name.startswith("planning_") or name.endswith("_capacity"):
            same(solved(m_block, name), solved(m_twin, name), name)
    for name in powers:
        twin_name = re.sub(r"_block_\d+h_", "_", name)
        block = to_hours(solved(m_block, name), tb)
        tech = next(d for d in block.dims if d.endswith("_tech"))
        hourly = solved(m_twin, twin_name).sel({tech: block.indexes[tech]})
        same(block, hourly, name)


def units_of(obj) -> list[tuple]:
    """``(area, tech)`` pairs along the technology or unit dimension of a variable or constraint."""
    dims = obj.labels.dims if hasattr(obj, "labels") else obj.dims
    dim = next(d for d in dims if d.endswith(("_tech", "_tech_unit")))
    if dim.endswith("_unit"):  # sparse: a MultiIndex of (area, tech)
        return [tuple(u) for u in obj.indexes[dim]]
    return [(a, t) for a in obj.indexes["area"] for t in obj.indexes[dim]]


def variables_in(m, con) -> set[str]:
    """Names of the variables with a live term in the left-hand side of ``con``."""
    used = set(np.unique(con.lhs.vars.values)) - {-1}
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
