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
    ("transport_time_block_hrs", "transport_tech"),
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
    transport: dict[str, float] | None = None,
) -> xr.Dataset:
    """
    ``p`` with the five block parameters set (NaN / ``"min"`` elsewhere).

    A parameter whose dimension is not in ``p`` is left out.
    """
    p = p.copy()
    given = (conversion, combined, resource, transport)
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


def transport_every_row_dataset() -> xr.Dataset:
    """
    T0(b): every row of the transport module on three 1 h steps, no block length.

    Ramps up and down, an availability that varies with the hour, capacity bounds, investment bounds, delays of 1 h
    and 2 h on two pairs, a linepack on two other pairs. The power line has no delay, no linepack, no availability
    and no investment bound: it carries a ramp up on link_1 and a capacity maximum on link_2 only.
    """
    p = xr.merge([_fixture("parameters_dispatch_invest"), _fixture("transport")])
    p["transport_hurdle_costs"] = p.transport_hurdle_costs / 8760
    p = p.sel(
        conversion_tech=["wind_onshore"],
        hour=[0, 1, 2],
        resource=["electricity", "methane"],
        year_dec=[2040, 2050],
        year_inv=[2030],
        year_op=[2030],
    ).copy(deep=True)
    p["conversion_availability"] = p.conversion_availability.expand_dims(area=p.area).copy()
    p.conversion_availability.loc[dict(conversion_tech="wind_onshore")] = [[0.6, 0.3, 0.1], [0.4, 0.7, 0.9]]

    def grid(fill, *dims):
        coords = {d: p[d] for d in dims}
        return xr.DataArray(np.full([p.sizes[d] for d in dims], fill), dims=dims, coords=coords)

    lt, lty = ("link", "transport_tech"), ("link", "transport_tech", "year_op")
    p["transport_ramp_up"] = grid(np.nan, *lty)
    p["transport_ramp_up"][0, :, :] = 0.5
    p["transport_ramp_down"] = grid(np.nan, *lty)
    p["transport_ramp_down"][1, 0, :] = 0.7
    p["transport_power_capacity_max"] = grid(np.nan, *lty)
    p["transport_power_capacity_max"][1, :, :] = 5000.0
    p["transport_power_capacity_min"] = grid(np.nan, *lty)
    p["transport_power_capacity_min"][0, 1, :] = 0.0
    p["transport_availability"] = grid(np.nan, "link", "transport_tech", "hour", "year_op")
    p["transport_availability"][0, 0, :, :] = np.array([0.9, 0.6, 0.8])[:, None]
    p["transport_power_capacity_investment_min"] = grid(np.nan, *lt)
    p["transport_power_capacity_investment_max"] = grid(np.nan, *lt)
    p["transport_power_capacity_investment_min"][0, 0], p["transport_power_capacity_investment_max"][0, 0] = 1.0, 10.0
    p["transport_power_capacity_investment_min"][1, 0], p["transport_power_capacity_investment_max"][1, 0] = 2.0, 2.0
    p["transport_delay"] = grid(0.0, *lt)
    p["transport_delay"][0, 1] = 1.0
    p["transport_delay"][1, 0] = 2.0
    p["transport_linepack_capacity"] = grid(np.nan, *lty)
    p["transport_linepack_capacity"][0, 0, :] = 5.0
    p["transport_linepack_capacity"][1, 1, :] = 12.0
    return p


def transport_toy(dt: list[float], rate: list[float] | None = None, **kw) -> xr.Dataset:
    """
    Hourly transport toy (two areas, two links, one year, no carbon, no flexdem).

    Areas ``a`` and ``b``, links ``ab`` and ``ba``, resources ``idle``, ``cargo`` and ``oil``. Conversion, both
    hourly: ``mill`` makes cargo at ``a`` only (its capacity at ``b`` is 0) and ``bunker`` makes oil. Ships carry
    cargo; by default an hourly ``H`` first and ``S`` second, the tech of index i with the link ends of index 0
    swapped when i is odd (``S`` runs ``b`` to ``a`` on ``ab`` and ``a`` to ``b`` on ``ba``), so that a position
    read on the full tech axis picks the wrong links. No length is set: ``with_blocks(p, transport={"S": 24})``.
    The cargo demand at ``b`` is ``rate`` (MW, cycled) times the step. Fixed costs and hurdle costs are finite,
    distinct and positive per (link, tech), and every spillage is priced, so that the optimum is unique.

    Keywords (a dict maps a tech to its value): ``techs`` (names), ``eff`` (efficiency), ``ramp`` ((up, down)),
    ``availability`` (list per hour), ``fuel`` (fuel consumption: the tech burns ``oil``), ``fuel_area`` (area the fuel
    is drawn at), ``delay`` (hours), ``linepack`` (capacity); ``silo`` (bool): an hourly storage ``silo`` on cargo.
    """
    from pommes.model.data_validation import ref_inputs

    dt = np.array(dt, dtype="float64")
    n = len(dt)
    rate = np.resize(np.array(HEAT_RATE if rate is None else rate), n)
    techs = list(kw.pop("techs", ["H", "S"]))
    eff, ramp, availability = kw.pop("eff", {}), kw.pop("ramp", {}), kw.pop("availability", {})
    fuel, fuel_area, delay = kw.pop("fuel", {}), kw.pop("fuel_area", {}), kw.pop("delay", {})
    linepack, silo = kw.pop("linepack", {}), kw.pop("silo", False)
    assert not kw, kw
    k = len(techs)
    ds = xr.Dataset(
        coords=dict(
            area=["a", "b"],
            hour=np.arange(n, dtype="int64"),
            resource=["idle", "cargo", "oil"],
            conversion_tech=["mill", "bunker"],
            transport_tech=techs,
            link=["ab", "ba"],
            year_dec=np.array([2040]),
            year_inv=np.array([2020]),
            year_op=np.array([2030], dtype="int64"),
        )
    )
    ds["conversion"] = True
    ds["transport"] = True
    ds["time_step_duration"] = ("hour", dt)
    ds["operation_year_duration"] = ("year_op", [8760.0])
    ds["operation_day_duration"] = np.array(24, dtype="int64")
    ds["discount_rate"] = ("year_op", [0.0])
    ds["discount_factor"] = ("year_op", [1.0])
    demand = np.zeros((2, n, 3, 1))
    demand[1, :, 1, 0] = rate * dt
    ds["demand"] = (("area", "hour", "resource", "year_op"), demand)
    ds["load_shedding_max_capacity"] = ("resource", [0.0, 0.0, 0.0])
    ds["load_shedding_cost"] = ("resource", [0.0, 0.0, 0.0])
    ds["spillage_max_capacity"] = ("resource", [np.nan, np.nan, np.nan])
    ds["spillage_cost"] = ("resource", [1.0, 2.0, 3.0])
    ds["conversion_factor"] = (("conversion_tech", "resource"), [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ds["conversion_variable_cost"] = ("conversion_tech", [0.5, 0.9])
    ds["conversion_fixed_cost"] = ("conversion_tech", [0.7, 1.1])
    ds["conversion_invest_cost"] = ("conversion_tech", [0.0, 0.0])
    ds["conversion_emission_factor"] = ("conversion_tech", [0.0, 0.0])
    ds["conversion_power_capacity_max"] = (("area", "conversion_tech"), [[np.nan, np.nan], [0.0, np.nan]])
    ds["conversion_power_capacity_min"] = (("area", "conversion_tech"), np.zeros((2, 2)))
    ds["conversion_life_span"] = ("conversion_tech", [20.0, 20.0])
    ds["conversion_end_of_life"] = (("conversion_tech", "year_inv"), [[2040.0], [2040.0]])
    ds["conversion_annuity_cost"] = (("conversion_tech", "year_dec", "year_inv"), np.zeros((2, 1, 1)))
    ds["conversion_early_decommissioning"] = np.array(False)
    ds["annuity_full_life"] = np.array(False)
    # transport: tech i, link j
    lt = lambda f: np.array([[f(i, j) for i in range(k)] for j in range(2)], dtype="float64")  # noqa: E731
    ends = np.array([[["a", "b"] if (i % 2 == 0) == (j == 0) else ["b", "a"] for i in range(k)] for j in range(2)])
    ds["transport_area_from"] = (("link", "transport_tech"), ends[:, :, 0])
    ds["transport_area_to"] = (("link", "transport_tech"), ends[:, :, 1])
    ds["transport_resource"] = (("link", "transport_tech"), np.full((2, k), "cargo"))
    ds["transport_fixed_cost"] = (("link", "transport_tech", "year_op"), lt(lambda i, j: 0.31 + 0.06 * i + 0.02 * j)[..., None])
    ds["transport_hurdle_costs"] = (("link", "transport_tech", "year_op"), lt(lambda i, j: 0.011 + 0.005 * i + 0.002 * j)[..., None])
    ds["transport_life_span"] = (("link", "transport_tech", "year_inv"), np.full((2, k, 1), 20.0))
    ds["transport_end_of_life"] = (("link", "transport_tech", "year_inv"), np.full((2, k, 1), 2040))
    ds["transport_annuity_cost"] = (("link", "transport_tech", "year_dec", "year_inv"), np.zeros((2, k, 1, 1)))
    ds["transport_early_decommissioning"] = np.array(False)

    def per_tech_grid(values, fill, *dims):
        da = xr.DataArray(
            np.full([ds.sizes[d] for d in dims], fill, dtype="float64"), dims=dims, coords={d: ds[d].values for d in dims}
        )
        for tech, v in values.items():
            da.loc[{"transport_tech": tech}] = v
        return da

    if eff:
        ds["transport_efficiency"] = per_tech_grid(eff, 1.0, "link", "transport_tech", "year_op")
    if ramp:
        ds["transport_ramp_up"] = per_tech_grid({t: v[0] for t, v in ramp.items()}, np.nan, "link", "transport_tech", "year_op")
        ds["transport_ramp_down"] = per_tech_grid({t: v[1] for t, v in ramp.items()}, np.nan, "link", "transport_tech", "year_op")
    if availability:
        ds["transport_availability"] = per_tech_grid(
            {t: np.asarray(v)[None, :, None] for t, v in availability.items()}, np.nan, "link", "transport_tech", "hour", "year_op"
        )
    if fuel:
        ds["transport_fuel_consumption"] = per_tech_grid(fuel, np.nan, "link", "transport_tech", "year_op")
        ds["transport_fuel_resource"] = (
            ("link", "transport_tech"),
            np.where(np.isin(techs, list(fuel))[None, :], "oil", "nan").repeat(2, axis=0),
        )
    if fuel_area:
        ds["transport_fuel_area"] = (
            ("link", "transport_tech"),
            np.array([[fuel_area.get(t, "nan") for t in techs]] * 2),
        )
    if delay:
        ds["transport_delay"] = per_tech_grid(delay, 0.0, "link", "transport_tech")
    if linepack:
        ds["transport_linepack_capacity"] = per_tech_grid(linepack, np.nan, "link", "transport_tech", "year_op")
    if silo:
        ds["storage"] = np.array(True)
        ds = ds.assign_coords(storage_tech=["silo"])
        ds["storage_main_resource"] = ("storage_tech", ["cargo"])
        ds["storage_factor_in"] = (("storage_tech", "resource"), [[0.0, -1.0, 0.0]])
        ds["storage_factor_out"] = (("storage_tech", "resource"), [[0.0, 1.0, 0.0]])
        ds["storage_factor_keep"] = (("storage_tech", "resource"), [[0.0, 0.0, 0.0]])
        ds["storage_dissipation"] = ("storage_tech", [0.0])
        ds["storage_life_span"] = ("storage_tech", [20.0])
        ds["storage_end_of_life"] = (("storage_tech", "year_inv"), [[2040.0]])
        ds["storage_early_decommissioning"] = np.array(False)
        for name in ("storage_annuity_cost_energy", "storage_annuity_cost_power"):
            ds[name] = (("storage_tech", "year_dec", "year_inv"), np.zeros((1, 1, 1)))
        for name, v in (("fixed_cost_energy", 0.2), ("fixed_cost_power", 0.3)):
            ds[f"storage_{name}"] = ("storage_tech", [v])
        for name in ("invest_cost_energy", "invest_cost_power"):
            ds[f"storage_{name}"] = ("storage_tech", [0.0])
    for name, spec in ref_inputs.items():
        if name not in ds.variables:
            ds[name] = np.array(spec["default"], dtype=spec["type"])
    return ds


def _transport_t2(dt, caps, techs=("H", "S"), **kw):
    """
    ``transport_toy`` made non-degenerate for the twin tests: cargo may be shed at a finite cost, and every ship is
    capped (``caps``: tech -> capacity maximum, MW, on every link) below the demand, which is always above 7 MW, so
    that the shedding and the ship capacities bind in every hour; the mill is fixed above its output (no conversion row
    binds). The optimum and the adequacy duals are then unique.
    """
    rate = [x + 6.0 for x in HEAT_RATE]
    p = transport_toy(dt, rate, techs=list(techs), **kw)
    p["load_shedding_max_capacity"] = ("resource", [0.0, 50.0, 0.0])
    p["load_shedding_cost"] = ("resource", [0.0, 4.0, 0.0])
    cap = np.full((2, len(techs), 1), np.nan)
    for tech, value in caps.items():
        cap[:, list(techs).index(tech), :] = value
    p["transport_power_capacity_max"] = (("link", "transport_tech", "year_op"), cap)
    # the mill's capacity is fixed above its output, so that no conversion row binds and the balance duals are unique
    p["conversion_power_capacity_max"] = (("area", "conversion_tech"), [[10.0, np.nan], [0.0, np.nan]])
    p["conversion_power_capacity_min"] = (("area", "conversion_tech"), [[10.0, 0.0], [0.0, 0.0]])
    return p


def transport_readers_dataset() -> xr.Dataset:
    """
    Readers case: ships ``Z, H, S, A`` (not alphabetical), the link ends swapped on the odd ones, every ship capped
    below the demand (the capacities are unique), only the link ``ab`` carries cargo. Hourly: add blocks on some techs.
    """
    p = _transport_t2([1.0] * 48, {"Z": 1.0, "H": 1.5, "S": 3.0, "A": 0.7}, techs=("Z", "H", "S", "A"))
    res = p.transport_resource.copy()
    res.loc[{"link": "ba"}] = "nothing"
    return p.assign(transport_resource=res)


def transport_availability_dual_dataset() -> xr.Dataset:
    """Three hourly steps, one ship ``S`` capped at 3 MW, availability 1, 0.5, 1 (add ``transport={"S": 3}``)."""
    return _transport_t2([1.0] * 3, {"S": 3.0}, techs=("S",), availability={"S": [1.0, 0.5, 1.0]})


def transport_t2_efficiency_dataset() -> xr.Dataset:
    """T2: 0.8 on the link ``ba`` of the block ship ``S`` (a to b) and 1 on ``ab``; the hourly ``H`` has 0.9 on ``ab``."""
    p = _transport_t2([1.0] * 48, {"H": 1.5, "S": 3.0}, eff={"H": 0.9, "S": 0.8})
    e = p["transport_efficiency"].copy()
    e.loc[{"link": "ab", "transport_tech": "S"}] = 1.0
    e.loc[{"link": "ba", "transport_tech": "H"}] = 1.0
    return with_blocks(p.assign(transport_efficiency=e), transport={"S": 24})


def transport_t2_availability_dataset() -> xr.Dataset:
    """T2: the availability of ``S`` is 1 for 20 h of each day, then 0.5, 0.5 and two NaN on day 1, 0.8, 0.8 and two NaN on day 2 (NaN read as 1: block minima 0.5 and 0.8)."""
    day = [1.0] * 20 + [0.5, 0.5, np.nan, np.nan] + [1.0] * 20 + [0.8, 0.8, np.nan, np.nan]
    p = _transport_t2([1.0] * 48, {"H": 1.5, "S": 3.0}, availability={"S": day})
    return with_blocks(p, transport={"S": 24})


def transport_t2_short_last_block_dataset() -> xr.Dataset:
    """T2: 60 h and L = 24, so blocks of 24, 24 and 12 h."""
    return with_blocks(_transport_t2([1.0] * 60, {"H": 1.5, "S": 3.0}), transport={"S": 24})


def transport_t2_two_lengths_dataset() -> xr.Dataset:
    """T2: ships of 12 h (``S``) and 24 h (``T``) and an hourly one (``H``): three groups on one link."""
    p = _transport_t2([1.0] * 48, {"H": 1.0, "S": 1.5, "T": 2.0}, techs=("H", "S", "T"))
    return with_blocks(p, transport={"S": 12, "T": 24})


def transport_t2_mixed_group_costs_dataset() -> xr.Dataset:
    """T2: an hourly (``H``) and a block (``S``) ship on each link: every (area, tech, year) cost cell is defined once."""
    return with_blocks(_transport_t2([1.0] * 48, {"H": 1.5, "S": 3.0}), transport={"S": 24})


def transport_t2_hurdle_dataset() -> xr.Dataset:
    """T2: two operating years, a hurdle cost that differs per link and per year, on the block ship ``S``."""
    p = _transport_t2([1.0] * 48, {"H": 1.5, "S": 3.0})
    p = p.sel(year_op=[2030, 2030]).assign_coords(year_op=[2030, 2035])
    hurdle = p["transport_hurdle_costs"].copy()
    hurdle.loc[{"link": "ba", "transport_tech": "S", "year_op": 2030}] = 0.05
    hurdle.loc[{"link": "ab", "transport_tech": "S", "year_op": 2030}] = 0.02
    hurdle.loc[{"link": "ba", "transport_tech": "S", "year_op": 2035}] = 0.13
    hurdle.loc[{"link": "ab", "transport_tech": "S", "year_op": 2035}] = 0.07
    hurdle.loc[{"link": "ab", "transport_tech": "H", "year_op": 2035}] = 0.09
    return with_blocks(p.assign(transport_hurdle_costs=hurdle), transport={"S": 24})


def transport_t2_ramps_uneven_steps_dataset() -> xr.Dataset:
    """
    T2: steps of 2, 1 and 1 h repeated, L = 4 (each block starts with its long step: the ramp divisor 2 is neither
    4/3 nor 4); ``S`` has availability 1, 0.4, 1, 0.4 per block, efficiency 0.8 and ramps 0.1 (up) and 0.12 (down), so that both
    ramp rows bind, a row that reads the other ramp is seen, and the power is not the power in.
    """
    avail = np.repeat([1.0, 0.4, 1.0, 0.4], 3)
    p = _transport_t2([2.0, 1.0, 1.0] * 4, {"H": 1.5, "S": 3.0}, eff={"S": 0.8}, ramp={"S": (0.1, 0.12)}, availability={"S": avail})
    return with_blocks(p, transport={"S": 4})


def transport_t2_interior_dataset() -> xr.Dataset:
    """
    T2 interior: ships ``S`` (24 h, capped) and ``T`` (12 h, capacity free: its power maximum is slack in some blocks) and an hourly ``H``, a
    demand that varies inside the blocks, steps of 2, 1 and 1 h, availability of ``S`` partly NaN, efficiency below 1,
    60 h (a short last block of 12 h for ``S``).
    """
    dt = [2.0, 1.0, 1.0] * 15
    avail = np.where(np.arange(45) % 5 == 2, np.nan, np.where(np.arange(45) % 7 == 0, 0.8, 1.0))
    p = transport_toy(dt, [x + 1.0 for x in HEAT_RATE], techs=["H", "S", "T"], eff={"S": 0.9, "T": 0.8}, availability={"S": avail})
    p["load_shedding_max_capacity"] = ("resource", [0.0, 50.0, 0.0])
    p["load_shedding_cost"] = ("resource", [0.0, 4.0, 0.0])
    cap = np.full((2, 3, 1), np.nan)
    cap[:, 0], cap[:, 1] = 1.5, 3.2  # H and S are capped, T is not: the three ships carry
    p["transport_power_capacity_max"] = (("link", "transport_tech", "year_op"), cap)
    return with_blocks(p, transport={"S": 24, "T": 12})


def _t2_fuel(dt, caps, fuel, resource="oil", area=None, bunker=(100.0, 0.0), techs=("H", "S"), **kw):
    """
    ``_transport_t2`` with ships that burn fuel: ``fuel`` (tech -> consumption) of ``resource`` (``oil``: the bunker
    conversion makes it, or ``cargo``), drawn at ``area`` (tech -> area, default the "from" end). The bunker has the
    fixed capacity ``bunker`` (MW at ``a``, at ``b``: its fixed cost is then a constant and the oil duals are the
    variable cost), so that the optimum is unique and the fuel is made where it is drawn.
    """
    p = _transport_t2(dt, caps, techs=techs, fuel=fuel, fuel_area=area or {}, **kw)
    p["transport_fuel_resource"] = p.transport_fuel_resource.where(p.transport_fuel_resource != "oil", resource)
    cap = np.array([[10.0, bunker[0]], [0.0, bunker[1]]])
    p["conversion_power_capacity_max"] = (("area", "conversion_tech"), cap)
    p["conversion_power_capacity_min"] = (("area", "conversion_tech"), cap)
    return p


def transport_t2_fuel_from_end_dataset() -> xr.Dataset:
    """T2: the ships burn the cargo itself (``H`` 0.1, block ``S`` 0.2) at the "from" end of each link: area ``a`` (``S`` runs a to b on ``ba``)."""
    p = _t2_fuel([1.0] * 48, {"H": 1.5, "S": 3.0}, {"H": 0.1, "S": 0.2}, resource="cargo")
    return with_blocks(p, transport={"S": 24})


def transport_t2_fuel_area_bunker_dataset() -> xr.Dataset:
    """T2: ``S`` burns oil drawn at area ``b`` (``transport_fuel_area``), the only area whose bunker has a capacity; the cargo goes a to b."""
    p = _t2_fuel([1.0] * 48, {"H": 1.5, "S": 3.0}, {"S": 0.2}, area={"S": "b"}, bunker=(0.0, 100.0))
    return with_blocks(p, transport={"S": 24})


def transport_t2_fuel_other_resource_dataset() -> xr.Dataset:
    """T2: the ships (``H`` 0.1, block ``S`` 0.2) burn oil, a resource other than their cargo, which the hourly bunker makes at ``a``; every resource is hourly."""
    p = _t2_fuel([1.0] * 48, {"H": 1.5, "S": 3.0}, {"H": 0.1, "S": 0.2})
    return with_blocks(p, transport={"S": 24})


def lp_text(m) -> str:
    """The LP file of ``m``, read back as text."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.lp"
        m.to_file(path, io_api="lp")
        return path.read_text()


def assert_t0(p: xr.Dataset) -> None:
    """
    Without block inputs the model has no trace of them (T0 plumbing pin).

    Builds ``p`` without the five parameters and with them all NaN: same LP
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


def with_transport_emissions(p: xr.Dataset, taxed=None, total=None, **carbon) -> xr.Dataset:
    """
    ``p`` with the two emission factors of the fuel burnt by transport and carbon on (``with_carbon``: ``tax``, ``goal``).

    ``taxed`` is ``transport_fuel_emission_factor`` (seen by the carbon tax), ``total`` is
    ``transport_fuel_total_emission_factor`` (seen by the goals): a number for every tech or ``{tech: value}`` (the other
    techs get 0); None leaves the parameter as it is.
    """
    out = {}
    for name, value in (("transport_fuel_emission_factor", taxed), ("transport_fuel_total_emission_factor", total)):
        if value is not None:
            da = xr.zeros_like(p.transport_fixed_cost)
            for tech in p.indexes["transport_tech"]:
                da.loc[{"transport_tech": tech}] = value.get(tech, 0.0) if isinstance(value, dict) else value
            out[name] = da
    return with_carbon(p.assign(out), **carbon)


EMISSION_TAXED, EMISSION_TOTAL = 0.25, 0.4  # tCO2 per MWh of fuel burnt: seen by the tax, seen by the goals


def _fuel_emissions(p: xr.Dataset, **carbon) -> xr.Dataset:
    return with_transport_emissions(p, EMISSION_TAXED, EMISSION_TOTAL, tax=7.0, **carbon)


def transport_t2_fuel_emissions_dataset() -> xr.Dataset:
    """T2: the ships (``H`` 0.1, block ``S`` 0.2) burn oil from the hourly bunker; both factors (0.25 and 0.4), a carbon tax of 7 and a goal of 2000 tCO2 a year in area ``b`` (the goal binds: it is 2628 without it)."""
    p = _fuel_emissions(_t2_fuel([1.0] * 48, {"H": 1.5, "S": 3.0}, {"H": 0.1, "S": 0.2}))
    return with_blocks(p.assign(carbon_goal_area=xr.DataArray([np.nan, 2000.0], dims="area", coords={"area": p.area})), transport={"S": 24})


def transport_t2_fuel_emissions_efficiency_dataset() -> xr.Dataset:
    """T2: as ``t2_fuel_emissions`` without a goal, and the block ship ``S`` has an efficiency of 0.8 (``power_in`` is not ``power``)."""
    p = _fuel_emissions(_t2_fuel([1.0] * 48, {"H": 1.5, "S": 3.0}, {"H": 0.1, "S": 0.2}, eff={"S": 0.8}))
    return with_blocks(p, transport={"S": 24})


def transport_t2_fuel_supplier_emissions_dataset() -> xr.Dataset:
    """T2: as ``t2_fuel_emissions`` without a goal, the hourly ``bunker`` emits 0.3 tCO2 per MWh of oil (the supplier's own factor)."""
    p = _t2_fuel([1.0] * 48, {"H": 1.5, "S": 3.0}, {"H": 0.1, "S": 0.2})
    return with_blocks(with_transport_emissions(p, EMISSION_TAXED, EMISSION_TOTAL, conversion={"bunker": 0.3}, tax=7.0), transport={"S": 24})


def transport_t2_fuel_short_last_block_dataset() -> xr.Dataset:
    """T2: fuel and both emission factors (a tax of 7), 60 h and L = 24 (blocks of 24, 24 and 12 h)."""
    p = _fuel_emissions(_t2_fuel([1.0] * 60, {"H": 1.5, "S": 3.0}, {"H": 0.1, "S": 0.2}))
    return with_blocks(p, transport={"S": 24})


def transport_t2_fuel_uneven_steps_dataset() -> xr.Dataset:
    """T2: fuel and both emission factors (a tax of 7), steps of 2, 1 and 1 h repeated, L = 4 (each block starts with its long step)."""
    p = _fuel_emissions(_t2_fuel([2.0, 1.0, 1.0] * 4, {"H": 1.5, "S": 3.0}, {"H": 0.1, "S": 0.2}))
    return with_blocks(p, transport={"S": 4})


# --- Task 7: whole-model datasets (T1, T2 conditions / multi-horizon / retrofit / ramps, T3) ---


def transport_t1_dataset() -> xr.Dataset:
    """
    T1: steps of 3 h and ``S`` on blocks of 3 h (a block is a step), with efficiency 0.8, fuel (oil from the bunker),
    ramps 0.1 (up and down), a hurdle cost and an availability that alternates 1 and 0.4 (the hourly ``H`` burns too).
    """
    p = _t2_fuel(
        [3.0] * 16,
        {"H": 1.5, "S": 3.0},
        {"H": 0.1, "S": 0.2},
        eff={"S": 0.8},
        ramp={"S": (0.1, 0.1)},
        availability={"S": [1.0, 0.4] * 8},
    )
    return with_blocks(p, transport={"S": 3})


def transport_t2_conditions_dataset() -> xr.Dataset:
    """
    T2: two operating conditions (weights 0.4 and 0.6) with another demand (1 and 1.3 times) and another availability
    of ``S``: block minima 0.5 and 0.8 under ``a``, 0.7 and 0.6 under ``b``; efficiency 0.8 on ``S``.
    """
    day_a = [1.0] * 20 + [0.5, 0.5, np.nan, np.nan] + [1.0] * 20 + [0.8, 0.8, np.nan, np.nan]
    day_b = [1.0] * 8 + [0.7] * 4 + [1.0] * 12 + [1.0] * 6 + [0.6] * 6 + [1.0] * 12
    a = _transport_t2([1.0] * 48, {"H": 1.5, "S": 3.0}, eff={"S": 0.8}, availability={"S": day_a})
    b = _transport_t2([1.0] * 48, {"H": 1.5, "S": 3.0}, eff={"S": 0.8}, availability={"S": day_b})
    p = with_conditions(
        a,
        [0.4, 0.6],
        demand=[a.demand, 1.3 * a.demand],
        transport_availability=[a.transport_availability, b.transport_availability],
    )
    return with_blocks(p, transport={"S": 24})


def _transport_on_years(p: xr.Dataset, years: dict) -> xr.Dataset:
    """
    ``p`` (one year of each kind) on ``years`` (``year_inv``, ``year_dec``, ``year_op``): the transport and conversion
    technologies live 20 years, at an annuity that is distinct and positive per technology, link, decommissioning year
    and vintage. ``year_op`` repeats the first year's data.
    """
    years = {k: np.array(v, dtype="int64") for k, v in years.items()}
    out = p.sel(year_op=[p.year_op.values[0]] * years["year_op"].size).assign_coords(year_op=years["year_op"])
    out = out.drop_vars(
        ["conversion_end_of_life", "conversion_annuity_cost", "transport_life_span", "transport_end_of_life", "transport_annuity_cost"]
    ).drop_vars(["year_inv", "year_dec"]).assign_coords(year_inv=years["year_inv"], year_dec=years["year_dec"])
    for module, dims in (("conversion", ["conversion_tech"]), ("transport", ["link", "transport_tech"])):
        sizes = [out.sizes[d] for d in dims]
        out[f"{module}_end_of_life"] = ((*dims, "year_inv"), np.broadcast_to(years["year_inv"] + 20.0, (*sizes, years["year_inv"].size)).copy())
        k = np.arange(np.prod(sizes)).reshape(sizes)
        # the vintage term is not linear in the year (a linear one makes two vintages cost the same as one of the middle year)
        vintage = np.array([0.0, 0.037, 0.095, 0.16])[: years["year_inv"].size]
        cost = 0.4 + 0.05 * k[..., None, None] + 0.02 * np.arange(years["year_dec"].size)[:, None] + vintage[None, :]
        out[f"{module}_annuity_cost"] = ((*dims, "year_dec", "year_inv"), cost)
    out["transport_life_span"] = (("link", "transport_tech", "year_inv"), np.full((out.sizes["link"], out.sizes["transport_tech"], years["year_inv"].size), 20.0))
    out["conversion_life_span"] = ("conversion_tech", np.full(out.sizes["conversion_tech"], 20.0))
    return out


MULTI_HORIZON_YEARS = dict(year_inv=[2020, 2030, 2040], year_dec=[2040, 2050, 2060], year_op=[2030, 2040])


def transport_t2_multi_horizon_dataset() -> xr.Dataset:
    """
    T2: investment, early decommissioning and two operation years (2030 and 2040).

    The ships are capped at 1.5 (``H``) and 3 (``S``) MW in 2030 and at 1 and 2 MW in 2040, and the 2020 vintage may not
    be built, so that the 2030 vintage serves the first year and part of it leaves in 2040 (early decommissioning:
    year_dec 2040, before the end of its life in 2050), the 2040 vintage completing the second year. The annuities,
    hurdle costs and demand are distinct per technology, link, vintage, decommissioning year and operation year.
    """
    p = _transport_t2([1.0] * 48, {"H": 1.5, "S": 3.0}, eff={"S": 0.8})
    p = _transport_on_years(p, MULTI_HORIZON_YEARS)
    p["transport_early_decommissioning"] = np.array(True)
    cap = p["transport_power_capacity_max"].copy()
    cap.loc[{"year_op": 2040}] = cap.loc[{"year_op": 2040}] * (2.0 / 3.0)
    p["transport_power_capacity_max"] = cap
    invest = xr.full_like(p.transport_annuity_cost.isel(year_dec=0, drop=True), np.nan)
    invest.loc[{"year_inv": 2020}] = 0.0
    p["transport_power_capacity_investment_max"] = invest
    hurdle = p["transport_hurdle_costs"].copy()
    hurdle.loc[{"year_op": 2040}] = hurdle.loc[{"year_op": 2040}] * 1.7
    p["transport_hurdle_costs"] = hurdle
    p["demand"] = p.demand * xr.DataArray([1.0, 0.9], dims="year_op", coords={"year_op": p.year_op})
    return with_blocks(p, transport={"S": 24})


def transport_t2_retrofit_dataset() -> xr.Dataset:
    """
    T2 retrofit: ``H`` (hourly, buildable in 2020 only, decommissioned in 2040) serves 2030; ``S`` (block ship) cannot
    be built and gets its capacity only by retrofit of ``H`` in 2040, the second year of operation (demand grows 1.6
    times). ``S`` runs on the same ends as ``H`` (the retrofit is per link). Early decommissioning off, distinct costs.
    """
    p = _transport_t2([1.0] * 48, {"H": 1.5, "S": 3.0}, eff={"S": 0.8})
    p = _transport_on_years(p, {k: v for k, v in RETROFIT_YEARS.items() if k != "year_ret"})
    p["transport_area_from"] = p.transport_area_from.where(p.transport_tech != "S", p.transport_area_from.sel(transport_tech="H"))
    p["transport_area_to"] = p.transport_area_to.where(p.transport_tech != "S", p.transport_area_to.sel(transport_tech="H"))
    p["demand"] = p.demand * xr.DataArray(RETROFIT_DEMAND_GROWTH, dims="year_op", coords={"year_op": p.year_op})
    invest = xr.full_like(p.transport_annuity_cost.isel(year_dec=0, drop=True), np.nan)
    invest.loc[{"transport_tech": "H", "year_inv": [2030, 2040]}] = 0.0  # H: the 2020 vintage only
    invest.loc[{"transport_tech": "S"}] = 0.0  # S: retrofit only
    p["transport_power_capacity_investment_max"] = invest
    p = _with_retrofit(p, "transport", ("H", "S"))
    return with_blocks(p, transport={"S": 24})


def transport_t2_ramps_short_last_block_dataset() -> xr.Dataset:
    """
    T2: 20 h of steps 2, 1 and 1 h, L = 8 (blocks of 8, 8 and 4 h, each starting with its long step: the ramp divisor
    is 2); ``S`` has efficiency 0.8, availability 1, 0.4 and 1 per block and ramps 0.1 (up) and 0.12 (down): both ramp
    rows bind, also into and out of the short last block.
    """
    avail = np.repeat([1.0, 0.4, 1.0], [6, 6, 3])
    p = _transport_t2([2.0, 1.0, 1.0] * 5, {"H": 1.5, "S": 3.0}, eff={"S": 0.8}, ramp={"S": (0.1, 0.12)}, availability={"S": avail})
    return with_blocks(p, transport={"S": 8})


def transport_all_block_dataset() -> xr.Dataset:
    """The ship ``S`` alone, on 24 h blocks: the model has no hourly transport variable."""
    return with_blocks(_transport_t2([1.0] * 48, {"S": 3.0}, techs=("S",)), transport={"S": 24})


def transport_hand_toy() -> xr.Dataset:
    """
    T3(a): an all-block model by hand (no hourly transport technology), two days of 24 steps of 1 h.

    ``mill`` at ``a`` (24 h blocks, cost 2 per MWh, no fixed cost, no capacity bound); the ship ``S`` a to b (24 h,
    efficiency 0.8, hurdle cost 0.5, no fixed cost); the demand of ``cargo`` at ``b`` is 0 for 12 h and 2 MW for 12 h
    each day; spillage is free, shedding is not allowed; ``operation_year_duration`` is 48 (normalisation 1). Cargo has
    block contributors only, so its balance is daily: per day 24 delivered, 30 injected and 30 produced, 2 * 30 + 0.5 *
    24 = 72, objective 144; hourly balances (the flat twin) give 288.
    """
    p = transport_toy([1.0] * 48, [0.0] * 12 + [2.0] * 12, techs=["S"], eff={"S": 0.8})
    p = p.sel(resource=["idle", "cargo"], conversion_tech=["mill"]).copy(deep=True)
    p["operation_year_duration"] = ("year_op", [48.0])
    p["spillage_cost"] = ("resource", [0.0, 0.0])
    p["load_shedding_max_capacity"] = ("resource", [0.0, 0.0])
    p["load_shedding_cost"] = ("resource", [0.0, 0.0])
    p["conversion_variable_cost"] = ("conversion_tech", [2.0])
    p["conversion_fixed_cost"] = ("conversion_tech", [0.0])
    p["transport_fixed_cost"] = xr.zeros_like(p.transport_fixed_cost)
    p["transport_hurdle_costs"] = xr.full_like(p.transport_hurdle_costs, 0.5)
    return with_blocks(p, conversion={"mill": 24}, transport={"S": 24})


def _t3b_base(**kw) -> xr.Dataset:
    """
    The ship ``S`` alone on a cargo demand that alternates 0.5 and 3.5 MW every 6 h (mean 2): the flow held over a block
    cannot follow it, so a balance summed over the block is cheaper than the hourly ones.
    """
    p = transport_toy([1.0] * 48, [0.5] * 6 + [3.5] * 6, techs=["S"], **kw)
    p["load_shedding_max_capacity"] = ("resource", [0.0, 50.0, 0.0])
    p["load_shedding_cost"] = ("resource", [0.0, 4.0, 0.0])
    return p


def transport_t3b_forced_dataset() -> xr.Dataset:
    """T3(b): ``S`` on 24 h blocks, an hourly ``mill`` and an hourly storage ``silo`` on cargo, no length forced (the test forces 24, 168 and 12 h)."""
    return with_blocks(_t3b_base(silo=True), transport={"S": 24})


def transport_t3b_derived_finer_dataset() -> xr.Dataset:
    """T3(b): ``S`` on 24 h and ``mill`` on 12 h, no storage: the derived balance of cargo is 12 h, finer than the ship."""
    return with_blocks(_t3b_base(), conversion={"mill": 12}, transport={"S": 24})


T3C_AVAILABILITY = [  # per block of 4 h: minima 0.6, 1, 0.8, 0.5, 1, 0.7 (NaN reads as 1)
    *[1.0, 0.6, np.nan, 1.0],
    *[1.0, 1.0, np.nan, np.nan],
    *[0.9, 0.8, 1.0, 0.95],
    *[0.5, 1.0, 1.0, 1.0],
    *[1.0] * 4,
    *[0.7, 0.7, 1.0, np.nan],
]


def transport_t3c_uniform_dataset() -> xr.Dataset:
    """
    T3(c): ship, mill and bunker all on blocks of 4 h, 24 steps of 1 h, no ramp: cargo and oil are derived at 4 h.

    ``S`` has efficiency 0.8, burns oil from the bunker at the "from" end and has an availability that varies inside the
    blocks (``T3C_AVAILABILITY``, partly NaN); the hurdle cost is the toy's. The demand varies from step to step.
    ``coarse_step_reference(p, 4)`` is today's model with steps of 4 h.
    """
    p = _t2_fuel([1.0] * 24, {"S": 3.0}, {"S": 0.2}, techs=("S",), eff={"S": 0.8}, availability={"S": T3C_AVAILABILITY})
    for name in _NO_TRANSPORT_RAMPS:
        p[name] = xr.DataArray(np.nan)
    p = with_blocks(p, conversion={"mill": 4, "bunker": 4}, transport={"S": 4})
    check_coarse_step_contract(p)
    return p


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


def flat_twin(p: xr.Dataset, conversion=None, combined=None, transport=None):
    """
    The flat twin (section 15.1): the hourly model of ``p`` without block
    inputs, plus ``power(h) = power(h-1)`` for every step that is not the
    first of its block, per technology (per unit and mode for combined) of
    ``conversion`` / ``combined`` / ``transport`` (``{tech: length}``; a
    transport tech: per existing pair). Not solved.
    """
    clean = p.drop_vars(BLOCK_PARAMETERS, errors="ignore")
    m = build_model(clean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb = time_blocks(with_blocks(clean, conversion=conversion, combined=combined, transport=transport))
    for module, lengths in (("conversion", conversion), ("combined", combined), ("transport", transport)):
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
            mask = same & (v.labels >= 0) if module == "transport" else same
            m.add_constraints(v - v.shift(hour=1) == 0, mask=mask, name=f"twin_{module}_{tech}")
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
# the transport ramps, NaN and checked on the transport datasets only (``coarse_step_dataset`` NaNs ``_NO_RAMPS`` on
# datasets with no transport module)
_NO_TRANSPORT_RAMPS = ("transport_ramp_up", "transport_ramp_down")
_STEP_SUMS = ("demand", "flexdem_demand")  # energies per step: summed per block
_STEP_MINS = ("transport_availability",)  # per-hour parameters: minimum per block, NaN read as 1 first (section 5)
_STEP_MEANS = (  # per-hour parameters: Δ-weighted mean per block, NaN read as 1 first (section 6.1)
    "conversion_availability",
    "net_import_import_price",
    "net_import_export_price",
    "net_import_emission_factor",
    "net_import_total_emission_factor",
)


def _check_no_ramps(p: xr.Dataset) -> None:
    for name in (*_NO_RAMPS, *_NO_TRANSPORT_RAMPS):
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
    are the Δ-weighted mean per block (``block_mean``, NaN read as 1 first, as the block model does, section 6.1);
    ``transport_availability`` is the minimum per block (NaN read as 1 first). The
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
        elif name in _STEP_MINS:
            out = xr.where(np.isfinite(da), da, 1.0).groupby(tb.block_of_hour[L]).min()
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
    one of the two limits. The objective is unique; the split of the flexible demand between blocks is not (only its
    sum over the blocks is), so tests compare that sum.
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
    margin, the others (steps (0.5, 0.7) and (0.2, 0.4)) the furnace's variable cost, so the coarse prices are 7n, vn,
    7n, vn (n = operation_year_normalization).
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


# --- Transport technologies with modes ---

BLEND = {"blend": {"ch4": 0.90, "h2": 0.10}, "methane": {"ch4": 1.0}}
LEAN = {"lean": {"ch4": 0.90, "h2": 0.03}}
PURE = {"hydrogen": {"h2": 1.0}, "methane": {"ch4": 1.0}}


def with_modes(p, modes: dict, order: list[str] | None = None, ramp: dict | None = None) -> xr.Dataset:
    """
    ``p`` with the coordinate ``transport_mode``, the table ``transport_mode_factor`` and, with ``ramp``, the two ramps.

    ``modes`` is ``{tech: {mode: {resource: factor}}}``; a technology that is not in it has no mode (factor 0). The
    coordinate is ``order`` or else the modes in the order they are first seen. ``ramp`` is ``{tech: (up, down)}``,
    on every link and operation year of the technology (NaN elsewhere).
    """
    seen = list(dict.fromkeys(m for table in modes.values() for m in table))
    labels = seen if order is None else list(order)
    assert set(seen) <= set(labels), (seen, labels)
    techs, resources = list(p.transport_tech.values), list(p.resource.values)
    factor = np.zeros((len(techs), len(labels), len(resources)))
    for tech, table in modes.items():
        for mode, shares in table.items():
            for resource, value in shares.items():
                factor[techs.index(tech), labels.index(mode), resources.index(resource)] = value
    p = p.assign_coords(transport_mode=labels)
    p["transport_mode_factor"] = (("transport_tech", "transport_mode", "resource"), factor)
    for name, k in (("transport_mode_ramp_up", 0), ("transport_mode_ramp_down", 1)):
        if ramp is not None:
            values = np.full((p.sizes["link"], len(techs), p.sizes["year_op"]), np.nan)
            for tech, pair in ramp.items():
                values[:, techs.index(tech), :] = pair[k]
            p[name] = (("link", "transport_tech", "year_op"), values)
    return p


def as_modal(p, techs: list[str]) -> xr.Dataset:
    """``p`` with the technologies ``techs`` modal, one mode ``only`` with the factor 1 on their ``transport_resource`` (6.1)."""
    modes = {}
    for tech in techs:
        carried = [r for r in np.unique(p.transport_resource.sel(transport_tech=tech).values.astype(str)) if r in p.resource.values]
        assert len(carried) == 1, (tech, carried)
        modes[tech] = {"only": {carried[0]: 1.0}}
    return with_modes(p, modes)


def transport_modes_toy(dt=(1.0, 1.0), **kw) -> xr.Dataset:
    """
    The hand toy of section 6.3: a pipe from ``a`` to ``b`` that can carry ``ch4`` and ``h2``, heat to make at ``b``.

    Areas ``a`` and ``b``, one link ``ab``, resources ``idle``, ``h2``, ``ch4``, ``heat``, one year. ``electrolyser``
    makes ``h2`` at 1 EUR/MWh and ``well`` makes ``ch4`` at 5 EUR/MWh, both at ``a`` only; ``boiler_h2`` and
    ``boiler_ch4`` make heat from them 1 to 1 at no cost, at ``b`` only. Every conversion capacity is pinned
    (min = max = 20 where the technology lives, 0 elsewhere), so that S masks units. The heat demand at ``b`` is
    ``demand`` MW times the step; spillage costs 100; no fixed, invest or hurdle cost. The pipe carries ``ch4`` as
    ``transport_resource`` and its capacity is pinned to ``capacity``. ``operation_year_duration`` is ``sum(dt)``.

    Keywords: ``modes`` (a mode table of the pipe, as ``BLEND``; ``{tech: table}`` when there are several ``techs``;
    default None: the plain pipe), ``order`` (the coordinate), ``techs`` (default ``("pipe",)``, all on ``ab``),
    ``eta`` (efficiency), ``capacity`` (MW), ``availability`` (of the electrolyser, per hour), ``ramp`` ((up, down) of
    the mode ramp, for every modal technology), ``shed`` (heat shedding at 100), ``demand`` (MW of heat per step).
    """
    from pommes.model.data_validation import ref_inputs

    modes, order, ramp = kw.pop("modes", None), kw.pop("order", None), kw.pop("ramp", None)
    techs = list(kw.pop("techs", ("pipe",)))
    eta, capacity, availability = kw.pop("eta", 1.0), kw.pop("capacity", 20.0), kw.pop("availability", None)
    shed, demand = kw.pop("shed", False), kw.pop("demand", 10.0)
    assert not kw, kw
    dt = np.array(dt, dtype="float64")
    n, k = len(dt), len(techs)
    resources = ["idle", "h2", "ch4", "heat"]
    ds = xr.Dataset(
        coords=dict(
            area=["a", "b"],
            hour=np.arange(n, dtype="int64"),
            resource=resources,
            conversion_tech=["electrolyser", "well", "boiler_h2", "boiler_ch4"],
            transport_tech=techs,
            link=["ab"],
            year_dec=np.array([2040]),
            year_inv=np.array([2020]),
            year_op=np.array([2030], dtype="int64"),
        )
    )
    ds["conversion"] = True
    ds["transport"] = True
    ds["time_step_duration"] = ("hour", dt)
    ds["operation_year_duration"] = ("year_op", [float(dt.sum())])
    ds["operation_day_duration"] = np.array(24, dtype="int64")
    ds["discount_rate"] = ("year_op", [0.0])
    ds["discount_factor"] = ("year_op", [1.0])
    heat = np.zeros((2, n, 4, 1))
    heat[1, :, 3, 0] = demand * dt
    ds["demand"] = (("area", "hour", "resource", "year_op"), heat)
    ds["load_shedding_max_capacity"] = ("resource", [0.0, 0.0, 0.0, np.nan if shed else 0.0])
    ds["load_shedding_cost"] = ("resource", [0.0, 0.0, 0.0, 100.0])
    ds["spillage_max_capacity"] = ("resource", [np.nan] * 4)
    ds["spillage_cost"] = ("resource", [100.0] * 4)
    ds["conversion_factor"] = (
        ("conversion_tech", "resource"),
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [0.0, 0.0, -1.0, 1.0]],
    )
    ds["conversion_variable_cost"] = ("conversion_tech", [1.0, 5.0, 0.0, 0.0])
    ds["conversion_fixed_cost"] = ("conversion_tech", [0.0] * 4)
    ds["conversion_invest_cost"] = ("conversion_tech", [0.0] * 4)
    ds["conversion_emission_factor"] = ("conversion_tech", [0.0] * 4)
    pinned = [[20.0, 20.0, 0.0, 0.0], [0.0, 0.0, 20.0, 20.0]]  # a: the producers; b: the boilers
    ds["conversion_power_capacity_max"] = (("area", "conversion_tech"), pinned)
    ds["conversion_power_capacity_min"] = (("area", "conversion_tech"), pinned)
    ds["conversion_life_span"] = ("conversion_tech", [20.0] * 4)
    ds["conversion_end_of_life"] = (("conversion_tech", "year_inv"), [[2040.0]] * 4)
    ds["conversion_annuity_cost"] = (("conversion_tech", "year_dec", "year_inv"), np.zeros((4, 1, 1)))
    ds["conversion_early_decommissioning"] = np.array(False)
    ds["annuity_full_life"] = np.array(False)
    if availability is not None:
        values = np.full((2, 4, n, 1), np.nan)
        values[:, 0, :, 0] = np.asarray(availability, dtype="float64")
        ds["conversion_availability"] = (("area", "conversion_tech", "hour", "year_op"), values)
    ds["transport_area_from"] = (("link", "transport_tech"), np.full((1, k), "a"))
    ds["transport_area_to"] = (("link", "transport_tech"), np.full((1, k), "b"))
    ds["transport_resource"] = (("link", "transport_tech"), np.full((1, k), "ch4"))
    for name in ("transport_fixed_cost", "transport_hurdle_costs"):
        ds[name] = (("link", "transport_tech", "year_op"), np.zeros((1, k, 1)))
    ds["transport_power_capacity_min"] = (("link", "transport_tech", "year_op"), np.full((1, k, 1), float(capacity)))
    ds["transport_power_capacity_max"] = (("link", "transport_tech", "year_op"), np.full((1, k, 1), float(capacity)))
    if eta != 1.0:
        ds["transport_efficiency"] = (("link", "transport_tech", "year_op"), np.full((1, k, 1), float(eta)))
    ds["transport_life_span"] = (("link", "transport_tech", "year_inv"), np.full((1, k, 1), 20.0))
    ds["transport_end_of_life"] = (("link", "transport_tech", "year_inv"), np.full((1, k, 1), 2040))
    ds["transport_annuity_cost"] = (("link", "transport_tech", "year_dec", "year_inv"), np.zeros((1, k, 1, 1)))
    ds["transport_early_decommissioning"] = np.array(False)
    if modes is not None:
        modes = {techs[0]: modes} if k == 1 else modes
        ds = with_modes(ds, modes, order, None if ramp is None else {t: ramp for t in modes})
    for name, spec in ref_inputs.items():
        if name not in ds.variables:
            ds[name] = np.array(spec["default"], dtype=spec["type"])
    return ds
