"""Synthetic N-area, N-hour dataset: a configurable number of areas and of hours.

Deterministic (no randomness) and built with NumPy and xarray only. Used by
test_benchmark_3nodes.py (a grid of n_areas x n_hours), test_flexdem.py and
test_time_steps.py.

Tech-pool sizes (3 common + 5 restricted conversion techs, 1 common + 2
restricted storage techs, 2 restricted combined techs) are kept FIXED
regardless of n_areas -- restricted techs are assigned to areas
round-robin (area i gets restricted-pool tech ``i % pool_size``), so more
than one area can share a given restricted tech's name, each only with
capacity in its own row. This keeps variable/constraint counts scaling
close to linearly with n_areas (dominated by the *common* techs, which are
genuinely dense across all areas) rather than also growing the tech
dimension itself -- important to keep the larger grid cells (7 areas x
8760h) from blowing up in size/RAM.

Restricted (area, tech) pairs get an explicit 0 (not NaN) capacity at
every *other* area, matching active_area_tech_mask's documented
assumption (a genuine hard exclusion, not "NaN happens to never be
built").

Every unit lives 20 years, so the vintages 2020 and 2030 end their life in 2040
and 2050, on the ``year_dec`` grid: with ``early_decommissioning`` off a unit
exists only if its end of life is a ``year_dec``. The optimum builds conversion,
storage and combined capacity. Before October 2026 the life spans were 25 years
(15 to 40 for storage), whose ends of life (2045, 2055, ...) fell on no
``year_dec``: nothing could be built and the optimum shed all the demand, so
the solve times of the benchmark grids run on that version measure a trivial
problem (the solver's presolve ends it), not the model.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

# life span of every unit, in years: the vintages 2020 and 2030 then end on the year_dec 2040 and 2050
LIFE_SPAN = 20.0
# cost of spilling one MWh of electricity
ELECTRICITY_SPILLAGE_COST = 1.0


def _annuity_multiplier(n_dec: int, n_inv: int) -> np.ndarray:
    """A simple, deterministic (year_dec, year_inv) annuity-share matrix."""
    shape = max(n_dec, n_inv + 1)
    mat = np.full((shape, shape), np.nan)
    for offset, value in {0: 1 / 20, 1: 1 / 12, 2: 1 / 8}.items():
        for i in range(shape - offset):
            mat[i + offset, i] = value
    return mat[:n_dec, 1 : n_inv + 1]


def build_nnode_parameters(n_areas: int = 3, n_hours: int = 365) -> xr.Dataset:
    area = np.array([f"N{i + 1}" for i in range(n_areas)], dtype=str)
    hour = np.arange(n_hours, dtype="int64")
    resource = np.array(["electricity", "heat", "methane"], dtype=str)
    mode = np.array(["m1", "m2"], dtype=str)

    common_conv = ["wind_onshore", "solar_pv", "ocgt"]
    restricted_conv = ["biomass", "electrolysis", "chp_waste", "hydro_ror", "coal"]
    conversion_tech = np.array(common_conv + restricted_conv, dtype=str)
    n_conv = len(conversion_tech)

    restricted_storage = ["phs", "h2_tank"]
    storage_tech = np.array(["battery"] + restricted_storage, dtype=str)
    n_storage = len(storage_tech)

    combined_tech = np.array(["chp_gas", "chp_biomass"], dtype=str)
    n_combined = len(combined_tech)

    year_dec = np.array([2030, 2040, 2050], dtype="int64")
    year_inv = np.array([2020, 2030], dtype="int64")
    year_op = np.array([2030], dtype="int64")

    coords = xr.Coordinates(
        {
            "area": area,
            "hour": hour,
            "resource": resource,
            "mode": mode,
            "conversion_tech": conversion_tech,
            "storage_tech": storage_tech,
            "combined_tech": combined_tech,
            "year_dec": year_dec,
            "year_inv": year_inv,
            "year_op": year_op,
        }
    )
    ds = xr.Dataset(coords=coords)

    n_dec, n_inv = len(year_dec), len(year_inv)
    annuity_multiplier = _annuity_multiplier(n_dec, n_inv)

    # ------------------------------------------------------------------
    # Shared / time / eco
    # ------------------------------------------------------------------
    ds["conversion"] = True
    ds["storage"] = True
    ds["combined"] = True
    ds["carbon"] = True

    ds["time_step_duration"] = ("hour", np.ones(n_hours, dtype="float64"))
    ds["operation_year_duration"] = ("year_op", np.array([float(n_hours)]))
    ds["operation_day_duration"] = np.array(24, dtype="int64")
    ds["discount_rate"] = ("year_op", np.zeros(1))
    ds["discount_factor"] = ("year_op", np.ones(1))
    ds["carbon_tax"] = np.array(50.0)
    ds["carbon_goal"] = np.array(np.nan)
    ds["carbon_goal_area"] = np.array(np.nan)

    hr = hour.astype("float64")
    base = {"electricity": 120.0, "heat": 60.0, "methane": 20.0}
    amp = {"electricity": 40.0, "heat": 30.0, "methane": 5.0}
    demand = np.empty((n_areas, n_hours, 3, 1))
    for ai in range(n_areas):
        phase = 0.6 * ai
        for ri, r in enumerate(resource):
            demand[ai, :, ri, 0] = base[r] + amp[r] * (
                0.5 + 0.5 * np.sin(2 * np.pi * hr / 24 + phase)
            )
    ds["demand"] = (("area", "hour", "resource", "year_op"), demand)

    ds["load_shedding_max_capacity"] = ("resource", np.full(3, np.nan))
    # methane is bought by shedding it, at 20 per MWh: the fuel of the combined units
    ds["load_shedding_cost"] = ("resource", np.array([3000.0, 1000.0, 20.0]))
    ds["spillage_max_capacity"] = ("resource", np.full(3, np.nan))
    # spilling electricity costs a little, so that curtailing wind and solar is the only optimum (an optimum that
    # produces more and spills it would otherwise cost the same)
    ds["spillage_cost"] = ("resource", np.array([ELECTRICITY_SPILLAGE_COST, 0.0, 0.0]))

    # ------------------------------------------------------------------
    # Conversion: common techs (wind/solar/ocgt) in every area, plus a
    # fixed 5-tech restricted pool assigned round-robin across areas.
    # ------------------------------------------------------------------
    cap_max = np.zeros((n_areas, n_conv))
    cap_min = np.zeros((n_areas, n_conv))
    common_cap = {"wind_onshore": 150.0, "solar_pv": 120.0, "ocgt": 200.0}
    for ti, tech in enumerate(conversion_tech):
        if tech in common_cap:
            cap_max[:, ti] = common_cap[tech]
        else:
            ri = restricted_conv.index(tech)
            for ai in range(n_areas):
                if ai % len(restricted_conv) == ri:
                    cap_max[ai, ti] = 100.0
    ds["conversion_power_capacity_max"] = (("area", "conversion_tech"), cap_max)
    ds["conversion_power_capacity_min"] = (("area", "conversion_tech"), cap_min)

    ds["conversion_early_decommissioning"] = np.array(False)
    ds["conversion_annuity_perfect_foresight"] = np.array(False)
    ds["conversion_must_run"] = ("conversion_tech", np.zeros(n_conv))
    ds["conversion_finance_rate"] = ("conversion_tech", np.full(n_conv, 0.05))
    ds["conversion_life_span"] = ("conversion_tech", np.full(n_conv, LIFE_SPAN))
    ds["conversion_power_capacity_investment_max"] = (
        "conversion_tech",
        np.full(n_conv, np.nan),
    )
    ds["conversion_power_capacity_investment_min"] = (
        "conversion_tech",
        np.full(n_conv, np.nan),
    )
    ds["conversion_max_yearly_production"] = (
        ("area", "conversion_tech", "year_op"),
        np.full((n_areas, n_conv, 1), np.nan),
    )
    ds["conversion_max_daily_production"] = (
        ("area", "conversion_tech", "year_op"),
        np.full((n_areas, n_conv, 1), np.nan),
    )
    ds["conversion_ramp_up"] = ("conversion_tech", np.full(n_conv, np.nan))
    ds["conversion_ramp_down"] = ("conversion_tech", np.full(n_conv, np.nan))
    ds["conversion_max_capacity_factor"] = np.array(np.nan)  # off (parameter added in pommes 0.2.0)

    var_cost = {"wind_onshore": 0, "solar_pv": 0, "ocgt": 45, "biomass": 20,
                "electrolysis": 5, "chp_waste": 10, "hydro_ror": 0, "coal": 30}
    fixed_cost = {"wind_onshore": 25, "solar_pv": 15, "ocgt": 8, "biomass": 30,
                  "electrolysis": 12, "chp_waste": 20, "hydro_ror": 18, "coal": 22}
    invest_cost = {"wind_onshore": 1200, "solar_pv": 700, "ocgt": 500, "biomass": 2500,
                   "electrolysis": 900, "chp_waste": 1800, "hydro_ror": 2000, "coal": 1500}
    emission_factor = {"wind_onshore": 0, "solar_pv": 0, "ocgt": 0.4, "biomass": 0.05,
                       "electrolysis": 0, "chp_waste": 0.1, "hydro_ror": 0, "coal": 0.9}
    ds["conversion_variable_cost"] = (
        "conversion_tech",
        np.array([var_cost[t] for t in conversion_tech], dtype="float64"),
    )
    ds["conversion_fixed_cost"] = (
        "conversion_tech",
        np.array([fixed_cost[t] for t in conversion_tech], dtype="float64"),
    )
    ds["conversion_invest_cost"] = (
        "conversion_tech",
        np.array([invest_cost[t] for t in conversion_tech], dtype="float64"),
    )
    ds["conversion_emission_factor"] = (
        "conversion_tech",
        np.array([emission_factor[t] for t in conversion_tech], dtype="float64"),
    )

    conv_factor = np.zeros((n_conv, 3))
    conv_factor[:, 0] = 1.0  # every conversion tech produces electricity
    ds["conversion_factor"] = (("conversion_tech", "resource"), conv_factor)

    conv_avail = np.full((n_conv, n_hours), np.nan)
    solar = np.clip(np.sin(2 * np.pi * hr / 24 - np.pi / 2), 0, None)
    wind = 0.35 + 0.3 * np.sin(2 * np.pi * hr / 60) + 0.1 * np.sin(2 * np.pi * hr / 11)
    wind = np.clip(wind, 0.02, 0.95)
    conv_avail[list(conversion_tech).index("wind_onshore")] = wind
    conv_avail[list(conversion_tech).index("solar_pv")] = solar
    ds["conversion_availability"] = (("conversion_tech", "hour"), conv_avail)

    ds["conversion_end_of_life"] = (
        ("conversion_tech", "year_inv"),
        ds.conversion_life_span.to_numpy()[:, None] + ds.year_inv.to_numpy()[None, :],
    )
    ds["conversion_annuity_cost"] = (
        ("conversion_tech", "year_dec", "year_inv"),
        ds.conversion_invest_cost.to_numpy()[:, None, None]
        * annuity_multiplier[None, :, :],
    )

    # ------------------------------------------------------------------
    # Storage: "battery" common to every area, plus a fixed 2-tech
    # restricted pool assigned round-robin across areas.
    # ------------------------------------------------------------------
    p_inv_max = np.zeros((n_areas, n_storage, n_inv))
    p_inv_min = np.zeros((n_areas, n_storage, n_inv))
    e_inv_max = np.zeros((n_areas, n_storage, n_inv))
    e_inv_min = np.zeros((n_areas, n_storage, n_inv))
    for ti, tech in enumerate(storage_tech):
        if tech == "battery":
            p_inv_max[:, ti, :] = 60.0
            e_inv_max[:, ti, :] = 240.0
        else:
            ri = restricted_storage.index(tech)
            for ai in range(n_areas):
                if ai % len(restricted_storage) == ri:
                    p_inv_max[ai, ti, :] = 80.0
                    e_inv_max[ai, ti, :] = 500.0
    ds["storage_power_capacity_investment_max"] = (
        ("area", "storage_tech", "year_inv"),
        p_inv_max,
    )
    ds["storage_power_capacity_investment_min"] = (
        ("area", "storage_tech", "year_inv"),
        p_inv_min,
    )
    ds["storage_energy_capacity_investment_max"] = (
        ("area", "storage_tech", "year_inv"),
        e_inv_max,
    )
    ds["storage_energy_capacity_investment_min"] = (
        ("area", "storage_tech", "year_inv"),
        e_inv_min,
    )
    ds["storage_energy_power_ratio"] = np.array(np.nan)
    ds["storage_power_capacity_max"] = np.array(np.nan)
    ds["storage_power_capacity_min"] = np.array(np.nan)
    ds["storage_energy_capacity_max"] = np.array(np.nan)
    ds["storage_energy_capacity_min"] = np.array(np.nan)

    ds["storage_early_decommissioning"] = np.array(False)
    ds["storage_annuity_perfect_foresight"] = np.array(False)
    ds["storage_finance_rate"] = ("storage_tech", np.full(n_storage, 0.05))
    ds["storage_life_span"] = ("storage_tech", np.full(n_storage, LIFE_SPAN))
    ds["storage_fixed_cost_power"] = ("storage_tech", np.array([5.0, 8.0, 4.0]))
    ds["storage_fixed_cost_energy"] = ("storage_tech", np.array([2.0, 1.0, 3.0]))
    ds["storage_invest_cost_power"] = ("storage_tech", np.array([300.0, 900.0, 400.0]))
    ds["storage_invest_cost_energy"] = ("storage_tech", np.array([150.0, 60.0, 250.0]))
    ds["storage_dissipation"] = ("storage_tech", np.array([0.02, 0.0, 0.05]))
    ds["storage_main_resource"] = (
        "storage_tech",
        np.array(["electricity", "electricity", "hydrogen"], dtype=str),
    )
    ds["storage_charge_ramp"] = ("storage_tech", np.full(n_storage, np.nan))
    ds["storage_discharge_ramp"] = ("storage_tech", np.full(n_storage, np.nan))

    factor_in = np.zeros((n_storage, 3))
    factor_out = np.zeros((n_storage, 3))
    factor_keep = np.zeros((n_storage, 3))
    eff_in = {"battery": -1.05, "phs": -1.1, "h2_tank": -1.3}
    eff_out = {"battery": 0.95, "phs": 0.85, "h2_tank": 0.6}
    for ti, tech in enumerate(storage_tech):
        factor_in[ti, 0] = eff_in[tech]
        factor_out[ti, 0] = eff_out[tech]
    ds["storage_factor_in"] = (("storage_tech", "resource"), factor_in)
    ds["storage_factor_out"] = (("storage_tech", "resource"), factor_out)
    ds["storage_factor_keep"] = (("storage_tech", "resource"), factor_keep)

    ds["storage_end_of_life"] = (
        ("storage_tech", "year_inv"),
        ds.storage_life_span.to_numpy()[:, None] + ds.year_inv.to_numpy()[None, :],
    )
    ds["storage_annuity_cost_power"] = (
        ("storage_tech", "year_dec", "year_inv"),
        ds.storage_invest_cost_power.to_numpy()[:, None, None]
        * annuity_multiplier[None, :, :],
    )
    ds["storage_annuity_cost_energy"] = (
        ("storage_tech", "year_dec", "year_inv"),
        ds.storage_invest_cost_energy.to_numpy()[:, None, None]
        * annuity_multiplier[None, :, :],
    )

    # ------------------------------------------------------------------
    # Combined: 2-tech restricted pool assigned round-robin across areas
    # (no common combined tech, matching the original 3-node dataset).
    # ------------------------------------------------------------------
    comb_cap_max = np.zeros((n_areas, n_combined))
    comb_cap_min = np.zeros((n_areas, n_combined))
    for ti in range(n_combined):
        for ai in range(n_areas):
            if ai % n_combined == ti:
                comb_cap_max[ai, ti] = 90.0
    ds["combined_power_capacity_max"] = (("area", "combined_tech"), comb_cap_max)
    ds["combined_power_capacity_min"] = (("area", "combined_tech"), comb_cap_min)

    ds["combined_early_decommissioning"] = np.array(False)
    ds["combined_annuity_perfect_foresight"] = np.array(False)
    ds["combined_must_run"] = np.array(np.nan)
    ds["combined_max_load"] = np.array(1.0)
    ds["combined_min_load"] = np.array(0.0)
    ds["combined_ramp_up"] = ("combined_tech", np.full(n_combined, np.nan))
    ds["combined_ramp_down"] = ("combined_tech", np.full(n_combined, np.nan))
    ds["combined_mode_ramp_up"] = np.array(np.nan)  # off (mode-shift ramps, opt-in)
    ds["combined_mode_ramp_down"] = np.array(np.nan)
    ds["combined_finance_rate"] = ("combined_tech", np.full(n_combined, 0.05))
    ds["combined_life_span"] = ("combined_tech", np.full(n_combined, LIFE_SPAN))
    ds["combined_variable_cost"] = ("combined_tech", np.array([15.0, 22.0]))
    ds["combined_fixed_cost"] = ("combined_tech", np.array([18.0, 24.0]))
    ds["combined_invest_cost"] = ("combined_tech", np.array([1400.0, 1900.0]))
    ds["combined_power_capacity_investment_max"] = (
        "combined_tech",
        np.full(n_combined, np.nan),
    )
    ds["combined_power_capacity_investment_min"] = (
        "combined_tech",
        np.full(n_combined, np.nan),
    )
    ds["combined_emission_factor"] = (
        ("combined_tech", "mode"),
        np.array([[0.35, 0.35], [0.05, 0.05]]),
    )
    combined_factor = np.zeros((n_combined, 2, 3))
    combined_factor[:, 0, 0] = 1.0
    combined_factor[:, 0, 2] = -1.7
    combined_factor[:, 1, 0] = 1.0
    combined_factor[:, 1, 1] = 0.6
    combined_factor[:, 1, 2] = -2.2
    ds["combined_factor"] = (("combined_tech", "mode", "resource"), combined_factor)

    ds["combined_end_of_life"] = (
        ("combined_tech", "year_inv"),
        ds.combined_life_span.to_numpy()[:, None] + ds.year_inv.to_numpy()[None, :],
    )
    ds["combined_annuity_cost"] = (
        ("combined_tech", "year_dec", "year_inv"),
        ds.combined_invest_cost.to_numpy()[:, None, None]
        * annuity_multiplier[None, :, :],
    )

    # ------------------------------------------------------------------
    # Other modules left disabled -- not needed for this benchmark.
    # ------------------------------------------------------------------
    for module in ["process", "transport", "net_import", "flexdem", "turpe", "retrofit"]:
        ds[module] = False

    return ds


if __name__ == "__main__":
    ds = build_nnode_parameters(n_areas=7, n_hours=8760)
    print(ds)
    print()
    print("sizes:", dict(ds.sizes))
