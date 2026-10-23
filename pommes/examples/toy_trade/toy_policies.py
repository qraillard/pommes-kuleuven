"""Policies on the toy trade model (brief section 12, step 4): CBAM, EU ETS maritime, fixed point on the EU carbon price.

Same three nodes as toy_trade.py (EU, NAF, AUS), one representative hour for 2030. The EU needs hydrogen, which it makes
itself (SMR or electrolysis: the source of its own emissions), and ammonia. Ammonia can be made
green (electrolysis) or grey (steam methane reforming, 0.28 tCO2 per MWh of hydrogen), in the EU or abroad, and shipped
as two commodities, `ammonia_green` and `ammonia_grey`, so that the CBAM charge stays linear (brief 5.4: one tradable
variant per emission-intensity class). Ships burn fuel oil bought at the origin.

Policies, all linear:
  EU carbon price     `carbon_tax` on emissions in the EU (exogenous case), or an EU cap `carbon_goal_area` whose dual is
                      the price (endogenous case).
  CBAM                on each ship link carrying grey ammonia: embedded intensity x (EU price - origin price) x phase-in,
                      as a hurdle cost per MWh delivered.
  EU ETS maritime     on voyages into the EU: 50 % of the ship's fuel emissions x EU price, as a hurdle cost per MWh.
With a cap, the CBAM charge needs the price the cap produces: `fixed_point` solves, reads the cap's dual, resets the
hurdles and solves again until the price settles.

Hurdle costs are EUR per MWh carried; pommes-kul weights them by the year normalisation like the other operating costs,
so the single modelled hour counts for the whole year without any scaling here.

Run: python pommes/examples/toy_trade/toy_policies.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from pommes.io.build_input_dataset import compute_annuity_cost, compute_discount, compute_end_of_life
from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs

HOURS = 8760.0
YEAR, RATE = 2030, 0.07
AREAS = ["AUS", "EU", "NAF"]
RESOURCES = ["ammonia", "ammonia_green", "ammonia_grey", "electricity", "fuel_oil", "hydrogen", "hydrogen_final",
             "hydrogen_grey", "methane"]
LINKS = {"AUS_EU": ("AUS", "EU"), "NAF_EU": ("NAF", "EU")}
SHIPS = {"ship_green": "ammonia_green", "ship_grey": "ammonia_grey"}
BUNKER = {"NAF_EU": 0.013, "AUS_EU": 0.075}                  # MWh fuel per MWh ammonia (World Trade, Rotterdam-Arzew / Port Hedland)
SHIP_CAPEX = {"NAF_EU": 95.1, "AUS_EU": 390.5}               # EUR per kW of ammonia flow, FOM 4 %, 20 years
SMR_EF, FUEL_OIL_EF = 0.28, 0.279                            # tCO2 per MWh of hydrogen (POMMES-EU) / per MWh of fuel oil
H2_PER_NH3 = 1.1484
GREY_INTENSITY = SMR_EF * H2_PER_NH3                         # tCO2 per MWh of grey ammonia (direct emissions)
DEMAND_NH3 = 500.0                                           # MW in the EU
DEMAND_H2 = 1000.0                                           # MW in the EU, made there (hydrogen is not shipped in this toy)
SOLAR_CF = {"EU": 0.12, "NAF": 0.24, "AUS": 0.26}
SOLAR_CAPEX = {"EU": (466.0, 11.65), "NAF": (758.0, 9.3), "AUS": (778.0, 9.9)}
GAS_PRICE = {"EU": 35.0, "NAF": 20.0, "AUS": 25.0}           # EUR/MWh, illustrative
FUEL_OIL_PRICE = 40.0

# tech: (factors, capex EUR/kW, FOM EUR/kW/yr, life, emission factor, areas allowed)
TECHS = {
    "solar_pv": ({"electricity": 1.0}, None, None, 40, 0.0, AREAS),
    "electrolyser": ({"hydrogen": 1.0, "electricity": -1 / 0.69}, 1451 / 0.69, 58.06 / 0.69, 25, 0.0, AREAS),
    "smr": ({"hydrogen_grey": 1.0, "methane": -1.31, "electricity": -0.017}, 850.0, 144.0, 25, SMR_EF, AREAS),
    "hb_green": ({"ammonia_green": 1.0, "hydrogen": -H2_PER_NH3, "electricity": -0.2473}, 1294 * H2_PER_NH3, 1294 * H2_PER_NH3 * 0.029, 30, 0.0, AREAS),
    "hb_grey": ({"ammonia_grey": 1.0, "hydrogen_grey": -H2_PER_NH3, "electricity": -0.2473}, 1294 * H2_PER_NH3, 1294 * H2_PER_NH3 * 0.029, 30, 0.0, AREAS),
    "gas_supply": ({"methane": 1.0}, 0.0, 0.0, 40, 0.0, AREAS),
    "fuel_oil_supply": ({"fuel_oil": 1.0}, 0.0, 0.0, 40, FUEL_OIL_EF, ["AUS", "NAF"]),
    "use_green": ({"ammonia": 1.0, "ammonia_green": -1.0}, 0.0, 0.0, 40, 0.0, ["EU"]),
    "use_grey": ({"ammonia": 1.0, "ammonia_grey": -1.0}, 0.0, 0.0, 40, 0.0, ["EU"]),
    "h2_use_green": ({"hydrogen_final": 1.0, "hydrogen": -1.0}, 0.0, 0.0, 40, 0.0, ["EU"]),
    "h2_use_grey": ({"hydrogen_final": 1.0, "hydrogen_grey": -1.0}, 0.0, 0.0, 40, 0.0, ["EU"]),
}


def build(eu_price: float | None = 100.0, eu_cap: float | None = None, cbam_price: float | None = None,
          cbam_phase_in: float = 1.0, ets_maritime: bool = True) -> xr.Dataset:
    """eu_price: carbon tax in the EU (EUR/t) or None with eu_cap (tCO2 a year); cbam_price: EU price used for CBAM."""
    ds = xr.Dataset(coords=dict(
        area=np.array(AREAS, dtype=str), resource=np.array(RESOURCES, dtype=str),
        conversion_tech=np.array(list(TECHS), dtype=str), transport_tech=np.array(list(SHIPS), dtype=str),
        link=np.array(list(LINKS), dtype=str), hour=np.array([0], dtype="int64"), year_op=np.array([YEAR], dtype="int64"),
        year_inv=np.array([YEAR], dtype="int64"), year_dec=np.arange(2035, 2085, 5, dtype="int64")))
    ds = ds.assign(conversion=True, transport=True, carbon=True, combined=False, storage=False, net_import=False,
                   turpe=False, process=False, flexdem=False)

    def da(dims, fill=np.nan):
        return xr.DataArray(np.full([ds.sizes[d] for d in dims], fill, dtype="float64"), coords={d: ds[d] for d in dims})

    demand = da(["area", "resource"], 0.0)
    demand.loc[dict(area="EU", resource="ammonia")] = DEMAND_NH3
    demand.loc[dict(area="EU", resource="hydrogen_final")] = DEMAND_H2
    factor, life, ef = da(["conversion_tech", "resource"], 0.0), da(["conversion_tech"]), da(["conversion_tech"], 0.0)
    invest, fixed, var = da(["area", "conversion_tech"], 0.0), da(["area", "conversion_tech"], 0.0), da(["area", "conversion_tech", "year_op"], 0.0)
    invest_max = da(["area", "conversion_tech", "year_inv"])
    for tech, (f, capex, fom, lifetime, e, areas) in TECHS.items():
        for r, v in f.items():
            factor.loc[dict(conversion_tech=tech, resource=r)] = v
        life.loc[dict(conversion_tech=tech)], ef.loc[dict(conversion_tech=tech)] = lifetime, e
        for a in AREAS:
            c, o = SOLAR_CAPEX[a] if tech == "solar_pv" else (capex, fom)
            invest.loc[dict(area=a, conversion_tech=tech)] = c * 1000
            fixed.loc[dict(area=a, conversion_tech=tech)] = o * 1000
            if a not in areas:
                invest_max.loc[dict(area=a, conversion_tech=tech)] = 0.0
        if tech == "gas_supply":
            for a, g in GAS_PRICE.items():
                var.loc[dict(area=a, conversion_tech=tech)] = g
        if tech == "fuel_oil_supply":
            var.loc[dict(conversion_tech=tech)] = FUEL_OIL_PRICE
    availability = da(["area", "conversion_tech", "hour", "year_op"])
    for a, cf in SOLAR_CF.items():
        availability.loc[dict(area=a, conversion_tech="solar_pv")] = cf

    tax = da(["area", "year_op"], 0.0)
    if eu_price is not None:
        tax.loc[dict(area="EU")] = eu_price
    cap = da(["area", "year_op"])
    if eu_cap is not None:
        cap.loc[dict(area="EU")] = eu_cap
    p_cbam = eu_price if cbam_price is None else cbam_price
    hurdle = da(["link", "transport_tech", "year_op"], 0.0)
    for link in LINKS:
        maritime = 0.5 * FUEL_OIL_EF * BUNKER[link] * p_cbam if ets_maritime else 0.0
        hurdle.loc[dict(link=link)] = maritime
        hurdle.loc[dict(link=link, transport_tech="ship_grey")] += GREY_INTENSITY * p_cbam * cbam_phase_in
    ship_res = xr.DataArray(np.array([list(SHIPS.values())] * len(LINKS), dtype=str), coords=dict(link=ds.link, transport_tech=ds.transport_tech))
    capex = xr.DataArray([SHIP_CAPEX[l] * 1000 for l in LINKS], coords=dict(link=ds.link))
    fuel = xr.DataArray([BUNKER[l] for l in LINKS], coords=dict(link=ds.link))
    ds = ds.assign(
        demand=demand, load_shedding_cost=xr.DataArray(1e5), spillage_cost=xr.DataArray(0.0),
        time_step_duration=xr.DataArray([1.0], coords=dict(hour=ds.hour)), operation_year_duration=xr.DataArray(HOURS),
        discount_rate=xr.DataArray(RATE), year_ref=xr.DataArray(YEAR),
        conversion_factor=factor, conversion_invest_cost=invest, conversion_fixed_cost=fixed, conversion_variable_cost=var,
        conversion_life_span=life, conversion_finance_rate=xr.DataArray(RATE), conversion_availability=availability,
        conversion_power_capacity_investment_max=invest_max, conversion_emission_factor=ef,
        carbon_tax=tax, carbon_goal_area=cap, carbon_goal=xr.DataArray(np.nan),
        transport_area_from=xr.DataArray([LINKS[l][0] for l in LINKS], coords=dict(link=ds.link)).astype(str),
        transport_area_to=xr.DataArray([LINKS[l][1] for l in LINKS], coords=dict(link=ds.link)).astype(str),
        transport_resource=ship_res, transport_invest_cost=capex.broadcast_like(ship_res).astype("float64"),
        transport_fixed_cost=(capex * 0.04).broadcast_like(ship_res).astype("float64"), transport_life_span=xr.DataArray(20.0),
        transport_finance_rate=xr.DataArray(RATE), transport_hurdle_costs=hurdle,
        transport_fuel_consumption=fuel.broadcast_like(ship_res).astype("float64"),
        transport_fuel_resource=xr.DataArray("fuel_oil"),
    )
    config = {"pre_process": {
        "annuity_computation": {
            "conversion": {"conversion_annuity_cost": {"invest_cost": "conversion_invest_cost", "finance_rate": "conversion_finance_rate", "life_span": "conversion_life_span"}},
            "transport": {"transport_annuity_cost": {"invest_cost": "transport_invest_cost", "finance_rate": "transport_finance_rate", "life_span": "transport_life_span"}}},
        "discount_factor": {"discount_rate": "discount_rate", "year_ref": "year_ref"}}}
    return check_inputs(compute_discount(compute_annuity_cost(compute_end_of_life(ds), config), config))


def solve(ds: xr.Dataset) -> dict:
    model = build_model(ds)
    _, condition = model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True)
    shipped = s.operation_transport_power.to_pandas()
    # pommes-milp keeps the dense (area, conversion_tech) dimensions (pommes-kul: one entry per sparse unit)
    use = s.operation_conversion_power.transpose("area", "conversion_tech").to_pandas()
    production = use.stack().rename_axis(["area", "tech"])
    price = None
    if "operation_carbon_goal_area_constraint" in model.constraints:
        dual = model.dual["operation_carbon_goal_area_constraint"].squeeze(drop=True)
        # the cap constrains a year of emissions (t) in the annual objective (EUR): its dual is EUR/t
        price = float(-dual.sel(area="EU")) if "area" in dual.dims else float(-dual)
    emissions = (s.operation_total_carbon_emissions * HOURS).to_pandas()
    return {"condition": condition, "shipped": shipped, "production": production, "eu_price_from_cap": price,
            "emissions": emissions, "objective": model.objective.value}


def fixed_point(eu_cap: float, start: float = 50.0, tol: float = 0.1, max_iter: int = 30) -> tuple[float, list[float], dict]:
    """Solve with the cap, set CBAM and ETS maritime on the cap's price, repeat until the price moves less than tol."""
    price, history = start, [start]
    for _ in range(max_iter):
        r = solve(build(eu_price=None, eu_cap=eu_cap, cbam_price=price))
        new = r["eu_price_from_cap"]
        history.append(new)
        if abs(new - price) < tol:
            return new, history, r
        price = new
    raise RuntimeError(f"no convergence: {history}")


def main() -> None:
    for label, kwargs in (("no carbon price", dict(eu_price=0.0)), ("EU 100 EUR/t, no CBAM", dict(eu_price=100.0, cbam_phase_in=0.0, ets_maritime=False)),
                          ("EU 100 EUR/t with CBAM and ETS maritime", dict(eu_price=100.0))):
        r = solve(build(**kwargs))
        prod = r["production"][r["production"] > 1]
        print(f"--- {label}: {r['condition']}"); print(prod.round(0).to_string()); print(r["shipped"].round(1).to_string())
    cap = 0.5 * float(solve(build(eu_price=0.0))["emissions"].loc["EU"].sum())
    price, history, r = fixed_point(max(cap, 1.0))
    print(f"--- EU cap {cap:.0f} t: fixed point at {price:.1f} EUR/t after {len(history) - 1} solves: {np.round(history, 1)}")


if __name__ == "__main__":
    main()
