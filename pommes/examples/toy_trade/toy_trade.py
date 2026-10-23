"""Toy trade model for POMMES-World (brief section 12, step 3): 3 nodes, hydrogen and ammonia, one year.

Nodes: EU (the importer, demand only), NAF (North Africa, close) and AUS (Australia, far). Each node can make
electricity from solar PV, hydrogen by electrolysis and ammonia by Haber-Bosch; the EU can also crack ammonia
back to hydrogen. Trade links towards the EU:
  NAF->EU  ammonia ship   and  hydrogen pipeline
  AUS->EU  ammonia ship
The ships burn ammonia as bunker fuel at the origin (`transport_fuel_consumption`), the pipeline loses 2 % of
its hydrogen (`transport_efficiency`) and uses electricity for compression at the origin. Costs are EUR 2024 for
2030 from the POMMES-World pre-processing (technology cost picks, hydrogen picks, World Trade shipping costs for
Rotterdam-Arzew and Rotterdam-Port Hedland); capacity factors are round illustrative numbers.

One representative hour stands for the year (operation_year_duration 8760), so the duals of the adequacy
constraint are prices in EUR per MW-year; divided by 8760 they are EUR/MWh.

Run: python pommes/examples/toy_trade/toy_trade.py   (writes toy_trade_results.csv and toy_trade_sankey.html)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pommes.io.build_input_dataset import compute_annuity_cost, compute_discount, compute_end_of_life
from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs

HERE = Path(__file__).resolve().parent
HOURS = 8760.0
AREAS = ["AUS", "EU", "NAF"]
RESOURCES = ["ammonia", "electricity", "hydrogen"]
TECHS = ["cracker", "electrolyser", "haber_bosch", "solar_pv"]
TRANSPORT = ["h2_pipe", "nh3_ship"]
LINKS = {"AUS_EU": ("AUS", "EU"), "NAF_EU": ("NAF", "EU")}
YEAR = 2030
FINANCE_RATE = 0.07

DEMAND = {("EU", "hydrogen"): 1000.0, ("EU", "ammonia"): 500.0}          # MW, all year
SOLAR_CF = {"EU": 0.12, "NAF": 0.24, "AUS": 0.26}                          # illustrative

# conversion: output factor 1 on the first resource; inputs negative (MWh per MWh of output)
CONVERSION = {
    #               factors                                              capex EUR/kW out  FOM EUR/kW/yr  life
    "solar_pv": ({"electricity": 1.0}, None, None, 40),
    "electrolyser": ({"hydrogen": 1.0, "electricity": -1 / 0.69}, 1451 / 0.69, 58.06 / 0.69, 25),
    "haber_bosch": ({"ammonia": 1.0, "hydrogen": -1.1484, "electricity": -0.2473}, 1294 * 1.1484, 1294 * 1.1484 * 0.029, 30),
    "cracker": ({"hydrogen": 1.0, "ammonia": -1.46}, 1378, 1378 * 0.043, 25),
}
SOLAR_CAPEX = {"EU": (466.0, 11.65), "NAF": (758.0, 9.3), "AUS": (778.0, 9.9)}   # EUR/kW, EUR/kW/yr (picks, 2030)

# transport: (link, tech) -> capex EUR/kW delivered, FOM share, life, efficiency, fuel resource, fuel per MWh injected
TRANSPORT_DATA = {
    ("NAF_EU", "nh3_ship"): (95.1, 0.04, 20, 1.0, "ammonia", 0.013),
    ("AUS_EU", "nh3_ship"): (390.5, 0.04, 20, 1.0, "ammonia", 0.075),
    ("NAF_EU", "h2_pipe"): (931.0, 0.0317, 40, 0.98, "electricity", 0.0475),
}


def build_toy_dataset() -> xr.Dataset:
    year_dec = np.arange(2035, 2085, 5, dtype="int64")
    ds = xr.Dataset(coords=dict(
        area=np.array(AREAS, dtype=str), resource=np.array(RESOURCES, dtype=str),
        conversion_tech=np.array(TECHS, dtype=str), transport_tech=np.array(TRANSPORT, dtype=str),
        link=np.array(list(LINKS), dtype=str), hour=np.array([0], dtype="int64"),
        year_op=np.array([YEAR], dtype="int64"), year_inv=np.array([YEAR], dtype="int64"), year_dec=year_dec,
    ))
    ds = ds.assign(conversion=True, transport=True, carbon=False, combined=False, storage=False, net_import=False,
                   turpe=False, process=False, flexdem=False)

    def da(dims, fill=np.nan):
        return xr.DataArray(np.full([ds.sizes[d] for d in dims], fill, dtype="float64"), coords={d: ds[d] for d in dims})

    demand = da(["area", "resource"], 0.0)
    for (a, r), v in DEMAND.items():
        demand.loc[dict(area=a, resource=r)] = v
    factor = da(["conversion_tech", "resource"], 0.0)
    invest, fixed, life = da(["area", "conversion_tech"]), da(["area", "conversion_tech"]), da(["conversion_tech"])
    for tech, (f, capex, fom, lifetime) in CONVERSION.items():
        for r, v in f.items():
            factor.loc[dict(conversion_tech=tech, resource=r)] = v
        life.loc[dict(conversion_tech=tech)] = lifetime
        for a in AREAS:
            c, o = SOLAR_CAPEX[a] if tech == "solar_pv" else (capex, fom)
            invest.loc[dict(area=a, conversion_tech=tech)] = c * 1000          # EUR/MW
            fixed.loc[dict(area=a, conversion_tech=tech)] = o * 1000
    availability = da(["area", "conversion_tech", "hour", "year_op"])
    for a, cf in SOLAR_CF.items():
        availability.loc[dict(area=a, conversion_tech="solar_pv")] = cf
    invest_max = da(["area", "conversion_tech", "year_inv"])
    invest_max.loc[dict(area=["AUS", "NAF"], conversion_tech="cracker")] = 0.0   # cracking only at the importer

    t_invest, t_fixed, t_life = da(["link", "transport_tech"], 0.0), da(["link", "transport_tech"], 0.0), da(["link", "transport_tech"], 20.0)
    t_eff, t_fuel, t_max = da(["link", "transport_tech"]), da(["link", "transport_tech"]), da(["link", "transport_tech", "year_inv"], 0.0)
    t_fuel_res = xr.DataArray(np.full((len(LINKS), len(TRANSPORT)), "none", dtype=object), coords=dict(link=ds.link, transport_tech=ds.transport_tech))
    t_res = xr.DataArray(np.array([["hydrogen", "ammonia"]] * len(LINKS), dtype=str), coords=dict(link=ds.link, transport_tech=ds.transport_tech))
    for (link, tech), (capex, fom, lifetime, eff, fuel_res, fuel) in TRANSPORT_DATA.items():
        sel = dict(link=link, transport_tech=tech)
        t_invest.loc[sel], t_fixed.loc[sel], t_life.loc[sel] = capex * 1000, capex * 1000 * fom, lifetime
        t_eff.loc[sel], t_fuel.loc[sel], t_fuel_res.loc[sel] = eff, fuel, fuel_res
        t_max.loc[sel] = np.nan                                                    # this link may be built
    ds = ds.assign(
        demand=demand, load_shedding_cost=xr.DataArray(1e4), spillage_cost=xr.DataArray(0.0),
        time_step_duration=xr.DataArray([1.0], coords=dict(hour=ds.hour)), operation_year_duration=xr.DataArray(HOURS),
        discount_rate=xr.DataArray(FINANCE_RATE), year_ref=xr.DataArray(YEAR),
        conversion_factor=factor, conversion_invest_cost=invest, conversion_fixed_cost=fixed,
        conversion_life_span=life, conversion_finance_rate=xr.DataArray(FINANCE_RATE), conversion_availability=availability,
        conversion_power_capacity_investment_max=invest_max,
        transport_area_from=xr.DataArray([LINKS[l][0] for l in LINKS], coords=dict(link=ds.link)).astype(str),
        transport_area_to=xr.DataArray([LINKS[l][1] for l in LINKS], coords=dict(link=ds.link)).astype(str),
        transport_resource=t_res, transport_invest_cost=t_invest, transport_fixed_cost=t_fixed,
        transport_life_span=t_life, transport_finance_rate=xr.DataArray(FINANCE_RATE),
        transport_power_capacity_investment_max=t_max, transport_hurdle_costs=xr.DataArray(0.0),
        transport_efficiency=t_eff, transport_fuel_consumption=t_fuel, transport_fuel_resource=t_fuel_res.astype(str),
    )
    config = {"pre_process": {
        "annuity_computation": {
            "conversion": {"conversion_annuity_cost": {"invest_cost": "conversion_invest_cost", "finance_rate": "conversion_finance_rate", "life_span": "conversion_life_span"}},
            "transport": {"transport_annuity_cost": {"invest_cost": "transport_invest_cost", "finance_rate": "transport_finance_rate", "life_span": "transport_life_span"}},
        },
        "discount_factor": {"discount_rate": "discount_rate", "year_ref": "year_ref"}}}
    ds = compute_end_of_life(ds)
    ds = compute_annuity_cost(ds, config)
    ds = compute_discount(ds, config)
    return check_inputs(ds)


def solve_toy(ds: xr.Dataset | None = None) -> dict:
    ds = build_toy_dataset() if ds is None else ds
    model = build_model(ds)
    status, condition = model.solve(solver_name="highs")
    s = model.solution.squeeze(drop=True)
    price = (model.dual["operation_adequacy_constraint"].squeeze(drop=True) / HOURS).to_pandas()   # EUR/MWh
    flows = []
    for link, (a, b) in LINKS.items():
        for tech in TRANSPORT:
            delivered = float(s.operation_transport_power.sel(link=link, transport_tech=tech))
            if delivered < 1e-6:
                continue
            injected = float(s.operation_transport_power_in.sel(link=link, transport_tech=tech))
            res = str(ds.transport_resource.sel(link=link, transport_tech=tech).item())
            fuel_res = str(ds.transport_fuel_resource.sel(link=link, transport_tech=tech).item())
            fuel = injected * float(ds.transport_fuel_consumption.sel(link=link, transport_tech=tech))
            flows.append({"link": link, "tech": tech, "resource": res, "from": a, "to": b, "delivered_mw": delivered,
                          "injected_mw": injected, "fuel_resource": fuel_res, "fuel_mw": fuel,
                          "price_from": price.loc[a, res], "price_to": price.loc[b, res],
                          "fuel_price": price.loc[a, fuel_res],
                          # EUR a year: value delivered minus value taken at origin (cargo and fuel)
                          "trade_margin_eur": HOURS * (delivered * price.loc[b, res] - injected * price.loc[a, res]
                                                       - fuel * price.loc[a, fuel_res])})
    link_cost = (s.operation_transport_costs + s.planning_transport_costs).sum("area").to_pandas()
    capacity = s.operation_conversion_power_capacity.to_pandas()
    return {"status": status, "condition": condition, "objective": model.objective.value, "price": price,
            "flows": pd.DataFrame(flows), "link_cost": link_cost, "capacity": capacity, "model": model}


def sankey_html(flows: pd.DataFrame, path: Path) -> None:
    """Energy flows between nodes as a Sankey (d3-sankey from jsDelivr), for a quick look."""
    nodes = sorted(set(flows["from"]) | set(flows["to"]))
    links = [{"source": nodes.index(r["from"]), "target": nodes.index(r["to"]), "value": round(r["delivered_mw"], 1),
              "label": f"{r['resource']} by {r['tech']}"} for r in flows.to_dict("records")]
    html = f"""<!doctype html><meta charset="utf-8"><title>Toy trade Sankey</title>
<style>body{{font-family:sans-serif;margin:16px}}</style><h3>Toy trade: delivered flows (MW, all year)</h3><svg width="640" height="320"></svg>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script><script src="https://cdn.jsdelivr.net/npm/d3-sankey@0.12"></script>
<script>
const data = {{nodes: {json.dumps([{"name": n} for n in nodes])}, links: {json.dumps(links)}}};
const svg = d3.select("svg"); const {{nodes, links}} = d3.sankey().nodeWidth(14).nodePadding(24).extent([[10,10],[630,310]])(data);
svg.append("g").selectAll("path").data(links).join("path").attr("d", d3.sankeyLinkHorizontal()).attr("fill","none")
  .attr("stroke","#2a78d6").attr("stroke-opacity",0.4).attr("stroke-width", d => Math.max(1, d.width)).append("title").text(d => d.label + ": " + d.value + " MW");
svg.append("g").selectAll("rect").data(nodes).join("rect").attr("x", d => d.x0).attr("y", d => d.y0).attr("height", d => d.y1 - d.y0).attr("width", d => d.x1 - d.x0).attr("fill","#444");
svg.append("g").selectAll("text").data(nodes).join("text").attr("x", d => d.x0 < 300 ? d.x1 + 6 : d.x0 - 6).attr("y", d => (d.y0 + d.y1) / 2).attr("dy","0.35em").attr("text-anchor", d => d.x0 < 300 ? "start" : "end").text(d => d.name);
</script>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    r = solve_toy()
    print(r["status"], r["condition"], f"objective {r['objective'] / 1e6:.1f} M EUR a year")
    print("prices, EUR/MWh:"); print(r["price"].round(1).to_string())
    print("trade flows:"); print(r["flows"].round(2).to_string())
    print("link costs, EUR a year:"); print(r["link_cost"].round(0).to_string())
    print("capacities, MW:"); print(r["capacity"].round(0).to_string())
    r["flows"].to_csv(HERE / "toy_trade_results.csv", index=False)
    sankey_html(r["flows"], HERE / "toy_trade_sankey.html")


if __name__ == "__main__":
    main()
