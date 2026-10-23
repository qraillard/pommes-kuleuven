"""CO2 accounting and cumulative CO2 storage on a toy (POMMES-EU-Industry task M01).

Two nodes: IND, an industry node making cement, and STO, a CO2 storage site reached by a pipeline.
One representative hour per operation year, scaled to the year by the normalisation. Units: t of
cement, MWh of energy, t of CO2.

Accounting (POMMES-EU-Industry brief, section 8.1):
  biomass supply      negative emission factor: the biogenic carbon is credited upstream
  kilns               count every tonne they emit, fossil, process or biogenic; with capture the
                      factor is net of capture and the captured tonnes leave as the resource `co2`
  storage site        consumes `co2`, factor 0
  e-fuel synthesis    consumes `co2`, factor +1 per t (released when the fuel is burnt)
  DAC                 produces `co2`, factor -1 per t
The goal counts every tonne, negative ones included. The tax costs max(0, tax x net emissions) per
area and year: removals offset the area's own emissions down to zero, and are not paid beyond.
Permanent storage removes CO2 from the system through a sink: a conversion technology with a `co2`
input only (no storage level), which carries an injection capacity, a yearly limit and a
cumulative limit, unlike spillage.
Spillage is closed for every resource (`spillage_max_capacity = 0`): captured CO2 must be stored or
used, and credited biomass must be burnt. `open_spillage` reopens it to show the leak.
The storage site has a yearly injection limit and a cumulative capacity over the operation years of
the run, each operation year standing for `operation_year_weight` calendar years.

Run: python pommes/examples/toy_co2/toy_co2.py
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from pommes.io.build_input_dataset import compute_annuity_cost, compute_discount, compute_end_of_life
from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs

HOURS = 8760.0
RATE = 0.05
AREAS = ["IND", "STO"]
RESOURCES = ["biomass", "cement", "co2", "efuel", "electricity", "methane"]
CEMENT = 100.0              # t/h
PROCESS_EF = 0.5            # t CO2 per t cement (calcination)
GAS_EF = 0.2                # t CO2 per MWh methane burnt
BIO_C = 0.4                 # t biogenic CO2 per MWh biomass
CAPTURE = 0.9
KILN_GAS = GAS_EF * 1.0 + PROCESS_EF          # 0.70 t/t, 1.0 MWh methane per t
KILN_BIO = BIO_C * 1.1 + PROCESS_EF           # 0.94 t/t, 1.1 MWh biomass per t
EFUEL_CO2 = 0.25            # t CO2 per MWh e-fuel

# tech: (factors per unit of the tech's power, emission factor, variable cost EUR/unit, area)
TECHS = {
    "gas_supply": ({"methane": 1.0}, 0.0, 30.0, "IND"),
    "biomass_supply": ({"biomass": 1.0}, -BIO_C, 40.0, "IND"),
    "power_supply": ({"electricity": 1.0}, 0.0, 50.0, "IND"),
    "kiln_gas": ({"cement": 1.0, "methane": -1.0, "electricity": -0.1}, KILN_GAS, 0.0, "IND"),
    "kiln_gas_ccs": ({"cement": 1.0, "methane": -1.0, "electricity": -0.3, "co2": CAPTURE * KILN_GAS},
                     (1 - CAPTURE) * KILN_GAS, 0.0, "IND"),
    "kiln_bio": ({"cement": 1.0, "biomass": -1.1, "electricity": -0.1}, KILN_BIO, 0.0, "IND"),
    "kiln_bio_ccs": ({"cement": 1.0, "biomass": -1.1, "electricity": -0.3, "co2": CAPTURE * KILN_BIO},
                     (1 - CAPTURE) * KILN_BIO, 0.0, "IND"),
    "efuel_synthesis": ({"efuel": 1.0, "electricity": -1.5, "co2": -EFUEL_CO2}, EFUEL_CO2, 0.0, "IND"),
    "dac": ({"co2": 1.0, "electricity": -2.0}, -1.0, 0.0, "IND"),
    "co2_storage": ({"co2": -1.0}, 0.0, 10.0, "STO"),
}
SUPPLY = ["gas_supply", "biomass_supply", "power_supply", "co2_storage"]


def build(techs: list[str], tax: float = 100.0, year_ops: tuple[int, ...] = (2030,), efuel: float = 0.0,
          dac_max: float = 0.0, injection_max: float = np.nan, cumulative: float = np.nan,
          weight: float | None = 10.0, open_spillage: tuple[str, ...] = ()) -> xr.Dataset:
    """Dataset with the kilns and options in `techs` (supplies and storage always present).

    tax: EUR/t on every area; efuel: e-fuel demand at IND, MWh/h; dac_max: DAC capacity, t/h;
    injection_max: storage injection limit, t/yr; cumulative: storage capacity over the run, t;
    weight: calendar years per operation year (None leaves it unset); open_spillage: resources
    whose spillage is allowed at no cost.
    """
    techs = SUPPLY + [t for t in techs if t not in SUPPLY]
    first = year_ops[0]
    ds = xr.Dataset(coords=dict(
        area=np.array(AREAS, dtype=str), resource=np.array(RESOURCES, dtype=str),
        conversion_tech=np.array(list(TECHS), dtype=str), transport_tech=np.array(["co2_pipe"], dtype=str),
        link=np.array(["IND_STO"], dtype=str), hour=np.array([0], dtype="int64"),
        year_op=np.array(year_ops, dtype="int64"), year_inv=np.array(year_ops, dtype="int64"),
        year_dec=np.arange(first + 10, 2110, 10, dtype="int64")))
    ds = ds.assign(conversion=True, transport=True, carbon=True, combined=False, storage=False, net_import=False,
                   turpe=False, process=False, flexdem=False)

    def da(dims, fill=np.nan):
        return xr.DataArray(np.full([ds.sizes[d] for d in dims], fill, dtype="float64"), coords={d: ds[d] for d in dims})

    demand = da(["area", "resource"], 0.0)
    demand.loc[dict(area="IND", resource="cement")] = CEMENT
    demand.loc[dict(area="IND", resource="efuel")] = efuel
    spill_max = da(["area", "resource"], 0.0)
    for r in open_spillage:
        spill_max.loc[dict(resource=r)] = np.nan
    factor, ef = da(["conversion_tech", "resource"], 0.0), da(["conversion_tech"], 0.0)
    var = da(["area", "conversion_tech", "year_op"], 0.0)
    cap_max = da(["area", "conversion_tech", "year_op"], 0.0)
    for tech, (f, e, cost, area) in TECHS.items():
        for r, v in f.items():
            factor.loc[dict(conversion_tech=tech, resource=r)] = v
        ef.loc[dict(conversion_tech=tech)] = e
        var.loc[dict(area=area, conversion_tech=tech)] = cost
        if tech in techs:
            cap_max.loc[dict(area=area, conversion_tech=tech)] = dac_max if tech == "dac" else np.nan
    yearly = da(["area", "conversion_tech", "year_op"])
    yearly.loc[dict(area="STO", conversion_tech="co2_storage")] = injection_max
    cumul = da(["area", "conversion_tech"])
    cumul.loc[dict(area="STO", conversion_tech="co2_storage")] = cumulative

    ds = ds.assign(
        demand=demand, load_shedding_cost=xr.DataArray(1e5), spillage_cost=xr.DataArray(0.0),
        spillage_max_capacity=spill_max,
        time_step_duration=xr.DataArray([1.0], coords=dict(hour=ds.hour)), operation_year_duration=xr.DataArray(HOURS),
        discount_rate=xr.DataArray(RATE), year_ref=xr.DataArray(first),
        conversion_factor=factor, conversion_invest_cost=da(["area", "conversion_tech"], 0.0),
        conversion_fixed_cost=da(["area", "conversion_tech"], 0.0), conversion_variable_cost=var,
        conversion_life_span=da(["conversion_tech"], 40.0), conversion_finance_rate=xr.DataArray(RATE),
        conversion_power_capacity_max=cap_max, conversion_emission_factor=ef,
        conversion_max_yearly_production=yearly, conversion_max_cumulative_production=cumul,
        carbon_tax=da(["area", "year_op"], tax), carbon_goal=xr.DataArray(np.nan),
        transport_area_from=xr.DataArray(["IND"], coords=dict(link=ds.link)).astype(str),
        transport_area_to=xr.DataArray(["STO"], coords=dict(link=ds.link)).astype(str),
        transport_resource=xr.DataArray([["co2"]], coords=dict(link=ds.link, transport_tech=ds.transport_tech)).astype(str),
        transport_invest_cost=xr.DataArray(0.0), transport_fixed_cost=xr.DataArray(0.0),
        transport_life_span=xr.DataArray(40.0), transport_finance_rate=xr.DataArray(RATE),
    )
    if weight is not None:
        ds = ds.assign(operation_year_weight=da(["year_op"], weight))
    config = {"pre_process": {
        "annuity_computation": {
            "conversion": {"conversion_annuity_cost": {"invest_cost": "conversion_invest_cost", "finance_rate": "conversion_finance_rate", "life_span": "conversion_life_span"}},
            "transport": {"transport_annuity_cost": {"invest_cost": "transport_invest_cost", "finance_rate": "transport_finance_rate", "life_span": "transport_life_span"}}},
        "discount_factor": {"discount_rate": "discount_rate", "year_ref": "year_ref"}}}
    return check_inputs(compute_discount(compute_annuity_cost(compute_end_of_life(ds), config), config))


def solve(ds: xr.Dataset) -> dict:
    """Solve; yearly quantities per operation year (t/yr), conversion power per (area, tech, year_op)."""
    model = build_model(ds)
    _, condition = model.solve(solver_name="highs", output_flag=False)
    s = model.solution
    norm = float(ds.operation_year_duration) / float(ds.time_step_duration.sum())
    power = s.operation_conversion_power.sum("hour")
    # shed and spilled energy [t] = time_step_duration x power, summed over the steps
    spill = (s.operation_spillage_power * ds.time_step_duration).sum("hour").fillna(0) * norm
    return {
        "condition": condition, "model": model,
        "emissions": (s.operation_total_carbon_emissions.sum(["area", "hour"]) * norm).to_series(),
        "carbon_costs": s.operation_carbon_costs.to_series(),
        "shed": float((s.operation_load_shedding_power * ds.time_step_duration).sum().fillna(0)) * norm,
        "power": power,                                                            # per hour
        "stored": (power.sel(area="STO", conversion_tech="co2_storage") * norm).to_series(),
        "spilled": spill.sum("area").to_dataframe("t").t.unstack("resource"),
    }


def cumulative_used(ds: xr.Dataset, result: dict) -> float:
    """CO2 stored over the run, t: yearly injection x calendar years per operation year."""
    return float((result["stored"].to_xarray() * ds.operation_year_weight).sum())


def main() -> None:
    cases = {
        "gas kiln": dict(techs=["kiln_gas"]),
        "biomass kiln": dict(techs=["kiln_bio"]),
        "gas kiln + CCS": dict(techs=["kiln_gas_ccs"]),
        "biomass kiln + CCS (BECCS)": dict(techs=["kiln_bio_ccs"]),
        "gas kiln + CCS, CO2 to e-fuel": dict(techs=["kiln_gas_ccs", "efuel_synthesis"], efuel=50.0),
        "gas kiln + CCS, CO2 spillage open": dict(techs=["kiln_gas_ccs"], open_spillage=("co2",)),
    }
    for label, kwargs in cases.items():
        r = solve(build(**kwargs))
        print(f"{label:38s} {r['condition']:8s} emissions {r['emissions'].iloc[0] / 1e3:9.1f} kt/yr"
              f"   stored {r['stored'].iloc[0] / 1e3:7.1f} kt/yr")


if __name__ == "__main__":
    main()
