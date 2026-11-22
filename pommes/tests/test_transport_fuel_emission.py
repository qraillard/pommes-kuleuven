"""Emissions of the fuel burnt by transport (tCO2 per MWh of fuel burnt).

A link that burns fuel (``transport_fuel_consumption`` per MWh injected) emits ``factor x fuel burnt`` at each step, in the
arrival area of the link whatever ``transport_fuel_area`` is, at the step the fuel is burnt (the injection step). Two
optional factors choose where they count: ``transport_fuel_emission_factor`` enters the emissions that the carbon tax
sees, ``transport_fuel_total_emission_factor`` those of the carbon goals. Without them (absent, 0 or NaN), or without
fuel burnt, the model is that of 0.7.1 without them.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import pommes
from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import with_conditions
from pommes.tests.time_blocks_helper import lp_text

TAXED = 0.25  # tCO2 per MWh of fuel burnt, seen by the carbon tax
TOTAL = 0.4  # tCO2 per MWh of fuel burnt, seen by the carbon goals
CONSUMPTION = 0.1  # MWh of fuel per MWh injected
TAX = 7.0  # per tCO2

_TOY = Path(pommes.__file__).parent / "examples" / "toy_trade" / "toy_trade.py"
_FACTORS = ("transport_fuel_emission_factor", "transport_fuel_total_emission_factor")
_ROWS = ("operation_carbon_emissions_def", "operation_total_carbon_emissions_def")


def _is_todays_model(q):
    """q builds as it does without the two factors: same LP text and same number of terms in both emission rows
    (linopy drops zero coefficients from the text, so a term that is all zeros shows only in the number of terms)."""
    mp, mq = build_model(q.drop_vars([n for n in _FACTORS if n in q])), build_model(q)
    assert lp_text(mp) == lp_text(mq)
    assert {r: mp.constraints[r].lhs.nterm for r in _ROWS} == {r: mq.constraints[r].lhs.nterm for r in _ROWS}


@pytest.fixture()
def parameters_transport(parameters_dispatch_invest, transport, carbon):
    flags = {f"carbon_goal{x}_expected": np.array(False) for x in ("", "_area", "_group")}
    return xr.merge([parameters_dispatch_invest, transport, carbon, xr.Dataset(flags)])


def _one_way(parameters_transport, wind_area_1=(1.0, 1.0, 1.0), year_op=(2030,)):
    """Link_1 from area_1 to area_2 over three hours; area 2 cannot generate; the link burns electricity."""
    p = parameters_transport.sel(
        conversion_tech=["wind_onshore"],
        transport_tech=["power_line"],
        link=["link_1"],
        hour=[0, 1, 2],
        resource=["electricity"],
        year_dec=[2040, 2050],
        year_inv=[2030],
        year_op=list(year_op),
    ).copy(deep=True)
    p["transport_hurdle_costs"] = p.transport_hurdle_costs / 8760
    p["conversion_power_capacity_investment_max"] = p.conversion_power_capacity_investment_max.expand_dims(
        area=p.area, conversion_tech=p.conversion_tech
    ).copy()
    p.conversion_power_capacity_investment_max.loc[dict(area="area_2", conversion_tech="wind_onshore")] = 0
    p["conversion_availability"] = p.conversion_availability.expand_dims(area=p.area).copy()
    p.conversion_availability.loc[dict(area="area_1", conversion_tech="wind_onshore")] = list(wind_area_1)
    p["carbon_goal"] = np.nan
    p["carbon_tax"] = 0.0
    p["transport_fuel_consumption"] = CONSUMPTION
    p["transport_fuel_resource"] = xr.DataArray("electricity")
    return p


def _solve(p, taxed=None, total=None):
    for name, value in zip(_FACTORS, (taxed, total)):
        if value is not None:
            p = p.assign({name: value})
    m = build_model(p)
    m.solve(solver_name="highs")
    return m, m.solution


def _burnt(s, p):
    """MWh of fuel burnt at each step, summed over the links and techs."""
    return (
        s.operation_transport_power_in.sum(["link", "transport_tech"]) * CONSUMPTION * p.time_step_duration
    ).squeeze(drop=True)


def _extra(s, base, name):
    return (s[name] - base[name]).squeeze(drop=True)


def test_emissions_in_the_arrival_area(parameters_transport):
    p = _one_way(parameters_transport)
    _, base = _solve(p)
    _, s = _solve(p, TAXED, TOTAL)
    burnt = _burnt(s, p)
    assert float(burnt.sum()) > 0
    for name, factor in (("operation_carbon_emissions", TAXED), ("operation_total_carbon_emissions", TOTAL)):
        extra = _extra(s, base, name)
        np.testing.assert_allclose(extra.sel(area="area_2").values, factor * burnt.values, atol=1e-9)
        np.testing.assert_allclose(extra.sel(area="area_1").values, 0.0, atol=1e-9)


def test_time_step_duration_scales_the_emissions(parameters_transport):
    p = _one_way(parameters_transport)
    p["time_step_duration"] = xr.DataArray([2.0, 2.0, 2.0], coords=[p.hour])
    _, base = _solve(p)
    _, s = _solve(p, TAXED, TOTAL)
    burnt = _burnt(s, p)  # injected power x consumption x 2 h
    assert float(burnt.sum()) > 0
    for name, factor in (("operation_carbon_emissions", TAXED), ("operation_total_carbon_emissions", TOTAL)):
        extra = _extra(s, base, name)
        np.testing.assert_allclose(extra.sel(area="area_2").values, factor * burnt.values, atol=1e-9)


def _toy_with_carbon(fuel_area=None):
    """The toy trade model (3 areas, ships burning ammonia at the origin) with carbon on and no tax."""
    spec = importlib.util.spec_from_file_location("toy_trade_fuel_emission", _TOY)
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    ds = toy.build_toy_dataset()
    ds = ds.assign(carbon=True)
    for name in ("carbon_tax", "carbon_goal_area", "carbon_goal"):
        ds = ds.drop_vars(name, errors="ignore")
    ds = toy.check_inputs(ds)
    ds["carbon_tax"] = xr.DataArray(0.0)
    ds["carbon_goal"] = xr.DataArray(np.nan)
    ds["transport_fuel_area"] = xr.DataArray(
        np.full((ds.sizes["link"], ds.sizes["transport_tech"]), fuel_area or "nan", dtype=object),
        coords=dict(link=ds.link, transport_tech=ds.transport_tech),
    ).astype(str)
    return ds


def test_fuel_area_does_not_move_the_emissions():
    """The fuel is drawn at a third area (AUS) for the NAF_EU ships; the emissions stay in the arrival area (EU)."""
    ds = _toy_with_carbon(fuel_area="AUS")
    taxed = xr.zeros_like(ds.transport_fuel_consumption.fillna(0))
    total = xr.zeros_like(taxed)
    taxed.loc[dict(link="NAF_EU", transport_tech="nh3_ship")] = 0.3
    total.loc[dict(link="NAF_EU", transport_tech="nh3_ship")] = 0.5
    _, base = _solve(ds)
    _, s = _solve(ds, taxed, total)
    burnt = float(s.operation_transport_power_in.sel(link="NAF_EU", transport_tech="nh3_ship").squeeze()) * float(
        ds.transport_fuel_consumption.sel(link="NAF_EU", transport_tech="nh3_ship")
    )
    assert burnt > 0
    for name, factor in (("operation_carbon_emissions", 0.3), ("operation_total_carbon_emissions", 0.5)):
        extra = _extra(s, base, name)
        np.testing.assert_allclose(extra.sel(area="EU").item(), factor * burnt, atol=1e-9)
        for area in ("NAF", "AUS"):
            np.testing.assert_allclose(extra.sel(area=area).item(), 0.0, atol=1e-9)


def test_carbon_tax_charges_the_taxed_factor_only(parameters_transport):
    p = _one_way(parameters_transport)
    p["carbon_tax"] = TAX
    m_base, base = _solve(p)
    m, s = _solve(p, taxed=TAXED)
    burnt = _burnt(s, p)
    normalization = float(p.operation_year_duration.squeeze()) / float(p.time_step_duration.sum())
    cost = s.operation_carbon_costs.squeeze(drop=True)
    assert float(cost.sel(area="area_1")) == pytest.approx(0.0, abs=1e-6)
    assert float(cost.sel(area="area_2")) == pytest.approx(TAX * normalization * TAXED * float(burnt.sum()), rel=1e-6)
    # the cost rows are in the objective: it rises by the discounted carbon costs
    rise = m.objective.value - m_base.objective.value
    assert rise == pytest.approx(float(p.discount_factor.squeeze()) * float(cost.sum()), rel=1e-6)
    # the taxed factor does not feed the goals row
    np.testing.assert_allclose(
        s.operation_total_carbon_emissions.values, base.operation_total_carbon_emissions.values, atol=1e-9
    )


def test_total_factor_alone_is_not_taxed_but_binds_the_goals(parameters_transport):
    p = _one_way(parameters_transport)
    p["carbon_tax"] = TAX
    p["load_shedding_cost"] = xr.DataArray(1000.0)
    p["load_shedding_max_capacity"] = xr.DataArray(np.nan)
    m_base, base = _solve(p)
    m, s = _solve(p, total=TOTAL)
    assert m.objective.value == pytest.approx(m_base.objective.value, rel=1e-9)
    np.testing.assert_allclose(s.operation_carbon_emissions.values, base.operation_carbon_emissions.values, atol=1e-9)
    burnt = _burnt(s, p)
    assert float(burnt.sum()) > 0
    extra = _extra(s, base, "operation_total_carbon_emissions")
    np.testing.assert_allclose(extra.sel(area="area_2").values, TOTAL * burnt.values, atol=1e-9)
    # a goal of the arrival area binds on them: half of what the free solution emits there
    normalization = float(p.operation_year_duration.squeeze()) / float(p.time_step_duration.sum())
    free = normalization * float(s.operation_total_carbon_emissions.sel(area="area_2").sum())
    goal = xr.DataArray([[np.nan], [0.5 * free]], coords=[p.area, p.year_op])
    m_goal, s_goal = _solve(p.assign(carbon_goal_area=goal), total=TOTAL)
    bound = normalization * float(s_goal.operation_total_carbon_emissions.sel(area="area_2").sum())
    assert bound == pytest.approx(0.5 * free, rel=1e-6)
    assert m_goal.objective.value > m.objective.value


def test_emissions_sit_at_the_injection_step_with_a_delay(parameters_transport):
    p = _one_way(parameters_transport, wind_area_1=(1.0, 0.0, 1.0))
    p["transport_delay"] = 1.0
    p["load_shedding_cost"] = xr.DataArray(1000.0)
    p["load_shedding_max_capacity"] = xr.DataArray(np.nan)
    _, base = _solve(p)
    _, s = _solve(p, TAXED, TOTAL)
    injected = s.operation_transport_power_in.squeeze(drop=True).transpose("hour").values
    delivered = s.operation_transport_power.squeeze(drop=True).transpose("hour").values
    assert not np.allclose(injected, delivered)
    for name, factor in (("operation_carbon_emissions", TAXED), ("operation_total_carbon_emissions", TOTAL)):
        extra = _extra(s, base, name).sel(area="area_2")
        np.testing.assert_allclose(extra.values, factor * CONSUMPTION * injected, atol=1e-9)


def test_two_operating_conditions(parameters_transport):
    p = _one_way(parameters_transport)
    p["conversion_availability"] = p.conversion_availability.broadcast_like(p.hour).copy()
    low = p.conversion_availability.copy()
    low.loc[dict(area="area_1", conversion_tech="wind_onshore")] = [1.0, 0.0, 1.0]
    p = with_conditions(p, [0.5, 0.5], conversion_availability=[p.conversion_availability, low])
    p["load_shedding_cost"] = xr.DataArray(1000.0)
    p["load_shedding_max_capacity"] = xr.DataArray(np.nan)
    _, base = _solve(p)
    _, s = _solve(p, TAXED, TOTAL)
    injected = s.operation_transport_power_in.sum(["link", "transport_tech"]).squeeze(drop=True)
    assert float(injected.sum()) > 0
    for name, factor in (("operation_carbon_emissions", TAXED), ("operation_total_carbon_emissions", TOTAL)):
        extra = _extra(s, base, name)
        np.testing.assert_allclose(
            extra.sel(area="area_2").transpose("operating_condition", "hour").values,
            (factor * CONSUMPTION * injected).transpose("operating_condition", "hour").values,
            atol=1e-9,
        )
        np.testing.assert_allclose(extra.sel(area="area_1").values, 0.0, atol=1e-9)


@pytest.mark.parametrize("name", _FACTORS)
@pytest.mark.parametrize("value", [0.0, np.nan])
def test_zero_or_nan_factor_is_todays_model(parameters_transport, name, value):
    p = _one_way(parameters_transport)
    _is_todays_model(p.assign({name: value}))
    _is_todays_model(p.assign({n: value for n in _FACTORS}))


def test_absent_factors_is_todays_model(parameters_transport):
    """A dataset without the two variables (not through check_inputs, or a 0.7.0 file) builds the same model."""
    p = _one_way(parameters_transport)
    assert all(name in p for name in _FACTORS)
    _is_todays_model(p.drop_vars(list(_FACTORS)))
    _is_todays_model(p.assign({n: 0.0 for n in _FACTORS}))


def test_each_term_is_in_its_own_row_only(parameters_transport):
    p = _one_way(parameters_transport)
    base, taxed, total = (
        lp_text(build_model(q))
        for q in (p, p.assign(transport_fuel_emission_factor=TAXED), p.assign(transport_fuel_total_emission_factor=TAXED))
    )
    assert len({base, taxed, total}) == 3


def test_no_term_without_carbon_or_transport(parameters_transport):
    p = _one_way(parameters_transport)
    both = {name: TAXED for name in _FACTORS}
    for module in ("carbon", "transport"):
        q = p.assign({module: np.array(False)})
        assert lp_text(build_model(q)) == lp_text(build_model(q.assign(both)))


def test_no_term_where_the_pair_does_not_exist_or_burns_no_fuel(parameters_transport):
    """Three techs, one resource in the model: the methane techs have no pair; the power line burns no fuel."""
    p = parameters_transport.sel(
        conversion_tech=["wind_onshore"],
        link=["link_1"],
        hour=[0],
        resource=["electricity"],
        year_dec=[2040, 2050],
        year_inv=[2030],
        year_op=[2030],
    ).copy(deep=True)
    p["carbon_goal"] = np.nan
    ghost = xr.DataArray(
        [[CONSUMPTION, CONSUMPTION, np.nan]], coords=[p.link, p.transport_tech]
    )  # the methane techs: no pair; the power line: no fuel
    p["transport_fuel_consumption"] = ghost
    p["transport_fuel_resource"] = xr.DataArray("electricity")
    factors = {name: xr.DataArray(TAXED).broadcast_like(ghost) for name in _FACTORS}
    _is_todays_model(p.assign(factors))
    # a zero consumption burns no fuel either
    p0 = p.assign(transport_fuel_consumption=ghost.fillna(0.0).where(p.transport_tech == "power_line", 0.0))
    _is_todays_model(p0.assign(factors))


def test_factor_non_zero_in_one_year_only(parameters_transport):
    """Two operating years: a factor that is zero in one of them emits only in the other, in each row."""
    p = _one_way(parameters_transport, year_op=(2030, 2040))
    taxed = xr.DataArray([0.0, TAXED], coords=[p.year_op])
    total = xr.DataArray([TOTAL, 0.0], coords=[p.year_op])
    _, base = _solve(p)
    _, s = _solve(p, taxed, total)
    burnt = s.operation_transport_power_in.sum(["link", "transport_tech"]) * CONSUMPTION * p.time_step_duration
    assert float(burnt.sel(year_op=2030).sum()) > 0 and float(burnt.sel(year_op=2040).sum()) > 0
    for name, factor in (("operation_carbon_emissions", taxed), ("operation_total_carbon_emissions", total)):
        extra = (s[name] - base[name]).sel(area="area_2")
        np.testing.assert_allclose(extra.transpose("year_op", "hour").values, (factor * burnt).transpose("year_op", "hour").values, atol=1e-9)
        np.testing.assert_allclose((s[name] - base[name]).sel(area="area_1").values, 0.0, atol=1e-9)
