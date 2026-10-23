"""Policies on the toy trade model: carbon leakage, CBAM, EU ETS maritime, fixed point on the EU carbon price."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "toy_policies", Path(__file__).resolve().parents[1] / "examples" / "toy_trade" / "toy_policies.py"
)
toy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(toy)


def _shipped(r, link, tech):
    return float(r["shipped"].loc[link, tech])


def test_without_cbam_the_eu_imports_grey_ammonia():
    r = toy.solve(toy.build(eu_price=100.0, cbam_phase_in=0.0, ets_maritime=False))
    assert r["condition"] == "optimal"
    assert _shipped(r, "NAF_EU", "ship_grey") == pytest.approx(toy.DEMAND_NH3)
    # the EU's own hydrogen turns green under its carbon price: the emissions move abroad
    assert r["production"].loc[("EU", "electrolyser")] == pytest.approx(toy.DEMAND_H2)


def test_cbam_switches_imports_to_green():
    r = toy.solve(toy.build(eu_price=100.0))
    assert _shipped(r, "NAF_EU", "ship_green") == pytest.approx(toy.DEMAND_NH3)
    assert _shipped(r, "NAF_EU", "ship_grey") == pytest.approx(0, abs=1e-6)


def test_policy_hurdles_are_linear_in_the_price():
    ds = toy.build(eu_price=100.0)
    grey = float(ds.transport_hurdle_costs.sel(link="NAF_EU", transport_tech="ship_grey").squeeze())
    green = float(ds.transport_hurdle_costs.sel(link="NAF_EU", transport_tech="ship_green").squeeze())
    maritime = 0.5 * toy.FUEL_OIL_EF * toy.BUNKER["NAF_EU"] * 100.0
    assert green == pytest.approx(maritime)
    assert grey == pytest.approx(maritime + toy.GREY_INTENSITY * 100.0)
    doubled = toy.build(eu_price=200.0)
    assert float(doubled.transport_hurdle_costs.sel(link="NAF_EU", transport_tech="ship_grey").squeeze()) == pytest.approx(2 * grey)


def test_fixed_point_on_the_eu_carbon_price():
    uncapped = toy.solve(toy.build(eu_price=0.0))
    cap = 0.5 * float(uncapped["emissions"].loc["EU"].sum())
    price, history, r = toy.fixed_point(cap)
    assert price > 0 and len(history) <= 6
    assert float(r["emissions"].loc["EU"].sum()) == pytest.approx(cap, rel=1e-6)
    # at the fixed point the CBAM charge uses the price the cap produces
    assert history[-1] == pytest.approx(history[-2], abs=0.1)
