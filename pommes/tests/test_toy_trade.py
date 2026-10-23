"""The toy trade model (pommes/examples/toy_trade): duals, trade margins, bunker fuel and link losses."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "toy_trade", Path(__file__).resolve().parents[1] / "examples" / "toy_trade" / "toy_trade.py"
)
toy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(toy)


@pytest.fixture(scope="module")
def base():
    return toy.solve_toy()


def test_toy_solves_and_meets_demand(base):
    assert base["condition"] == "optimal"
    flows = base["flows"].set_index(["link", "tech"])
    # the EU makes nothing itself here: all its hydrogen and ammonia come in
    assert flows.loc[("NAF_EU", "h2_pipe"), "delivered_mw"] == pytest.approx(1000)
    assert flows.loc[("NAF_EU", "nh3_ship"), "delivered_mw"] == pytest.approx(500)


def test_trade_margin_pays_exactly_for_the_link(base):
    """Zero-profit condition of the LP: price gap x flow = the link's annual cost."""
    for r in base["flows"].itertuples():
        assert r.trade_margin_eur == pytest.approx(base["link_cost"][r.tech], rel=1e-6)


def test_losses_and_bunker_fuel(base):
    flows = base["flows"].set_index(["link", "tech"])
    pipe, ship = flows.loc[("NAF_EU", "h2_pipe")], flows.loc[("NAF_EU", "nh3_ship")]
    assert pipe["injected_mw"] == pytest.approx(1000 / 0.98)
    assert pipe["fuel_mw"] == pytest.approx(1000 / 0.98 * 0.0475)
    assert ship["fuel_mw"] == pytest.approx(500 * 0.013)
    price = base["price"]
    # the importer pays at least the origin price plus the fuel burnt
    assert price.loc["EU", "ammonia"] > price.loc["NAF", "ammonia"] * 1.013


def test_without_north_africa_ammonia_comes_from_australia():
    ds = toy.build_toy_dataset()
    ds["conversion_availability"].loc[dict(area="NAF", conversion_tech="solar_pv")] = 0.0
    r = toy.solve_toy(ds)
    flows = r["flows"].set_index(["link", "tech"])
    assert ("AUS_EU", "nh3_ship") in flows.index and ("NAF_EU", "h2_pipe") not in flows.index
    aus = flows.loc[("AUS_EU", "nh3_ship")]
    assert aus["fuel_mw"] == pytest.approx(aus["injected_mw"] * 0.075)
    assert aus["trade_margin_eur"] == pytest.approx(r["link_cost"]["nh3_ship"], rel=1e-6)
    assert np.isfinite(r["price"].loc["EU", "hydrogen"])
