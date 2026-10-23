"""CO2 accounting and cumulative CO2 storage on the toy of pommes/examples/toy_co2 (POMMES-EU-Industry M01)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pommes.model.build_model import build_model

_spec = importlib.util.spec_from_file_location(
    "toy_co2", Path(__file__).resolve().parents[1] / "examples" / "toy_co2" / "toy_co2.py"
)
toy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(toy)

YEAR = toy.CEMENT * toy.HOURS          # t of cement a year


def _emissions(r, year=2030):
    return float(r["emissions"].loc[year])


def test_unabated_biomass_nets_to_its_process_emissions():
    """Biogenic CO2 of the kiln is cancelled by the upstream credit: only calcination is left."""
    r = toy.solve(toy.build(["kiln_bio"]))
    assert r["condition"] == "optimal"
    assert _emissions(r) == pytest.approx(toy.PROCESS_EF * YEAR, rel=1e-9)


def test_fossil_capture_counts_the_uncaptured_share():
    r = toy.solve(toy.build(["kiln_gas_ccs"]))
    assert _emissions(r) == pytest.approx((1 - toy.CAPTURE) * toy.KILN_GAS * YEAR, rel=1e-9)
    assert float(r["stored"].loc[2030]) == pytest.approx(toy.CAPTURE * toy.KILN_GAS * YEAR, rel=1e-9)


def test_beccs_is_net_negative_and_pays_no_tax():
    """Net-negative area under a positive tax: no carbon cost, and no payment for the removal.

    With carbon cost = tax x emissions and a lower bound of 0, net emissions were forced >= 0:
    the solver shed the credited biomass at the load-shedding cost instead."""
    r = toy.solve(toy.build(["kiln_bio_ccs"], tax=100.0))
    assert r["condition"] == "optimal"
    expected = ((1 - toy.CAPTURE) * toy.KILN_BIO - toy.BIO_C * 1.1) * YEAR
    assert expected < 0
    assert _emissions(r) == pytest.approx(expected, rel=1e-9)
    assert float(r["carbon_costs"].sum()) == pytest.approx(0, abs=1e-6)
    assert r["shed"] == pytest.approx(0, abs=1e-6)


def test_removals_offset_the_area_tax_down_to_zero_only():
    """DAC (110 EUR/t with storage) runs under a 200 EUR/t tax until the area is at net zero, not beyond."""
    r = toy.solve(toy.build(["kiln_gas", "dac"], tax=200.0, dac_max=100.0))
    kiln = toy.KILN_GAS * toy.CEMENT                                 # 70 t/h
    assert float(r["power"].sel(area="IND", conversion_tech="dac").squeeze()) == pytest.approx(kiln)
    assert _emissions(r) == pytest.approx(0, abs=1e-6)
    assert float(r["carbon_costs"].sum()) == pytest.approx(0, abs=1e-6)


def test_co2_used_in_efuel_counts_as_released():
    """Captured fossil CO2 sent to e-fuel is emitted when the fuel is burnt; the rest is stored."""
    efuel = 50.0
    r = toy.solve(toy.build(["kiln_gas_ccs", "efuel_synthesis"], efuel=efuel))
    captured = toy.CAPTURE * toy.KILN_GAS * toy.CEMENT
    used = toy.EFUEL_CO2 * efuel
    assert float(r["stored"].loc[2030]) == pytest.approx((captured - used) * toy.HOURS, rel=1e-9)
    physical = toy.KILN_GAS * toy.CEMENT - (captured - used)        # fossil CO2 not stored, t/h
    assert _emissions(r) == pytest.approx(physical * toy.HOURS, rel=1e-9)


def test_dac_with_storage_removes_one_tonne_per_tonne():
    r = toy.solve(toy.build(["kiln_gas", "dac"], tax=200.0, dac_max=10.0))
    assert float(r["power"].sel(area="IND", conversion_tech="dac").squeeze()) == pytest.approx(10.0)
    assert _emissions(r) == pytest.approx((toy.KILN_GAS * toy.CEMENT - 10.0) * toy.HOURS, rel=1e-9)


def test_open_co2_spillage_hides_emissions():
    """Why the study closes spillage of co2: spilled CO2 leaves the books (assembly rule)."""
    closed = toy.solve(toy.build(["kiln_gas_ccs"]))
    leak = toy.solve(toy.build(["kiln_gas_ccs"], open_spillage=("co2",)))
    captured = toy.CAPTURE * toy.KILN_GAS * YEAR
    assert float(closed["spilled"].sum().sum()) == 0
    assert float(leak["spilled"].loc[2030, "co2"]) == pytest.approx(captured, rel=1e-9)
    assert float(leak["stored"].loc[2030]) == pytest.approx(0, abs=1e-6)
    # reported emissions are the same, but the spilled tonnes were really emitted
    assert _emissions(leak) == pytest.approx(_emissions(closed), rel=1e-9)


def test_cumulative_storage_limit_over_two_operation_years():
    """Each operation year stands for 10 years; the site holds 15 years of the plant's capture.

    The tax is paid earlier at a higher discount factor, so 2030 stores all it captures (10 years)
    and 2040 the rest (half its capture over its 10 years)."""
    annual = toy.CAPTURE * toy.KILN_GAS * YEAR
    ds = toy.build(["kiln_gas", "kiln_gas_ccs"], year_ops=(2030, 2040), cumulative=15 * annual)
    r = toy.solve(ds)
    assert r["condition"] == "optimal"
    assert float(r["stored"].loc[2030]) == pytest.approx(annual, rel=1e-9)
    assert float(r["stored"].loc[2040]) == pytest.approx(0.5 * annual, rel=1e-9)
    assert toy.cumulative_used(ds, r) == pytest.approx(15 * annual, rel=1e-9)
    assert "operation_conversion_cumulative_production_max_constraint" in r["model"].constraints


def test_cumulative_storage_limit_in_myopic_runs():
    """Myopic chain: the 2040 run gets the capacity the 2030 run left; same result as above here."""
    annual = toy.CAPTURE * toy.KILN_GAS * YEAR
    total = 15 * annual
    ds30 = toy.build(["kiln_gas", "kiln_gas_ccs"], year_ops=(2030,), cumulative=total)
    r30 = toy.solve(ds30)
    used = toy.cumulative_used(ds30, r30)
    assert used == pytest.approx(10 * annual, rel=1e-9)
    ds40 = toy.build(["kiln_gas", "kiln_gas_ccs"], year_ops=(2040,), cumulative=total - used)
    r40 = toy.solve(ds40)
    assert float(r40["stored"].loc[2040]) == pytest.approx(0.5 * annual, rel=1e-9)
    assert used + toy.cumulative_used(ds40, r40) == pytest.approx(total, rel=1e-9)


def test_yearly_injection_limit():
    annual = toy.CAPTURE * toy.KILN_GAS * YEAR
    r = toy.solve(toy.build(["kiln_gas", "kiln_gas_ccs"], injection_max=0.25 * annual))
    assert float(r["stored"].loc[2030]) == pytest.approx(0.25 * annual, rel=1e-9)


def test_cumulative_limit_needs_the_year_weight():
    ds = toy.build(["kiln_gas_ccs"], cumulative=1e6, weight=None)
    assert np.isnan(ds.operation_year_weight).all()
    with pytest.raises(ValueError, match="operation_year_weight"):
        build_model(ds)


def test_no_cumulative_constraint_without_a_limit():
    model = build_model(toy.build(["kiln_gas_ccs"]))
    assert "operation_conversion_cumulative_production_max_constraint" not in model.constraints
