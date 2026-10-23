"""Shut-downs and ramps of the process units (unit commitment) in pommes/model/process.py.

An EAF-like technology: 2 units of 1 MW, min load 0.75, max load 1.0, no minimum times. The
state is the number of units on after the step, so a step can stop the units on before it (the
last one included), and the ramps let a started (stopped) unit jump to (drop from) its min load,
while a unit that stays on still ramps at most ``ramp`` per hour (a start-up/shut-down pair buys no
extra ramp, and a stopping unit cannot borrow the ramp of the others). Instances from
``test_process_min_times._instance`` (methane -> heat, spillage free, heat shedding at 1000).
"""
import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.tests.test_process_min_times import HEAT, RESOURCES, _instance, _solve


def _eaf(parameters_dispatch_invest, process, heat, ramp=np.nan):
    """One tech, 2 units of 1 MW, min load 0.75; ``heat``: demand (MWh) a step."""
    d = np.zeros((1, len(heat), len(RESOURCES)))
    d[0, :, RESOURCES.index(HEAT)] = heat
    p = _instance(parameters_dispatch_invest, process, d, [dict(out=HEAT, size=1.0, cap=2.0)])
    p["process_min_load"] = xr.zeros_like(p["process_min_load"]) + 0.75
    p["process_ramp_up"] = xr.DataArray(ramp)
    p["process_ramp_down"] = xr.DataArray(ramp)
    return p


def _state_power(m):
    state = m.solution["operation_process_state"].squeeze(["area", "process_tech", "year_op"])
    power = m.solution["operation_process_power"].sum("mode").squeeze(["area", "process_tech", "year_op"])
    return state.values, power.values


def test_the_whole_fleet_stops_and_restarts(parameters_dispatch_invest, process):
    """No demand at hours 4-7: both units stop at once (2 -> 0) and restart at hour 8."""
    p = _eaf(parameters_dispatch_invest, process, [2.0] * 4 + [0.0] * 4 + [2.0] * 4)
    state, _ = _state_power(_solve(build_model(p)))
    assert list(state) == pytest.approx([2] * 4 + [0] * 4 + [2] * 4)


def test_the_last_unit_drops_from_min_load_to_zero_and_back(parameters_dispatch_invest, process):
    """Ramp 0.3/h < min load 0.75: the last unit still stops (0.75 -> 0) and starts (0 -> 0.75) in one hour."""
    p = _eaf(parameters_dispatch_invest, process, [0.75] * 4 + [0.0] * 4 + [0.75] * 4, ramp=0.3)
    state, power = _state_power(_solve(build_model(p)))
    assert list(state) == pytest.approx([1] * 4 + [0] * 4 + [1] * 4)
    assert list(power) == pytest.approx([0.75] * 4 + [0.0] * 4 + [0.75] * 4)


def test_a_unit_stops_from_min_load_and_the_others_ramp_at_most_ramp(parameters_dispatch_invest, process):
    """2 MW (both units at 1.0) -> 0.75 MW wanted from hour 4, ramp 0.3: a unit can only stop from its
    min load and may not borrow the ramp of the one staying on, so both go to 0.75 at hour 4 (a 0.25
    drop each), then one stops at hour 5."""
    p = _eaf(parameters_dispatch_invest, process, [2.0] * 4 + [0.75] * 4, ramp=0.3)
    state, power = _state_power(_solve(build_model(p)))
    assert list(state) == pytest.approx([2] * 5 + [1] * 3)
    assert list(power) == pytest.approx([2.0] * 4 + [1.5] + [0.75] * 3)


def test_a_plant_without_unit_size_stops_and_restarts_despite_a_slow_ramp(parameters_dispatch_invest, process):
    """Copper-smelter-like plant without unit size (one on/off block of 1 MW): min load 0.9, ramp 0.01/h.
    No demand at hours 4-7: running costs methane, so it stops and restarts, the start-up and shut-down
    moving it by its min load although the ramp alone would take 90 h."""
    d = np.zeros((1, 12, len(RESOURCES)))
    d[0, :, RESOURCES.index(HEAT)] = [0.9] * 4 + [0.0] * 4 + [0.9] * 4
    p = _instance(parameters_dispatch_invest, process, d, [dict(out=HEAT, size=0.0, cap=1.0)])
    p["process_min_load"] = xr.zeros_like(p["process_min_load"]) + 0.9
    p["process_ramp_up"] = xr.DataArray(0.01)
    p["process_ramp_down"] = xr.DataArray(0.01)
    state, power = _state_power(_solve(build_model(p)))
    assert list(state) == pytest.approx([1] * 4 + [0] * 4 + [1] * 4)
    assert list(power) == pytest.approx([0.9] * 4 + [0.0] * 4 + [0.9] * 4)


@pytest.mark.parametrize("swap, target, feasible", [(0, 1.7, True), (1, 1.65, False), (1, 1.6, True)])
def test_a_start_up_shut_down_pair_buys_no_ramp(parameters_dispatch_invest, process, swap, target, feasible):
    """2 units on at 1.5 MW, ramp 0.1/h, free start-ups/shut-downs, a surplus spilled for free (only the
    ramps limit hour 4). Without a swap the fleet ramps 2 x 0.1; stopping one unit while starting the other
    (state stays 2) leaves one unit ramping, so 1.5 -> 1.6 at most, not the extra min load (0.75) a
    start-up term alone would grant."""
    p = _eaf(parameters_dispatch_invest, process, [1.5] * 8, ramp=0.1)
    m = build_model(p)
    power = m.variables["operation_process_power"].sum("mode")
    m.add_constraints(m.variables["operation_process_startup"].sel(hour=4) == swap)
    m.add_constraints(m.variables["operation_process_shutdown"].sel(hour=4) == swap)
    m.add_constraints(power.sel(hour=3) == 1.5)
    m.add_constraints(power.sel(hour=4) >= target)
    _, condition = m.solve(solver_name="highs")
    assert (condition == "optimal") == feasible


@pytest.mark.parametrize("swap, target, feasible", [(0, 1.55, True), (1, 1.6, False), (1, 1.65, True)])
def test_a_start_up_shut_down_pair_buys_no_ramp_down(parameters_dispatch_invest, process, swap, target, feasible):
    """Mirror of the above, downwards: 2 units on at 1.75 MW (one at min load 0.75, so it may stop), ramp
    0.1/h. Without a swap the fleet drops 2 x 0.1; with a swap only the unit kept on ramps, so 1.75 -> 1.65
    at least."""
    p = _eaf(parameters_dispatch_invest, process, [1.5] * 8, ramp=0.1)
    m = build_model(p)
    power = m.variables["operation_process_power"].sum("mode")
    m.add_constraints(m.variables["operation_process_startup"].sel(hour=4) == swap)
    m.add_constraints(m.variables["operation_process_shutdown"].sel(hour=4) == swap)
    m.add_constraints(power.sel(hour=3) == 1.75)
    m.add_constraints(power.sel(hour=4) <= target)
    _, condition = m.solve(solver_name="highs")
    assert (condition == "optimal") == feasible
