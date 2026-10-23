"""flexdem windows in a rolling horizon: ``flexdem_window_offset_hrs`` puts the windows on the calendar,
``flexdem_shifted_energy_initial`` carries the shifted energy committed before the horizon into its first window.

Toy built on add_flexdem alone: one area, 8 hourly steps (hours 1..8), flexible baseline 1 MWh per step for heat
(4 h windows) and steel (no window), max load ratio 3, and a price falling over the hours so the demand is pushed to the
end of each window.
"""

import numpy as np
import pytest
import xarray as xr
from linopy import Model

from pommes.model.flexdem import add_flexdem

N, WINDOW, CARRY = 8, 4, 1.5
WHOLE = "operation_flexdem_demand_conservation"
WINDOWS = f"operation_flexdem_demand_conservation_{WINDOW}h"


def _params(**extra):
    dims = ("area", "resource", "year_op")
    coords = dict(area=["a"], resource=["heat", "steel"], year_op=[2030], hour=np.arange(1, N + 1))
    per_res = lambda v: (dims, np.full((1, 2, 1), v))  # noqa: E731
    return xr.Dataset(
        dict(
            flexdem_demand=(("area", "resource", "hour", "year_op"), np.ones((1, 2, N, 1))),
            flexdem_conservation_hrs=(dims, np.array([[[WINDOW], [0]]], dtype="int64")),
            flexdem_maxload_ratio=per_res(3.0),
            flexdem_minload_ratio=per_res(0.0),
            flexdem_ramp_up=per_res(np.nan),
            flexdem_ramp_down=per_res(np.nan),
            time_step_duration=("hour", np.ones(N)),
            **extra,
        ),
        coords=coords,
    )


def _model(p):
    m = Model()
    supply = m.add_variables(lower=0, coords=[p.area, p.resource, p.hour, p.year_op], name="supply")
    adequacy = m.add_constraints(supply == 0, name="adequacy")  # becomes supply == flexdem demand
    add_flexdem(m, p, None, adequacy)
    m.objective = ((N + 1 - p.hour) * supply).sum()
    return m


def _solve(p):
    m = _model(p)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


def _window_sums(m, p, resource):
    shifted = (m.solution["operation_flexdem_demand"] - p.flexdem_demand).sel(resource=resource).squeeze()
    return shifted.values.reshape(-1).tolist()


def _steps_per_window(con):
    return (con.vars.sel(resource="heat").squeeze() != -1).sum("_term").values.tolist()


def test_offset_shifts_the_window_boundaries():
    assert _steps_per_window(_model(_params()).constraints[WINDOWS]) == [4, 4]
    # Start 2 h into a window: hours 1-2 close it, 3-6 and 7-8 (cut by the horizon end) follow.
    assert _steps_per_window(_model(_params(flexdem_window_offset_hrs=2.0)).constraints[WINDOWS]) == [2, 4, 2]


def test_carry_in_closes_the_first_window():
    p = _params(flexdem_window_offset_hrs=2.0,
                flexdem_shifted_energy_initial=(("area", "resource", "year_op"), np.array([[[CARRY], [0.0]]])))
    m = _solve(p)
    s = _window_sums(m, p, "heat")
    assert [sum(s[:2]), sum(s[2:6]), sum(s[6:])] == pytest.approx([-CARRY, 0.0, 0.0], abs=1e-6)
    assert sum(_window_sums(m, p, "steel")) == pytest.approx(0.0, abs=1e-6)  # nothing carried in
    assert max(abs(v) for v in s) > 0.4  # demand is shifted: the test is not vacuous


def test_whole_horizon_conservation_is_kept_and_counts_the_carry():
    m = _model(_params())
    assert (m.constraints[WHOLE].labels.squeeze() != -1).all()  # a stand-alone model keeps its rows
    windows = m.constraints[WINDOWS].labels.squeeze()
    assert (windows.sel(resource="heat") != -1).all() and (windows.sel(resource="steel") == -1).all()
    # With a carry, the whole-horizon row is the sum of the window rows: still feasible, nothing forced per block.
    p = _params(flexdem_shifted_energy_initial=(("area", "resource", "year_op"), np.full((1, 2, 1), CARRY)))
    m = _solve(p)
    assert sum(_window_sums(m, p, "heat")) == pytest.approx(-CARRY, abs=1e-6)
    assert sum(_window_sums(m, p, "steel")) == pytest.approx(-CARRY, abs=1e-6)  # no window: the horizon is its window


def test_without_the_inputs_the_model_is_unchanged():
    without = _model(_params())
    default = _model(_params(flexdem_window_offset_hrs=0.0, flexdem_shifted_energy_initial=0.0))
    assert list(without.constraints) == list(default.constraints)
    for name in without.constraints:
        a, b = without.constraints[name], default.constraints[name]
        for field in ("labels", "coeffs", "vars", "rhs"):
            np.testing.assert_array_equal(getattr(a, field).values, getattr(b, field).values, err_msg=name)
