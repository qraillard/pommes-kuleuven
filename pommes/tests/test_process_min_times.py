"""Minimum start-up / shut-down times of the process units (unit commitment).

Three properties of the constraints in pommes/model/process.py:

- each distinct minimum time v applies only to the units whose minimum time is v (a unit does not
  inherit the longest minimum time of the other units);
- only the units whose constraints are not empty are constrained: process_no_shutdown == False and
  a finite unit size (a unit size of 0 is kept: the plant is then on or off through the state and
  the windows bind when process_min_load > 0);
- the window over the last v steps is written with a cumulative count of the start-ups (resp.
  shut-downs), C(t) = C(t-1) + startup(t), and C(t) - C(t-v) <= state(t): 3 terms a row instead of
  v + 1, same feasible set as the rolling sum.

``_add_reference`` keeps the formulation of the rolling sums, with the mask of the minimum time
fixed (``fixed``) or as before (``master``: every unit whose minimum time is above 1 gets every
window), to compare optima. The process module is only valid at 1 h steps.

The constraints count units: the units started in the last v steps must still be on (state >= sum
of the start-ups of the window), identical units being exchangeable. A step can switch off any of
the units that were on before it, the last one included. The hand-made
instances have 3 units of 1 MW per technology, one always on (it covers the base demand of 1 MW)
and two switched on at the demand peaks (3 MW). ``_extra_unit_hours`` counts the unit-hours above
the always-on unit.
"""
import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model

HEAT, HYDROGEN = "heat", "hydrogen"
RESOURCES = ["electricity", "heat", "hydrogen", "methane"]


# ---------------------------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------------------------

def _instance(parameters_dispatch_invest, process, demand, techs, years=(2020,)):
    """Process techs ``techs`` (list of dicts), one area, fossil mode only: methane -> ``out``.

    demand: array (year_op, hour, resource) in MWh per 1 h step. A tech has keys out (resource it
    produces), size, up, down, nosd, cap (power capacity, fixed), cu, cd (start-up / shut-down cost).
    """
    p = xr.merge([parameters_dispatch_invest, process])
    p = p.sel(
        area=["area_1"],
        conversion_tech=["wind_onshore"],
        resource=RESOURCES,
        year_dec=[2040] if len(years) == 1 else [2030, 2040, 2050],
        year_inv=[2020] if len(years) == 1 else [2020, 2030],
        year_op=list(years),
    ).copy(deep=True)
    n_tech, n_hour = len(techs), demand.shape[1]
    p = p.isel(process_tech=[0] * n_tech).assign_coords(process_tech=[f"tech_{i}" for i in range(n_tech)])
    p = p.drop_dims("hour").assign_coords(hour=np.arange(n_hour, dtype="int64"))
    p["time_step_duration"] = ("hour", np.ones(n_hour))
    p["conversion_availability"] = (("conversion_tech", "hour"), np.ones((1, n_hour)))
    p["demand"] = (("year_op", "hour", "resource"), np.asarray(demand, dtype="float64"))
    p["operation_year_duration"] = ("year_op", np.full(len(years), float(n_hour)))

    p["load_shedding_cost"] = xr.zeros_like(p["load_shedding_cost"]) + 1000.0
    p["load_shedding_cost"].loc[dict(resource="methane")] = 5.0  # methane is bought at 5 per MWh
    p["conversion_power_capacity_investment_max"] = 0.0
    p["conversion_power_capacity_investment_min"] = 0.0
    p["conversion_annuity_cost"] = 0.0
    p["conversion_fixed_cost"] = 0.0

    factor = np.full((n_tech, 2, len(RESOURCES)), 0.0)  # mode 0 (electric): all zero, not defined
    for i, t in enumerate(techs):
        factor[i, 1, RESOURCES.index(t["out"])] = 1.0
        factor[i, 1, RESOURCES.index("methane")] = -1.5
    p["process_factor"] = (("process_tech", "mode", "resource"), factor)
    p["process_emission_factor"] = xr.zeros_like(p["process_emission_factor"])

    def per_tech(key, default, dtype="float64"):
        return ("process_tech", np.array([t.get(key, default) for t in techs], dtype=dtype))

    p["process_unit_size"] = per_tech("size", 1.0)
    p["process_power_capacity_min"] = per_tech("cap", 2.0)
    p["process_power_capacity_max"] = per_tech("cap", 2.0)
    p["process_min_load"] = xr.zeros_like(p["process_min_load"]) + 1.0  # a unit that is on produces its size
    p["process_max_load"] = xr.zeros_like(p["process_max_load"]) + 1.0
    p["process_minimum_startup_time"] = per_tech("up", 1.0)
    p["process_minimum_shutdown_time"] = per_tech("down", 1.0)
    p["process_no_shutdown"] = per_tech("nosd", False, "bool")
    p["process_startup_cost"] = per_tech("cu", 0.0)
    p["process_shutdown_cost"] = per_tech("cd", 0.0)
    return p


def _peaks(n_hour, peaks, resource, base=1.0, extra=2.0):
    """Demand of ``base`` MWh a step, ``base + extra`` at the hours ``peaks`` (single year)."""
    d = np.zeros((1, n_hour, len(RESOURCES)))
    d[0, :, RESOURCES.index(resource)] = base
    d[0, list(peaks), RESOURCES.index(resource)] += extra
    return d


def _solve(m):
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


def _extra_unit_hours(m):
    """Unit-hours above the always-on unit, per technology: sum of the states minus the hours."""
    state = m.solution["operation_process_state"]  # (area, process_tech, hour, year_op)
    return (state.sum("hour") - state.sizes["hour"]).squeeze(["area", "year_op"]).values


# ---------------------------------------------------------------------------------------------
# Reference formulation: rolling sums
# ---------------------------------------------------------------------------------------------

def _unit_values(m, p, name):
    """Values of the (process_tech) parameter ``name`` of ``p``, as a (process_tech) array (one area, one year)."""
    return xr.DataArray(
        [float(p[name].sel(process_tech=t)) for t in p.process_tech.values],
        coords={"process_tech": p.process_tech.values},
    )


def _remove_min_time_blocks(m):
    for name in [n for n in m.constraints if n.startswith("minimum_") or n.endswith("_cumulative_def")]:
        m.remove_constraints(name)
    for name in [n for n in m.variables if n.endswith("_cumulative")]:
        m.remove_variables(name)


def _add_reference(m, p, mask="fixed"):
    """Replace the minimum-time blocks by rolling sums (``fixed`` or ``master`` mask)."""
    _remove_min_time_blocks(m)
    state = m.variables["operation_process_state"]
    start = m.variables["operation_process_startup"]
    stop = m.variables["operation_process_shutdown"]
    nb = m.variables["operation_process_nb_units"]
    size = _unit_values(m, p, "process_unit_size")
    nosd = _unit_values(m, p, "process_no_shutdown")
    up = _unit_values(m, p, "process_minimum_startup_time")
    down = _unit_values(m, p, "process_minimum_shutdown_time")
    if mask == "fixed":
        on_off = np.isfinite(size) & (nosd == 0)
    else:  # master: every unit with a minimum time above 1, whatever its unit size or no_shutdown
        on_off = xr.ones_like(size, dtype=bool)
    for v in np.unique(up.values):
        if v > 1:
            sel = ((up == v) if mask == "fixed" else (up > 1)) & on_off
            m.add_constraints(start.rolling(hour=int(v), min_periods=1).sum() - state <= 0,
                              name=f"ref_uptime_{v}", mask=sel)
    for v in np.unique(down.values):
        if v > 1:
            sel = ((down == v) if mask == "fixed" else (down > 1)) & on_off
            m.add_constraints(stop.rolling(hour=int(v), min_periods=1).sum() + state - nb <= 0,
                              name=f"ref_downtime_{v}", mask=sel)
    return m


# ---------------------------------------------------------------------------------------------
# (a) hand-computable dispatch: two techs with different minimum times
# ---------------------------------------------------------------------------------------------

N = 24


@pytest.fixture(scope="module")
def two_techs(parameters_dispatch_invest, process):
    def make(up, down, peaks):
        d = _peaks(N, peaks[0], HEAT) + _peaks(N, peaks[1], HYDROGEN)
        techs = [dict(out=HEAT, size=1.0, cap=3.0, up=up[0], down=down[0]),
                 dict(out=HYDROGEN, size=1.0, cap=3.0, up=up[1], down=down[1])]
        return _instance(parameters_dispatch_invest, process, d, techs)
    return make


def test_shorter_minimum_uptime_binds_for_its_own_tech(two_techs):
    """One peak of 3 MW at hour 10 (2 units started): at least 1 extra unit stays on for the v - 1
    following hours, so the extra unit-hours are 2 + (v - 1) = v + 1."""
    p = two_techs(up=(4, 8), down=(1, 1), peaks=([10], [10]))
    m = _solve(build_model(p))
    assert list(_extra_unit_hours(m)) == pytest.approx([5, 9])
    # master: both techs get both windows (4 and 8), so the first one has the 8 h window too
    m = _solve(_add_reference(build_model(p), p, mask="master"))
    assert list(_extra_unit_hours(m)) == pytest.approx([9, 9])


def test_shorter_minimum_downtime_binds_for_its_own_tech(two_techs):
    """Peaks of 3 MW at hours 8 and 12. Down 2 h: switched off at 9 (3 -> 1), back at 12, 3 -> 1 at
    13: 2 + 2 = 4 unit-hours. Down 6 h: a switch off at 9 forbids 3 units until 14, so the 2 extra
    units stay on from 8 to 12: 10 unit-hours."""
    p = two_techs(up=(1, 1), down=(2, 6), peaks=([8, 12], [8, 12]))
    m = _solve(build_model(p))
    assert list(_extra_unit_hours(m)) == pytest.approx([4, 10])
    m = _solve(_add_reference(build_model(p), p, mask="master"))
    assert list(_extra_unit_hours(m)) == pytest.approx([10, 10])


# ---------------------------------------------------------------------------------------------
# (b) random MILP: cumulative formulation == rolling sums with the fixed mask
# ---------------------------------------------------------------------------------------------

def _random_instance(parameters_dispatch_invest, process, seed, n_hour=14, years=(2020, 2030)):
    rng = np.random.default_rng(seed)
    d = np.zeros((len(years), n_hour, len(RESOURCES)))
    d[:, :, RESOURCES.index(HEAT)] = rng.integers(0, 5, size=(len(years), n_hour))
    techs = []
    for _ in range(3):
        size = float(rng.choice([1.0, 2.0]))
        techs.append(dict(out=HEAT, size=size, cap=size * int(rng.integers(2, 4)),
                          up=float(rng.choice([1, 2, 3, 5])), down=float(rng.choice([1, 2, 4])),
                          cu=float(rng.choice([0, 2, 5])), cd=float(rng.choice([0, 1]))))
    techs[0]["up"], techs[1]["up"] = 3.0, 5.0  # at least two distinct minimum times
    return _instance(parameters_dispatch_invest, process, d, techs, years)


def _without_minimum_times(p):
    free = p.copy(deep=True)
    free["process_minimum_startup_time"] = free["process_minimum_startup_time"] * 0 + 1.0
    free["process_minimum_shutdown_time"] = free["process_minimum_shutdown_time"] * 0 + 1.0
    return free


@pytest.mark.parametrize("seed", range(6))
def test_cumulative_formulation_matches_rolling_sums(parameters_dispatch_invest, process, seed):
    p = _random_instance(parameters_dispatch_invest, process, seed)
    cumulative = _solve(build_model(p)).objective.value
    reference = _solve(_add_reference(build_model(p), p, mask="fixed")).objective.value
    assert cumulative == pytest.approx(reference, rel=1e-7, abs=1e-6)
    assert _solve(build_model(_without_minimum_times(p))).objective.value <= cumulative + 1e-6


def test_the_random_instances_are_not_trivial(parameters_dispatch_invest, process):
    """The minimum times raise the optimum on at least one of the random instances."""
    gaps = []
    for seed in range(6):
        p = _random_instance(parameters_dispatch_invest, process, seed)
        gaps.append(_solve(build_model(p)).objective.value
                    - _solve(build_model(_without_minimum_times(p))).objective.value)
    assert max(gaps) > 1e-6


# ---------------------------------------------------------------------------------------------
# (c) units without on/off variables get no minimum-time constraint
# ---------------------------------------------------------------------------------------------

def test_no_shutdown_and_nan_unit_size_units_are_not_constrained(parameters_dispatch_invest, process):
    rng = np.random.default_rng(3)
    d = np.zeros((1, 14, len(RESOURCES)))
    d[0, :, RESOURCES.index(HEAT)] = rng.integers(0, 5, size=14)
    techs = [dict(out=HEAT, size=1.0, cap=3.0, up=3.0, down=3.0, cu=2.0),             # on/off
             dict(out=HEAT, size=1.0, cap=3.0, up=3.0, down=3.0, nosd=True),          # no_shutdown
             dict(out=HEAT, size=np.nan, cap=3.0, up=3.0, down=3.0, cu=2.0),          # no unit size
             dict(out=HEAT, size=0.0, cap=3.0, up=3.0, down=3.0, cu=2.0)]             # unit size 0: kept
    p = _instance(parameters_dispatch_invest, process, d, techs)
    m = build_model(p)
    rows = {f"tech_{i}": 0 for i in range(4)}
    for name in ("minimum_uptime_constraint_3.0", "minimum_downtime_constraint_3.0"):
        labels = m.constraints[name].labels  # -1 where masked
        for tech in rows:
            rows[tech] += int((labels.sel(process_tech=tech) != -1).sum())
    assert rows["tech_0"] > 0 and rows["tech_3"] > 0
    assert rows["tech_1"] == 0 and rows["tech_2"] == 0
    cumulative = _solve(m).objective.value
    # master constrained all of them (one distinct minimum time, so its mask was right): same optimum
    master = _solve(_add_reference(build_model(p), p, mask="master")).objective.value
    assert cumulative == pytest.approx(master, rel=1e-7, abs=1e-6)


# ---------------------------------------------------------------------------------------------
# (d) size of the minimum-time blocks (reported, not asserted)
# ---------------------------------------------------------------------------------------------

def min_time_block_size(m):
    """(rows, non-zeros) of the constraints that carry the minimum times (window rows and the
    definition rows of the cumulative counts)."""
    rows = nnz = 0
    for name in m.constraints:
        if name.startswith("minimum_") or name.endswith("_cumulative_def"):
            con = m.constraints[name]
            labels = con.labels.values.reshape(-1)
            vars_ = np.moveaxis(con.vars.values, con.vars.dims.index("_term"), -1).reshape(labels.size, -1)
            coeffs = np.moveaxis(con.coeffs.values, con.coeffs.dims.index("_term"), -1).reshape(labels.size, -1)
            live = labels != -1
            rows += int(live.sum())
            nnz += int(((vars_ != -1) & (coeffs != 0))[live].sum())
    return rows, nnz


def test_report_size_of_the_minimum_time_blocks(parameters_dispatch_invest, process, capsys):
    p = _random_instance(parameters_dispatch_invest, process, 0, n_hour=48)
    rows, nnz = min_time_block_size(build_model(p))
    with capsys.disabled():
        print(f"\nMIN_TIME_BLOCKS rows={rows} nonzeros={nnz}")
    assert rows > 0


# ---------------------------------------------------------------------------------------------
# (d) scalar (dimensionless) minimum times, e.g. the default of an omitted optional parameter
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scalar_up_only", [False, True])
@pytest.mark.parametrize("value", [0.0, 1.0, 3.0])
@pytest.mark.parametrize("n_techs", [1, 2])
def test_scalar_minimum_times_build_and_match_the_per_tech_values(
    parameters_dispatch_invest, process, value, n_techs, scalar_up_only
):
    """A scalar minimum time (no dimension) must build and solve, with per-tech unit sizes; it is
    the same as the same value given per tech."""
    d = _peaks(N, [8, 12], HEAT)
    techs = [dict(out=HEAT, size=1.0 + i, cap=3.0 * (1.0 + i)) for i in range(n_techs)]
    p = _instance(parameters_dispatch_invest, process, d, techs)
    per_tech = _solve(build_model(p.assign(
        process_minimum_startup_time=p["process_minimum_startup_time"] * 0 + value,
        process_minimum_shutdown_time=p["process_minimum_shutdown_time"] * 0 + value,
    ))).objective.value
    scalar = p.assign(process_minimum_startup_time=xr.DataArray(value))
    if not scalar_up_only:
        scalar = scalar.assign(process_minimum_shutdown_time=xr.DataArray(value))
    else:
        scalar = scalar.assign(process_minimum_shutdown_time=p["process_minimum_shutdown_time"] * 0 + value)
    assert _solve(build_model(scalar)).objective.value == pytest.approx(per_tech, rel=1e-7, abs=1e-6)
