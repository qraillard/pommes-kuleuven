"""flexdem: the vectorised window conservation matches the original per-window loop, and handles
time steps longer than 1 h.

Model: the synthetic 3-area dataset (pommes/tests/dataset_nnodes.py) over two weeks, with
its heat demand moved to a flexible demand (24 h window, max load ratio 3, no minimum load).
"""
import numpy as np
import pytest

import pommes.model.build_model as build_module
from pommes.model.build_model import build_model
from pommes.model.data_validation import ref_inputs
from pommes.tests.dataset_nnodes import build_nnode_parameters

N_HOURS = 336
WINDOW = 24


def _with_defaults(ds):
    """Add the parameters the benchmark dataset predates, at their dataset_description.yaml default
    (what check_inputs does; check_inputs itself rejects this dataset's 40-year storage life)."""
    for name, spec in ref_inputs.items():
        if name not in ds.variables:
            ds[name] = np.array(spec["default"], dtype=spec["type"])
    # The benchmark dataset carries annuity tables by year_dec: previous convention
    # (annuity_mode: until_decommissioning).
    ds["annuity_full_life"] = np.array(False)
    return ds


def _with_flexdem(ds, window=WINDOW):
    """Move the heat demand to a flexible demand with a conservation window of ``window`` hours."""
    ds = ds.copy()
    heat = ds.demand.sel(resource="heat")
    ds["flexdem"] = True
    ds["flexdem_demand"] = (ds.demand * 0).where(ds.resource != "heat", heat)
    ds["demand"] = ds.demand.where(ds.resource != "heat", 0.0)
    shape = (ds.sizes["area"], ds.sizes["resource"], ds.sizes["year_op"])
    dims = ("area", "resource", "year_op")
    ds["flexdem_conservation_hrs"] = (dims, np.full(shape, window, dtype="int64"))
    ds["flexdem_maxload_ratio"] = (dims, np.full(shape, 3.0))
    ds["flexdem_minload_ratio"] = (dims, np.zeros(shape))
    ds["flexdem_ramp_up"] = (dims, np.full(shape, np.nan))
    ds["flexdem_ramp_down"] = (dims, np.full(shape, np.nan))
    return ds


ENERGY_PER_STEP = ("demand", "flexdem_demand")  # summed when steps merge; the rest is averaged


def _to_steps(ds, step):
    """Merge every ``step`` hours into one time step of ``step`` hours.

    pommes-kul convention: demand, flexdem_demand (and the adequacy balance) are energies per
    time step, so they are summed; availabilities and prices are averaged.
    """
    out = ds.coarsen(hour=step).mean()
    for name in ENERGY_PER_STEP:
        out[name] = ds[name].coarsen(hour=step).sum()
    out["hour"] = np.arange(out.sizes["hour"], dtype="int64")
    out["time_step_duration"] = ("hour", np.full(out.sizes["hour"], float(step)))
    for name in ds.data_vars:  # coarsen drops the variables without an hour dim
        if "hour" not in ds[name].dims:
            out[name] = ds[name]
    out["operation_year_duration"] = ds.operation_year_duration
    return out


def _add_flexdem_master(model, model_parameters, annualised_totex_def, operation_adequacy_constraint):
    """Reference: the conservation part of flexdem.py as on master (one constraint per window)."""
    m, p = model, model_parameters
    flexdem_demand = p.flexdem_demand
    for dim in ["area", "resource", "year_op", "hour"]:
        if dim not in flexdem_demand.dims:
            flexdem_demand = flexdem_demand.expand_dims(dim={dim: p[dim]})
    conservation_hrs = p.flexdem_conservation_hrs
    x = m.add_variables(name="operation_flexdem_demand", lower=0,
                        coords=[p.area, p.resource, p.hour, p.year_op],
                        mask=(flexdem_demand != 0).any("hour"))
    operation_adequacy_constraint.lhs += -x
    m.add_constraints(x - p.flexdem_demand * p.flexdem_maxload_ratio <= 0, name="max",
                      mask=np.isfinite(p.flexdem_maxload_ratio))
    m.add_constraints(x.sum("hour") - flexdem_demand.sum("hour") == 0, name="yearly")
    for res in p.resource.values:
        for area in p.area.values:
            for year in p.year_op.values:
                cons_hr = int(conservation_hrs.sel(year_op=year, resource=res, area=area))
                if cons_hr > 0 and (flexdem_demand.sel(resource=res, area=area, year_op=year) != 0).any():
                    for hr in range(p.hour.values.min(), p.hour.values.max() - cons_hr + 1, cons_hr):
                        if hr + cons_hr <= p.hour.values.max():
                            rng = list(range(hr, hr + cons_hr))
                            m.add_constraints(
                                x.sel(hour=rng, resource=res, area=area, year_op=year).sum("hour")
                                - flexdem_demand.sel(hour=rng, resource=res, area=area, year_op=year).sum("hour")
                                == 0, name=f"w_{hr}_{res}_{area}_{year}")
    return m


def _solve(ds):
    m = build_model(ds)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


@pytest.fixture(scope="module")
def ds():
    return _with_defaults(_with_flexdem(build_nnode_parameters(n_areas=3, n_hours=N_HOURS)))


def test_same_optimum_as_master_loop(ds, monkeypatch):
    new = _solve(ds)
    monkeypatch.setattr(build_module, "add_flexdem", _add_flexdem_master)
    old = _solve(ds)
    assert new.objective.value == pytest.approx(old.objective.value, rel=1e-7)


def test_windows_hold_and_flexibility_is_used(ds):
    m = _solve(ds)
    x = m.solution["operation_flexdem_demand"].sel(resource="heat", year_op=2030)
    d = ds.flexdem_demand.sel(resource="heat", year_op=2030)
    xs = x.coarsen(hour=WINDOW).sum()
    ds_ = d.coarsen(hour=WINDOW).sum()
    np.testing.assert_allclose(xs.values, ds_.transpose(*xs.dims).values, rtol=1e-6)
    assert float(abs(x - d).max()) > 1.0  # the model does shift demand: the test is not vacuous


def test_one_constraint_block_per_window_length(ds):
    m = build_model(ds)
    names = [n for n in m.constraints if n.startswith("operation_flexdem_demand_conservation_")]
    assert names == [f"operation_flexdem_demand_conservation_{WINDOW}h"]
    con = m.constraints[names[0]]
    assert con.labels.sizes["window"] == N_HOURS // WINDOW


@pytest.mark.parametrize("step", [2, 3])
def test_time_steps_longer_than_one_hour(ds, step):
    """With step-hour time steps, a 24 h window holds 24 / step steps; energy is conserved per window."""
    coarse = _to_steps(ds, step)
    m = _solve(coarse)
    x = m.solution["operation_flexdem_demand"].sel(resource="heat", year_op=2030)
    d = coarse.flexdem_demand.sel(resource="heat", year_op=2030)
    per = WINDOW // step
    np.testing.assert_allclose(x.coarsen(hour=per).sum().values,
                               d.coarsen(hour=per).sum().transpose(*x.dims).values, rtol=1e-6)
    con = m.constraints[f"operation_flexdem_demand_conservation_{WINDOW}h"]
    assert con.labels.sizes["window"] == N_HOURS // WINDOW
