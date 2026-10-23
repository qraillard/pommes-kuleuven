"""Time steps longer than 1 h.

pommes-kul convention: demand, flexdem demand and the adequacy balance are energies per time step
(net generation = time_step_duration x power), so a 3 h model built from data summed over 3 h
blocks reaches the optimum of the 1 h model whose data are constant over the same blocks. Windows
(flexdem conservation, daily production limit) must be counted in hours, not in time steps.

Model: the synthetic 3-area benchmark (benchmarks/dataset_nnodes.py) over two weeks, with flexible
heat (24 h windows), a daily production limit on the OCGT, and no storage dissipation (so that a
3 h step and three identical 1 h steps are exactly the same storage dynamics). Its capacities are
built (end of life aligned with the decommissioning years): the tests assert it.

Load shedding and spillage are powers (caps in MW): the adequacy balance multiplies them by the
step duration and their costs are per MWh of the energy shed or spilled.
"""
import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.tests.test_flexdem import N_HOURS, _to_steps, _with_defaults, _with_flexdem
from dataset_nnodes import build_nnode_parameters  # noqa: E402  (path set by test_flexdem)

STEP = 3


def _instance():
    ds = _with_flexdem(build_nnode_parameters(n_areas=3, n_hours=N_HOURS))
    # The benchmark's end of life (investment year + 25 or 15..40 years) matches none of its
    # decommissioning years (2030, 2040, 2050), so nothing could be built and every demand was shed.
    # An investment of 2020 (resp. 2030) ends in 2040 (resp. 2050).
    for tech in ("conversion", "storage"):
        eol = ds[f"{tech}_end_of_life"]
        ds[f"{tech}_end_of_life"] = xr.zeros_like(eol) + ds.year_inv + 20
    ds["storage_dissipation"] = ds.storage_dissipation * 0.0
    daily = np.full((ds.sizes["area"], ds.sizes["conversion_tech"], ds.sizes["year_op"]), np.nan)
    daily[:, list(ds.conversion_tech.values).index("ocgt"), :] = 1500.0  # MWh per day
    ds["conversion_max_daily_production"] = (("area", "conversion_tech", "year_op"), daily)
    return _with_defaults(ds)


def _piecewise_constant(ds, step):
    """Hourly data constant over each ``step``-hour block (block means repeated ``step`` times)."""
    coarse = ds.coarsen(hour=step).mean()
    out = ds.copy()
    for name in ds.data_vars:
        if "hour" in ds[name].dims:
            out[name] = coarse[name].isel(hour=np.repeat(np.arange(coarse.sizes["hour"]), step)) \
                .assign_coords(hour=ds.hour).transpose(*ds[name].dims)
    return out


def _solve(ds):
    m = build_model(ds)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


@pytest.fixture(scope="module")
def ds():
    return _instance()


def test_same_optimum_at_1h_and_3h(ds):
    hourly = _solve(_piecewise_constant(ds, STEP))
    coarse = _solve(_to_steps(ds, STEP))
    for m in (hourly, coarse):  # the comparison is only meaningful if capacity is built and used
        assert float(m.solution["planning_conversion_power_capacity"].sum()) > 100.0
        assert float(m.solution["planning_storage_power_capacity"].sum()) > 10.0
    produced = [float(m.solution["operation_conversion_power"].sum()) * dt  # MWh = power x step hours
                for m, dt in ((hourly, 1), (coarse, STEP))]
    assert produced[0] > 1e4
    assert produced[1] == pytest.approx(produced[0], rel=1e-3)
    assert coarse.objective.value == pytest.approx(hourly.objective.value, rel=1e-6)


def test_daily_limit_groups_the_steps_of_24_hours(ds):
    """Each row of the daily limit holds exactly the 8 three-hour steps that start in its day."""
    m = build_model(_to_steps(ds, STEP))
    con = m.constraints["operation_conversion_daily_production_max_constraint"]
    lab = m.variables["operation_conversion_power"].labels
    hour_index = np.indices(lab.shape)[lab.dims.index("hour")]
    step_of_label = dict(zip(lab.values.ravel().tolist(), hour_index.ravel().tolist()))
    others = [d for d in con.labels.dims if d != "day"]
    n_days = con.labels.sizes["day"]
    row_labels = con.labels.transpose(*others, "day").values.reshape(-1, n_days)
    terms = con.vars.transpose(*others, "day", "_term").values.reshape(-1, n_days, con.vars.sizes["_term"])
    rows = 0
    for r in range(row_labels.shape[0]):
        for d in range(n_days):
            if row_labels[r, d] == -1:  # masked: no limit for this unit
                continue
            rows += 1
            steps = sorted(step_of_label[int(v)] for v in terms[r, d] if v != -1)
            assert steps == list(range(d * 24 // STEP, (d + 1) * 24 // STEP)), (r, d, steps)
    assert rows == 3 * (N_HOURS // 24)  # one OCGT per area, every day


def test_windows_are_counted_in_hours(ds):
    m = build_model(_to_steps(ds, STEP))
    assert m.constraints["operation_conversion_daily_production_max_constraint"].labels.sizes["day"] \
        == N_HOURS // 24
    assert m.constraints["operation_flexdem_demand_conservation_24h"].labels.sizes["window"] == N_HOURS // 24


def test_import_emissions_do_not_depend_on_the_step(ds):
    """Methane imports with an emission factor: same tonnes and optimum at 1 h and 3 h."""
    d = ds.copy()
    d["net_import"] = True
    res = list(d.resource.values)
    dims = ("area", "hour", "resource", "year_op")
    shape = tuple(d.sizes[k] for k in dims)
    price, ef = np.zeros(shape), np.zeros(shape)
    price[:, :, res.index("methane"), :] = 40.0
    ef[:, :, res.index("methane"), :] = 0.2  # t CO2 per MWh imported
    d["net_import_import_price"] = (dims, price)
    d["net_import_emission_factor"] = (dims, ef)
    d["net_import_total_emission_factor"] = (dims, ef)
    limit = np.full((d.sizes["area"], d.sizes["resource"], d.sizes["year_op"]), np.nan)
    limit[:, res.index("electricity"), :] = 0.0  # only methane is imported
    limit[:, res.index("heat"), :] = 0.0
    d["net_import_max_yearly_energy_import"] = (("area", "resource", "year_op"), limit)
    hourly = _solve(_piecewise_constant(d, STEP))
    coarse = _solve(_to_steps(d, STEP))
    imported = float(hourly.solution["operation_net_import_import"].sel(resource="methane").sum())
    assert imported > 0
    assert float(coarse.solution["operation_total_carbon_emissions"].sum()) == pytest.approx(
        float(hourly.solution["operation_total_carbon_emissions"].sum()), rel=1e-6)
    assert coarse.objective.value == pytest.approx(hourly.objective.value, rel=1e-6)


# ---------------------------------------------------------------------------------------------
# Load shedding and spillage are powers: energy = time_step_duration x power
# ---------------------------------------------------------------------------------------------

def _shedding_instance():
    """The 3-area instance with too little capacity (shedding) and a revenue for spilling electricity.

    The caps are powers [MW]; data are constant within the 3 h blocks, so the 1 h and the 3 h models
    are the same model.
    """
    ds = _instance()
    ds["conversion_power_capacity_investment_max"] = ds.conversion_power_capacity_investment_max * 0.25
    ds["conversion_availability"] = ds.conversion_availability.where(ds.conversion_tech != "wind_onshore", 1.0)
    ds["load_shedding_cost"] = ds.load_shedding_cost * 0 + 500.0
    # A negative cost (a revenue, EUR per MWh spilled) makes the model spill electricity up to the cap.
    ds["spillage_cost"] = ("resource", [-5.0, 0.0, 0.0])
    ds["load_shedding_max_capacity"] = ("resource", np.full(3, 150.0))  # MW
    ds["spillage_max_capacity"] = ("resource", np.full(3, 40.0))  # MW
    return ds


def _energy(m, ds, name):
    """Energy [MWh] of a power variable: sum of time_step_duration x power."""
    return float((m.solution[name] * ds.time_step_duration).sum())


def test_shedding_and_spillage_energies_do_not_depend_on_the_step():
    ds = _shedding_instance()
    ds_hourly, ds_coarse = _piecewise_constant(ds, STEP), _to_steps(ds, STEP)
    hourly, coarse = _solve(ds_hourly), _solve(ds_coarse)
    shed = [_energy(m, d, "operation_load_shedding_power") for m, d in ((hourly, ds_hourly), (coarse, ds_coarse))]
    spill = [_energy(m, d, "operation_spillage_power") for m, d in ((hourly, ds_hourly), (coarse, ds_coarse))]
    assert shed[0] > 100.0 and spill[0] > 100.0  # both really occur
    assert shed[1] == pytest.approx(shed[0], rel=1e-6)
    assert spill[1] == pytest.approx(spill[0], rel=1e-6)
    assert coarse.objective.value == pytest.approx(hourly.objective.value, rel=1e-6)


def test_caps_are_powers():
    """At a 3 h step the caps (MW) bound the powers, not the energies per step: they are reached, not exceeded."""
    coarse = _solve(_to_steps(_shedding_instance(), STEP))
    shed = float(coarse.solution["operation_load_shedding_power"].max())
    spill = float(coarse.solution["operation_spillage_power"].max())
    assert shed == pytest.approx(150.0, abs=1e-6)  # binds: the shortage is larger
    assert spill == pytest.approx(40.0, abs=1e-6)
