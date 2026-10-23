"""Minimum energy-to-power ratio of a storage (``storage_energy_to_power_min``, hours).

Toy (single horizon 2020, area_1, wind_onshore, hours 2 and 3, electricity only): the wind farm charges the storage in
hour 2 and the storage serves the demand in hour 3 (no load shedding), so a storage power of a few tens of MW is needed.
The storage is made of cheap power and dear energy, so that without the parameter the optimum takes an energy-to-power
ratio far below 300 h. With ``storage_energy_to_power_min = 300`` the energy capacity must be at least 300 times the
power capacity, which costs more.
"""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs

RATIO = 300.0
CONSTRAINT = "operation_storage_energy_to_power_min_constraint"


def _toy(parameters_dispatch_invest, storage, techs=("battery",)):
    """Two hours, electricity only, storage techs ``techs`` with cheap power and dear energy, no dissipation."""
    p = (
        xr.merge([parameters_dispatch_invest, storage])
        .sel(
            area=["area_1"],
            conversion_tech=["wind_onshore"],
            hour=[2, 3],
            resource=["electricity"],
            storage_tech=list(techs),
            year_dec=[2040],
            year_inv=[2020],
            year_op=[2020],
        )
        .copy(deep=True)
    )
    p["storage_factor_in"] = -1.0
    p["storage_factor_out"] = 1.0
    p["storage_dissipation"] = 0.0
    p["storage_invest_cost_power"] = 10.0
    p["storage_invest_cost_energy"] = 1000.0
    # room for E = 300 P; the power investment is bounded per tech (see the tests)
    p["storage_energy_capacity_investment_max"] = 1e6
    return p


def _solve(p):
    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.termination_condition == "optimal"
    return model, model.solution.squeeze()


def _rows(model):
    """Number of rows the model really holds for the constraint (a fully masked constraint keeps its name, with labels -1)."""
    if CONSTRAINT not in model.constraints:
        return 0
    return int((model.constraints[CONSTRAINT].labels.to_numpy() != -1).sum())


def _ratio(s):
    return s.operation_storage_energy_capacity / s.operation_storage_power_capacity


@pytest.fixture()
def toy(parameters_dispatch_invest, storage):
    return _toy(parameters_dispatch_invest, storage)


def test_the_ratio_raises_the_energy_capacity_and_the_cost(toy):
    """Without the parameter E/P is below 300; with 300 the solution has E >= 300 P and the cost is higher."""
    free_model, free = _solve(toy)
    assert _rows(free_model) == 0
    assert float(_ratio(free)) < RATIO
    assert float(free.operation_storage_power_capacity) > 0

    p = toy.copy(deep=True)
    p["storage_energy_to_power_min"] = xr.DataArray([RATIO], coords=[p.storage_tech])
    model, s = _solve(p)
    assert _rows(model) == 1
    power = float(s.operation_storage_power_capacity)
    energy = float(s.operation_storage_energy_capacity)
    assert power > 0
    assert energy >= RATIO * power - 1e-6
    # the power needed is the same, only the energy is oversized
    assert power == pytest.approx(float(free.operation_storage_power_capacity), rel=1e-6)
    assert model.objective.value > free_model.objective.value
    # the extra cost is the extra energy at its annuity: the LP takes the cheapest way, E = 300 P exactly
    assert energy == pytest.approx(RATIO * power, rel=1e-6)


def test_the_ratio_is_a_lower_bound_not_an_equality(toy):
    """A ratio below the optimum E/P changes nothing: the constraint is built but not binding."""
    free_model, free = _solve(toy)
    p = toy.copy(deep=True)
    p["storage_energy_to_power_min"] = xr.DataArray(
        [0.5 * float(_ratio(free))], coords=[p.storage_tech]
    )
    model, s = _solve(p)
    assert _rows(model) == 1
    assert model.objective.value == pytest.approx(free_model.objective.value, rel=1e-9)
    assert float(_ratio(s)) >= 0.5 * float(_ratio(free)) - 1e-9


def test_nan_for_one_tech_and_300_for_another(parameters_dispatch_invest, storage):
    """Two electricity storages in the same toy: ``a`` (NaN) keeps its own ratio, ``b`` (300) is bound.

    ``a`` is limited to 4 MW (the toy needs 10 MW of power) so that ``b`` is needed too; the row exists for ``b`` only.
    """
    p = _toy(parameters_dispatch_invest, storage, techs=("battery", "tank_hydrogen"))
    p["storage_main_resource"] = xr.DataArray(["electricity"] * 2, coords=[p.storage_tech])
    p["storage_power_capacity_investment_max"] = xr.DataArray(
        [4.0, 1000.0], coords=[p.storage_tech]
    )
    p["storage_energy_to_power_min"] = xr.DataArray([np.nan, RATIO], coords=[p.storage_tech])

    model, s = _solve(p)
    rows = model.constraints[CONSTRAINT].labels
    has_row = (rows != -1).any(["area", "year_op"])
    assert list(has_row.storage_tech.to_numpy()[has_row.to_numpy()]) == ["tank_hydrogen"]

    power = s.operation_storage_power_capacity.to_series()
    energy = s.operation_storage_energy_capacity.to_series()
    assert power["battery"] == pytest.approx(4.0, rel=1e-6)
    assert power["tank_hydrogen"] == pytest.approx(6.0, rel=1e-6)
    assert energy["tank_hydrogen"] >= RATIO * power["tank_hydrogen"] - 1e-6
    assert energy["battery"] / power["battery"] < RATIO


def test_the_ratio_applies_per_operation_year(parameters_dispatch_invest, storage):
    """A ratio given in 2020 and NaN in 2030 builds the row for 2020 only."""
    p = (
        xr.merge([parameters_dispatch_invest, storage])
        .sel(
            area=["area_1"],
            conversion_tech=["wind_onshore"],
            hour=[2, 3],
            resource=["electricity"],
            storage_tech=["battery"],
            year_dec=[2030, 2040, 2050],
            year_inv=[2020, 2030],
            year_op=[2020, 2030],
        )
        .copy(deep=True)
    )
    p["conversion_annuity_cost"] = 0
    p["conversion_power_capacity_investment_min"] = xr.DataArray([1000, 0], coords=[p.year_inv])
    p["conversion_power_capacity_investment_max"] = xr.DataArray([1000, 0], coords=[p.year_inv])
    p["storage_factor_in"] = -1.0
    p["storage_factor_out"] = 1.0
    p["storage_dissipation"] = 0.0
    p["storage_energy_capacity_investment_max"] = 1e6
    p["storage_energy_to_power_min"] = xr.DataArray(
        [[RATIO, np.nan]], coords=[p.storage_tech, p.year_op]
    )
    model, s = _solve(p)
    labels = model.constraints[CONSTRAINT].labels
    assert labels.sel(year_op=2020).to_numpy().min() >= 0
    assert (labels.sel(year_op=2030).to_numpy() == -1).all()
    e, pw = s.operation_storage_energy_capacity, s.operation_storage_power_capacity
    assert float(e.sel(year_op=2020)) >= RATIO * float(pw.sel(year_op=2020)) - 1e-6


def test_without_the_parameter_the_model_is_unchanged(parameters_dispatch_invest, storage):
    """The parameter absent from the dataset, then present with all NaN: same constraints, same objective."""
    base = _toy(parameters_dispatch_invest, storage).drop_vars("storage_energy_to_power_min")
    assert "storage_energy_to_power_min" not in base
    # check_inputs completes the dataset with the default of dataset_description.yaml: a 0-d NaN
    completed = check_inputs(base)
    assert completed.storage_energy_to_power_min.dims == ()
    assert np.isnan(float(completed.storage_energy_to_power_min))
    model_a, _ = _solve(completed)
    # and a dataset that never went through check_inputs builds the same model without the constraint
    model_raw, _ = _solve(base)
    assert CONSTRAINT not in model_raw.constraints
    assert model_raw.constraints.ncons == model_a.constraints.ncons
    assert model_raw.objective.value == model_a.objective.value

    p = base.copy(deep=True)
    p["storage_energy_to_power_min"] = xr.DataArray([np.nan], coords=[p.storage_tech])
    model_b, _ = _solve(check_inputs(p))

    assert _rows(model_a) == 0
    assert _rows(model_b) == 0
    assert sorted(model_a.constraints) == sorted(model_b.constraints)
    assert model_a.objective.value == model_b.objective.value
    assert model_a.constraints.ncons == model_b.constraints.ncons
    assert model_a.variables.nvars == model_b.variables.nvars
