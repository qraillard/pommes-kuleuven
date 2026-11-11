"""Operating conditions in the storage module.

One area, four hours, electricity only. A run-of-river plant of fixed size (20 MW) is the only
source; the demand is 10 MW. A battery with unit efficiencies and no dissipation moves energy
inside a condition, never between two.
"""
import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import (
    assert_condition_dims_consistent,
    with_conditions,
)
from pommes.tests.dataset_nnodes import build_nnode_parameters
from pommes.tests.test_flexdem import _with_defaults

N_HOURS = 4
DEMAND = 10.0
SHEDDING_COST = 3000.0
OTHER_DIMS = ("hour", "operating_condition", "year_op")


def _base():
    ds = build_nnode_parameters(n_areas=1, n_hours=N_HOURS)
    ds = ds.sel(
        conversion_tech=["hydro_ror"],
        storage_tech=["battery"],
        resource=["electricity"],
    )
    for module in ("combined", "carbon"):
        ds[module] = False
    ds = _with_defaults(ds)
    ds["demand"] = ds.demand * 0 + DEMAND
    for bound in ("min", "max"):
        key = f"conversion_power_capacity_{bound}"
        ds[key] = xr.full_like(ds.conversion_power_capacity_max, 20.0)
    ds["storage_factor_in"] = ds.storage_factor_in * 0 - 1.0
    ds["storage_factor_out"] = ds.storage_factor_out * 0 + 1.0
    ds["storage_dissipation"] = ds.storage_dissipation * 0
    # finite and slack ramps: they build the ramp constraints without changing the solution
    for key in ("storage_charge_ramp", "storage_discharge_ramp"):
        ds[key] = xr.full_like(ds[key], 1.0)
    return ds


def _availability(ds, values):
    return xr.DataArray(
        [values],
        dims=("conversion_tech", "hour"),
        coords=dict(conversion_tech=ds.conversion_tech, hour=ds.hour),
    )


def _two_conditions(ds, weights=(0.5, 0.5)):
    # a: the plant runs in the first two hours (a surplus), then stops (a deficit); b: it never
    # runs
    return with_conditions(
        ds,
        list(weights),
        conversion_availability=[
            _availability(ds, [1, 1, 0, 0]),
            _availability(ds, [0, 0, 0, 0]),
        ],
    )


def _solve(ds):
    m = build_model(ds)
    if "operating_condition" in ds.coords:
        assert_condition_dims_consistent(m)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


def _by_hour(m, name, condition):
    da = m.solution[name].sel(operating_condition=condition)
    return da.sum([d for d in da.dims if d not in OTHER_DIMS]).squeeze()


def test_storage_cannot_move_energy_between_conditions():
    m = _solve(_two_conditions(_base()))
    shed = m.solution["operation_load_shedding_power"]
    shed = shed.sum([d for d in shed.dims if d not in OTHER_DIMS])
    # b has no source: its whole demand is shed, a's surplus cannot serve it
    assert float(shed.sel(operating_condition="b").sum()) == pytest.approx(
        DEMAND * N_HOURS
    )
    # a shifts its own surplus (hours 0-1) to its own deficit (hours 2-3): nothing is shed
    assert float(shed.sel(operating_condition="a").sum()) == pytest.approx(0.0)
    assert float(
        _by_hour(m, "operation_storage_power_out", "a").sum()
    ) == pytest.approx(2 * DEMAND)


def test_storage_cycle_closed_in_each_condition():
    m = _solve(_two_conditions(_base()))
    for condition in ("a", "b"):
        level = _by_hour(m, "operation_storage_level", condition)
        net = _by_hour(m, "operation_storage_power_in", condition) - _by_hour(
            m, "operation_storage_power_out", condition
        )
        # the level of the first hour follows the last one: the cycle closes inside the condition
        assert float(level.isel(hour=0)) == pytest.approx(
            float(level.isel(hour=-1) + net.isel(hour=0))
        )
        assert float(net.sum()) == pytest.approx(0.0)


def test_identical_conditions_leave_the_objective_unchanged():
    # the plant runs in the first two hours only, so a battery is built and its costs are paid once
    ds = _base()
    pattern = [1, 1, 0, 0]
    ds["conversion_availability"] = _availability(ds, pattern)
    reference = _solve(ds)
    same = with_conditions(
        ds, [0.5, 0.5], conversion_availability=[_availability(ds, pattern)] * 2
    )
    m = _solve(same)
    assert float(m.solution["planning_storage_power_capacity"].sum()) > 0
    assert float(m.solution["operation_storage_costs"].sum()) > 0
    assert m.objective.value == pytest.approx(reference.objective.value, rel=1e-9)


def test_an_absent_pair_has_no_storage_row_and_weight_zero_keeps_them():
    for weights, absent in (([1.0, np.nan], True), ([1.0, 0.0], False)):
        m = build_model(_two_conditions(_base(), weights))
        assert_condition_dims_consistent(m)
        checked = []
        for container in (m.variables, m.constraints):
            for name in container:
                if "storage" not in name:
                    continue
                labels = container[name].labels
                if "operating_condition" not in labels.dims:
                    continue
                checked.append(name)
                if absent:
                    assert (labels.sel(operating_condition="b") == -1).all(), name
                    if name in WITH_CONDITION:
                        assert (labels.sel(operating_condition="a") != -1).any(), name
                elif name in WITH_CONDITION:
                    assert (labels.sel(operating_condition="b") != -1).any(), name
        assert set(WITH_CONDITION) <= set(checked)


WITH_CONDITION = [
    "operation_storage_level",
    "operation_storage_power_in",
    "operation_storage_power_out",
    "operation_storage_net_generation",
    "operation_storage_costs",
    "operation_storage_level_def",
    "operation_storage_power_in_max_constraint",
    "operation_storage_power_out_max_constraint",
    "operation_storage_level_max_constraint",
    "operation_storage_charge_ramp_constraint",
    "operation_storage_discharge_ramp_constraint",
    "operation_storage_net_generation_def",
    "operation_storage_costs_def",
]
