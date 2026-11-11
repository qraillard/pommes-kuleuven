"""Operating conditions in the process module.

One area, one year, methane (bought by shedding at 5 per MWh) turned into heat by process units of 1 MW that run at
their size or not at all (min load = max load = 1, heat shedding at 1000). The integer commitment variables are
per condition, so a unit can start in one condition and stay off in the other.
"""
import warnings

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import (
    assert_condition_dims_consistent,
    with_conditions,
)
from pommes.tests.test_process_min_times import (
    HEAT,
    RESOURCES,
    _instance,
    _solve,
)

N_HOURS = 6
START_COST = 3.0


def _heat(hourly):
    """Demand (year_op, hour, resource): the heat demand per hour given."""
    d = np.zeros((1, N_HOURS, len(RESOURCES)))
    d[0, :, RESOURCES.index(HEAT)] = hourly
    return xr.DataArray(d, dims=("year_op", "hour", "resource"))


def _base(parameters_dispatch_invest, process, techs=None):
    techs = techs or [dict(out=HEAT, size=1.0, cap=1.0, cu=START_COST)]
    return _instance(
        parameters_dispatch_invest,
        process,
        _heat([0.0] * N_HOURS),
        techs,
    )


def _with_demands(p, weights, a, b):
    return with_conditions(
        p, weights, demand=[_demand(p, a), _demand(p, b)]
    )


def _demand(p, hourly):
    return _heat(hourly).assign_coords(year_op=p.year_op, hour=p.hour)


def _solved(p):
    m = build_model(p)
    if "operating_condition" in p.coords:
        assert_condition_dims_consistent(m)
    return _solve(m)


def _priced(p, carbon):
    """Carbon on: process emissions of 0.7 t per MWh of power, taxed at 30, no goal."""
    p = xr.merge([p, carbon])
    p["carbon_goal"] = 1e9
    p["carbon_tax"] = 30.0
    p["process_emission_factor"] = xr.zeros_like(p["process_emission_factor"]) + 0.7
    return p


def _cost(parameters_dispatch_invest, process, hourly, carbon=None):
    p = _base(parameters_dispatch_invest, process)
    if carbon is not None:
        p = _priced(p, carbon)
    p["demand"] = _demand(p, hourly)
    return _solved(p).objective.value


A = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]  # two starts
B = [0.0] * N_HOURS  # no heat


def test_unit_starts_only_in_the_condition_with_demand(
    parameters_dispatch_invest, process
):
    p = _base(parameters_dispatch_invest, process)
    m = _solved(_with_demands(p, [0.5, 0.5], A, B))
    startup = m.solution["operation_process_startup"]
    other = [d for d in startup.dims if d != "operating_condition"]
    per_condition = startup.sum(other)
    assert per_condition.sel(operating_condition="a") == 2
    assert per_condition.sel(operating_condition="b") == 0
    state = m.solution["operation_process_state"]
    assert float(state.sel(operating_condition="a").sum()) == 3
    assert float(state.sel(operating_condition="b").sum()) == 0


def test_process_cost_is_the_weighted_mean(parameters_dispatch_invest, process):
    # same capacity in both conditions, so its fixed cost comes back once
    a, b = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    p = _base(parameters_dispatch_invest, process)
    m = _solved(_with_demands(p, [0.6, 0.4], a, b))
    cost_a = _cost(parameters_dispatch_invest, process, a)
    cost_b = _cost(parameters_dispatch_invest, process, b)
    assert cost_a != cost_b
    assert m.objective.value == pytest.approx(
        0.6 * cost_a + 0.4 * cost_b, rel=1e-9
    )


def test_process_emissions_are_taxed_per_condition(
    parameters_dispatch_invest, process, carbon
):
    # the process power carries the condition into carbon's emission definitions
    a, b = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    p = _priced(_base(parameters_dispatch_invest, process), carbon)
    m = _solved(_with_demands(p, [0.6, 0.4], a, b))
    cost_a = _cost(parameters_dispatch_invest, process, a, carbon)
    cost_b = _cost(parameters_dispatch_invest, process, b, carbon)
    assert cost_a != _cost(parameters_dispatch_invest, process, a)  # emissions are priced
    assert m.objective.value == pytest.approx(
        0.6 * cost_a + 0.4 * cost_b, rel=1e-9
    )


def test_weight_zero_keeps_its_rows_and_costs_nothing(
    parameters_dispatch_invest, process
):
    p = _base(parameters_dispatch_invest, process)
    m = _solved(_with_demands(p, [1.0, 0.0], A, [1.0] * N_HOURS))
    assert m.objective.value == pytest.approx(
        _cost(parameters_dispatch_invest, process, A), rel=1e-9
    )
    for name in (
        "operation_process_state",
        "operation_process_costs",
    ):
        labels = m.variables[name].labels
        assert (labels.sel(operating_condition="b") != -1).all(), name
    labels = m.constraints["operation_process_costs_def"].labels
    assert (labels.sel(operating_condition="b") != -1).all()


# One tech of every kind: sized with minimum times, without unit size, sized and
# without unit size with no shutdown. Finite ramps and mode ramps build the ramp rows.
TECHS = [
    dict(out=HEAT, size=1.0, cap=2.0, up=2.0, down=2.0, cu=1.0, cd=1.0),
    dict(out=HEAT, size=0.0, cap=1.0, up=2.0, down=2.0),
    dict(out=HEAT, size=1.0, cap=2.0, nosd=True),
    dict(out=HEAT, size=0.0, cap=1.0, nosd=True),
]

VARIABLES = [
    "operation_process_power",
    "operation_process_net_generation",
    "operation_process_state",
    "operation_process_startup",
    "operation_process_shutdown",
    "operation_process_startup_cumulative",
    "operation_process_shutdown_cumulative",
    "operation_process_costs",
]

CONSTRAINTS = [
    "operation_process_power_max_constraint",
    "operation_process_power_max_constraint_no_unit_size",
    "operation_process_power_max_bigM_constraint",
    "operation_process_power_min_constraint",
    "operation_process_power_min_constraint_no_unit_size",
    "operation_process_ramp_up_constraint",
    "operation_process_startup_capability_constraint",
    "operation_process_shutdown_capability_constraint",
    "operation_process_ramp_up_constraint_no_unit_size",
    "operation_process_ramp_down_constraint",
    "operation_process_ramp_down_constraint_no_unit_size",
    "operation_process_state_nb_units",
    "operation_process_no_shutdown_constraint",
    "operation_process_no_shutdown_constraint_no_unit_size",
    "operation_process_startup_nb_units_constraint",
    "operation_process_shutdown_nb_units_constraint",
    "operation_process_power_undefined_mode",
    "operation_process_net_generation_def",
    "operation_process_startupshutdown_def",
    "operation_process_state_def",
    "operation_process_startup_cumulative_def",
    "operation_process_shutdown_cumulative_def",
    "minimum_uptime_constraint_2.0",
    "minimum_downtime_constraint_2.0",
    "operation_process_costs_def",
]
MODE_RAMPS = [
    "operation_process_mode_ramp_up_constraint",
    "operation_process_mode_ramp_up_constraint_no_unit_size",
    "operation_process_mode_ramp_down_constraint",
    "operation_process_mode_ramp_down_constraint_no_unit_size",
]


def test_an_absent_pair_has_no_process_row(parameters_dispatch_invest, process):
    p = _base(parameters_dispatch_invest, process, TECHS)
    for ramp in ("ramp_up", "ramp_down", "mode_ramp_up", "mode_ramp_down"):
        p[f"process_{ramp}"] = xr.full_like(p[f"process_{ramp}"], 0.5)
    p = with_conditions(
        p, [1.0, np.nan], demand=[_demand(p, A), _demand(p, A)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # mode ramps on units that can stop
        m = build_model(p)
    assert_condition_dims_consistent(m)
    mode_a, mode_b = p.mode.values[:2]
    expected_constraints = CONSTRAINTS + [
        f"{name}_{mode_a}_{mode_b}" for name in MODE_RAMPS
    ]
    for container, names in (
        (m.variables, VARIABLES),
        (m.constraints, expected_constraints),
    ):
        for name in names:
            assert name in container, name
            labels = container[name].labels
            assert "operating_condition" in labels.dims, name
            assert (labels.sel(operating_condition="b") == -1).all(), name
            assert (labels.sel(operating_condition="a") != -1).any(), name
    # nothing of the module that carries the dimension is left unchecked
    for container, names in (
        (m.variables, VARIABLES),
        (m.constraints, expected_constraints),
    ):
        carried = [
            n
            for n in container
            if "process" in n or n.startswith("minimum_")
            if "operating_condition" in container[n].labels.dims
        ]
        assert set(carried) <= set(names), set(carried) - set(names)
