import warnings

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal as assert_equal

from pommes.model.build_model import build_model
from pommes.model.net_import import net_import_groups
from pommes.tests.time_blocks_helper import (
    assert_t0,
    combined_toy,
    coarse_step_dataset,
    coarse_step_reference,
    hand_toy,
    net_import_pairs,
    net_import_toy,
    solve_model,
    solved,
    toy_n,
    with_blocks,
)
from pommes.utils import time_blocks


@pytest.fixture()
def parameter_net_import(parameters_dispatch_invest, net_import):
    p = xr.merge([parameters_dispatch_invest, net_import])
    return p


@pytest.fixture()
def parameter_net_import_single_horizon(parameter_net_import):
    p = parameter_net_import.sel(
        area=["area_1"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    return p


def test_net_import(parameter_net_import_single_horizon):
    p = parameter_net_import_single_horizon.sel(
        conversion_tech=["ocgt", "wind_onshore"],
        hour=[3, 4, 5],
        resource=["electricity", "methane"],
    )
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.dropna(dim="year_dec", how="all").squeeze()

    assert_equal(
        s.planning_conversion_power_capacity.to_numpy(), np.array([10, 25])
    )
    assert_equal(
        s.operation_conversion_power.to_numpy(),
        np.array([[10.0, 5.0, 0.0], [0.0, 5.0, 10.0]]),
    )
    assert_equal(
        s.operation_net_import_import.sel(resource="methane").to_numpy(),
        np.array([15, 7.5, 0]),
    )

    assert_equal(
        s.operation_load_shedding_power.to_numpy(),
        np.array([[np.nan, 0], [np.nan, 0], [np.nan, 0]]),
    )
    assert_equal(
        s.operation_spillage_power.to_numpy(), np.array([[0, 0], [0, 0], [0, 0]])
    )

    assert model.objective.value == 1120.0


def test_net_import_with_differentiated_prices(
    parameter_net_import_single_horizon,
):
    """
    Test the net import behavior of the model when export prices differ from import prices.
    """
    parameters = parameter_net_import_single_horizon.sel(
        conversion_tech=["wind_onshore"],
        hour=[3, 4, 5],
        resource=["electricity"],
    )

    # No limit for imports and export
    parameters.net_import_max_yearly_energy_import.loc[
        {"resource": "electricity"}
    ] = 1000
    parameters.net_import_max_yearly_energy_export.loc[
        {"resource": "electricity"}
    ] = 1000

    # Max wind capacity of 50 MW
    parameters["conversion_power_capacity_max"] = xr.DataArray(
        [50],
        dims="conversion_tech",
        coords={"conversion_tech": ["wind_onshore"]},
    )

    # Import price < wind LCOE, to avoid importing all the electricity
    parameters.net_import_import_price.loc[{"resource": "electricity"}] = 20
    # Export price > wind LCOE, to install the max wind capacity
    parameters.net_import_export_price.loc[{"resource": "electricity"}] = 18

    parameters.operation_year_duration[:] = (
        parameters.time_step_duration.sum().values
    )
    model = build_model(parameters)
    model.solve(solver_name="highs")
    solution = model.solution.dropna(dim="year_dec", how="all").squeeze()

    assert_equal(
        solution.operation_net_import_import.to_numpy(), np.array([10, 0, 0])
    )
    assert_equal(
        solution.operation_net_import_export.to_numpy(), np.array([0, 0, 10])
    )
    assert_equal(
        solution.planning_conversion_power_capacity.to_numpy(), np.array([50])
    )

    export_volumes = solution.operation_net_import_export.to_numpy()
    import_volumes = solution.operation_net_import_import.to_numpy()
    total_costs = (
        (import_volumes * parameters.net_import_import_price.values).sum()
        - (export_volumes * parameters.net_import_export_price.values).sum()
        + solution.planning_conversion_power_capacity.values * 10
    )
    assert np.isclose(
        model.objective.value, total_costs
    ), "Objective value mismatch"


# --- multi-resolution: net_import on the resource's blocks (sections 6.5, 16.2) ---

BLOCK = "operation_net_import_block_4h_"
STEPS = [1.0, 3.0]  # a 4 h block of two steps


def _objective(p):
    return solve_model(build_model(p)).objective.value


def test_t0_net_import_has_no_trace_of_blocks(parameter_net_import_single_horizon):
    p = parameter_net_import_single_horizon.sel(
        conversion_tech=["ocgt", "wind_onshore"],
        hour=[3, 4, 5],
        resource=["electricity", "methane"],
    )
    assert_t0(p)


def test_net_import_groups_t0_is_the_input(parameter_net_import_single_horizon):
    p = parameter_net_import_single_horizon
    groups = net_import_groups(p, time_blocks(p))
    assert len(groups) == 1
    assert groups[0][0] is None
    assert groups[0][1] is p  # not a copy, no sel (C5, C11)


def _hand(demand, price):
    """The hand toy (steps 1 h and 3 h) with steel forced to one 4 h block, and its hourly twin."""
    p = net_import_toy(STEPS, demand, price)
    return with_blocks(p, resource={"steel": 4}), p


def test_block_import_constant_price():
    """Case A: the block twin is exact, 10 n, as the hourly balance (section 16.2)."""
    block, hourly = _hand([2, 0], [5, 5])
    m = solve_model(build_model(block))
    assert m.objective.value == pytest.approx(10 * toy_n(block), rel=1e-9)
    assert solved(m, f"{BLOCK}import").sel(resource="steel").item() == pytest.approx(2.0)
    assert _objective(hourly) == pytest.approx(m.objective.value, rel=1e-9)


def test_block_import_varying_price():
    """Case B: block mean price (1 + 3 * 10) / 4 = 7.75, so 15.5 n against 2 n hourly: the restriction is real."""
    block, hourly = _hand([2, 0], [1, 10])
    n = toy_n(block)
    assert _objective(block) == pytest.approx(15.5 * n, rel=1e-9)
    assert _objective(hourly) == pytest.approx(2 * n, rel=1e-9)


def test_block_import_reversed_demand():
    """Case C: demand on the dear step, block 15.5 n, hourly 20 n: against the hourly balance there is no sign."""
    block, hourly = _hand([0, 2], [1, 10])
    n = toy_n(block)
    assert _objective(block) == pytest.approx(15.5 * n, rel=1e-9)
    assert _objective(hourly) == pytest.approx(20 * n, rel=1e-9)


@pytest.mark.filterwarnings("ignore:Time blocks give no gain")
def test_one_group_path_keeps_todays_names():
    """Block technologies on hourly balances only: one group, today's net_import names."""
    p = with_blocks(combined_toy([1.0] * 8), combined={"boiler_block": 4}).assign(
        net_import=np.array(True)
    )
    tb = time_blocks(p)
    assert tb.active and not tb.balance_lengths
    groups = net_import_groups(p, tb)
    assert len(groups) == 1 and groups[0][1] is p
    m = build_model(p)
    names = [n for n in [*m.variables, *m.constraints] if "net_import" in n]
    assert "operation_net_import_import" in names
    assert not [n for n in names if "_block_" in n]


def test_derived_weekly_with_net_import():
    """Net_import does not count: steel with a 4 h furnace stays derived at 4 h, without a warning."""
    p = hand_toy([1.0] * 8, [0, 2, 0, 2, 0, 2, 0, 2])
    p = with_blocks(p.assign(net_import=np.array(True)), conversion={"furnace": 4})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tb = time_blocks(p)
    assert tb.resource_length.sel(resource="steel") == 4
    assert [n for n in build_model(p).variables if n.startswith(BLOCK)]


def test_equal_size_subsets_costs_partner():
    """H1: two hourly and two coarse resources of equal size, each cost on its own resource."""
    p = net_import_pairs()
    tb = time_blocks(p)
    assert tb.resources(None) == ["idle", "h1"] and tb.resources(4) == ["c1", "c2"]
    m = solve_model(build_model(p))
    n = toy_n(p)
    assert m.objective.value == pytest.approx(44.5 * n, rel=1e-9)
    costs = solved(m, "operation_net_import_costs").sel(area="a", year_op=2030)
    expected = {"idle": 0.0, "h1": 14.0, "c1": 15.5, "c2": 15.0}  # distinct, so a swapped pair shows
    for resource, value in expected.items():
        assert costs.sel(resource=resource).item() == pytest.approx(value * n, rel=1e-9), resource
    assert list(m.constraints[f"{BLOCK}costs_def"].labels.indexes["resource"]) == ["c1", "c2"]
    assert list(m.constraints["operation_net_import_costs_def"].labels.indexes["resource"]) == ["idle", "h1"]


def test_no_hour_parameter_left_in_a_coarse_group():
    p = net_import_pairs()
    tb = time_blocks(p)
    by_length = dict(net_import_groups(p, tb))
    assert list(by_length) == [None, 4]
    assert list(by_length[None].indexes["resource"]) == ["idle", "h1"]
    assert any("hour" in da.dims for da in by_length[None].data_vars.values())
    coarse = by_length[4]
    assert list(coarse.indexes["resource"]) == ["c1", "c2"]
    assert not [n for n, da in coarse.data_vars.items() if "hour" in da.dims]
    # the means are Delta-weighted: c1 emission factor (0.3 * 1 + 0.4 * 3) / 4
    value = coarse.net_import_emission_factor.sel(resource="c1", area="a", year_op=2030)
    assert value.item() == pytest.approx(0.375)
    assert coarse.net_import_import_price.sel(resource="c2").item() == pytest.approx(5.0)


def _binding_rows(m, rows):
    for name in rows:
        dual = m.constraints[name].dual
        live = (m.constraints[name].labels != -1)
        assert live.any(), name
        assert (abs(dual.where(live)) > 1e-9).any(), f"{name} does not bind"


@pytest.mark.parametrize("L, horizon", [(4, 24), (6, 24)])
def test_block_net_import_equals_coarse_step_reference(L, horizon):
    """T3(c): constant-in-block prices, binding yearly import and export maxima, against steps of L hours."""
    p = coarse_step_dataset(L, horizon, net_import=True)
    block = f"operation_net_import_block_{L}h_"
    m = solve_model(build_model(p))
    reference = solve_model(build_model(coarse_step_reference(p, L)))
    assert m.objective.value == pytest.approx(reference.objective.value, rel=1e-7)
    _binding_rows(m, [f"{block}import_yearly_max_constraint", f"{block}export_yearly_max_constraint"])
    _binding_rows(
        reference,
        ["operation_net_import_import_yearly_max_constraint", "operation_net_import_export_yearly_max_constraint"],
    )
    for kind, resource in (("import", "heat"), ("export", "gas")):
        got = solved(m, f"{block}{kind}").sel(resource=resource).transpose("area", f"time_block_{L}h", "year_op")
        want = solved(reference, f"operation_net_import_{kind}").sel(resource=resource).transpose("area", "hour", "year_op")
        np.testing.assert_allclose(got.values, want.values, atol=1e-6)
        assert got.values.max() > 0.1  # the rest of the world is used
