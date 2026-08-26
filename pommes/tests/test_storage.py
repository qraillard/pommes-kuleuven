import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model


@pytest.fixture()
def parameters_storage(parameters_dispatch_invest, storage):
    return xr.merge([parameters_dispatch_invest, storage])


@pytest.fixture()
def parameters_storage_single_horizon(parameters_storage):
    p = parameters_storage
    p = p.sel(
        area=["area_1"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    return p.copy(deep=True)


@pytest.fixture()
def parameters_storage_multi_horizon(parameters_storage):
    p = parameters_storage.sel(
        area=["area_1"],
        conversion_tech=["wind_onshore"],
        hour=[2, 3],
        resource=["electricity"],
        storage_tech=["battery"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030],
    ).copy(deep=True)

    p["conversion_annuity_cost"] = 0
    p["conversion_power_capacity_investment_min"] = xr.DataArray(
        [1000, 0], coords=[p.year_inv]
    )
    p["conversion_power_capacity_investment_max"] = xr.DataArray(
        [1000, 0], coords=[p.year_inv]
    )
    p["conversion_availability"] = p.conversion_availability.broadcast_like(
        p.year_op
    ).copy()

    p["storage_fixed_cost_power"] = 10
    p["storage_fixed_cost_energy"] = 10

    p["storage_factor_in"] = -1
    p["storage_factor_out"] = 1
    p["storage_dissipation"] = 0

    p.conversion_availability.loc[
        dict(conversion_tech="wind_onshore", hour=3, year_op=2030)
    ] = np.array(0.1)

    return p


def test_storage_level(parameters_storage_single_horizon):
    p = parameters_storage_single_horizon
    p = p.sel(
        conversion_tech=["wind_onshore"],
        hour=[2, 3],
        resource=["electricity"],
        storage_tech=["battery"],
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_conversion_power_capacity.to_numpy(), np.array([600])
    )
    np.testing.assert_array_equal(
        s.operation_storage_energy_capacity.to_numpy(), np.array([25])
    )
    np.testing.assert_array_equal(
        s.operation_storage_power_capacity.to_numpy(), np.array([25])
    )
    np.testing.assert_array_equal(
        s.operation_storage_power_in.to_numpy(), np.array([25.0, 0.0])
    )
    np.testing.assert_array_equal(
        s.operation_storage_level.to_numpy(), np.array([25.0, 0.0])
    )
    np.testing.assert_array_equal(
        s.operation_storage_power_out.to_numpy(), np.array([0.0, 20])
    )

    assert model.objective.value == 9750.0


def test_storage_level_loop(parameters_storage_single_horizon):
    p = parameters_storage_single_horizon
    p = p.sel(
        conversion_tech=["wind_onshore"],
        hour=[3, 4],
        resource=["electricity"],
        storage_tech=["battery"],
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_conversion_power_capacity.to_numpy(), np.array([300])
    )
    np.testing.assert_array_equal(
        s.operation_storage_energy_capacity.to_numpy(), np.array([25])
    )
    np.testing.assert_array_equal(
        s.operation_storage_power_capacity.to_numpy(), np.array([25])
    )
    np.testing.assert_array_equal(
        s.operation_storage_power_in.to_numpy(), np.array([0, 25])
    )
    np.testing.assert_array_equal(
        s.operation_storage_level.to_numpy(), np.array([0, 25])
    )
    np.testing.assert_array_equal(
        s.operation_storage_power_out.to_numpy(), np.array([20, 0])
    )

    assert model.objective.value == 6750


def test_storage_factor_keep(parameters_storage_single_horizon):
    p = parameters_storage_single_horizon
    p = p.sel(
        area=["area_1"],
        conversion_tech=["electrolysis", "wind_onshore"],
        hour=[2, 3],
        resource=["electricity", "hydrogen"],
        storage_tech=["battery", "tank_hydrogen"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_conversion_power_capacity.to_numpy(), np.array([6, 720])
    )
    np.testing.assert_array_equal(
        s.operation_storage_energy_capacity.to_numpy(), np.array([25, 3])
    )
    np.testing.assert_array_equal(
        s.operation_storage_power_capacity.to_numpy(), np.array([25, 3])
    )
    np.testing.assert_array_equal(
        s.operation_storage_power_in.to_numpy(), np.array([[25, 3], [0, 0]])
    )
    np.testing.assert_array_equal(
        s.operation_storage_level.to_numpy(), np.array([[25, 3], [0, 0]])
    )
    np.testing.assert_array_equal(
        s.operation_storage_power_out.to_numpy(), np.array([[0, 0], [20, 3]])
    )

    assert model.objective.value == 12150


def test_storage_multi_horizon(parameters_storage_multi_horizon):
    p = parameters_storage_multi_horizon

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_storage_costs.to_numpy(), np.array([1500, 1500])
    )

    np.testing.assert_array_equal(
        s.planning_storage_power_capacity.to_numpy(),
        np.array([[np.nan, np.nan], [10, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.planning_storage_energy_capacity.to_numpy(),
        np.array([[np.nan, np.nan], [10, np.nan], [np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_storage_power_capacity.to_numpy(), np.array([10, 10])
    )
    np.testing.assert_array_equal(
        s.operation_storage_energy_capacity.to_numpy(), np.array([10, 10])
    )

    np.testing.assert_array_equal(
        s.operation_storage_level.to_numpy(), np.array([[10, 0], [0, 0]])
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.transpose("year_op", "hour").to_numpy(),
        np.array([[80, 0], [90, 90]]),
    )

    assert model.objective.value == 3400.0


def test_storage_early_decommissioning(parameters_storage_multi_horizon):
    p = parameters_storage_multi_horizon
    p = p.copy(deep=True)

    p["storage_early_decommissioning"] = True

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_storage_costs.to_numpy(), np.array([1500, 1500])
    )

    np.testing.assert_array_equal(
        s.planning_storage_power_capacity.transpose(
            "year_inv", "year_dec"
        ).to_numpy(),
        np.array([[10, 0, np.nan], [np.nan, np.nan, 0]]),
    )
    np.testing.assert_array_equal(
        s.planning_storage_energy_capacity.transpose(
            "year_inv", "year_dec"
        ).to_numpy(),
        np.array([[10, 0, np.nan], [np.nan, np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_storage_power_capacity.to_numpy(), np.array([10, 0])
    )
    np.testing.assert_array_equal(
        s.operation_storage_energy_capacity.to_numpy(), np.array([10, 0])
    )

    np.testing.assert_array_equal(
        s.operation_storage_level.to_numpy(), np.array([[10, 0], [0, 0]])
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.transpose("year_op", "hour").to_numpy(),
        np.array([[80, 0], [90, 90]]),
    )

    assert model.objective.value == 3200.0


def test_storage_annuity_perfect_foresight(parameters_storage_multi_horizon):
    p = parameters_storage_multi_horizon
    p = p.copy(deep=True)

    p["storage_early_decommissioning"] = True
    p["storage_annuity_perfect_foresight"] = True

    p.conversion_availability.loc[
        dict(conversion_tech="wind_onshore", hour=3, year_op=2030)
    ] = np.array(0.1)

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    np.testing.assert_array_equal(
        s.planning_storage_costs.to_numpy(), np.array([3000, 0])
    )

    np.testing.assert_array_equal(
        s.planning_storage_power_capacity.transpose(
            "year_inv", "year_dec"
        ).to_numpy(),
        np.array([[10, 0, np.nan], [np.nan, np.nan, 0]]),
    )
    np.testing.assert_array_equal(
        s.planning_storage_energy_capacity.transpose(
            "year_inv", "year_dec"
        ).to_numpy(),
        np.array([[10, 0, np.nan], [np.nan, np.nan, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_storage_power_capacity.to_numpy(), np.array([10, 0])
    )
    np.testing.assert_array_equal(
        s.operation_storage_energy_capacity.to_numpy(), np.array([10, 0])
    )

    np.testing.assert_array_equal(
        s.operation_storage_level.to_numpy(),
        np.array([[10, 0], [0, 0]]),
    )

    np.testing.assert_array_equal(
        s.operation_spillage_power.transpose("year_op", "hour").to_numpy(),
        np.array([[80, 0], [90, 90]]),
    )

    assert model.objective.value == 3200.0
