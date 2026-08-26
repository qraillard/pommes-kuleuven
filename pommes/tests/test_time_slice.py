import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal as assert_equal

from pommes.model.build_model import build_model


@pytest.fixture()
def parameters_conversion_single_horizon(parameters_dispatch_invest):
    p = parameters_dispatch_invest.sel(
        area=["area_1"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    return p


def test_conversion_costs(parameters_conversion_single_horizon):
    p = parameters_conversion_single_horizon.copy(deep=True)
    p = p.sel(
        conversion_tech=["ocgt", "wind_onshore"],
        hour=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        resource=["electricity"],
    )
    p = p.update(
        dict(
            conversion_power_capacity_investment_min=p.conversion_power_capacity_investment_min.expand_dims(
                conversion_tech=p.conversion_tech,
            ).copy(),
            conversion_power_capacity_investment_max=p.conversion_power_capacity_investment_max.expand_dims(
                conversion_tech=p.conversion_tech,
            ).copy(),
            demand=p.demand.expand_dims(hour=p.hour),
        )
    )
    p.conversion_power_capacity_investment_min[:] = [100, 10]
    p.conversion_power_capacity_investment_max[:] = [100, 10]

    # Test operation_year_duration consideration in the conversion costs
    p.operation_year_duration[:] = 10
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    target_operation_costs = (
        10
        / sum(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )  # operation_year_duration normalization
        * sum(
            8
            * np.array(
                [5.0, 3.0, 9.0, 10.0, 8.0, 6.0, 2.0, 1.0, 4.0, 7.0]
            )  # Variable operation conversion  costs
            * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )  # time_step_duration ponderation
        + 100 * 10
    )  # Fixed operation conversion costs
    assert_equal(
        s.operation_conversion_costs.to_numpy(),
        np.array([target_operation_costs, -0.0]),
    )

    # Test operation_year_duration consideration in the conversion costs
    p.operation_year_duration[:] = 8760
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    target_operation_costs = (
        8760
        / sum(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )  # operation_year_duration normalization
        * sum(
            8
            * np.array(
                [5.0, 3.0, 9.0, 10.0, 8.0, 6.0, 2.0, 1.0, 4.0, 7.0]
            )  # Variable operation conversion  costs
            * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )  # time_step_duration ponderation
        + 100 * 10
    )  # Fixed operation conversion costs
    assert_equal(
        s.operation_conversion_costs.to_numpy(),
        np.array([target_operation_costs, -0.0]),
    )

    # Test time_step_duration consideration in the conversion costs
    p.time_step_duration[:] = [
        336,
        456,
        572,
        590,
        590,
        12,
        747,
        839,
        4112,
        506,
    ]
    # p.time_step_duration[:] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*2
    p.demand.values = np.array(
        [[[10 * val]] for val in p.time_step_duration.values]
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    target_operation_costs = (
        8760
        / sum(
            np.array([336, 456, 572, 590, 590, 12, 747, 839, 4112, 506])
        )  # operation_year_duration normalization
        * sum(
            8
            * np.array(
                [5.0, 3.0, 9.0, 10.0, 8.0, 6.0, 2.0, 1.0, 4.0, 7.0]
            )  # Variable operation conversion  costs
            * np.array([336, 456, 572, 590, 590, 12, 747, 839, 4112, 506])
        )  # time_step_duration ponderation
        + 100 * 10
    )  # Fixed operation conversion costs
    assert_equal(
        s.operation_conversion_costs.to_numpy(),
        np.array([target_operation_costs, -0.0]),
    )


def test_conversion_ramp(parameters_conversion_single_horizon):
    p = parameters_conversion_single_horizon.copy(deep=True)
    p = p.sel(
        conversion_tech=["ocgt"],
        hour=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        resource=["electricity"],
    )
    p = p.update(
        dict(
            conversion_power_capacity_investment_min=p.conversion_power_capacity_investment_min.expand_dims(
                conversion_tech=p.conversion_tech,
            ).copy(),
            conversion_power_capacity_investment_max=p.conversion_power_capacity_investment_max.expand_dims(
                conversion_tech=p.conversion_tech,
            ).copy(),
            conversion_ramp_up=p.conversion_ramp_up.expand_dims(
                conversion_tech=p.conversion_tech,
            ).copy(),
            conversion_ramp_down=p.conversion_ramp_down.expand_dims(
                conversion_tech=p.conversion_tech,
            ).copy(),
            demand=p.demand.expand_dims(
                hour=p.hour,
            ).copy(),
            spillage_max_capacity=p.spillage_max_capacity.expand_dims(
                resource=p.resource,
            ).copy(),
        )
    )
    p.conversion_power_capacity_investment_min[:] = [10]
    p.conversion_power_capacity_investment_max[:] = [10]

    p.load_shedding_cost[:] = 3000
    p.load_shedding_max_capacity[:] = 1000
    p.spillage_max_capacity[:] = 0
    p.conversion_ramp_up[0] = 0.1
    p.conversion_ramp_down[0] = 0.1
    p.demand[0] = 1
    p.demand[9] = 1
    p.time_step_duration[:] = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    p.operation_year_duration[:] = 10

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    assert_allclose(
        s.operation_conversion_power.to_numpy(),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        rtol=1e-3,
        atol=1e-3,
    )
    p.time_step_duration[:] = [
        2.0,
        2.0,
        3.0,
        5.0,
        1.0,
        1.0,
        1.0,
        6.0,
        1.0,
        1.0,
    ]
    p.demand.values = np.array(
        [[[10 * val]] for val in p.time_step_duration.values]
    )
    p.demand[0] = 2
    p.demand[9] = 1
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    assert_allclose(
        s.operation_conversion_power.to_numpy(),
        np.array([1.0, 3.0, 6.0, 10.0, 10.0, 10.0, 9.0, 3.0, 2.0, 1.0]),
        rtol=1e-3,
        atol=1e-3,
    )


def test_load_shedding_spillage_costs(parameters_conversion_single_horizon):
    p = parameters_conversion_single_horizon.copy(deep=True)
    p = p.sel(
        conversion_tech=["wind_onshore"],
        hour=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        resource=["electricity"],
    )

    p = p.update(
        dict(
            conversion_power_capacity_investment_min=p.conversion_power_capacity_investment_min.expand_dims(
                conversion_tech=p.conversion_tech,
            ).copy(),
            conversion_power_capacity_investment_max=p.conversion_power_capacity_investment_max.expand_dims(
                conversion_tech=p.conversion_tech,
            ).copy(),
            demand=p.demand.expand_dims(
                hour=p.hour,
            ).copy(),
        )
    )
    p.conversion_power_capacity_investment_min[:] = [14]
    p.conversion_power_capacity_investment_max[:] = [14]

    p.time_step_duration[:] = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    p.operation_year_duration[:] = 10
    p.load_shedding_cost[:] = 1
    p["spillage_cost"] = xr.DataArray(2)
    p.load_shedding_max_capacity[:] = 1000
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    load_shedding = (
        10
        - np.minimum(
            np.array([0.5, 0.7, 0.1, 0.0, 0.2, 0.4, 0.8, 0.9, 0.6, 0.3]) * 14,
            10,
        )
    ) * p.time_step_duration.to_numpy()
    assert_equal(
        s.operation_load_shedding_costs.to_numpy(), np.sum(load_shedding)
    )
    spillage = (
        np.maximum(
            np.array([0.5, 0.7, 0.1, 0.0, 0.2, 0.4, 0.8, 0.9, 0.6, 0.3]) * 14,
            10,
        )
        - 10
    ) * p.time_step_duration.to_numpy()
    assert_equal(s.operation_spillage_power.to_numpy(), spillage)

    p.time_step_duration[:] = [
        1.0,
        1.0,
        2.0,
        3.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    p.demand.values = np.array(
        [[[10 * val]] for val in p.time_step_duration.values]
    )
    p.operation_year_duration[:] = 15
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    load_shedding = (
        10
        - np.minimum(
            np.array([0.5, 0.7, 0.1, 0.0, 0.2, 0.4, 0.8, 0.9, 0.6, 0.3]) * 14,
            10,
        )
    ) * p.time_step_duration.to_numpy()
    load_shedding = np.sum(load_shedding) * p.load_shedding_cost.to_numpy()
    assert_allclose(
        s.operation_load_shedding_costs.to_numpy(),
        np.sum(load_shedding) * 15 / 13,
        rtol=1e-3,
        atol=1e-3,
    )
    spillage = (
        np.maximum(
            np.array([0.5, 0.7, 0.1, 0.0, 0.2, 0.4, 0.8, 0.9, 0.6, 0.3]) * 14,
            10,
        )
        - 10
    ) * p.time_step_duration.to_numpy()
    assert_allclose(
        s.operation_spillage_costs.to_numpy(),
        np.sum(spillage) * 2 * 15 / 13,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.fixture()
def parameters_storage_single_horizon(parameters_dispatch_invest, storage):
    p = xr.merge([parameters_dispatch_invest, storage])
    p = p.sel(
        area=["area_1"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    return p.copy(deep=True)


def test_storages(parameters_storage_single_horizon):
    p = parameters_storage_single_horizon.copy(deep=True)

    p = p.sel(
        conversion_tech=["wind_onshore"],
        storage_tech=["battery"],
        # hour=[0,1,2,3,4,5,6,7,8,9],
        resource=["electricity"],
    )
    p = p.update(
        dict(
            conversion_power_capacity_investment_min=p.conversion_power_capacity_investment_min.expand_dims(
                conversion_tech=p.conversion_tech,
            ).copy(),
            conversion_power_capacity_investment_max=p.conversion_power_capacity_investment_max.expand_dims(
                conversion_tech=p.conversion_tech,
            ).copy(),
            demand=p.demand.expand_dims(hour=p.hour).copy(),
        ),
    )
    p.conversion_power_capacity_investment_min[:] = [100]
    p.conversion_power_capacity_investment_max[:] = [100]

    # Test operation_year_duration consideration in the conversion costs
    p.operation_year_duration[:] = 10
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    assert_equal(s.planning_storage_power_capacity.to_numpy(), 20)
    assert_equal(s.planning_storage_energy_capacity.to_numpy(), 31.25)

    # Test storages self-decharge and balancing

    p.time_step_duration[:] = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    p.operation_year_duration[:] = 10
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    assert_equal(
        s.operation_storage_power_in.to_numpy(),
        np.array(
            [14.0625, 20.0, 0.0, 0.0, -0.0, 20.0, -0.0, -0.0, -0.0, -0.0]
        ),
    )
    assert_equal(
        s.operation_storage_level.to_numpy(),
        np.array([14.0625, 31.25, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert_equal(s.planning_conversion_power_capacity, 100)
    for i in p.hour:
        assert_allclose(
            s.operation_storage_level[i],
            (
                s.operation_storage_level[i - 1]
                * (1 - 0.2) ** p.time_step_duration[i]
                + (
                    s.operation_storage_power_in[i]
                    - s.operation_storage_power_out[i]
                )
                * p.time_step_duration[i]
            ),
            rtol=1e-3,
            atol=1e-3,
        )

    p.time_step_duration[:] = [
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ]
    p.demand.values = np.array(
        [[[10 * val]] for val in p.time_step_duration.values]
    )
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    assert_allclose(
        s.operation_storage_power_in.to_numpy(),
        np.array(
            [20.0, 20.0, 0.0, 0.0, 5.48791605, 15.0, 20.0, 20.0, 20.0, 10.0]
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    assert_allclose(
        s.operation_storage_power_out.to_numpy(),
        np.array([0.0, 0.0, -0.0, 20.0, 1.951664, 0.0, 0.0, 0.0, 0.0, 0.0]),
        rtol=1e-3,
        atol=1e-3,
    )
    assert_allclose(
        s.operation_storage_level.to_numpy(),
        np.array(
            [
                90.08789062,
                97.65625,
                62.5,
                0.0,
                7.07250369,
                34.52640236,
                62.09689751,
                79.74201441,
                91.03488922,
                78.2623291,
            ]
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    assert_equal(s.planning_conversion_power_capacity, 100)
    # hours 3 and 4 test the inflows and outflows
    for i in p.hour:
        assert_allclose(
            s.operation_storage_level[i],
            (
                s.operation_storage_level[i - 1]
                * (1 - 0.2) ** p.time_step_duration[i]
                + (
                    s.operation_storage_power_in[i]
                    - s.operation_storage_power_out[i]
                )
                * p.time_step_duration[i]
            ),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.fixture()
def parameters_emission_single_horizon(parameters_dispatch_invest, carbon):
    p = xr.merge([parameters_dispatch_invest, carbon])
    p = p.sel(
        area=["area_1"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    return p.copy(deep=True)


def test_emissions(parameters_emission_single_horizon):
    p = parameters_emission_single_horizon.copy(deep=True)
    p = p.sel(
        # hour=[0,1,2,],
        conversion_tech=["ocgt", "wind_onshore"],
        resource=["electricity"],
    )
    p["carbon_goal"] = xr.DataArray(1000000)

    p.time_step_duration[:] = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    p.operation_year_duration[:] = 10
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()
    costs = p.carbon_tax * (
        p.operation_year_duration
        / p.time_step_duration.sum("hour")
        * (p.time_step_duration * s.operation_carbon_emissions).sum("hour")
    )
    assert_equal(s.operation_carbon_costs, costs)

    list(s.operation_carbon_emissions.to_numpy())

    p.time_step_duration[:] = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    p.operation_year_duration[:] = 8760
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    costs = p.carbon_tax * (
        p.operation_year_duration
        / p.time_step_duration.sum("hour")
        * (p.time_step_duration * s.operation_carbon_emissions).sum("hour")
    )

    assert_equal(s.operation_carbon_costs, costs)

    p.update(
        dict(demand=p.demand.expand_dims(hour=p.hour).copy()),
    )
    p.time_step_duration[:] = [2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 24, 2.0, 2.0, 2.0]
    p.operation_year_duration[:] = p.time_step_duration.sum("hour")
    p.demand.values = np.array(
        [[[10 * val]] for val in p.time_step_duration.values]
    )
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    costs = p.carbon_tax * (
        p.operation_year_duration
        / p.time_step_duration.sum("hour")
        * s.operation_carbon_emissions.sum("hour")
    )

    assert_allclose(s.operation_carbon_costs, costs, rtol=1e-3, atol=1e-3)


@pytest.fixture()
def parameter_net_import_single_horizon(
    parameters_dispatch_invest, net_import
):
    p = xr.merge([parameters_dispatch_invest, net_import])
    p = p.sel(
        area=["area_1"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    return p


def test_net_import(parameter_net_import_single_horizon):
    p = parameter_net_import_single_horizon.sel(
        conversion_tech=[
            "ocgt",
        ],
        hour=[3, 4, 5],
        resource=["electricity", "methane"],
    )
    p.update(
        dict(demand=p.demand.expand_dims(hour=p.hour).copy()),
    )

    p.time_step_duration[:] = [3, 4, 5]
    p.demand.sel(resource="methane").values = np.array(
        [[10 * val] for val in p.time_step_duration.values]
    )
    p.operation_year_duration[:] = 5
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.dropna(dim="year_dec", how="all").squeeze()
    import_costs = (
        p.operation_year_duration
        / p.time_step_duration.sum("hour")
        * (p.net_import_import_price * s.operation_net_import_import).sum(
            "hour"
        )
    )
    assert_equal(
        s.operation_net_import_import.sel(resource="methane").to_numpy(),
        np.array([15, 15, 15]),
    )
    assert_allclose(
        s.operation_net_import_costs.sel(resource="methane").to_numpy(),
        import_costs.sel(resource="methane").to_numpy(),
        rtol=1e-3,
        atol=1e-3,
    )


# TODO Carbon Goals
# TODO Max yearly import
