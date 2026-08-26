import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal as assert_equal

from pommes.model.build_model import build_model


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
