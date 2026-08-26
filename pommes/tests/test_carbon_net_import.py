import numpy as np
import xarray as xr
from numpy.testing import assert_array_almost_equal as assert_equal_epsilon
from numpy.testing import assert_array_equal as assert_equal

from pommes.model.build_model import build_model


def test_carbon(parameters_dispatch_invest, carbon):
    p = xr.merge([parameters_dispatch_invest, carbon])
    p = p.sel(
        area=["area_1"],
        conversion_tech=["ocgt", "wind_onshore"],
        resource=["electricity"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(
        s.operation_conversion_power_capacity.to_numpy(), np.array([10, 22])
    )
    assert_equal_epsilon(
        s.operation_conversion_power.sel(conversion_tech="ocgt").to_numpy(),
        np.array([0, 0, 7.8, 10, 5.6, 1.2, 0, 0, 0, 3.4]),
        12,
    )
    assert_equal_epsilon(
        s.operation_carbon_emissions.to_numpy(),
        np.array([0, 0, 3.9, 5, 2.8, 0.6, 0, 0, 0, 1.7]),
        12,
    )
    assert_equal_epsilon(
        model.dual.operation_carbon_goal_constraint.to_numpy(),
        np.array([-2.0]),
        10,
    )
    assert model.objective.value == 772


def test_carbon_import(parameters_dispatch_invest, carbon, net_import):
    p = xr.merge([parameters_dispatch_invest, carbon, net_import])
    p = p.sel(
        area=["area_1"],
        conversion_tech=["ocgt", "wind_onshore"],
        resource=["electricity", "methane"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    # Values here corresponds built for a 3 hours operation year duration
    p.operation_year_duration[:] = p.time_step_duration.sum().values

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(
        s.operation_conversion_power_capacity.to_numpy(), np.array([10, 50])
    )
    assert_equal_epsilon(
        s.operation_conversion_power.sel(conversion_tech="ocgt").to_numpy(),
        np.array([0, 0, 5, 10, 0, 0, 0, 0, 0, 0]),
        12,
    )
    assert_equal_epsilon(
        s.operation_carbon_emissions.to_numpy(),
        np.array([0, 0, 5, 10, 0, 0, 0, 0, 0, 0]) * (0.5 + 1.5 * 0.2),
        12,
    )

    assert_equal_epsilon(
        model.dual.operation_adequacy_constraint.squeeze().to_numpy(),
        np.array(
            [
                [-0, -0],
                [-0.0, -0.0],
                [39.6, 20.4],
                # 39.6 = electricity: var_cost + conv.cO2 * carbon.tax
                #        + eta_gas * (import_cost + import_emission * carbon.tax)
                [69.6, 20.4],
                [30.2, 14.13333333],
                # TODO: understand 14.13333 fro methane adequacy --> optimisation algorithm?
                [-0.0, -0.0],
                [-0.0, -0.0],
                [-0.0, -0.0],
                [-0.0, -0.0],
                [-0.0, -0.0],
            ]
        ),
        8,
    )
    assert_equal_epsilon(
        model.dual.operation_carbon_goal_constraint.to_numpy(),
        np.array([0]),
        10,
    )
    assert model.objective.value == 1394.0


def test_carbon_no_import(parameters_dispatch_invest, carbon, net_import):
    p = xr.merge([parameters_dispatch_invest, carbon, net_import])
    p = p.sel(
        area=["area_1"],
        conversion_tech=["ocgt", "wind_onshore"],
        resource=["electricity", "methane"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    )
    p = p.copy(deep=True)
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    p["net_import"] = False

    model = build_model(p)
    model.solve(solver_name="highs")
    s = model.solution.squeeze()

    assert_equal(
        s.operation_conversion_power_capacity.to_numpy(), np.array([10, 100])
    )
    assert_equal_epsilon(
        s.operation_conversion_power.sel(conversion_tech="ocgt").to_numpy(),
        np.array([0, 0, 0, 10, 0, 0, 0, 0, 0, 0]),
        12,
    )
    assert_equal_epsilon(
        s.operation_carbon_emissions.to_numpy(),
        np.array([0, 0, 0, 5, 0, 0, 0, 0, 0, 0]),
        12,
    )

    assert_equal_epsilon(
        model.dual.operation_carbon_goal_constraint.to_numpy(),
        np.array([0]),
        10,
    )
    assert model.objective.value == 16390.0
