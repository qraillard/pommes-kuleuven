import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs

ALLOWED = [
    "conversion_availability",
    "transport_availability",
    "demand",
    "flexdem_demand",
    "net_import_import_price",
    "net_import_export_price",
    "net_import_emission_factor",
    "net_import_total_emission_factor",
    "conversion_variable_cost",
    "combined_variable_cost",
    "process_variable_cost",
    "transport_hurdle_costs",
    "load_shedding_cost",
    "spillage_cost",
    "conversion_max_yearly_production",
    "net_import_max_yearly_energy_import",
    "net_import_max_yearly_energy_export",
]
FLAGS = [
    "carbon_goal_expected",
    "carbon_goal_area_expected",
    "carbon_goal_group_expected",
    "conversion_max_yearly_production_expected",
    "net_import_max_yearly_energy_import_expected",
    "net_import_max_yearly_energy_export_expected",
]


def with_conditions(parameters, names=(), weights=(0.5, 0.5), **extra):
    """The test case with an ``operating_condition`` coordinate (one value per weight)."""
    n = 2 if weights is None else len(weights)
    ds = parameters.assign_coords(operating_condition=list("abc")[:n])
    values = {
        name: xr.DataArray(np.arange(1.0, n + 1), dims="operating_condition")
        for name in names
    }
    if weights is not None:
        values["operating_condition_weight"] = xr.DataArray(
            list(weights), dims="operating_condition"
        )
    return ds.assign(**values, **extra)


@pytest.mark.parametrize("name", ALLOWED)
def test_condition_allowed_on_profiles_costs_and_ceilings(parameters, name):
    check_inputs(with_conditions(parameters, [name]))


@pytest.mark.parametrize(
    "name",
    [
        "conversion_invest_cost",
        "time_step_duration",
        "carbon_goal",
        "turpe_variable_cost",
    ],
)
def test_condition_refused_on_other_parameters(parameters, name):
    with pytest.raises(ValueError, match=name):
        check_inputs(with_conditions(parameters, [name]))


@pytest.mark.parametrize(
    "weights, ok",
    [
        ((0.5, 0.4), False),
        ((0.5, 0.3, 0.2), True),
        ((0.5, np.nan, 0.5), True),
        ((0.5, 0.5 + 5e-10), True),
        ((0.5, 0.5 + 1e-6), False),
    ],
)
def test_weights_must_sum_to_one(parameters, weights, ok):
    ds = with_conditions(parameters, weights=weights)
    if ok:
        check_inputs(ds)
    else:
        with pytest.raises(ValueError, match="operating_condition_weight"):
            check_inputs(ds)


def test_several_conditions_need_a_weight(parameters):
    with pytest.raises(ValueError, match="operating_condition_weight"):
        check_inputs(with_conditions(parameters, weights=None))


def test_single_condition_needs_no_weight(parameters):
    check_inputs(parameters.assign_coords(operating_condition=["a"]))


def test_expected_flag_refuses_a_ceiling_with_conditions(parameters):
    name = "conversion_max_yearly_production"
    with pytest.raises(ValueError, match=name):
        check_inputs(
            with_conditions(parameters, [name], **{f"{name}_expected": True})
        )
    check_inputs(
        with_conditions(parameters, [name], **{f"{name}_expected": False})
    )


def test_flags_default_to_false(parameters):
    ds = check_inputs(parameters.drop_vars(FLAGS))
    for flag in FLAGS:
        assert flag in ds and ds[flag].dtype == bool and not bool(ds[flag])


def with_year_weights(parameters, rows):
    """Weights per (year_op, operating_condition); ``rows`` gives the first years."""
    years = parameters.year_op.values
    values = np.tile([0.5, 0.5], (len(years), 1))
    values[: len(rows)] = rows
    return parameters.assign_coords(operating_condition=["a", "b"]).assign(
        operating_condition_weight=xr.DataArray(
            values,
            dims=("year_op", "operating_condition"),
            coords={"year_op": years},
        )
    )


@pytest.mark.parametrize(
    "rows, message",
    [
        ([[np.inf, 0.5]], "must be finite"),
        ([[1.5, -0.5]], "must be >= 0"),
        ([[np.nan, np.nan]], "every weight is nan"),
    ],
)
def test_weights_are_finite_non_negative_and_present(parameters, rows, message):
    ds = with_year_weights(parameters, rows)
    with pytest.raises(ValueError, match=message) as error:
        check_inputs(ds)
    assert str(int(parameters.year_op.values[0])) in str(error.value)


def test_dimension_without_coordinate_is_refused(parameters):
    """The natural xr.concat gives a dimension with no coordinate: refuse it."""
    a = parameters.conversion_availability
    ds = parameters.assign(
        conversion_availability=xr.concat(
            [a, a * 0.5], dim="operating_condition"
        ),
        operating_condition_weight=xr.DataArray(
            [0.5, 0.5], dims="operating_condition"
        ),
    )
    assert "operating_condition" not in ds.coords
    with pytest.raises(ValueError, match="without a coordinate"):
        check_inputs(ds)
    with pytest.raises(ValueError, match="without a coordinate"):
        build_model(ds)
