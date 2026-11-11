import numpy as np
import pytest
import xarray as xr

from pommes.model.data_validation.dataset_check import check_inputs

ALLOWED = [
    "conversion_availability", "transport_availability", "demand",
    "flexdem_demand", "net_import_import_price", "net_import_export_price",
    "net_import_emission_factor", "net_import_total_emission_factor",
    "conversion_variable_cost", "combined_variable_cost",
    "process_variable_cost", "transport_hurdle_costs", "load_shedding_cost",
    "spillage_cost", "conversion_max_yearly_production",
    "net_import_max_yearly_energy_import",
    "net_import_max_yearly_energy_export",
]
FLAGS = [
    "carbon_goal_expected", "carbon_goal_area_expected",
    "carbon_goal_group_expected", "conversion_max_yearly_production_expected",
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
        ((0.5, 0.5 + 2e-9), True),
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
