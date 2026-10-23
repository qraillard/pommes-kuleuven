import xarray as xr
from numpy.testing import assert_equal

from pommes import test_case_path
from pommes.io.build_input_dataset import (
    build_input_parameters,
    read_config_file,
)
from pommes.model.data_validation.dataset_check import check_inputs


def test_build_from_test_conf(parameters):
    config_ = read_config_file(file_path=f"{test_case_path}/config.yaml")
    # The fixtures hold annuity tables by year_dec: previous convention.
    config_["pre_process"]["annuity_mode"] = "until_decommissioning"

    p = build_input_parameters(config_)
    p = check_inputs(p)

    assert_equal(
        sorted([x for x in p.coords]), sorted([x for x in parameters.coords])
    )
    for coord in parameters.coords:
        coord1, coord2 = xr.broadcast(
            p[coord], parameters[coord]
        )  # Here coord converted to DataArray
        xr.testing.assert_equal(coord1, coord2)

    assert_equal(
        sorted([x for x in p.data_vars]),
        sorted([x for x in parameters.data_vars]),
    )

    for variable in parameters.data_vars:
        da1, da2 = xr.broadcast(p[variable], parameters[variable])
        # print(variable)
        xr.testing.assert_equal(da1, da2)


def test_build_from_test_conf_full_life_is_the_default(parameters):
    """Default annuity mode: invest cost x crf(rate, life span), no year_dec dimension."""
    config_ = read_config_file(file_path=f"{test_case_path}/config.yaml")
    assert "annuity_mode" not in config_["pre_process"]

    p = build_input_parameters(config_)

    assert bool(p.annuity_full_life)
    for category in ("combined", "conversion", "transport"):
        annuity = p[f"{category}_annuity_cost"]
        assert "year_dec" not in annuity.dims
        # finance rate 0 and life span 20 in the test case: crf = 1 / 20
        xr.testing.assert_allclose(
            annuity.broadcast_like(p[f"{category}_invest_cost"]),
            (p[f"{category}_invest_cost"] / 20).broadcast_like(annuity),
        )
    assert "year_dec" not in p.storage_annuity_cost_power.dims
    assert "year_dec" not in p.storage_annuity_cost_energy.dims


def test_build_from_test_conf_default(parameters):
    config_ = read_config_file(file_path=f"{test_case_path}/config.yaml")

    p = build_input_parameters(config_)
    p = p.drop_vars(["discount_factor", "discount_rate"])

    p = check_inputs(p)

    assert_equal(
        sorted([x for x in p.data_vars]),
        sorted([x for x in parameters.data_vars]),
    )
