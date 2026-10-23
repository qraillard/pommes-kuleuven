"""Unit tests for pommes.model.retrofit._spec."""

from pommes.model.retrofit._spec import RetrofitSpec

CONVERSION = RetrofitSpec("conversion", "area", ("power",))
STORAGE = RetrofitSpec("storage", "area", ("power", "energy"))
TRANSPORT = RetrofitSpec("transport", "link", ("power",))


def test_structural_names_derive_from_module():
    assert CONVERSION.tech_dim == "conversion_tech"
    assert CONVERSION.pair_dim == "retrofit_conversion_pair"
    assert CONVERSION.flag == "retrofit_conversion"
    assert TRANSPORT.site_dim == "link"


def test_host_names_always_carry_the_kind():
    assert (
        CONVERSION.host_planning_var("power")
        == "planning_conversion_power_capacity"
    )
    assert (
        CONVERSION.host_capacity_def("power")
        == "operation_conversion_power_capacity_def"
    )
    assert (
        STORAGE.host_capacity_def("energy")
        == "operation_storage_energy_capacity_def"
    )


def test_cost_params_suffix_the_kind_only_when_ambiguous():
    assert (
        CONVERSION.param("invest_cost", "power")
        == "retrofit_conversion_invest_cost"
    )
    assert (
        STORAGE.param("invest_cost", "power")
        == "retrofit_storage_invest_cost_power"
    )
    assert (
        STORAGE.param("ret_ratio", "energy")
        == "retrofit_storage_ret_ratio_energy"
    )


def test_kind_free_params_have_no_suffix():
    assert (
        STORAGE.param("eligible_share")
        == "retrofit_storage_eligible_share"
    )


def test_capacity_bounds_always_infix_the_kind():
    assert (
        CONVERSION.bound("power", "max")
        == "retrofit_conversion_power_capacity_investment_max"
    )
    assert (
        STORAGE.bound("energy", "min")
        == "retrofit_storage_energy_capacity_investment_min"
    )
