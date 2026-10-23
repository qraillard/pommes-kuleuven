"""Annuities use the exact finance rate of the dataset, never a grid value.

Year_dec tables of ``annuity_mode: until_decommissioning`` (previous convention).

``compute_annuity_cost`` used to look the capital recovery factor up on a fixed
grid of rates (0 to 0.145 by 0.005) and turned any other rate into 0, an
interest-free annuity. These tests compare the result with the closed form
``cost * r / (1 - (1 + r) ** -m)``, ``m = year_dec - year_start``, written here
independently of ``pommes.utils.crf``.
"""

import logging
from decimal import Decimal, localcontext

import numpy as np
import pytest
import xarray as xr

from pommes.io.build_input_dataset import (
    build_annuity_rate,
    compute_annuity_cost,
)
from pommes.model.data_validation.dataset_check import check_inputs
from pommes.utils import crf

YEARS_INV = [2020, 2030, 2040]
YEARS_DEC = [2020, 2030, 2040, 2050, 2060, 2070, 2080]
TECHS = ["a", "b"]
INVEST = np.array([[100.0, 110.0, 120.0], [200.0, 210.0, 220.0]])
LIFE = np.array([25.0, 40.0])


def crf_reference(rate, m):
    """Capital recovery factor with 80 digits (exact for rates near 0), independent of pommes.utils."""
    if rate == 0:
        return 1 / m
    with localcontext() as context:
        context.prec = 80
        r = Decimal(float(rate))
        return float(r / (1 - (1 + r) ** (-Decimal(float(m)))))


def closed_form(rate, life, cost, year_start, year_dec):
    """Annuity of one asset in one decommissioning year, EUR, NaN outside the life span."""
    m = year_dec - year_start
    if not (0 < m <= life) or not np.isfinite(rate):
        return np.nan
    return cost * crf_reference(rate, m)


def expected_conversion(rates):
    """Expected conversion_annuity_cost (tech, year_inv, year_dec), one rate per tech."""
    out = np.full((len(TECHS), len(YEARS_INV), len(YEARS_DEC)), np.nan)
    for i in range(len(TECHS)):
        for j, y_inv in enumerate(YEARS_INV):
            for k, y_dec in enumerate(YEARS_DEC):
                out[i, j, k] = closed_form(
                    float(rates[i]), LIFE[i], INVEST[i, j], y_inv, y_dec
                )
    return out


def conversion_dataset(rate, rate_dims=()):
    """Conversion module with two technologies; `rate` is a scalar or one value per tech."""
    return xr.Dataset(
        {
            "conversion": xr.DataArray(True),
            "conversion_invest_cost": xr.DataArray(
                INVEST, dims=["conversion_tech", "year_inv"]
            ),
            "conversion_life_span": xr.DataArray(
                LIFE, dims=["conversion_tech"]
            ),
            "conversion_finance_rate": xr.DataArray(rate, dims=rate_dims),
        },
        coords={
            "conversion_tech": TECHS,
            "year_inv": YEARS_INV,
            "year_dec": YEARS_DEC,
        },
    )


def annuity_config(
    category="conversion",
    year_start=None,
    name=None,
    mode="until_decommissioning",
):
    """Config of one annuity; these tests check the year_dec tables (previous convention)."""
    param = {
        "invest_cost": f"{category}_invest_cost",
        "finance_rate": f"{category}_finance_rate",
        "life_span": f"{category}_life_span",
    }
    if year_start is not None:
        param["year_start"] = year_start
    name = name or f"{category}_annuity_cost"
    return {
        "pre_process": {
            "annuity_mode": mode,
            "annuity_computation": {category: {name: param}},
        }
    }


def conversion_annuity(ds):
    out = compute_annuity_cost(ds, annuity_config())
    return out.conversion_annuity_cost.transpose(
        "conversion_tech", "year_inv", "year_dec"
    ).values


# Rates off the old grid (0.063), noisy floats, on the grid, exactly at its
# excluded upper end (0.15), above it, and zero.
@pytest.mark.parametrize(
    "rate",
    [0.063, 0.0617, 0.0754, 0.1 + 0.2, 0.05, 0.15, 0.2, 0.37, 0.0]
    + [1e-17, 1e-12, -1e-12, -0.3],
)
def test_annuity_equals_closed_form_for_any_scalar_rate(rate):
    got = conversion_annuity(conversion_dataset(rate))
    want = expected_conversion([rate, rate])
    # same NaN pattern (outside the life span and year_dec <= year_inv) ...
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    assert np.isfinite(got).sum() == 18  # 6 for a (25 y), 12 for b (40 y)
    # ... same values
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_off_grid_rate_is_not_interest_free():
    """The bug: 0.063 gave cost / m, now strictly above it."""
    got = conversion_annuity(conversion_dataset(0.063))
    m = np.array(YEARS_DEC)[None, :] - np.array(YEARS_INV)[:, None]
    interest_free = INVEST[0][:, None] / np.where(m > 0, m, np.nan)
    finite = np.isfinite(got[0])
    assert (got[0][finite] > interest_free[finite] * 1.01).all()


def test_rate_zero_keeps_one_over_m():
    got = conversion_annuity(conversion_dataset(0.0))
    y_inv, y_dec = 2020, 2040
    assert got[0, 0, YEARS_DEC.index(y_dec)] == pytest.approx(
        INVEST[0, 0] / (y_dec - y_inv), rel=1e-15
    )


def test_rates_differ_by_technology():
    rates = np.array([0.063, 0.0617])
    ds = conversion_dataset(rates, rate_dims=["conversion_tech"])
    np.testing.assert_allclose(
        conversion_annuity(ds), expected_conversion(rates), rtol=1e-12
    )


def test_rate_varying_with_investment_year():
    rates = np.array([0.05, 0.063, 0.0754])
    ds = conversion_dataset(rates, rate_dims=["year_inv"])
    want = np.array(
        [
            [
                [
                    closed_form(rates[j], LIFE[i], INVEST[i, j], y_inv, y_dec)
                    for y_dec in YEARS_DEC
                ]
                for j, y_inv in enumerate(YEARS_INV)
            ]
            for i in range(len(TECHS))
        ]
    )
    np.testing.assert_allclose(conversion_annuity(ds), want, rtol=1e-12)


def not_built(ds, tech):
    """Dataset where `tech` has no invest cost (not an asset of the study)."""
    cost = ds.conversion_invest_cost
    return ds.assign(
        conversion_invest_cost=cost.where(ds.conversion_tech != tech, 0.0)
    )


@pytest.mark.parametrize("bad_rate", [np.nan, np.inf, -np.inf])
def test_rate_not_finite_with_an_invest_cost_raises_with_the_count(bad_rate):
    """Used to give a NaN annuity, i.e. a capex of zero in the model."""
    ds = conversion_dataset(
        np.array([0.063, bad_rate]), rate_dims=["conversion_tech"]
    )
    with pytest.raises(
        ValueError, match=r"conversion_annuity_cost: 3 entries"
    ):
        conversion_annuity(ds)


def test_rate_not_finite_without_an_invest_cost_gives_nan_for_that_asset_only():
    for bad_rate in (np.nan, np.inf):
        ds = not_built(
            conversion_dataset(
                np.array([0.063, bad_rate]), rate_dims=["conversion_tech"]
            ),
            "b",
        )
        got = conversion_annuity(ds)
        assert np.isnan(got[1]).all()
        np.testing.assert_allclose(
            got[0], expected_conversion([0.063, 0.063])[0], rtol=1e-12
        )


def test_all_rates_nan_raises_with_an_invest_cost_and_is_all_nan_without():
    with pytest.raises(ValueError, match=r"6 entries"):
        conversion_annuity(conversion_dataset(np.nan))
    ds = conversion_dataset(np.nan)
    ds = ds.assign(conversion_invest_cost=ds.conversion_invest_cost * 0.0)
    assert np.isnan(conversion_annuity(ds)).all()


def test_nan_invest_cost_needs_no_rate():
    ds = conversion_dataset(np.nan)
    ds = ds.assign(conversion_invest_cost=ds.conversion_invest_cost * np.nan)
    assert np.isnan(conversion_annuity(ds)).all()


def test_nan_or_non_positive_life_span_with_an_invest_cost_warns(caplog):
    """Such a vintage holds no capacity, so its invest cost is ignored (no error)."""
    for life in (
        np.array([np.nan, 40.0]),
        np.array([0.0, 40.0]),
        np.array([-5.0, 40.0]),
    ):
        ds = conversion_dataset(0.063)
        ds = ds.assign(
            conversion_life_span=xr.DataArray(life, dims=["conversion_tech"])
        )
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            conversion_annuity(ds)
        assert "3 entries have an invest cost but a life span" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            conversion_annuity(not_built(ds, "a"))  # no invest cost: no warning
        assert "life span" not in caplog.text


@pytest.mark.parametrize(
    "rate", [0.0, 1e-17, 1e-12, 0.063, 0.37, -0.3, 1.0, 5.0]
)
@pytest.mark.parametrize("m", [1, 20, 40.5])
def test_crf_matches_the_80_digit_reference(rate, m):
    scalar = crf(rate, m)
    array = crf(np.array([rate]), np.array([m]))[0]
    want = crf_reference(rate, m)
    assert np.isfinite(scalar)
    assert scalar == pytest.approx(want, rel=1e-13)
    assert array == pytest.approx(want, rel=1e-13)


def test_crf_undefined_cases_are_nan_not_inf():
    for rate, m in [
        (0.05, 0),
        (0.05, -3),
        (-1.0, 10),
        (-2.0, 10),
        (np.nan, 10),
        (0.05, np.nan),
    ]:
        assert np.isnan(crf(rate, m))
        assert np.isnan(crf(np.array([rate]), np.array([m]))[0])


def test_crf_that_is_not_finite_raises(monkeypatch):
    """The guard of the legacy table, whatever crf returns."""
    monkeypatch.setattr(
        "pommes.io.build_input_dataset.crf",
        lambda r, m: np.inf if m > 0 else np.nan,
    )
    with pytest.raises(ValueError, match="not finite"):
        conversion_annuity(conversion_dataset(0.063))


def test_float32_rate_is_looked_up_exactly():
    rate = np.float32(0.063)
    got = conversion_annuity(conversion_dataset(rate))
    np.testing.assert_allclose(
        got, expected_conversion([float(rate)] * 2), rtol=1e-12
    )


@pytest.mark.parametrize("rate", [-1.0, -1.5])
def test_rate_that_cannot_be_computed_raises(rate):
    with pytest.raises(ValueError, match="finance"):
        conversion_annuity(conversion_dataset(rate))


def test_each_category_uses_its_own_exact_rate():
    ds = conversion_dataset(0.063).assign(
        transport=xr.DataArray(True),
        transport_invest_cost=xr.DataArray(INVEST[0], dims=["year_inv"]),
        transport_life_span=xr.DataArray(30.0),
        transport_finance_rate=xr.DataArray(0.0754),
    )
    config = annuity_config()
    config["pre_process"]["annuity_computation"].update(
        annuity_config("transport")["pre_process"]["annuity_computation"]
    )
    out = compute_annuity_cost(ds, config)
    np.testing.assert_allclose(
        out.conversion_annuity_cost.transpose(
            "conversion_tech", "year_inv", "year_dec"
        ).values,
        expected_conversion([0.063, 0.063]),
        rtol=1e-12,
    )
    want = np.array(
        [
            [
                closed_form(0.0754, 30.0, INVEST[0, j], y_inv, y_dec)
                for y_dec in YEARS_DEC
            ]
            for j, y_inv in enumerate(YEARS_INV)
        ]
    )
    np.testing.assert_allclose(
        out.transport_annuity_cost.transpose("year_inv", "year_dec").values,
        want,
        rtol=1e-12,
    )


def test_inactive_module_does_not_need_its_rate():
    """A module switched off may lack its finance rate variable altogether."""
    ds = conversion_dataset(0.063).assign(transport=xr.DataArray(False))
    config = annuity_config()
    config["pre_process"]["annuity_computation"].update(
        annuity_config("transport")["pre_process"]["annuity_computation"]
    )
    out = compute_annuity_cost(ds, config)
    assert "transport_annuity_cost" not in out


# Retrofit: the repayment starts at year_ret, not year_inv.
YEARS_RET = [2030, 2040]
YEARS_DEC_RET = [2030, 2040, 2050, 2060, 2070]


def retrofit_dataset(rate):
    return xr.Dataset(
        {
            "retrofit_conversion": xr.DataArray(True),
            "retrofit_conversion_invest_cost": xr.DataArray(100.0),
            "retrofit_conversion_life_span": xr.DataArray(25.0),
            "retrofit_conversion_finance_rate": xr.DataArray(rate),
        },
        coords={
            "year_inv": [2020],
            "year_ret": YEARS_RET,
            "year_dec": YEARS_DEC_RET,
        },
    )


@pytest.mark.parametrize("rate", [0.063, 0.0754, 0.2, 0.0])
def test_retrofit_annuity_uses_exact_rate_from_year_ret(rate):
    ds = retrofit_dataset(rate)
    out = compute_annuity_cost(
        ds,
        annuity_config("retrofit_conversion", year_start="year_ret"),
    ).retrofit_conversion_annuity_cost.transpose("year_ret", "year_dec")
    want = np.array(
        [
            [
                closed_form(rate, 25.0, 100.0, y_ret, y_dec)
                for y_dec in YEARS_DEC_RET
            ]
            for y_ret in YEARS_RET
        ]
    )
    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(want))
    np.testing.assert_allclose(out.values, want, rtol=1e-12)
    # year_ret 2040 -> year_dec 2050: 10 annuities, 2030 -> 2050: 20 annuities
    assert out.sel(year_ret=2040, year_dec=2050) != out.sel(
        year_ret=2030, year_dec=2050
    )


def test_build_annuity_rate_with_given_rates_has_exactly_those_rates():
    years_dec = xr.DataArray(
        YEARS_DEC, dims="year_dec", coords={"year_dec": YEARS_DEC}
    )
    years_inv = xr.DataArray(
        YEARS_INV, dims="year_inv", coords={"year_inv": YEARS_INV}
    )
    table = build_annuity_rate(
        years_dec, years_inv, finance_rates=[0.063, 0.2, 0.0]
    )
    np.testing.assert_array_equal(table.finance_rate.values, [0.0, 0.063, 0.2])
    m = 2050 - 2030
    assert float(
        table.sel(year_dec=2050, year_inv=2030, finance_rate=0.063)
    ) == pytest.approx(0.063 / (1 - 1.063 ** (-m)), rel=1e-12)
    assert float(
        table.sel(year_dec=2050, year_inv=2030, finance_rate=0.0)
    ) == pytest.approx(1 / m)
    assert np.isnan(
        table.sel(year_dec=2030, year_inv=2030, finance_rate=0.063)
    )


def test_build_annuity_rate_default_arguments_still_work():
    """Legacy call without finance_rates: the old grid, unchanged values."""
    years_dec = xr.DataArray(
        YEARS_DEC, dims="year_dec", coords={"year_dec": YEARS_DEC}
    )
    years_inv = xr.DataArray(
        YEARS_INV, dims="year_inv", coords={"year_inv": YEARS_INV}
    )
    table = build_annuity_rate(years_dec, years_inv)
    assert table.finance_rate.size == 30
    legacy = build_annuity_rate(
        years_dec, years_inv, max_finance_rate=0.1, finance_rate_step=0.01
    )
    assert legacy.finance_rate.size == 10


def test_on_grid_values_are_unchanged_by_the_exact_lookup():
    """The exact table gives the same numbers as the legacy grid on grid rates."""
    years_dec = xr.DataArray(
        YEARS_DEC, dims="year_dec", coords={"year_dec": YEARS_DEC}
    )
    years_inv = xr.DataArray(
        YEARS_INV, dims="year_inv", coords={"year_inv": YEARS_INV}
    )
    legacy = build_annuity_rate(years_dec, years_inv).sel(finance_rate=0.05)
    exact = build_annuity_rate(years_dec, years_inv, finance_rates=[0.05]).sel(
        finance_rate=0.05
    )
    xr.testing.assert_identical(
        legacy.drop_vars("finance_rate"), exact.drop_vars("finance_rate")
    )


# check_inputs: max life span without stacking every *_life_span variable.
def life_span_dataset():
    """Life spans (max 20) on pairwise different dimensions; horizon 2030 + 30 = 2060."""
    return xr.Dataset(
        {
            "conversion_life_span": xr.DataArray(
                [[20.0, np.nan], [10.0, 15.0]],
                dims=["area", "conversion_tech"],
            ),
            "storage_life_span": xr.DataArray(
                [[10.0, 20.0], [5.0, 12.0]], dims=["storage_tech", "year_inv"]
            ),
            "transport_life_span": xr.DataArray(
                [[np.nan, 18.0]], dims=["link", "transport_tech"]
            ),
            "combined_life_span": xr.DataArray(np.nan),
        },
        coords={
            "area": ["x", "y"],
            "conversion_tech": ["a", "b"],
            "storage_tech": ["s1", "s2"],
            "link": ["l"],
            "transport_tech": ["t1", "t2"],
            "year_inv": [2020, 2030],
            "year_dec": [2050, 2060],
        },
    )


LONGEST = {  # one value on the variable's own dimensions, NaN where the variable has no value
    "conversion_life_span": lambda v: xr.DataArray(
        [[np.nan, v], [np.nan, np.nan]], dims=["area", "conversion_tech"]
    ),
    "storage_life_span": lambda v: xr.DataArray(
        [[np.nan, np.nan], [v, np.nan]], dims=["storage_tech", "year_inv"]
    ),
    "transport_life_span": lambda v: xr.DataArray(
        [[v, np.nan]], dims=["link", "transport_tech"]
    ),
    "combined_life_span": lambda v: xr.DataArray(v),
}


def test_check_inputs_accepts_the_base_dataset():
    check_inputs(life_span_dataset())


@pytest.mark.parametrize("holder", list(LONGEST))
def test_check_inputs_max_life_span_is_the_maximum_over_all_variables(holder):
    """Same result as the stacked maximum: whichever variable holds the longest life decides."""
    ds = life_span_dataset()
    ds[holder] = LONGEST[holder](30.0)
    check_inputs(ds)  # 2030 + 30 = 2060: on the boundary, accepted
    ds = life_span_dataset()
    ds[holder] = LONGEST[holder](30.5)
    with pytest.raises(ValueError, match=r"Max life span: 30\.5"):
        check_inputs(ds)


def test_check_inputs_does_not_stack_the_life_span_variables(monkeypatch):
    """The union of the dimensions of all life spans (105 GiB on a large study) is never built."""

    def forbidden(self, *args, **kwargs):
        raise AssertionError(
            "check_inputs stacked the variables with to_dataarray"
        )

    monkeypatch.setattr(xr.Dataset, "to_dataarray", forbidden)
    check_inputs(life_span_dataset())


def test_check_inputs_without_life_span_or_with_only_nan():
    names = [
        "conversion_life_span",
        "storage_life_span",
        "transport_life_span",
        "combined_life_span",
    ]
    ds = life_span_dataset().drop_vars(names)
    check_inputs(ds)
    check_inputs(ds.assign(conversion_life_span=xr.DataArray(np.nan)))


def test_check_inputs_skips_a_life_span_variable_with_a_zero_length_dimension():
    """nanmax of an empty array raised ValueError."""
    ds = life_span_dataset().assign_coords(
        process_tech=np.array([], dtype=str)
    )
    ds["process_life_span"] = xr.DataArray(
        np.empty((2, 0)),
        dims=["area", "process_tech"],
        coords={"area": ["x", "y"], "process_tech": np.array([], dtype=str)},
    )
    check_inputs(ds)  # max 20, from the other variables
    longest = ds.copy(deep=True)
    longest["transport_life_span"] = LONGEST["transport_life_span"](30.5)
    with pytest.raises(ValueError, match=r"Max life span: 30\.5"):
        check_inputs(longest)


def test_check_inputs_with_only_an_empty_life_span_variable():
    ds = life_span_dataset().drop_vars(
        [
            "conversion_life_span",
            "storage_life_span",
            "transport_life_span",
            "combined_life_span",
        ]
    )
    ds = ds.assign_coords(process_tech=np.array([], dtype=str))
    ds["process_life_span"] = xr.DataArray(
        np.empty((2, 0)),
        dims=["area", "process_tech"],
        coords={"area": ["x", "y"], "process_tech": np.array([], dtype=str)},
    )
    check_inputs(ds)
