"""Annuities fixed at investment and paid over the full life (``annuity_mode: full_life``).

New default: the annuity of a vintage is ``invest_cost x crf(exact finance rate, life_span)``,
independent of the decommissioning year, and paid in every operation year of the full life even
if the capacity is decommissioned earlier (sunk cost). The previous convention stays available as
``annuity_mode: until_decommissioning``.
"""

import logging
from decimal import Decimal, localcontext

import numpy as np
import pytest
import xarray as xr

from pommes.io.build_input_dataset import (
    build_annuity_rate,
    compute_annuity_cost,
    compute_discount,
    compute_end_of_life,
)
from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs


def crf_closed_form(rate, m):
    """Capital recovery factor with 80 digits (exact for rates near 0), independent of pommes.utils."""
    if m <= 0 or not np.isfinite(rate):
        return np.nan
    if rate == 0:
        return 1 / m
    with localcontext() as context:
        context.prec = 80
        r = Decimal(float(rate))
        return float(r / (1 - (1 + r) ** (-Decimal(float(m)))))


# ----------------------------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------------------------
YEARS_INV = [2020, 2030, 2040]
YEARS_DEC = [2020, 2030, 2040, 2050, 2060, 2070, 2080]
TECHS = ["a", "b"]
INVEST = np.array([[100.0, 110.0, 120.0], [200.0, 210.0, 220.0]])
LIFE = np.array([25.0, 40.0])


def conversion_dataset(rate, rate_dims=(), life=LIFE, early=False):
    return xr.Dataset(
        {
            "conversion": xr.DataArray(True),
            "conversion_invest_cost": xr.DataArray(
                INVEST, dims=["conversion_tech", "year_inv"]
            ),
            "conversion_life_span": xr.DataArray(
                life, dims=["conversion_tech"]
            ),
            "conversion_finance_rate": xr.DataArray(rate, dims=rate_dims),
            "conversion_early_decommissioning": xr.DataArray(early),
        },
        coords={
            "conversion_tech": TECHS,
            "year_inv": YEARS_INV,
            "year_dec": YEARS_DEC,
        },
    )


def config(mode=None, category="conversion", year_start=None):
    param = {
        "invest_cost": f"{category}_invest_cost",
        "finance_rate": f"{category}_finance_rate",
        "life_span": f"{category}_life_span",
    }
    if year_start is not None:
        param["year_start"] = year_start
    pre_process = {
        "annuity_computation": {category: {f"{category}_annuity_cost": param}}
    }
    if mode is not None:
        pre_process["annuity_mode"] = mode
    return {"pre_process": pre_process}


def expected_full_life(rates, life=LIFE):
    """invest_cost x crf(rate, life_span) per (conversion_tech, year_inv)."""
    return np.array(
        [
            [
                INVEST[i, j] * crf_closed_form(float(rates[i]), life[i])
                for j in range(len(YEARS_INV))
            ]
            for i in range(len(TECHS))
        ]
    )


@pytest.mark.parametrize("mode", [None, "full_life"])
@pytest.mark.parametrize(
    "rate",
    [0.063, 0.0617, 0.0754, 0.065, 0.15, 0.2, 0.37, 0.0]
    + [1e-17, -1e-17, 1e-12, -0.3],  # rates near 0 gave inf or lost digits
)
def test_full_life_annuity_is_invest_times_crf_over_the_life_span(rate, mode):
    out = compute_annuity_cost(conversion_dataset(rate), config(mode))
    annuity = out.conversion_annuity_cost
    assert "year_dec" not in annuity.dims
    assert bool(out.annuity_full_life) is True
    np.testing.assert_allclose(
        annuity.transpose("conversion_tech", "year_inv").values,
        expected_full_life([rate, rate]),
        rtol=1e-12,
        atol=0,
    )


def test_full_life_annuity_does_not_depend_on_the_decommissioning_year_grid():
    short = conversion_dataset(0.063)
    long = short.reindex(year_dec=YEARS_DEC + [2090, 2100])
    a = compute_annuity_cost(short, config()).conversion_annuity_cost
    b = compute_annuity_cost(long, config()).conversion_annuity_cost
    xr.testing.assert_identical(a, b)


def not_built(ds, tech):
    """Dataset where `tech` has no invest cost (not an asset of the study)."""
    return ds.assign(
        conversion_invest_cost=ds.conversion_invest_cost.where(
            ds.conversion_tech != tech, 0.0
        )
    )


def annuity_by_tech(ds, mode=None):
    out = compute_annuity_cost(ds, config(mode))
    return out.conversion_annuity_cost.transpose(
        "conversion_tech", "year_inv"
    ).values


@pytest.mark.parametrize("mode", ["full_life", "until_decommissioning"])
@pytest.mark.parametrize("bad_rate", [np.nan, np.inf])
def test_rate_not_finite_with_an_invest_cost_raises_in_both_modes(
    mode, bad_rate
):
    """A NaN annuity is dropped by the model: the capex would be zero."""
    ds = conversion_dataset(np.array([0.063, bad_rate]), ["conversion_tech"])
    with pytest.raises(
        ValueError, match=r"conversion_annuity_cost: 3 entries.*not finite: 3"
    ):
        compute_annuity_cost(ds, config(mode))


@pytest.mark.parametrize("mode", ["full_life", "until_decommissioning"])
@pytest.mark.parametrize("life", [np.nan, 0.0, -5.0])
def test_life_span_missing_with_an_invest_cost_warns_in_both_modes(
    mode, life, caplog
):
    """A vintage without a positive life span holds no capacity: its cost is ignored, with a warning."""
    ds = conversion_dataset(np.array([np.nan, 0.063]), ["conversion_tech"], life=np.array([life, 40.0]))
    with caplog.at_level(logging.WARNING):
        got = compute_annuity_cost(ds, config(mode)).conversion_annuity_cost
    assert "conversion_annuity_cost: 3 entries have an invest cost but a life span" in caplog.text
    assert not np.isfinite(got.sel(conversion_tech="a")).any()
    assert np.isfinite(got.sel(conversion_tech="b")).any()


def test_the_error_counts_every_variable_and_entry():
    ds = conversion_dataset(np.nan).assign(
        transport=xr.DataArray(True),
        transport_invest_cost=xr.DataArray(INVEST[0], dims=["year_inv"]),
        transport_life_span=xr.DataArray(25.0),
        transport_finance_rate=xr.DataArray(np.nan),
    )
    conf = config()
    conf["pre_process"]["annuity_computation"].update(
        config(category="transport")["pre_process"]["annuity_computation"]
    )
    with pytest.raises(ValueError) as error:
        compute_annuity_cost(ds, conf)
    message = str(error.value)
    assert "conversion_annuity_cost: 6 entries" in message
    assert "transport_annuity_cost: 3 entries" in message


@pytest.mark.parametrize("mode", ["full_life", "until_decommissioning"])
def test_assets_without_invest_cost_need_neither_rate_nor_life_span(mode):
    """Invest cost 0 or NaN: no capex, so no rate or life span is needed; the annuity is NaN."""
    ds = conversion_dataset(
        np.array([0.063, np.nan]),
        ["conversion_tech"],
        life=np.array([25.0, np.nan]),
    )
    got = (
        annuity_by_tech(not_built(ds, "b"), mode)
        if mode == "full_life"
        else None
    )
    if got is not None:
        assert np.isnan(got[1]).all() and np.isfinite(got[0]).all()
    nan_cost = ds.assign(
        conversion_invest_cost=ds.conversion_invest_cost.where(
            ds.conversion_tech != "b", np.nan
        )
    )
    out = compute_annuity_cost(nan_cost, config(mode)).conversion_annuity_cost
    assert bool(np.isnan(out.sel(conversion_tech="b")).all())
    assert bool(np.isfinite(out.sel(conversion_tech="a")).any())


def test_full_life_crf_that_is_not_finite_raises(monkeypatch):
    monkeypatch.setattr(
        "pommes.io.build_input_dataset.crf",
        lambda r, m: np.where(m > 0, np.inf, np.nan),
    )
    with pytest.raises(ValueError, match="not finite"):
        compute_annuity_cost(conversion_dataset(0.063), config("full_life"))


def test_full_life_rate_varying_with_investment_year():
    rates = np.array([0.05, 0.063, 0.0754])
    ds = conversion_dataset(rates, ["year_inv"])
    got = compute_annuity_cost(ds, config()).conversion_annuity_cost.transpose(
        "conversion_tech", "year_inv"
    )
    want = np.array(
        [
            [
                INVEST[i, j] * crf_closed_form(rates[j], LIFE[i])
                for j in range(3)
            ]
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(got.values, want, rtol=1e-12)


@pytest.mark.parametrize("rate", [-1.0, -1.5])
def test_full_life_rate_that_cannot_be_used_raises(rate):
    with pytest.raises(ValueError, match="finance"):
        compute_annuity_cost(conversion_dataset(rate), config())


def test_unknown_annuity_mode_raises():
    with pytest.raises(ValueError, match="annuity_mode"):
        compute_annuity_cost(conversion_dataset(0.063), config("forever"))


def test_until_decommissioning_mode_keeps_the_year_dec_table_and_flags_it():
    out = compute_annuity_cost(
        conversion_dataset(0.063), config("until_decommissioning")
    )
    assert "year_dec" in out.conversion_annuity_cost.dims
    assert bool(out.annuity_full_life) is False


def test_year_dec_table_is_the_same_with_early_decommissioning_enabled_or_not():
    """Previous convention: crf over (year_dec - year_inv) years, whatever the early decommissioning flag.

    The off-grid rate 0.063 gives the closed form for every (year_inv, year_dec); the on-grid rate
    0.065 gives exactly what the legacy rate grid gives."""
    for early in (False, True):
        out = compute_annuity_cost(
            conversion_dataset(0.063, early=early),
            config("until_decommissioning"),
        )
        got = out.conversion_annuity_cost.transpose(
            "conversion_tech", "year_inv", "year_dec"
        ).values
        want = np.full(got.shape, np.nan)
        for i in range(2):
            for j, y_inv in enumerate(YEARS_INV):
                for k, y_dec in enumerate(YEARS_DEC):
                    if 0 < y_dec - y_inv <= LIFE[i]:
                        want[i, j, k] = INVEST[i, j] * crf_closed_form(
                            0.063, y_dec - y_inv
                        )
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)

    tables = {
        early: compute_annuity_cost(
            conversion_dataset(0.065, early=early),
            config("until_decommissioning"),
        ).conversion_annuity_cost
        for early in (False, True)
    }
    xr.testing.assert_identical(tables[False], tables[True])
    years = xr.DataArray(
        YEARS_DEC, dims="year_dec", coords={"year_dec": YEARS_DEC}
    )
    years_inv = xr.DataArray(
        YEARS_INV, dims="year_inv", coords={"year_inv": YEARS_INV}
    )
    legacy_grid = build_annuity_rate(years, years_inv).sel(
        finance_rate=0.065, drop=True
    )
    expected = conversion_dataset(0.065).conversion_invest_cost * legacy_grid
    want = xr.where(
        (conversion_dataset(0.065).year_inv < years)
        * (
            years
            <= conversion_dataset(0.065).year_inv
            + conversion_dataset(0.065).conversion_life_span
        ),
        expected,
        np.nan,
    )
    xr.testing.assert_allclose(tables[False].transpose(*want.dims), want)


def test_full_life_retrofit_annuity_needs_year_ret_and_ignores_the_start_year():
    ds = xr.Dataset(
        {
            "retrofit_conversion": xr.DataArray(True),
            "retrofit_conversion_invest_cost": xr.DataArray(100.0),
            "retrofit_conversion_life_span": xr.DataArray(25.0),
            "retrofit_conversion_finance_rate": xr.DataArray(0.063),
        },
        coords={
            "year_inv": [2020],
            "year_ret": [2030, 2040],
            "year_dec": [2040, 2050, 2060],
        },
    )
    conf = config(category="retrofit_conversion", year_start="year_ret")
    out = compute_annuity_cost(ds, conf)
    annuity = out.retrofit_conversion_annuity_cost
    assert "year_dec" not in annuity.dims
    assert float(annuity) == pytest.approx(
        100.0 * crf_closed_form(0.063, 25.0), rel=1e-12
    )
    with pytest.raises(ValueError, match="year_ret"):
        compute_annuity_cost(ds.drop_vars("year_ret"), conf)


def test_full_life_each_category_uses_its_own_rate():
    ds = conversion_dataset(0.063).assign(
        transport=xr.DataArray(True),
        transport_invest_cost=xr.DataArray(INVEST[0], dims=["year_inv"]),
        transport_life_span=xr.DataArray(30.0),
        transport_finance_rate=xr.DataArray(0.0754),
    )
    conf = config()
    conf["pre_process"]["annuity_computation"].update(
        config(category="transport")["pre_process"]["annuity_computation"]
    )
    out = compute_annuity_cost(ds, conf)
    np.testing.assert_allclose(
        out.transport_annuity_cost.values,
        INVEST[0] * crf_closed_form(0.0754, 30.0),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        out.conversion_annuity_cost.transpose(
            "conversion_tech", "year_inv"
        ).values,
        expected_full_life([0.063, 0.063]),
        rtol=1e-12,
    )


# ----------------------------------------------------------------------------------------
# Model: one technology, demand in the first two years only
# ----------------------------------------------------------------------------------------
YEAR_OP = [2020, 2030, 2040, 2050]
INVEST_COST, FIXED_COST, LIFE_SPAN = 1000.0, 10.0, 40.0


def mini_model(
    mode, early, perfect_foresight=False, rate=0.063, life=LIFE_SPAN
):
    """1 MW wanted in 2020 and 2030, nothing after; vintage 2020 only (later ones capped at 0).

    One hour per year, normalisation 1, no discounting: the objective is the sum over operation
    years of fixed cost x capacity + planning costs.
    """
    coords = dict(
        area=np.array(["A"]),
        resource=np.array(["e"]),
        conversion_tech=np.array(["gen"]),
        hour=np.array([0]),
        year_op=np.array(YEAR_OP),
        year_inv=np.array(YEAR_OP),
        year_dec=np.arange(2030, 2100, 10),
    )
    ds = xr.Dataset(coords=coords)
    ds = ds.assign(
        conversion=True,
        transport=False,
        carbon=False,
        combined=False,
        storage=False,
        net_import=False,
        turpe=False,
        process=False,
        flexdem=False,
    )

    def da(dims, value):
        return xr.DataArray(
            np.full([ds.sizes[d] for d in dims], value, dtype="float64"),
            coords={d: ds[d] for d in dims},
        )

    demand = da(["area", "hour", "resource", "year_op"], 0.0)
    demand.loc[dict(year_op=[2020, 2030])] = 1.0
    investment_max = da(["area", "conversion_tech", "year_inv"], 0.0)
    investment_max.loc[dict(year_inv=2020)] = np.nan
    ds = ds.assign(
        demand=demand,
        load_shedding_cost=xr.DataArray(1e6),
        spillage_cost=xr.DataArray(0.0),
        spillage_max_capacity=xr.DataArray(0.0),
        time_step_duration=xr.DataArray([1.0], coords=dict(hour=ds.hour)),
        operation_year_duration=xr.DataArray(1.0),
        discount_rate=xr.DataArray(0.0),
        year_ref=xr.DataArray(2020),
        conversion_factor=xr.DataArray(
            [[1.0]],
            coords=dict(
                conversion_tech=ds.conversion_tech, resource=ds.resource
            ),
        ),
        conversion_invest_cost=da(
            ["area", "conversion_tech", "year_inv"], INVEST_COST
        ),
        conversion_fixed_cost=da(
            ["area", "conversion_tech", "year_op"], FIXED_COST
        ),
        conversion_variable_cost=da(
            ["area", "conversion_tech", "year_op"], 0.0
        ),
        conversion_life_span=da(["area", "conversion_tech", "year_inv"], life),
        conversion_finance_rate=xr.DataArray(rate),
        conversion_power_capacity_investment_max=investment_max,
        conversion_early_decommissioning=xr.DataArray(early),
        conversion_annuity_perfect_foresight=xr.DataArray(perfect_foresight),
    )
    conf = config(mode)
    conf["pre_process"]["discount_factor"] = {
        "discount_rate": "discount_rate",
        "year_ref": "year_ref",
    }
    ds = check_inputs(
        compute_discount(
            compute_annuity_cost(compute_end_of_life(ds), conf), conf
        )
    )
    model = build_model(ds)
    status, condition = model.solve(solver_name="highs", output_flag=False)
    assert condition == "optimal"
    s = model.solution

    def by_year(da):
        # dense models have (area, conversion_tech) dims, sparse ones a conversion_tech_unit dim
        return da.sum([d for d in da.dims if d != "year_op"]).values

    return dict(
        objective=model.objective.value,
        costs=by_year(s.planning_conversion_costs),
        capacity=by_year(s.operation_conversion_power_capacity),
        ds=ds,
    )


def annuity_year(rate, years):
    """Closed-form annual annuity of 1 MW over `years` years."""
    return INVEST_COST * crf_closed_form(rate, years)


@pytest.mark.parametrize("rate", [0.063, 0.065])
def test_full_life_early_decommissioning_saves_fixed_cost_only(rate):
    """The annuity is still charged in 2040 and 2050 after the capacity is gone."""
    full = annuity_year(rate, 40)
    kept = mini_model("full_life", early=False, rate=rate)
    early = mini_model("full_life", early=True, rate=rate)

    np.testing.assert_allclose(kept["capacity"], [1, 1, 1, 1], atol=1e-9)
    np.testing.assert_allclose(kept["costs"], [full] * 4, rtol=1e-9)
    assert kept["objective"] == pytest.approx(
        4 * FIXED_COST + 4 * full, rel=1e-9
    )

    np.testing.assert_allclose(early["capacity"], [1, 1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(early["costs"], [full] * 4, rtol=1e-9)
    assert early["objective"] == pytest.approx(
        2 * FIXED_COST + 4 * full, rel=1e-9
    )
    assert kept["objective"] - early["objective"] == pytest.approx(
        2 * FIXED_COST, rel=1e-9
    )
    assert "year_dec" not in early["ds"].conversion_annuity_cost.dims


@pytest.mark.parametrize("rate", [0.063, 0.065])
@pytest.mark.parametrize("early", [False, True])
def test_until_decommissioning_without_perfect_foresight_is_also_sunk(
    rate, early
):
    """Previous convention, flag 0: cheapest annuity (full life) paid until the end of life."""
    full = annuity_year(rate, 40)
    r = mini_model(
        "until_decommissioning",
        early=early,
        perfect_foresight=False,
        rate=rate,
    )
    np.testing.assert_allclose(r["costs"], [full] * 4, rtol=1e-9)
    years_alive = 2 if early else 4
    assert r["objective"] == pytest.approx(
        years_alive * FIXED_COST + 4 * full, rel=1e-9
    )
    assert "year_dec" in r["ds"].conversion_annuity_cost.dims


@pytest.mark.parametrize("rate", [0.063, 0.065])
def test_until_decommissioning_with_perfect_foresight_stops_the_annuity_at_decommissioning(
    rate,
):
    """Previous convention, flag 1: annuity over (year_dec - year_inv) years, paid until year_dec."""
    kept = mini_model(
        "until_decommissioning", early=False, perfect_foresight=True, rate=rate
    )
    np.testing.assert_allclose(
        kept["costs"], [annuity_year(rate, 40)] * 4, rtol=1e-9
    )
    assert kept["objective"] == pytest.approx(
        4 * FIXED_COST + 4 * annuity_year(rate, 40), rel=1e-9
    )

    early = mini_model(
        "until_decommissioning", early=True, perfect_foresight=True, rate=rate
    )
    a20 = annuity_year(rate, 20)
    np.testing.assert_allclose(early["capacity"], [1, 1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(early["costs"], [a20, a20, 0, 0], rtol=1e-9)
    assert early["objective"] == pytest.approx(
        2 * FIXED_COST + 2 * a20, rel=1e-9
    )
    # dearer to keep the unit to 2060, and the early decommissioning year (2040) is the cheapest
    options = {
        d: n * FIXED_COST + n * annuity_year(rate, 10 * n)
        for d, n in {2040: 2, 2050: 3, 2060: 4}.items()
    }
    assert min(options, key=options.get) == 2040


def objective(ds):
    model = build_model(ds)
    status, condition = model.solve(solver_name="highs", output_flag=False)
    assert condition == "optimal"
    return model.objective.value


def test_full_life_flag_with_a_year_dec_annuity_table_raises():
    r = mini_model("full_life", early=True)
    ds = r["ds"]
    legacy_table = compute_annuity_cost(
        conversion_like(ds), config("until_decommissioning")
    ).conversion_annuity_cost
    assert "year_dec" in legacy_table.dims
    with pytest.raises(ValueError, match="year_dec"):
        build_model(ds.assign(conversion_annuity_cost=legacy_table))


@pytest.mark.parametrize("mode", ["full_life", "until_decommissioning"])
def test_dataset_without_the_flag_is_read_from_its_annuity_tables(mode):
    """Hand-built datasets (older studies, benchmarks) keep working: a year_dec table is the
    previous convention, a table without year_dec is full life."""
    flagged = mini_model(mode, early=True)["ds"]
    unflagged = flagged.drop_vars("annuity_full_life")
    assert objective(unflagged) == pytest.approx(objective(flagged), rel=1e-12)
    checked = check_inputs(unflagged)
    assert bool(checked.annuity_full_life) is (mode == "full_life")
    # a flag that is there is not overridden
    flipped = flagged.assign(
        annuity_full_life=xr.DataArray(mode != "full_life")
    )
    assert bool(check_inputs(flipped).annuity_full_life) is (
        mode != "full_life"
    )


def conversion_like(ds):
    """Dataset with the variables compute_annuity_cost reads, taken from a built model dataset."""
    keep = [
        "conversion",
        "conversion_invest_cost",
        "conversion_life_span",
        "conversion_finance_rate",
    ]
    return ds[keep].assign_coords(year_dec=ds.year_dec)


# ----------------------------------------------------------------------------------------
# Life spans that do not land on the year_dec grid (user decision 2026-10-06: the full-life
# rule is kept, the annuity runs the whole life span; a warning says so)
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("rate", [0.063, 0.065])
def test_off_grid_life_span_pays_the_annuity_over_the_full_life_span(
    rate, caplog
):
    """Life 25 on a 10-year year_dec grid: capacity ends at year_dec 2040 (rounded down from 2045),
    the annuity of crf(25 years) is paid in 2020, 2030 and 2040, the third year without capacity.

    Pins the current rule. The previous convention (flag 0) paid crf(20 years) in 2020 and 2030."""
    with caplog.at_level(
        logging.WARNING, logger="pommes.io.build_input_dataset"
    ):
        full = mini_model("full_life", early=False, rate=rate, life=25.0)
    a25, a20 = annuity_year(rate, 25), annuity_year(rate, 20)
    np.testing.assert_allclose(full["capacity"], [1, 1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(full["costs"], [a25, a25, a25, 0], rtol=1e-9)
    assert full["objective"] == pytest.approx(
        2 * FIXED_COST + 3 * a25, rel=1e-9
    )
    assert "year_start + life_span is not in year_dec" in caplog.text

    legacy = mini_model(
        "until_decommissioning", early=False, rate=rate, life=25.0
    )
    np.testing.assert_allclose(legacy["costs"], [a20, a20, 0, 0], rtol=1e-9)
    assert full["objective"] > legacy["objective"]


def warnings_of(ds, mode, caplog):
    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="pommes.io.build_input_dataset"
    ):
        compute_annuity_cost(ds, config(mode))
    return [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    ]


def test_off_grid_life_span_warning_has_counts_and_an_example(caplog):
    # tech a: life 25 (year_inv + 25 = 2045, 2055, 2065: off the 10-year grid), tech b: life 40 on the grid
    messages = warnings_of(conversion_dataset(0.063), "full_life", caplog)
    assert len(messages) == 1
    message = messages[0]
    assert "conversion_annuity_cost" in message
    assert "for 3 of 6 entries" in message
    assert (
        "'conversion_tech': 'a'" in message and "'year_inv': 2020" in message
    )
    assert "2045" in message and "capacity ends at 2040" in message
    assert "rounded-down year_dec" in message
    assert "finer year_dec grid" in message


def test_no_off_grid_warning_when_life_spans_land_on_the_grid_or_there_is_no_invest_cost(
    caplog,
):
    on_grid = conversion_dataset(0.063, life=np.array([30.0, 40.0]))
    assert warnings_of(on_grid, "full_life", caplog) == []
    off_grid_not_built = not_built(conversion_dataset(0.063), "a")
    assert warnings_of(off_grid_not_built, "full_life", caplog) == []
    zero_life = conversion_dataset(0.063, life=np.array([25.0, 40.0]))
    assert warnings_of(not_built(zero_life, "a"), "full_life", caplog) == []


def test_no_off_grid_warning_in_the_previous_convention(caplog):
    assert (
        warnings_of(conversion_dataset(0.063), "until_decommissioning", caplog)
        == []
    )


def test_a_yearly_year_dec_grid_removes_the_off_grid_warning(caplog):
    yearly = conversion_dataset(0.063).reindex(year_dec=np.arange(2020, 2090))
    assert warnings_of(yearly, "full_life", caplog) == []


def test_off_grid_warning_names_the_retrofit_start_year(caplog):
    ds = xr.Dataset(
        {
            "retrofit_conversion": xr.DataArray(True),
            "retrofit_conversion_invest_cost": xr.DataArray(100.0),
            "retrofit_conversion_life_span": xr.DataArray(25.0),
            "retrofit_conversion_finance_rate": xr.DataArray(0.063),
        },
        coords={
            "year_inv": [2020],
            "year_ret": [2030, 2040],
            "year_dec": [2040, 2050, 2060, 2070],
        },
    )
    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="pommes.io.build_input_dataset"
    ):
        compute_annuity_cost(
            ds, config(category="retrofit_conversion", year_start="year_ret")
        )
    text = caplog.text
    assert "retrofit_conversion_annuity_cost" in text
    assert "for 2 of 2 entries" in text
    assert "'year_ret': 2030" in text and "2055" in text
