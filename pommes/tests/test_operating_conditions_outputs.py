"""Operating conditions: outputs (dual scale, summaries, save and read a solution).

Summaries never sum over operating_condition: each total comes per condition, and the cost of a
condition is its planning costs plus its operation costs; the weighted mean over the conditions is
the annualised totex.
"""
import numpy as np
import pytest
import xarray as xr

from pommes.io import calculate_summaries as cs
from pommes.io.read_solution import read_solution
from pommes.io.save_solution import save_solution
from pommes.model.build_model import build_model
from pommes.post_process import get_costs, get_net_generation
from pommes.tests.conditions_helper import with_conditions
from pommes.tests.dataset_nnodes import build_nnode_parameters
from pommes.tests.test_flexdem import _with_defaults
from pommes.utils import (
    condition_dual_scale,
    expected_over_conditions,
    operating_condition_weight,
)

try:  # sparse engine: the units of a solution are unstacked to (area, tech)
    from pommes.model.sparsity import densify_any
except ImportError:  # dense engine: a solution is dense already

    def densify_any(x):
        return x


C = "operating_condition"
WEIGHTS = [0.5, 0.3, 0.2]
FACTORS = [1.0, 0.8, 1.25]


def _solved(p):
    m = build_model(p)
    status, _ = m.solve(solver_name="highs", output_flag=False)
    assert status == "ok"
    return m


def _dense(solution):
    return solution.map(densify_any)


def _perturbed(p, weights):
    """
    Conditions with different demands: one set of capacities, a different dispatch per condition.
    """
    return with_conditions(
        p, weights, demand=[p.demand * f for f in FACTORS[: len(weights)]]
    )


@pytest.fixture(scope="module")
def p_cond(parameters):
    return _perturbed(parameters, WEIGHTS)


@pytest.fixture(scope="module")
def model_cond(p_cond):
    return _solved(p_cond)


# --- duals ---------------------------------------------------------------------------------------


def _priced():
    """One area, ocgt of fixed capacity: a unique adequacy price (the dual is not unique where a
    capacity ties).
    """
    p = build_nnode_parameters(n_areas=1, n_hours=24)
    for module in ("storage", "combined", "carbon"):
        p[module] = False
    p = _with_defaults(p).sel(conversion_tech=["ocgt"])
    for name in (
        "conversion_max_yearly_production",
        "conversion_max_daily_production",
        "conversion_max_capacity_factor",
        "conversion_power_capacity_investment_max",
    ):
        p[name] = p[name] * np.nan
    p["conversion_power_capacity_max"] = (
        p["conversion_power_capacity_max"] * 0 + 120.0
    )
    noise = 1 + 0.3 * np.random.default_rng(0).random(p.demand.shape)
    p["demand"] = p.demand * noise
    return p


def test_dual_scale_gives_back_the_price():
    # three identical conditions: the raw dual of an adequacy row is the weight of its condition
    # times the price without conditions
    p0 = _priced()
    p = with_conditions(p0, WEIGHTS)
    name = "operation_adequacy_constraint"
    ref = _solved(p0).constraints[name].dual
    raw = _solved(p).constraints[name].dual
    assert float(ref.max() - ref.min()) > 0  # the prices differ: the test is not trivial
    price = raw / condition_dual_scale(p)
    for k in p[C].values:
        np.testing.assert_allclose(
            price.sel({C: k}).transpose(*ref.dims).values,
            ref.values,
            rtol=1e-6,
            atol=1e-9,
        )


def test_dual_scale_of_a_weight_zero_condition_gives_no_price():
    p = with_conditions(_priced(), [1.0, 0.0])
    scale = condition_dual_scale(p)
    assert (scale.sel({C: "b"}) == 0).all()
    raw = _solved(p).constraints["operation_adequacy_constraint"].dual
    assert not np.isfinite((raw / scale).sel({C: "b"})).any()


def test_dual_scale_without_conditions_is_one(parameters):
    assert float(condition_dual_scale(parameters)) == 1.0


# --- summaries -----------------------------------------------------------------------------------


@pytest.mark.parametrize("by_year_op", [True, False])
@pytest.mark.parametrize("by_area", [True, False])
def test_summaries_keep_the_condition(p_cond, model_cond, by_year_op, by_area):
    s = model_cond.solution
    kwargs = dict(by_year_op=by_year_op, by_area=by_area)
    for func in (
        cs.calculate_total_costs,
        cs.calculate_total_net_generation,
        cs.calculate_total_emissions,
    ):
        frame = func(s, p_cond, **kwargs)
        assert C in frame.index.names, func.__name__
        assert list(frame.index.unique(C)) == list(p_cond[C].values), (
            func.__name__
        )
        # every table is labelled by its own levels (they are concatenated by position)
        if by_area:
            assert set(frame.index.unique("area")) <= set(p_cond.area.values)
        if by_year_op:
            assert set(frame.index.unique("year_op")) <= set(p_cond.year_op.values)
    # capacities are shared by the conditions: they stay without the dimension (by_area=False
    # already fails without conditions on the transport links: not touched here)
    if by_area:
        capacity = cs.calculate_total_power_capacity(s, p_cond, **kwargs)
        assert C not in capacity.index.names
    for var in s.data_vars:
        assert C not in cs.get_sum_dims(s[var], by_year_op, by_area), var


def test_summaries_return_each_condition_not_their_sum(p_cond, model_cond):
    s = model_cond.solution
    frame = cs.calculate_total_net_generation(s, p_cond, by_area=False)
    # served energy of each condition, from the solution
    generation = s.operation_conversion_net_generation.sel(year_op=2020)
    per_condition = generation.sum(
        [d for d in generation.dims if d != C]
    ).values
    assert len(set(np.round(per_condition, 6))) == 3  # three different dispatches
    conversion = (
        frame.xs("conversion", level="type")
        .groupby(level=["year_op", C])
        .sum()
        .sum(axis=1)
    )
    for k, value in zip(p_cond[C].values, per_condition):
        assert conversion.loc[(2020, k)] == pytest.approx(value, rel=1e-9)


@pytest.fixture(
    scope="module", params=[WEIGHTS, [1.0, 0.0, np.nan]], ids=["weights", "zero_and_absent"]
)
def solved(request, parameters):
    p = _perturbed(parameters, request.param)
    return p, _solved(p)


def test_total_costs_are_planning_plus_operation_per_condition(solved):
    p, m = solved
    s = m.solution
    costs = cs.calculate_total_costs(s, p)
    # one value per condition: planning costs repeat in every condition (an absent pair has no
    # operation cost)
    per_condition = (
        costs.groupby(level=["operation_or_planning", "year_op", C])
        .sum()
        .sum(axis=1)
    )
    w = operating_condition_weight(p).to_series().fillna(0)
    for year in s.year_op.values:
        expected = sum(
            weight
            * (
                per_condition.loc[("operation", year, k)]
                + per_condition.loc[("planning", year, k)]
            )
            for (_, k), weight in w.xs(year, level="year_op", drop_level=False).items()
        )
        assert expected == pytest.approx(
            float(s.annualised_totex.sel(year_op=year).sum()), rel=1e-6
        )
    assert per_condition.xs("planning").groupby(level="year_op").nunique().eq(1).all()


def test_get_costs_and_get_net_generation_keep_the_condition(solved):
    p, m = solved
    s = _dense(m.solution)
    costs = get_costs(s, p)
    assert C in costs.dims
    # per condition: planning + operation(k); the weighted mean is the annualised totex
    mean = expected_over_conditions(costs.sum("asset"), p)
    np.testing.assert_allclose(
        mean.transpose(*s.annualised_totex.dims).values,
        s.annualised_totex.values,
        rtol=1e-6,
        atol=1e-6,
    )
    generation = get_net_generation(s, p)
    assert C in generation.dims
    assert not np.allclose(
        generation.sel({C: "a"}).fillna(0), generation.sel({C: "b"}).fillna(0)
    )


def test_generate_summary_dataframes_keep_the_condition(p_cond, model_cond):
    frames = cs.generate_summary_dataframes_from_results(
        model_cond.solution, p_cond
    )
    for name in ("Total costs - EUR", "CO2 emissions - tCO2eq", "Production - MWh"):
        assert C in frames[name].index.names, name
    assert C not in frames["Operation costs - EUR"].dims  # annualised_totex: an expectation


# --- save and read -------------------------------------------------------------------------------


def test_save_and_read_a_solution_with_conditions(p_cond, model_cond, tmp_path):
    out = tmp_path / "solution"
    solution = _dense(model_cond.solution.copy(deep=True))
    save_solution(model_cond, str(out), model_parameters=p_cond)
    # NetCDF: the dimension and the values come back as they are
    saved = xr.open_dataset(out / "solution.nc")
    assert C in saved.dims
    name = "operation_load_shedding_power"
    xr.testing.assert_allclose(saved[name], solution[name])
    # CSV: one more column, every variable read back with its values
    read = read_solution(str(out / "variables"))
    assert C in read.dims
    assert set(read.data_vars) <= set(solution.data_vars)
    assert name in read.data_vars
    for name in read.data_vars:
        got = read[name].to_series().dropna()
        ref = solution[name].to_series().dropna()
        ref = ref.reorder_levels(got.index.names).sort_index()
        assert got.sort_index().index.equals(ref.index), name
        np.testing.assert_allclose(
            got.sort_index().values, ref.values, rtol=1e-9, atol=1e-9
        )
