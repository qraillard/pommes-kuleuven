"""Operating conditions on the whole model (every module together).

Structure (which variables and constraints carry the dimension), neutrality of identical conditions, absent pairs
walked over every variable and constraint, the pseudo-year reference and the two-stage decomposition of the objective.
"""
import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs
from pommes.tests.conditions_helper import (
    assert_condition_dims_consistent,
    with_conditions,
)
from pommes.tests.dataset_nnodes import build_nnode_parameters
from pommes.tests.test_retrofit_conversion import (  # noqa: F401 (fixture)
    p_retrofit_value,
)
from pommes.utils import sort_coords_and_vars

C = "operating_condition"
LABELS = ["a", "b", "c"]
DEMAND_FACTORS = [1.0, 0.8, 1.25]
AVAILABILITY_FACTORS = [1.0, 0.6, 0.9]
FLAGS = [
    "carbon_goal_expected",
    "carbon_goal_area_expected",
    "carbon_goal_group_expected",
    "conversion_max_yearly_production_expected",
    "net_import_max_yearly_energy_import_expected",
    "net_import_max_yearly_energy_export_expected",
]


def _full(parameters, process):
    """Every module: the parameters, the process fixture, a flexible demand and finite annual ceilings."""
    p = xr.merge([parameters, process])
    flexible = xr.zeros_like(
        p.demand.broadcast_like(p.area).broadcast_like(p.hour)
    ).transpose("area", "hour", "resource", "year_op")
    flexible.loc[dict(area="area_1", resource="electricity")] = 2.0
    p["flexdem"] = xr.DataArray(True)
    p["flexdem_demand"] = flexible
    p["flexdem_conservation_hrs"] = xr.DataArray(5).astype("int64")
    p["flexdem_maxload_ratio"] = xr.DataArray(3.0)
    p["flexdem_minload_ratio"] = xr.DataArray(0.0)
    p["conversion_max_yearly_production"] = xr.DataArray(
        [np.nan, 30.0, np.nan, np.nan, np.nan],
        dims="conversion_tech",
        coords=dict(conversion_tech=p.conversion_tech),
    )
    p["net_import_max_yearly_energy_import"] = xr.DataArray(
        [np.nan, np.nan, np.nan, 40.0],
        dims="resource",
        coords=dict(resource=p.resource),
    )
    return check_inputs(sort_coords_and_vars(p))


def _perturbed(p, weights, n=None):
    """Conditions whose demand and conversion availability differ."""
    n = n or len(weights)
    return with_conditions(
        p,
        weights,
        labels=LABELS[:n],
        demand=[p.demand * f for f in DEMAND_FACTORS[:n]],
        conversion_availability=[
            p.conversion_availability * f for f in AVAILABILITY_FACTORS[:n]
        ],
    )


def _flags_true(p):
    p = p.copy()
    for flag in FLAGS:
        p[flag] = xr.DataArray(True)
    return p


def _solve(p, guard=True):
    m = build_model(p)
    if guard:
        assert_condition_dims_consistent(m)
    status, _ = m.solve(solver_name="highs", output_flag=False)
    assert status == "ok"
    return m


# --- 1. what carries the dimension ---------------------------------------------------------------------------------

CAPACITIES = [
    f"operation_{module}_power_capacity"
    for module in ("conversion", "combined", "process", "transport", "storage")
] + ["operation_storage_energy_capacity"]
CAPACITY_CONSTRAINTS = {
    f"{name}_{kind}"
    for name in CAPACITIES
    for kind in ("def", "max_constraint", "min_constraint")
}
WITHOUT_CONDITION_VARIABLES = {
    "annualised_totex",
    "operation_process_nb_units",
    "operation_turpe_contract_power",
    "operation_turpe_fixed_costs",
    *CAPACITIES,
}
WITHOUT_CONDITION_CONSTRAINTS = {
    "annualised_totex_def",
    *CAPACITY_CONSTRAINTS,
    "operation_storage_energy_to_power_min_constraint",
    "operation_process_nb_units_max_if_no_unit_size_constraint",
    "operation_process_power_nb_units_constraint",
    "operation_turpe_fixed_costs_def",
    "operation_turpe_increasing_contract_power_constraint",
}
EXPECTED_ANNUAL_CONSTRAINTS = {
    "operation_carbon_goal_constraint",
    "operation_carbon_goal_area_constraint",
    "operation_conversion_yearly_production_max_constraint",
    "operation_net_import_import_yearly_max_constraint",
    "operation_net_import_export_yearly_max_constraint",
}


def _outside_planning(names):
    return {
        n for n in names if not n.startswith(("planning_", "retrofit"))
    }


def _without_condition(items):
    return _outside_planning(n for n in items if C not in items[n].labels.dims)


@pytest.fixture(scope="module")
def full(parameters, process):
    return _full(parameters, process)


@pytest.mark.parametrize("flags", [False, True])
def test_every_operational_variable_has_the_condition(full, flags):
    p = _perturbed(full, [0.5, 0.5])
    if flags:
        p = _flags_true(p)
    m = build_model(p)
    assert_condition_dims_consistent(m)
    for name in m.variables:
        dims = m.variables[name].labels.dims
        hourly = "hour" in dims
        cost = name.startswith("operation_") and name.endswith("_costs")
        if name.startswith(("planning_", "retrofit")) or (
            name == "annualised_totex"
        ):
            assert C not in dims, name
        elif hourly or cost:
            assert C in dims or name in WITHOUT_CONDITION_VARIABLES, name
    assert _without_condition(m.variables) == WITHOUT_CONDITION_VARIABLES
    # constraints: exactly the capacity, planning, totex, nb_units and TURPE contract rows lack the dimension
    expected = WITHOUT_CONDITION_CONSTRAINTS | (
        EXPECTED_ANNUAL_CONSTRAINTS if flags else set()
    )
    assert _without_condition(m.constraints) == expected


# --- 2. identical conditions are neutral ---------------------------------------------------------------------------


def _capacities(m):
    return {n: m.solution[n] for n in CAPACITIES if n in m.solution}


def _assert_neutral(p, weights, capacities=True):
    m0 = _solve(p, guard=False)
    m1 = _solve(with_conditions(p, weights))
    assert m1.objective.value == pytest.approx(m0.objective.value, rel=1e-7)
    if not capacities:
        return
    capacities = _capacities(m0)
    assert capacities
    for name, values in capacities.items():
        np.testing.assert_allclose(
            m1.solution[name].values, values.values, rtol=1e-7, atol=1e-7
        )


def test_identical_conditions_are_neutral_on_test_case(full):
    # the optimum of test_case is degenerate in its capacities (its operational costs are almost all zero: another
    # conversion or storage capacity costs the same in total), so only the objective is compared
    _assert_neutral(full, [0.5, 0.3, 0.2], capacities=False)


def test_identical_conditions_are_neutral_on_nnodes():
    # three areas: combined and storage capacities and their fixed costs, and the folding over areas, are real here
    # (build_nnode_parameters has no transport technology: transport is in the whole model of test_case, and in the
    # transport file)
    _assert_neutral(build_nnode_parameters(n_areas=3, n_hours=24), [0.5, 0.3, 0.2])


# --- 3. an absent pair in the full model ---------------------------------------------------------------------------


def _assert_rows(m, reference, absent):
    """Every variable and constraint has the rows of the reference minus the absent (year_op, condition) pairs."""
    for kind in ("variables", "constraints"):
        items, ref_items = getattr(m, kind), getattr(reference, kind)
        assert set(items) == set(ref_items)
        carrying = 0
        for name in items:
            got = items[name].labels != -1
            want = ref_items[name].labels != -1
            if C in got.dims:
                carrying += 1
                want = want.reindex({C: got[C]}, fill_value=False)
                want = want & ~absent.broadcast_like(got)
            xr.testing.assert_equal(got, want)
        assert carrying


def _absent(p):
    return p.operating_condition_weight.isnull()


def _in_mode(p, flags):
    """The six annual constraints per condition (False) or on the expectation (True)."""
    return _flags_true(p) if flags else p


@pytest.mark.parametrize("flags", [False, True])
def test_an_absent_pair_in_the_full_model(full, flags):
    # b has other data than a: a row of b leaking into the model would change the objective
    p = _in_mode(_perturbed(full, [1.0, np.nan], n=2), flags)
    m = _solve(p)
    m_a = _solve(_in_mode(_perturbed(full, [1.0], n=1), flags))
    assert m.objective.value == pytest.approx(m_a.objective.value, rel=1e-7)
    _assert_rows(m, m_a, _absent(p))


@pytest.mark.parametrize("flags", [False, True])
def test_an_absent_pair_in_one_year_of_the_full_model(full, flags):
    year = int(full.year_op[1])
    weights = np.full((full.sizes["year_op"], 2), 0.5)
    weights[1] = [1.0, np.nan]
    # availability is the same in both conditions (it has no year): only the demand of b differs, and in `year` the
    # demand of b, absent there, is a copy of that of a in the reference
    demand_b = full.demand * 1.25
    copy_b = demand_b.where(full.year_op != year, full.demand)
    p = with_conditions(full, [0.5, 0.5], demand=[full.demand, demand_b])
    p = _in_mode(p, flags)
    p["operating_condition_weight"] = xr.DataArray(
        weights,
        coords={"year_op": full.year_op.values, C: LABELS[:2]},
        dims=("year_op", C),
    )
    reference = _in_mode(
        with_conditions(full, [0.5, 0.5], demand=[full.demand, copy_b]), flags
    )
    m = _solve(p)
    m_ref = _solve(reference)
    assert m.objective.value == pytest.approx(m_ref.objective.value, rel=1e-7)
    _assert_rows(m, m_ref, _absent(p))
    # the absent pair is exactly (year, b)
    labels = m.constraints["operation_adequacy_constraint"].labels
    b = labels.sel({C: "b"}, drop=True)
    assert not (b.sel(year_op=year) != -1).any()
    assert (b.drop_sel(year_op=year) != -1).all()
    assert (labels.sel({C: "a"}) != -1).all()


# --- 4. pseudo-year reference --------------------------------------------------------------------------------------

WEATHER = [(1.0, 1.0), (1.25, 0.7)]  # (demand factor, availability factor) of the two weather profiles
WEIGHTS = [0.6, 0.4]


def _one_vintage(p):
    """Investment in the vintage 2030 only (end of life 2050), no early decommissioning of any module."""
    p = p.copy(deep=True)
    for name in p.data_vars:
        if name.endswith("_early_decommissioning"):
            p[name] = xr.DataArray(False)
    for name in p.data_vars:
        if name.endswith("_investment_max"):
            p[name] = (p[name] * xr.ones_like(p.year_inv, dtype=float)).copy()
            p[name].loc[{"year_inv": 2020}] = 0.0
    return p


def _two_conditions(p):
    """One operation year (2030), the two weather profiles as conditions weighted 0.6 and 0.4."""
    return with_conditions(
        p,
        WEIGHTS,
        labels=LABELS[:2],
        demand=[p.demand * d for d, _ in WEATHER],
        conversion_availability=[p.conversion_availability * a for _, a in WEATHER],
    )


def _two_years(p):
    """The same profiles as two operation years (2030, 2040) whose discount factors are the weights, no conditions."""
    years = xr.concat([p, p], dim="year_op", data_vars="minimal", coords="minimal", compat="override")
    years = years.assign_coords(year_op=[2030, 2040])
    years["discount_factor"] = ("year_op", WEIGHTS)
    # not in the objective: any value but 1 shows a leak (it would change the objective)
    years["operation_year_weight"] = ("year_op", [5.0, 7.0])
    years["demand"] = xr.concat(
        [p.demand.squeeze("year_op", drop=True) * d for d, _ in WEATHER], dim=years.year_op
    )
    years["conversion_availability"] = xr.concat(
        [p.conversion_availability * a for _, a in WEATHER], dim=years.year_op
    )
    return years


def test_pseudo_year_reference():
    # three areas: the planning costs (annuities, fixed costs) and the operation costs of every module are real
    p = _one_vintage(build_nnode_parameters(n_areas=3, n_hours=24))
    m_conditions = _solve(_two_conditions(p))
    m_years = _solve(_two_years(p), guard=False)
    assert m_conditions.objective.value == pytest.approx(m_years.objective.value, rel=1e-7)
    capacities = [
        n for n in m_conditions.variables if n.startswith("planning_") and n.endswith("_capacity")
    ]
    assert len(capacities) == 4  # conversion, storage (power and energy), combined
    for name in capacities:
        np.testing.assert_allclose(
            m_conditions.solution[name].values, m_years.solution[name].values, rtol=1e-7, atol=1e-7
        )


# --- 5. two-stage decomposition ------------------------------------------------------------------------------------


def _with_tax_group(p, areas, rate):
    p = p.assign_coords(carbon_tax_group=["g"])
    p["carbon_tax_group_areas"] = xr.DataArray(
        [[a in areas] for a in p.area.values], coords=[p.area, p.carbon_tax_group]
    )
    p["carbon_tax_group_rate"] = xr.DataArray(
        [[rate] * p.sizes["year_op"]], coords=[p.carbon_tax_group, p.year_op]
    )
    return p


def test_two_stage_decomposition():
    """
    Stochastic objective == sum of weight x objective of each condition alone, the first-stage variables fixed.

    The variables without the condition (capacities, planning) are fixed to the stochastic optimum in the model of
    each condition alone (weight 1); what remains is the operation of that condition. Their weighted sum is the
    objective only if the expectation is taken once, with the right weights, and no row couples two conditions.
    """
    weights = [0.5, 0.3, 0.2]
    nnodes = build_nnode_parameters(n_areas=3, n_hours=24)
    p = _with_tax_group(_perturbed(nnodes, weights), nnodes.area.values[:2], 300.0)
    m = _solve(p)
    shared = [
        n for n in m.variables if C not in m.variables[n].labels.dims and n != "annualised_totex"
    ]
    assert shared
    total = 0.0
    for k, weight in zip(LABELS, weights):
        alone = p.sel({C: [k]})
        alone["operating_condition_weight"] = xr.ones_like(alone.operating_condition_weight)
        m_k = build_model(alone)
        for name in shared:
            v = m_k.variables[name]
            m_k.add_constraints(
                v == m.solution[name].fillna(0), name=f"fixed_{name}", mask=v.labels != -1
            )
        status, _ = m_k.solve(solver_name="highs", output_flag=False)
        assert status == "ok"
        total += weight * m_k.objective.value
    assert m.objective.value == pytest.approx(total, rel=1e-7)


# --- 6. retrofit: one decision for all the conditions --------------------------------------------------------------

HYDROGEN = 3.0  # demand in 2030: what "smr" (capped at 3) can supply alone
SMR_COST = [300.0, 0.0]  # variable cost of "smr" in the conditions a and b
RETROFIT = "planning_retrofit_conversion_power_capacity"


def _retrofit_market(p, annuity, weights, smr_cost=SMR_COST):
    """
    The retrofit smr -> smr_ccs as the only route to smr_ccs, worth it only where "smr" is dear.

    From the p_retrofit_value slice: a hydrogen demand "smr" covers alone, no direct investment in smr_ccs, no
    conversion annuity or fixed cost (the retrofit annuity is then the only planning cost) and the variable cost of
    "smr" per condition. smr_ccs also draws a little electricity, which costs it methane (shed at 1000): it is dearer
    than "smr" at a variable cost of 0 and cheaper at 300. A gas price cannot favour smr_ccs: both take 1.5 methane.
    """
    p = p.copy(deep=True)
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = HYDROGEN
    for name in ("conversion_annuity_cost", "conversion_fixed_cost"):
        p[name] = xr.where(p[name].notnull(), 0.0, np.nan)
    p["conversion_power_capacity_investment_max"].loc[
        dict(conversion_tech="smr_ccs")
    ] = 0.0
    p["retrofit_conversion_annuity_cost"] = xr.full_like(
        p.retrofit_conversion_annuity_cost, annuity
    )
    cost = lambda smr: p.conversion_variable_cost.where(  # noqa: E731
        p.conversion_tech != "smr", smr
    )
    return with_conditions(
        p,
        weights,
        labels=LABELS[:2],
        conversion_variable_cost=[cost(c) for c in smr_cost],
    )


def _saving(p, smr_cost):
    """What one unit of smr_ccs capacity saves in a condition where "smr" costs `smr_cost` per MWh, in a year."""
    f, vc = p.conversion_factor, p.conversion_variable_cost
    elec = "electricity"
    # electricity of one MWh of hydrogen from smr_ccs, made by ocgt from methane (the methane of smr and smr_ccs is
    # the same: it cancels)
    ocgt_per_mwh = float(
        -f.sel(conversion_tech="smr_ccs", resource=elec)
        / f.sel(conversion_tech="ocgt", resource=elec)
    )
    methane = float(-f.sel(conversion_tech="ocgt", resource="methane"))
    ccs_cost = float(vc.sel(conversion_tech="smr_ccs")) + ocgt_per_mwh * (
        float(vc.sel(conversion_tech="ocgt"))
        + methane * float(p.load_shedding_cost.sel(resource="methane"))
    )
    hours = float(
        p.operation_year_duration.sel(year_op=2030) / p.time_step_duration.sum()
    )
    return hours * max(0.0, smr_cost - ccs_cost)


def _retrofitted(m):
    return float(m.solution[RETROFIT].sum())


@pytest.mark.parametrize(
    "weights, share",
    [
        ((1.0, 0.0), 0.5),  # profitable in a only, weight 1 there
        ((0.0, 1.0), 0.01),  # a, where it pays, costs nothing in the expectation
        ((0.5, 0.5), 0.45),  # below the weighted saving
        ((0.5, 0.5), 0.55),  # above it
        ((0.8, 0.2), 0.7),  # below it, above the saving of equal weights
        ((0.8, 0.2), 0.9),  # above it
    ],
)
def test_retrofit_decision_is_shared(p_retrofit_value, weights, share):  # noqa: F811
    p = p_retrofit_value
    saving_a = _saving(p, SMR_COST[0])
    assert saving_a > 0 and _saving(p, SMR_COST[1]) == 0
    annuity = share * saving_a
    market = _retrofit_market(p, annuity, list(weights))
    m = _solve(market)
    # the retrofit pays when the weighted saving of a unit exceeds its annuity (nothing is saved in b)
    weighted = weights[0] * saving_a
    expected = HYDROGEN if weighted > annuity else 0.0
    assert _retrofitted(m) == pytest.approx(expected, abs=1e-6)
    # objective: the model without the retrofit, less what the retrofitted units save net of their annuity
    off = market.copy(deep=True)
    off["retrofit_conversion"] = xr.DataArray(False)
    m_off = _solve(off)
    assert m.objective.value == pytest.approx(
        m_off.objective.value - expected * (weighted - annuity), rel=1e-7
    )


def test_retrofit_variables_have_no_condition(p_retrofit_value):  # noqa: F811
    m = build_model(_retrofit_market(p_retrofit_value, 1.0, [0.6, 0.4]))
    assert_condition_dims_consistent(m)
    for kind in (m.variables, m.constraints):
        names = [n for n in kind if "retrofit" in n]
        assert names
        for name in names:
            assert C not in kind[name].labels.dims, name
    assert RETROFIT in m.variables
    assert "planning_retrofit_conversion_costs" in m.variables
    # while the dispatch around it does carry the condition
    assert C in m.variables["operation_conversion_power"].labels.dims


def _shortfall(p):
    """The scenario of test_retrofit_conversion.py: 3 units retrofitted and 4 of smr_ccs built directly."""
    p = p.copy(deep=True)
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
    )
    p["retrofit_conversion_annuity_cost"] = xr.full_like(
        p.retrofit_conversion_annuity_cost, 0.1
    )
    return p


def test_retrofit_fixture_with_identical_conditions_is_neutral(
    p_retrofit_value,  # noqa: F811
):
    p = _shortfall(p_retrofit_value)
    m0 = _solve(p, guard=False)
    m1 = _solve(with_conditions(p, [0.5, 0.3, 0.2]))
    assert _retrofitted(m0) == pytest.approx(3.0, abs=1e-6)
    assert m1.objective.value == pytest.approx(m0.objective.value, rel=1e-7)
    for name in (RETROFIT, "planning_conversion_power_capacity"):
        np.testing.assert_allclose(
            m1.solution[name].values,
            m0.solution[name].values,
            rtol=1e-7,
            atol=1e-7,
        )
