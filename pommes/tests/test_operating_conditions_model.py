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
    # three areas: transport, combined and storage capacities and their fixed costs are real here
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


def test_an_absent_pair_in_the_full_model(full):
    # b has other data than a: a row of b leaking into the model would change the objective
    p = _perturbed(full, [1.0, np.nan], n=2)
    m = _solve(p)
    m_a = _solve(_perturbed(full, [1.0], n=1))
    assert m.objective.value == pytest.approx(m_a.objective.value, rel=1e-7)
    _assert_rows(m, m_a, _absent(p))


def test_an_absent_pair_in_one_year_of_the_full_model(full):
    year = int(full.year_op[1])
    weights = np.full((full.sizes["year_op"], 2), 0.5)
    weights[1] = [1.0, np.nan]
    # availability is the same in both conditions (it has no year): only the demand of b differs, and in `year` the
    # demand of b, absent there, is a copy of that of a in the reference
    demand_b = full.demand * 1.25
    copy_b = demand_b.where(full.year_op != year, full.demand)
    p = with_conditions(full, [0.5, 0.5], demand=[full.demand, demand_b])
    p["operating_condition_weight"] = xr.DataArray(
        weights,
        coords={"year_op": full.year_op.values, C: LABELS[:2]},
        dims=("year_op", C),
    )
    reference = with_conditions(
        full, [0.5, 0.5], demand=[full.demand, copy_b]
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
    """Investment in the vintage 2030 only (end of life 2050, no early decommissioning)."""
    p = p.copy(deep=True)
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
