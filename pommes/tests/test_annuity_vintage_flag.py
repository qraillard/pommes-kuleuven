"""``until_decommissioning``: the perfect-foresight flag chooses the annuity per vintage.

The planning cost row of every module is ``-planning_costs + annuity_terms == 0``. With
``annuity_mode: until_decommissioning`` the annuity of a vintage is either the one up to the
decommissioning year (flag 1) or the cheapest one up to the end of life (flag 0), and the flag has
the dimensions (tech, year_inv). It used to be applied after the sum over ``year_inv``, which gave
the expression a ``year_inv`` dimension back: the constraint was written once per vintage, as
identical rows when the flag is constant per tech, and as contradicting rows when it differs
between the vintages of one tech. The choice is now made per vintage, then summed.

The checks compare the built constraint, row by row and coefficient by coefficient, with:

- the same dataset with a scalar flag (never affected by the bug), which the golden
  fingerprints below pin to the behaviour before the fix (also for ``full_life``);
- a tiny model solved with HiGHS, against closed-form annuities.
"""

import hashlib
from typing import NamedTuple

import numpy as np
import pytest
import xarray as xr

from pommes.io.build_input_dataset import (
    compute_annuity_cost,
    compute_discount,
    compute_end_of_life,
)
from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs

# Fixtures of the module tests, reused here (pytest resolves them by name).
from pommes.tests.test_annuity_full_life import (
    config,
    crf_closed_form,
)
from pommes.tests.test_combined import (  # noqa: F401
    parameters_combined,
    parameters_combined_multi_horizon,
)
from pommes.tests.test_conversion import (  # noqa: F401
    parameters_conversion_multi_horizon,
)
from pommes.tests.test_process import (  # noqa: F401
    parameters_process,
    parameters_process_multi_horizon,
)
from pommes.tests.test_retrofit_conversion import (
    p_retrofit_multi_vintage,  # noqa: F401
)
from pommes.tests.test_retrofit_other_modules import (  # noqa: F401
    p_retrofit_combined_multi_vintage,
    p_retrofit_storage_multi_vintage,
    p_retrofit_transport_multi_vintage,
)
from pommes.tests.test_storage import (  # noqa: F401
    parameters_storage,
    parameters_storage_multi_horizon,
)
from pommes.tests.test_transport import (  # noqa: F401
    parameters_transport,
    parameters_transport_multi_horizon,
)


class Spec(NamedTuple):
    fixture: str
    host: str  # prefix of the module whose planning cost row is built
    flag: str  # perfect-foresight parameter
    flag_dims: tuple  # its dimensions, without the vintage dimension
    vintages: tuple  # vintage dimensions the flag may have (the first is the default one)
    constraint: str
    costs: str  # planning cost variable (one row per element)


def module_spec(name, fixture, tech_dim, extra_dims=("area",)):
    return Spec(
        fixture=fixture,
        host=name,
        flag=f"{name}_annuity_perfect_foresight",
        flag_dims=(*extra_dims, tech_dim),
        vintages=("year_inv",),
        constraint=f"planning_{name}_costs_def",
        costs=f"planning_{name}_costs",
    )


def retrofit_spec(name, fixture):
    return Spec(
        fixture=fixture,
        host=name,
        flag=f"retrofit_{name}_annuity_perfect_foresight",
        flag_dims=(f"retrofit_{name}_pair",),
        vintages=("year_ret", "year_inv"),
        constraint=f"planning_retrofit_{name}_costs_def",
        costs=f"planning_retrofit_{name}_costs",
    )


SPECS = {
    "conversion": module_spec(
        "conversion", "parameters_conversion_multi_horizon", "conversion_tech"
    ),
    "combined": module_spec(
        "combined", "parameters_combined_multi_horizon", "combined_tech"
    ),
    "process": module_spec(
        "process", "parameters_process_multi_horizon", "process_tech"
    ),
    "storage": module_spec(
        "storage", "parameters_storage_multi_horizon", "storage_tech"
    ),
    "transport": module_spec(
        "transport",
        "parameters_transport_multi_horizon",
        "transport_tech",
        extra_dims=("link",),
    ),
    "retrofit_conversion": retrofit_spec(
        "conversion", "p_retrofit_multi_vintage"
    ),
    "retrofit_combined": retrofit_spec(
        "combined", "p_retrofit_combined_multi_vintage"
    ),
    "retrofit_storage": retrofit_spec(
        "storage", "p_retrofit_storage_multi_vintage"
    ),
    "retrofit_transport": retrofit_spec(
        "transport", "p_retrofit_transport_multi_vintage"
    ),
}

# (spec, vintage dimension of the flag) pairs with at least two vintages in the fixture
MIXED_CASES = [
    (name, vintage) for name, s in SPECS.items() for vintage in s.vintages
]


# ----------------------------------------------------------------------------------------
# Datasets and flags
# ----------------------------------------------------------------------------------------
def distinct_annuities(p):
    """Annuity tables whose cheapest entry differs from the others (so that the flag matters).

    ``100 + 1.5 x (year_dec - start)`` where ``year_dec > start`` (NaN elsewhere), halved for the
    energy part of storage: the cheapest annuity of a vintage is the one of the earliest
    decommissioning year.
    """
    q = p.copy(deep=True)
    for name in list(q.data_vars):
        a = q[name]
        if "annuity_cost" not in name or "year_dec" not in a.dims:
            continue
        start = q.year_ret if "year_ret" in a.dims else q.year_inv
        table = (100.0 + 1.5 * (q.year_dec - start)).where(q.year_dec > start)
        if "energy" in name:
            table = table / 2
        q[name] = table.broadcast_like(a).transpose(*a.dims)
    return q


def with_early_decommissioning(p, spec):
    q = p.copy(deep=True)
    q[f"{spec.host}_early_decommissioning"] = True
    if spec.flag.startswith("retrofit_"):
        q[f"retrofit_{spec.host}_early_decommissioning"] = True
    return q


def dataset(spec, request):
    p = request.getfixturevalue(spec.fixture)
    assert not bool(p.annuity_full_life)
    return distinct_annuities(with_early_decommissioning(p, spec))


def full_life_copy(p):
    """Same dataset with year_dec-free annuities (cheapest over year_dec) and annuity_full_life."""
    q = p.copy(deep=True)
    for name in list(q.data_vars):
        if "annuity_cost" in name and "year_dec" in q[name].dims:
            q[name] = q[name].min("year_dec")
    q["annuity_full_life"] = True
    return q


def flag_array(p, spec, vintage, by_vintage=None, value=False):
    """Flag of dims (flag_dims, vintage): constant `value`, or one value per vintage."""
    dims = [*spec.flag_dims, vintage]
    coords = {d: p[d].values for d in dims}
    flag = xr.DataArray(
        np.full([p.sizes[d] for d in dims], value, dtype=bool),
        dims=dims,
        coords=coords,
    )
    if by_vintage is not None:
        per_vintage = xr.DataArray(
            np.array(by_vintage, dtype=bool),
            dims=[vintage],
            coords={vintage: p[vintage].values},
        )
        flag = per_vintage.broadcast_like(flag).transpose(*dims)
    return flag


def with_flag(p, spec, flag):
    return p.assign({spec.flag: flag})


# ----------------------------------------------------------------------------------------
# The built rows, in a form that does not depend on linopy labels
# ----------------------------------------------------------------------------------------
def _plain(x):
    return x.item() if hasattr(x, "item") else x


def variable_keys(model):
    """{label: (variable name, ((dim, coordinate), ...))} of every variable element."""
    keys = {}
    for name, var in model.variables.items():
        labels = var.labels
        dims = list(labels.dims)
        indexes = [labels.indexes[d] for d in dims]
        values = labels.values
        for position in zip(*np.nonzero(values != -1)):
            key = (
                name,
                tuple(
                    (d, _plain(indexes[k][position[k]]))
                    for k, d in enumerate(dims)
                ),
            )
            keys[int(values[position])] = key
    return keys


def coefficient(x):
    return float(f"{x:.10g}")


def built_rows(model, name):
    """{row coordinates: ({variable key: coefficient}, rhs)} of the rows that exist."""
    con = model.constraints[name]
    keys = variable_keys(model)
    dims = list(con.labels.dims)
    labels = con.labels.values
    variables = con.vars.transpose(*dims, "_term").values
    coeffs = con.coeffs.transpose(*dims, "_term").values
    rhs = con.rhs.transpose(*dims).values
    indexes = [con.labels.indexes[d] for d in dims]
    rows = {}
    for position in zip(*np.nonzero(labels != -1)):
        row = tuple(
            (d, _plain(indexes[k][position[k]])) for k, d in enumerate(dims)
        )
        terms = {}
        for var, coeff in zip(variables[position], coeffs[position]):
            if var == -1 or not np.isfinite(coeff) or coeff == 0:
                continue
            terms[keys[int(var)]] = terms.get(keys[int(var)], 0.0) + coeff
        rows[row] = (
            {k: coefficient(c) for k, c in terms.items()},
            coefficient(rhs[position]),
        )
    return rows


def fingerprint(rows):
    """(number of rows, number of terms, sha256 of the sorted rows)."""
    lines = sorted(
        repr((row, sorted(terms.items()), rhs))
        for row, (terms, rhs) in rows.items()
    )
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
    return (len(rows), sum(len(t) for t, _ in rows.values()), digest)


def build(p):
    return build_model(p)


def rows_of(p, spec):
    return built_rows(build(p), spec.constraint)


_SCALAR_ROWS = {}


def scalar_rows(name, case, request):
    """Rows with a scalar flag (True or False) or full_life; built once per case (fixed datasets)."""
    if (name, case) not in _SCALAR_ROWS:
        spec = SPECS[name]
        p = dataset(spec, request)
        if case == "full_life":
            p = full_life_copy(p)
        else:
            p = p.assign({spec.flag: case == "scalar_true"})
        _SCALAR_ROWS[(name, case)] = rows_of(p, spec)
    return _SCALAR_ROWS[(name, case)]


# ----------------------------------------------------------------------------------------
# Rows that were right before the fix stay what they were (fingerprints taken at master)
# ----------------------------------------------------------------------------------------
# (rows, terms, sha256 prefix) of the planning cost rows built by master (0.4.0+kul) on the datasets above;
# full_life gives the rows of the flag-0 branch (life spans on the year_dec grid)
GOLDEN = {
    ("conversion", "scalar_false"): (2, 7, 'a2e44805da639866'),
    ("conversion", "scalar_true"): (2, 6, 'f4ecea7f6e072a13'),
    ("conversion", "full_life"): (2, 7, 'a2e44805da639866'),
    ("combined", "scalar_false"): (2, 7, '487cc7e43a96e8df'),
    ("combined", "scalar_true"): (2, 6, '1c3f04aa602b3769'),
    ("combined", "full_life"): (2, 7, '487cc7e43a96e8df'),
    ("process", "scalar_false"): (2, 7, '473777d0145f49d0'),
    ("process", "scalar_true"): (2, 6, 'a853b1531c34def0'),
    ("process", "full_life"): (2, 7, '473777d0145f49d0'),
    ("storage", "scalar_false"): (2, 12, '524e6a13d79f850b'),
    ("storage", "scalar_true"): (2, 10, 'a12de321e77f4b49'),
    ("storage", "full_life"): (2, 12, '524e6a13d79f850b'),
    ("transport", "scalar_false"): (4, 14, 'fdb2d7eebf0c10f8'),
    ("transport", "scalar_true"): (4, 12, '2972289015acc0c2'),
    ("transport", "full_life"): (4, 14, 'fdb2d7eebf0c10f8'),
    ("retrofit_conversion", "scalar_false"): (3, 9, 'b9fb29c07577b54d'),
    ("retrofit_conversion", "scalar_true"): (3, 8, '4937e306899f0d44'),
    ("retrofit_conversion", "full_life"): (3, 9, 'b9fb29c07577b54d'),
    ("retrofit_combined", "scalar_false"): (4, 12, '8a10b511c334f77b'),
    ("retrofit_combined", "scalar_true"): (4, 11, '35f18afd7bc327e8'),
    ("retrofit_combined", "full_life"): (4, 12, '8a10b511c334f77b'),
    ("retrofit_storage", "scalar_false"): (4, 20, 'e86d6850fd40c80b'),
    ("retrofit_storage", "scalar_true"): (4, 18, '1c24fbfcbd013c7a'),
    ("retrofit_storage", "full_life"): (4, 20, 'e86d6850fd40c80b'),
    ("retrofit_transport", "scalar_false"): (8, 24, 'aad71b39cd2cf776'),
    ("retrofit_transport", "scalar_true"): (8, 22, '49914d454c54374e'),
    ("retrofit_transport", "full_life"): (8, 24, 'aad71b39cd2cf776'),
}


@pytest.mark.parametrize("case", ["scalar_false", "scalar_true", "full_life"])
@pytest.mark.parametrize("name", list(SPECS))
def test_rows_of_the_branches_that_were_correct_are_unchanged(
    name, case, request
):
    """Scalar flags (no year_inv dimension to leak) and full_life: same rows as master."""
    rows = scalar_rows(name, case, request)
    assert rows, "the toy dataset must give planning cost rows"
    assert fingerprint(rows) == GOLDEN[(name, case)]


# ----------------------------------------------------------------------------------------
# a. One row per (unit, year_op): no year_inv dimension, no duplicates
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("name", list(SPECS))
def test_constant_flags_give_one_row_per_cost_element(name, value, request):
    spec = SPECS[name]
    p = dataset(spec, request)
    vintage = spec.vintages[0]
    constant = with_flag(p, spec, flag_array(p, spec, vintage, value=value))

    model = build(constant)
    con = model.constraints[spec.constraint]
    cost = model.variables[spec.costs]
    assert "year_inv" not in con.labels.dims
    assert vintage not in con.labels.dims
    assert set(con.labels.dims) == set(cost.labels.dims)
    assert int((con.labels != -1).sum()) == int((cost.labels != -1).sum())

    # the same rows as with the scalar flag, which was never duplicated
    case = "scalar_true" if value else "scalar_false"
    assert built_rows(model, spec.constraint) == scalar_rows(name, case, request)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize(
    "dims",
    [(), ("year_inv",), ("conversion_tech",), ("area", "conversion_tech")],
)
def test_a_flag_with_fewer_dimensions_is_broadcast_over_the_expression(
    dims, value, request
):
    """A scalar flag, or one without year_inv (tech only), gives the rows of the scalar flag."""
    spec = SPECS["conversion"]
    p = dataset(spec, request)
    flag = xr.DataArray(
        np.full([p.sizes[d] for d in dims], value, dtype=bool),
        dims=dims,
        coords={d: p[d].values for d in dims},
    )
    model = build(with_flag(p, spec, flag))
    assert "year_inv" not in model.constraints[spec.constraint].labels.dims
    case = "scalar_true" if value else "scalar_false"
    assert built_rows(model, spec.constraint) == scalar_rows(
        "conversion", case, request
    )


# ----------------------------------------------------------------------------------------
# b. Different flags for the vintages of one tech: each vintage keeps its own annuity
# ----------------------------------------------------------------------------------------
def alternate(n, first):
    return [(i % 2 == 0) == first for i in range(n)]


@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("name,vintage", MIXED_CASES)
def test_rows_take_each_vintage_from_its_own_flag(name, vintage, first, request):
    """PF annuity coefficient for the flagged vintages, cheapest annuity for the others."""
    spec = SPECS[name]
    p = dataset(spec, request)
    by_vintage = alternate(p.sizes[vintage], first)
    assert True in by_vintage and False in by_vintage
    mixed = with_flag(
        p, spec, flag_array(p, spec, vintage, by_vintage=by_vintage)
    )

    rows_pf = scalar_rows(name, "scalar_true", request)
    rows_cheapest = scalar_rows(name, "scalar_false", request)
    assert rows_pf != rows_cheapest, "the flag must matter on this dataset"
    flag_of = dict(zip(p[vintage].values.tolist(), by_vintage))

    expected = {}
    for row in rows_pf.keys() | rows_cheapest.keys():
        terms = {}
        sources = (rows_pf.get(row, ({}, 0.0)), rows_cheapest.get(row, ({}, 0.0)))
        for key in sources[0][0].keys() | sources[1][0].keys():
            value = dict(key[1]).get(vintage)
            source = sources[0] if value is None or flag_of[value] else sources[1]
            if key in source[0]:
                terms[key] = source[0][key]
        expected[row] = (terms, sources[0][1])
    assert expected != rows_pf and expected != rows_cheapest

    assert rows_of(mixed, spec) == expected


# ----------------------------------------------------------------------------------------
# b. The same on a model solved with HiGHS: closed-form annuities
# ----------------------------------------------------------------------------------------
YEAR_OP = [2020, 2030, 2040, 2050]
INVEST_COST, FIXED_COST, LIFE_SPAN, RATE = 1000.0, 10.0, 40.0, 0.063


def two_vintage_model(flag_2020, flag_2030):
    """Demand 1 MW in 2020, 2 MW in 2030, nothing after; 1 MW is built in 2020 and 1 MW in 2030.

    The 2020 vintage is wanted until 2040 (decommissioning year 2040), the 2030 vintage in 2030
    only (decommissioning year 2040). One hour per year, normalisation 1, no discounting.
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
    demand.loc[dict(year_op=2020)] = 1.0
    demand.loc[dict(year_op=2030)] = 2.0
    # one MW per vintage whatever the costs; only the decommissioning years are chosen
    investment_max = da(["area", "conversion_tech", "year_inv"], 0.0)
    investment_max.loc[dict(year_inv=[2020, 2030])] = 1.0
    investment_min = da(["area", "conversion_tech", "year_inv"], np.nan)
    investment_min.loc[dict(year_inv=[2020, 2030])] = 1.0
    flag = xr.DataArray(
        [flag_2020, flag_2030, False, False],
        dims=["year_inv"],
        coords=dict(year_inv=ds.year_inv),
    )
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
        conversion_life_span=da(
            ["area", "conversion_tech", "year_inv"], LIFE_SPAN
        ),
        conversion_finance_rate=xr.DataArray(RATE),
        conversion_power_capacity_investment_max=investment_max,
        conversion_power_capacity_investment_min=investment_min,
        conversion_early_decommissioning=xr.DataArray(True),
        conversion_annuity_perfect_foresight=flag,
    )
    conf = config("until_decommissioning")
    conf["pre_process"]["discount_factor"] = {
        "discount_rate": "discount_rate",
        "year_ref": "year_ref",
    }
    ds = check_inputs(
        compute_discount(
            compute_annuity_cost(compute_end_of_life(ds), conf), conf
        )
    )
    assert not bool(ds.annuity_full_life)
    model = build_model(ds)
    status, condition = model.solve(solver_name="highs", output_flag=False)
    assert condition == "optimal"
    s = model.solution

    def by_year(x):
        return x.sum([d for d in x.dims if d != "year_op"]).values

    return dict(
        model=model,
        costs=by_year(s.planning_conversion_costs),
        capacity=by_year(s.operation_conversion_power_capacity),
        objective=model.objective.value,
    )


def annuity(years):
    """Closed-form annual annuity of 1 MW over `years` years."""
    return INVEST_COST * crf_closed_form(RATE, years)


def test_two_vintages_with_different_flags_cost_the_pf_and_the_cheapest_annuity():
    a10, a20, a40 = annuity(10), annuity(20), annuity(40)

    # 2020 vintage perfect foresight: crf over 20 years (decommissioned in 2040), paid in 2020
    # and 2030; 2030 vintage cheapest: crf over its 40 years, paid until its end of life (2070)
    first = two_vintage_model(True, False)
    np.testing.assert_allclose(first["capacity"], [1, 2, 0, 0], atol=1e-9)
    np.testing.assert_allclose(
        first["costs"], [a20, a20 + a40, a40, a40], rtol=1e-9
    )

    # the other way round: 2020 vintage cheapest (crf over 40 years until 2060),
    # 2030 vintage perfect foresight (crf over 10 years, paid in 2030 only)
    second = two_vintage_model(False, True)
    np.testing.assert_allclose(second["capacity"], [1, 2, 0, 0], atol=1e-9)
    np.testing.assert_allclose(
        second["costs"], [a40, a40 + a10, a40, a40], rtol=1e-9
    )

    # constant flags are unchanged: the sums of the two pure choices
    both_pf = two_vintage_model(True, True)
    np.testing.assert_allclose(
        both_pf["costs"], [a20, a20 + a10, 0, 0], rtol=1e-9
    )
    neither = two_vintage_model(False, False)
    np.testing.assert_allclose(
        neither["costs"], [a40, 2 * a40, 2 * a40, 2 * a40], rtol=1e-9
    )

    # objective = fixed costs of the capacity + planning costs, in every case
    for r in (first, second, both_pf, neither):
        assert r["objective"] == pytest.approx(
            FIXED_COST * r["capacity"].sum() + r["costs"].sum(), rel=1e-9
        )
