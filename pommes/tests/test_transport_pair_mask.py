"""The transport variables exist only for the (link, tech) pairs that carry a resource.

A study with one transport tech per resource has a ``transport_resource(link, tech)`` that is a resource of the model on
a few links of each tech only; the other pairs carry nothing and would be 0 in any solution. ``add_transport`` leaves
them out: the flows, the capacities and the rows that define or bound them.

1. Where every pair carries a resource, the linear program is the very same as before the mask (rows, columns and
   coefficients, compared with a frozen description built by the code before the mask).
2. Where some pairs carry nothing, the variables and rows exist for the other pairs only, and the optimum is the one of
   the same data where those pairs carry a resource but cannot be built.
3. The masked pairs come back as NaN in the solution: the readers inside pommes (summaries, capacities, CSV files) do
   not let them leak into totals.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pommes.io.calculate_summaries import calculate_total_of_a_type
from pommes.io.save_solution import save_solution
from pommes.model.build_model import build_model
from pommes.post_process import get_capacities

TRANSPORT_VARIABLES = [
    "operation_transport_power_capacity",
    "operation_transport_power",
    "operation_transport_power_in",
    "planning_transport_power_capacity",
]

# The resource of each tech of the fixtures
RESOURCE_OF = {"big_methane_pipe": "methane", "methane_pipe": "methane", "power_line": "electricity"}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@pytest.fixture()
def p_base(parameters_dispatch_invest, transport):
    """Two areas, two links (one each way), three transport techs, three hours: wind in both areas, electricity and
    methane."""
    p = xr.merge([parameters_dispatch_invest, transport])
    # hurdle costs are weighted like the other variable costs: see test_transport.py
    p["transport_hurdle_costs"] = p.transport_hurdle_costs / 8760
    p = p.sel(
        conversion_tech=["wind_onshore"],
        hour=[0, 1, 2],
        resource=["electricity", "methane"],
        year_dec=[2040, 2050],
        year_inv=[2030],
        year_op=[2030],
    ).copy(deep=True)
    p.update(dict(conversion_availability=p.conversion_availability.expand_dims(area=p.area).copy()))
    p.conversion_availability.loc[dict(conversion_tech="wind_onshore")] = [[0.6, 0.3, 0.1], [0.4, 0.7, 0.9]]
    return p


def _link_tech(p, fill=np.nan):
    return xr.DataArray(
        np.full((p.sizes["link"], p.sizes["transport_tech"]), fill),
        coords=dict(link=p.link, transport_tech=p.transport_tech),
    )


def _link_tech_year(p, fill=np.nan):
    return xr.DataArray(
        np.full((p.sizes["link"], p.sizes["transport_tech"], p.sizes["year_op"]), fill),
        coords=dict(link=p.link, transport_tech=p.transport_tech, year_op=p.year_op),
    )


def _with_resources(p, exists):
    """``exists``: boolean (link, tech). The pairs that do not exist have no resource, and cannot be built."""
    resource = _link_tech(p, "").astype(object)
    for tech in p.transport_tech.values:
        resource.loc[dict(transport_tech=tech)] = RESOURCE_OF[tech]
    resource = resource.where(exists, np.nan)
    p = p.copy(deep=True)
    p["transport_resource"] = resource.astype(str)
    return p


def _blocked_unless(p, exists):
    """No investment on the pairs that are not in ``exists`` (whatever their resource)."""
    p = p.copy(deep=True)
    p["transport_power_capacity_investment_max"] = xr.where(exists, np.nan, 0.0)
    return p


def _every_kind_of_row(p):
    """Ramps, capacity bounds, an availability, investment bounds, delays and a linepack on some pairs: every
    constraint of the module."""
    p = p.copy(deep=True)
    ramp_up = _link_tech_year(p)
    ramp_up[0, :, :] = 0.5
    p["transport_ramp_up"] = ramp_up
    ramp_down = _link_tech_year(p)
    ramp_down[1, 0, :] = 0.7
    p["transport_ramp_down"] = ramp_down
    cap_max = _link_tech_year(p)
    cap_max[1, :, :] = 5000.0
    p["transport_power_capacity_max"] = cap_max
    cap_min = _link_tech_year(p)
    cap_min[0, 1, :] = 0.0
    p["transport_power_capacity_min"] = cap_min
    availability = xr.DataArray(
        np.full((p.sizes["link"], p.sizes["transport_tech"], p.sizes["hour"], p.sizes["year_op"]), np.nan),
        coords=dict(link=p.link, transport_tech=p.transport_tech, hour=p.hour, year_op=p.year_op),
    )
    availability[0, 0, :, :] = 0.8
    p["transport_availability"] = availability
    invest_min, invest_max = _link_tech(p), _link_tech(p)
    invest_min[0, 0], invest_max[0, 0] = 1.0, 10.0  # a lower and an upper bound
    invest_min[1, 0], invest_max[1, 0] = 2.0, 2.0  # pinned
    p["transport_power_capacity_investment_min"] = invest_min
    p["transport_power_capacity_investment_max"] = invest_max
    # delays of 1 and 2 hours on one pair each, a linepack on two others, and pairs with neither (the rows of the
    # plain injection definition)
    delay = _link_tech(p, 0.0)
    delay[0, 1] = 1.0
    delay[1, 0] = 2.0
    p["transport_delay"] = delay
    linepack = _link_tech_year(p)
    linepack[0, 0, :] = 5.0
    linepack[1, 1, :] = 12.0
    p["transport_linepack_capacity"] = linepack
    return p


def _exists(p):
    return p.transport_resource.broadcast_like(_link_tech(p)).isin(p.resource.values)


def _all_exist(p):
    return xr.DataArray(
        np.ones((p.sizes["link"], p.sizes["transport_tech"]), dtype=bool),
        coords=dict(link=p.link, transport_tech=p.transport_tech),
    )


def _some_exist(p):
    """The two methane pipes on the first link only, the power line on both."""
    exists = _all_exist(p)
    exists.loc[dict(link=p.link.values[1], transport_tech=["big_methane_pipe", "methane_pipe"])] = False
    return exists


# ---------------------------------------------------------------------------
# 1. Every pair carries a resource: the linear program is the one of the code before the mask
# ---------------------------------------------------------------------------


def _digest(obj):
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:12]


def _transport_description(model):
    """The transport part of the linear program, independent of the labels the other modules took before it.

    For every transport variable: the positions of its active columns with their bounds. For every constraint: the
    rows with transport terms, as (row position, the transport terms as (variable, position, coefficient), sign,
    right-hand side). Each is reduced to a count and a hash.
    """
    names = [name for name in model.variables if "transport" in name]
    position = {}
    description = {}
    for name in model.variables:
        labels = model.variables[name].labels.values.ravel()
        for pos, label in enumerate(labels):
            if label >= 0:
                position[int(label)] = (name, pos)
        if name in names:
            variable = model.variables[name]
            active = [
                (pos, float(lo), float(up))
                for pos, (label, lo, up) in enumerate(
                    zip(labels, variable.lower.values.ravel(), variable.upper.values.ravel())
                )
                if label >= 0
            ]
            description[name] = (len(active), _digest(active))
    for name, constraint in model.constraints.items():
        row_position = {int(label): pos for pos, label in enumerate(constraint.labels.values.ravel()) if label >= 0}
        flat = constraint.flat
        flat = flat[(flat.labels >= 0) & (flat.vars >= 0)]
        rows = {}
        for label, vars_, coeffs, sign, rhs in zip(flat.labels, flat.vars, flat.coeffs, flat.sign, flat.rhs):
            variable_name, variable_position = position[int(vars_)]
            if variable_name not in names:
                continue
            row = rows.setdefault(row_position[int(label)], [[], sign, round(float(rhs), 9)])
            row[0].append((variable_name, variable_position, round(float(coeffs), 9)))
        if rows:
            ordered = sorted((pos, sorted(terms), sign, rhs) for pos, (terms, sign, rhs) in rows.items())
            description[name] = (len(rows), _digest(ordered))
    return description


# Built by ``_transport_description`` on the dataset below with the code of ``add_transport`` before the mask
# (pommes 0.4.1+kul, head 78b56dc), where every pair carries a resource.
FROZEN = {
    "annualised_totex_def": (2, "8ceb98ad043c"),
    "operation_adequacy_constraint": (12, "f9399d72d858"),
    "operation_transport_costs": (6, "ec5a43c2d588"),
    "operation_transport_costs_def": (6, "7027635e28bf"),
    "operation_transport_linepack_level": (6, "eb03ae9a1880"),
    "operation_transport_linepack_level_def": (6, "b76786c1936a"),
    "operation_transport_linepack_level_max_constraint": (6, "d7642aed7575"),
    "operation_transport_net_generation": (18, "f1933667585e"),
    "operation_transport_net_generation_def": (18, "979f976876d7"),
    "operation_transport_power": (18, "8632a168ae63"),
    "operation_transport_power_capacity": (6, "3be97e83d385"),
    "operation_transport_power_capacity_def": (6, "6874d4cc058f"),
    "operation_transport_power_capacity_max_constraint": (3, "758635bf96af"),
    "operation_transport_power_capacity_min_constraint": (1, "8fb726dafa2e"),
    "operation_transport_power_in": (18, "8632a168ae63"),
    "operation_transport_power_in_def": (6, "2acf8420aa1e"),
    "operation_transport_power_in_def_delay_1": (3, "fb24ce7443b9"),
    "operation_transport_power_in_def_delay_2": (3, "3c72ce4fdfcf"),
    "operation_transport_power_in_max_constraint": (6, "26255d70c51f"),
    "operation_transport_power_max_constraint": (18, "2808f6b95cd1"),
    "operation_transport_ramp_down_constraint": (2, "39db8ed4d9ae"),
    "operation_transport_ramp_up_constraint": (6, "7da38c7fd908"),
    "planning_transport_costs": (6, "3be97e83d385"),
    "planning_transport_costs_def": (6, "9140747682bb"),
    "planning_transport_power_capacity": (6, "f578f89e42be"),
    "planning_transport_power_capacity_def": (1, "558dba4692de"),
    "planning_transport_power_capacity_max_constraint": (1, "d4efc08e2fed"),
    "planning_transport_power_capacity_min_constraint": (1, "017bb5c3dadb"),
}


def test_all_pairs_carrying_a_resource_build_the_lp_of_the_code_before_the_mask(p_base):
    p = _every_kind_of_row(_with_resources(p_base, _all_exist(p_base)))
    assert bool(_exists(p).all())
    assert _transport_description(build_model(p)) == FROZEN


def test_all_pairs_carrying_a_resource_have_every_variable(p_base):
    p = _every_kind_of_row(_with_resources(p_base, _all_exist(p_base)))
    model = build_model(p)
    for name in TRANSPORT_VARIABLES[:3]:
        assert (model.variables[name].labels.values >= 0).all(), name
    planned = model.variables["planning_transport_power_capacity"].labels >= 0
    assert bool(planned.any(["year_dec", "year_inv"]).all())


# ---------------------------------------------------------------------------
# 2. Some pairs carry nothing
# ---------------------------------------------------------------------------


def test_variables_and_rows_exist_for_the_pairs_that_carry_a_resource_only(p_base):
    p = _every_kind_of_row(_with_resources(p_base, _some_exist(p_base)))
    exists = _exists(p)
    assert not bool(exists.all()) and bool(exists.any())
    model = build_model(p)

    def live(name):
        container = model.variables if name in model.variables else model.constraints
        return container[name].labels >= 0

    for name in [
        "operation_transport_power_capacity",
        "operation_transport_power",
        "operation_transport_power_in",
        "planning_transport_power_capacity",
        "operation_transport_power_max_constraint",
        "operation_transport_power_capacity_def",
    ]:
        assert not bool(live(name).where(~exists, False).any()), name
        assert bool(live(name).where(exists, False).any()), name
    for name in ("operation_transport_power_capacity", "operation_transport_power", "operation_transport_power_in"):
        assert bool((live(name) == exists).all()), name
    assert bool((live("operation_transport_power_capacity_def") == exists).all())

    unmasked = build_model(_every_kind_of_row(_with_resources(p_base, _all_exist(p_base))))
    assert model.nvars < unmasked.nvars
    assert model.ncons < unmasked.ncons


@pytest.fixture()
def solved(p_base):
    """The same data twice: where two pairs carry nothing (``masked``), and where they carry a resource but cannot be
    built (``blocked``)."""
    exists = _some_exist(p_base)
    models = {}
    for key, p in (
        ("masked", _blocked_unless(_with_resources(p_base, exists), exists)),
        ("blocked", _blocked_unless(_with_resources(p_base, _all_exist(p_base)), exists)),
    ):
        model = build_model(p)
        assert model.solve(solver_name="highs")[0] == "ok"
        models[key] = (p, model)
    return exists, models["masked"], models["blocked"]


def test_masked_model_has_the_optimum_of_the_one_with_blocked_pairs(solved):
    exists, (_, masked), (_, blocked) = solved

    assert masked.objective.value == pytest.approx(blocked.objective.value, rel=1e-9)
    assert masked.objective.value > 0
    s_blocked, s_masked = blocked.solution, masked.solution
    for name in s_blocked.data_vars:
        a, b = s_blocked[name], s_masked[name]
        if name in TRANSPORT_VARIABLES:
            _check_nan_where_the_pair_does_not_exist(b, name, exists)
            np.testing.assert_allclose(
                b.where(exists).fillna(0).values, a.where(exists).fillna(0).values, atol=1e-6, err_msg=name
            )
        else:
            np.testing.assert_allclose(np.nan_to_num(b.values), np.nan_to_num(a.values), atol=1e-6, err_msg=name)
    # the price of electricity (the one resource with a demand and a unique dual solution)
    np.testing.assert_allclose(
        masked.dual["operation_adequacy_constraint"].sel(resource="electricity").values,
        blocked.dual["operation_adequacy_constraint"].sel(resource="electricity").values,
        atol=1e-6,
    )
    # something is carried by the power line on both links
    assert float(s_masked.operation_transport_power.sum()) > 1


# ---------------------------------------------------------------------------
# 3. The readers of the solution skip the masked pairs
# ---------------------------------------------------------------------------


def _check_nan_where_the_pair_does_not_exist(value, name, exists):
    """NaN on every entry of a pair that carries nothing; no NaN on the pairs that carry one (for the planned
    capacity, which has masked vintages of its own: some value on each pair)."""
    assert bool(value.where(~exists).isnull().all()), name
    if name == "planning_transport_power_capacity":
        assert bool((value.notnull().any(["year_dec", "year_inv"]) == exists).all()), name
    else:
        assert not bool(value.where(exists, 0.0).isnull().any()), name


def test_the_masked_pairs_are_nan_in_the_solution(solved):
    exists, (_, masked), _ = solved
    for name in TRANSPORT_VARIABLES:
        _check_nan_where_the_pair_does_not_exist(masked.solution[name], name, exists)


def _frames_equal(a, b):
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index(), check_exact=False, atol=1e-6)


def test_summaries_do_not_leak_nan_into_totals(solved):
    exists, (p, masked), (_, blocked) = solved
    for by_year_op in (True, False):
        for kind, operation_or_planning in (("power_capacity", "operation"), ("costs", "operation"), ("costs", "planning")):
            a = calculate_total_of_a_type(kind, masked.solution, p, operation_or_planning, by_year_op=by_year_op)
            b = calculate_total_of_a_type(kind, blocked.solution, p, operation_or_planning, by_year_op=by_year_op)
            assert a is not None and len(a) > 0
            _frames_equal(a, b)
            if kind == "power_capacity":
                rows = a[a.index.get_level_values("type") == "transport"]
                assert len(rows) > 0
                assert not rows[kind].isna().any()
                assert rows[kind].sum() > 0


def test_get_capacities_skips_the_masked_pairs(solved):
    exists, (p, masked), (_, blocked) = solved
    both = _link_tech(p, 0.0)
    p = p.assign(
        transport_area_from=p.transport_area_from.broadcast_like(both).transpose("link", "transport_tech"),
        transport_area_to=p.transport_area_to.broadcast_like(both).transpose("link", "transport_tech"),
    )
    a = get_capacities(masked.solution, p)
    b = get_capacities(blocked.solution, p)
    transport_rows = a.sel(asset=[x for x in a.asset.values if str(x[0]).startswith("transport")])
    assert not bool(transport_rows.isnull().any())
    assert float(transport_rows.sum()) > 0
    np.testing.assert_allclose(
        np.nan_to_num(a.transpose(*b.dims).values), np.nan_to_num(b.values), atol=1e-6
    )


def test_saved_files_have_no_row_for_the_masked_pairs(solved, tmp_path):
    exists, (p, masked), (_, blocked) = solved
    save_solution(masked, str(tmp_path / "masked"), export_csv=True, save_input=False)
    save_solution(blocked, str(tmp_path / "blocked"), export_csv=True, save_input=False)

    for name in TRANSPORT_VARIABLES:
        a = pd.read_csv(tmp_path / "masked" / "variables" / f"{name}.csv", sep=";")
        b = pd.read_csv(tmp_path / "blocked" / "variables" / f"{name}.csv", sep=";")
        assert not a.isna().any().any(), name
        assert len(a) < len(b), name
        assert a[name].sum() == pytest.approx(b[name].sum(), abs=1e-6), name
        assert len(a[["link", "transport_tech"]].drop_duplicates()) <= int(exists.sum()), name
