"""Carbon tax on groups of areas (``carbon_tax_group``).

By default pommes charges ``max(0, carbon_tax x net emissions)`` per area. With the ``carbon_tax_group`` dimension, a group
of areas pays ``max(0, rate x sum of the net emissions of its members)``: a removal in one member offsets the emissions of
the others, and a group with negative net emissions pays nothing (a removal is never paid as such). The per-area tax is
masked for the member areas of a group that has a rate, so nothing is charged twice.

The toy has two areas, 10 steps of 1 h, one year, a tax of 100 per t:

* ``area_1`` serves its electricity demand with ``ocgt`` (0.5 t per MWh): ``demand_1`` MWh per step gives 5 x ``demand_1``
  t over the 10 steps (2 MWh per step: 10 t);
* ``area_2`` has a must-run ``wind_onshore`` of fixed capacity, availability 0.5, emission factor -0.4 t per MWh: it
  stands for a removal unit (2 MW: 1 MW x 10 steps x -0.4 = -4 t; 5 MW: -10 t). It has no demand, the energy is spilled.

Each case is compared with the same toy without any tax, so the differences below are the carbon costs alone.
"""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model

TAX = 100.0
AREAS = ["area_1", "area_2"]
TECHS = ["ocgt", "wind_onshore"]


def _toy(base, carbon, demand_1=2.0, wind_mw=2.0, tax=TAX, discount=1.0):
    """The two-area toy (see the module docstring); per-area ``carbon_tax`` ``tax`` in both areas (NaN: none)."""
    p = (
        xr.merge([base, carbon])
        .sel(
            area=AREAS,
            conversion_tech=TECHS,
            resource=["electricity"],
            year_dec=[2040],
            year_inv=[2020],
            year_op=[2020],
        )
        .copy(deep=True)
    )
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    p["carbon_goal"] = np.nan
    p["discount_factor"] = xr.DataArray([discount], coords=[p.year_op])
    p["carbon_tax"] = xr.DataArray(
        [[tax], [tax]], coords={"area": p.area, "year_op": p.year_op}
    )

    def by_area_tech(a1, a2):
        return xr.DataArray(
            [a1, a2], coords={"area": p.area, "conversion_tech": p.conversion_tech}
        )

    # area_1 can only build ocgt, area_2 only the wind farm, at the capacity given
    p["conversion_power_capacity_max"] = by_area_tech([1000.0, 0.0], [0.0, 1000.0])
    p["conversion_power_capacity_investment_max"] = by_area_tech([1000.0, 0.0], [0.0, wind_mw])
    p["conversion_power_capacity_investment_min"] = by_area_tech([0.0, 0.0], [0.0, wind_mw])
    p["conversion_emission_factor"] = xr.DataArray([0.5, -0.4], coords={"conversion_tech": p.conversion_tech})
    p["conversion_availability"].loc[dict(conversion_tech="wind_onshore")] = 0.5
    demand = np.zeros((2, len(p.hour), 1, 1))
    demand[0] = demand_1
    p["demand"] = xr.DataArray(
        demand,
        coords={"area": p.area, "hour": p.hour, "resource": p.resource, "year_op": p.year_op},
    )
    return p


def _groups(p, members, rate):
    """Add carbon tax groups: ``members`` {group: [areas]}, ``rate`` {group: rate in the year (NaN: none)}."""
    names = list(members)
    p = p.assign_coords(carbon_tax_group=xr.DataArray(names, dims="carbon_tax_group"))
    p["carbon_tax_group_areas"] = xr.DataArray(
        [[a in members[g] for g in names] for a in p.area.values],
        coords=[p.area, p.carbon_tax_group],
    )
    p["carbon_tax_group_rate"] = xr.DataArray(
        [[rate[g]] for g in names], coords=[p.carbon_tax_group, p.year_op]
    )
    return p


def _solve(p):
    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.termination_condition == "optimal"
    return model, model.solution.squeeze()


def _net_emissions(s):
    """Net emissions [t] of area_1 and area_2 over the 10 steps."""
    return tuple(s.operation_carbon_emissions.sel(area=a).sum().item() for a in AREAS)


def _carbon_cost(base, carbon, build, **toy):
    """Objective of the toy with the taxes of ``build(p)`` minus the objective of the same toy without any tax."""
    free = _solve(_toy(base, carbon, tax=np.nan, **toy))[0].objective.value
    model, s = _solve(build(_toy(base, carbon, **toy)))
    return model.objective.value - free, model, s


def test_group_nets_the_removal_against_the_emissions(parameters_dispatch_invest, carbon):
    """area_1 emits 10 t, area_2 removes 4 t, both in one group: the group pays rate x 6, not rate x 10."""
    both = lambda p: _groups(p, {"g": AREAS}, {"g": TAX})
    cost, model, s = _carbon_cost(parameters_dispatch_invest, carbon, both)
    assert _net_emissions(s) == pytest.approx((10.0, -4.0), abs=1e-6)
    assert s.operation_carbon_tax_group_costs.item() == pytest.approx(TAX * 6, rel=1e-6)
    assert cost == pytest.approx(TAX * 6, rel=1e-6)
    # the per-area tax of the members is masked: nothing is charged twice
    assert s.operation_carbon_costs.values == pytest.approx(0.0, abs=1e-6)

    # the same toy with per-area taxes only: area_1 pays rate x 10, area_2 is not paid for its removal
    per_area, _, s0 = _carbon_cost(parameters_dispatch_invest, carbon, lambda p: p)
    assert per_area == pytest.approx(TAX * 10, rel=1e-6)
    assert s0.operation_carbon_costs.sel(area="area_1").item() == pytest.approx(TAX * 10, rel=1e-6)
    assert s0.operation_carbon_costs.sel(area="area_2").item() == pytest.approx(0.0, abs=1e-6)


def test_group_with_negative_net_emissions_pays_nothing(parameters_dispatch_invest, carbon):
    """Net -5 t (+5 t of area_1, -10 t of area_2): the group pays 0 and earns nothing for the removal."""
    both = lambda p: _groups(p, {"g": AREAS}, {"g": TAX})
    cost, model, s = _carbon_cost(parameters_dispatch_invest, carbon, both, demand_1=1.0, wind_mw=5.0)
    assert _net_emissions(s) == pytest.approx((5.0, -10.0), abs=1e-6)
    assert s.operation_carbon_tax_group_costs.item() == pytest.approx(0.0, abs=1e-6)
    assert cost == pytest.approx(0.0, abs=1e-6)  # not negative: the removals are not paid
    # with per-area taxes area_1 would pay rate x 5
    per_area, _, _ = _carbon_cost(parameters_dispatch_invest, carbon, lambda p: p, demand_1=1.0, wind_mw=5.0)
    assert per_area == pytest.approx(TAX * 5, rel=1e-6)


def test_mixed_removal_group_does_not_offset_an_area_outside_it(parameters_dispatch_invest, carbon):
    """Group {area_2} (net -4 t) pays 0; area_1, outside, pays its own tax on 10 t."""
    only_2 = lambda p: _groups(p, {"g": ["area_2"]}, {"g": TAX})
    cost, model, s = _carbon_cost(parameters_dispatch_invest, carbon, only_2)
    assert s.operation_carbon_tax_group_costs.item() == pytest.approx(0.0, abs=1e-6)
    assert s.operation_carbon_costs.sel(area="area_1").item() == pytest.approx(TAX * 10, rel=1e-6)
    assert cost == pytest.approx(TAX * 10, rel=1e-6)


def test_mixed_member_area_pays_through_its_group_only(parameters_dispatch_invest, carbon):
    """Group {area_1} at rate 50 (area_1 emits 10 t) with a per-area tax of 100 on both areas: the group pays 500, the
    per-area tax of area_1 is masked, area_2 (outside, net -4 t) pays 0."""
    only_1 = lambda p: _groups(p, {"g": ["area_1"]}, {"g": 50.0})
    cost, model, s = _carbon_cost(parameters_dispatch_invest, carbon, only_1)
    assert s.operation_carbon_tax_group_costs.item() == pytest.approx(500.0, rel=1e-6)
    assert s.operation_carbon_costs.sel(area="area_1").item() == pytest.approx(0.0, abs=1e-6)
    assert s.operation_carbon_costs.sel(area="area_2").item() == pytest.approx(0.0, abs=1e-6)
    assert cost == pytest.approx(500.0, rel=1e-6)


def test_group_rate_0_is_a_tax_free_group(parameters_dispatch_invest, carbon):
    """A rate of 0 is a rate: the members do not pay the per-area tax either."""
    both = lambda p: _groups(p, {"g": AREAS}, {"g": 0.0})
    cost, model, s = _carbon_cost(parameters_dispatch_invest, carbon, both)
    assert cost == pytest.approx(0.0, abs=1e-6)


def test_group_rate_nan_leaves_the_per_area_tax(parameters_dispatch_invest, carbon):
    """No group rate (NaN) in the year: no group tax, the members keep their own carbon_tax."""
    both = lambda p: _groups(p, {"g": AREAS}, {"g": np.nan})
    cost, model, s = _carbon_cost(parameters_dispatch_invest, carbon, both)
    assert s.operation_carbon_tax_group_costs.item() == pytest.approx(0.0, abs=1e-6)
    assert cost == pytest.approx(TAX * 10, rel=1e-6)


def test_group_costs_are_discounted_like_the_area_costs(parameters_dispatch_invest, carbon):
    """With a discount factor of 0.5 the group cost weighs 0.5 x 600 in the objective, as the per-area cost 0.5 x 1000."""
    both = lambda p: _groups(p, {"g": AREAS}, {"g": TAX})
    cost, _, _ = _carbon_cost(parameters_dispatch_invest, carbon, both, discount=0.5)
    assert cost == pytest.approx(0.5 * TAX * 6, rel=1e-6)
    per_area, _, _ = _carbon_cost(parameters_dispatch_invest, carbon, lambda p: p, discount=0.5)
    assert per_area == pytest.approx(0.5 * TAX * 10, rel=1e-6)


def test_without_groups_the_model_is_the_per_area_model(parameters_dispatch_invest, carbon):
    """No carbon_tax_group dimension: no group variable and no group constraint; a group nobody belongs to changes nothing
    but the (zero) group variable."""
    p = _toy(parameters_dispatch_invest, carbon)
    model, s = _solve(p)
    assert "carbon_tax_group" not in model.parameters.dims
    assert "operation_carbon_tax_group_costs" not in model.variables
    assert "operation_carbon_tax_group_costs_def" not in model.constraints

    empty = _groups(p, {"g": []}, {"g": TAX})
    model_g, s_g = _solve(empty)
    assert model_g.objective.value == pytest.approx(model.objective.value, rel=1e-9)
    assert s_g.operation_carbon_tax_group_costs.item() == pytest.approx(0.0, abs=1e-6)
    assert set(model_g.variables) - set(model.variables) == {"operation_carbon_tax_group_costs"}
    assert set(model_g.constraints) - set(model.constraints) == {"operation_carbon_tax_group_costs_def"}
