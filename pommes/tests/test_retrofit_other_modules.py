"""Retrofit behaviour on the combined, process and storage modules.

Combined and process are area-sited, single-capacity-kind adapters
built the same way as the conversion adapter (see
test_retrofit_conversion.py), so the tests for them are deliberately
narrower: they check that the generic block wires up correctly for
each host, plus the one genuine wrinkle process has that conversion
does not -- a unit size that quantises operational capacity to whole
units.

The shared ``coords`` fixture gives ``combined_tech`` and
``process_tech`` exactly one technology each, so every value test for
those two uses a SELF-pair (a technology retrofitted into itself)
rather than a genuine A -> B transfer. Each test says explicitly what
that does and does not prove; the conversion tests already cover
genuine two-technology economics (e.g. retrofit undercutting an
overpriced direct build), which a self-pair cannot exercise since
source and target share identical costs.

Storage is different: it is the one host with two capacity kinds
(power and energy), each converting at its own ratio, and ``coords``
gives ``storage_tech`` three technologies, so its tests below use a
genuine A -> B pair (tank_methane -> tank_hydrogen) and focus on
proving the two kinds transfer independently.

Transport is different again: it plans on ``link`` rather than
``area``, so its transfer variable is link-indexed while its cost
must still reach ``area`` to join the objective. ``coords`` gives
``transport_tech`` three technologies, so the tests below use a
genuine A -> B pair (methane_pipe -> big_methane_pipe) and focus on
proving that a link's retrofit cost is split evenly across the two
areas it connects, not attributed wholly to one.
"""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model


@pytest.fixture()
def p_retrofit_combined(parameters_retrofit_combined):
    # year_inv includes 2030 solely so year_ret (2030, 2040) is a
    # subset of it, as chaining requires (see _check_year_ret_grid).
    return parameters_retrofit_combined.sel(
        area=["area_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030],
        year_ret=[2030, 2040],
    ).copy(deep=True)


def test_retrofit_combined_declares_variables(p_retrofit_combined):
    """
    Structural only: proves the adapter builds under ``build_model`` and
    the transfer variable carries the dimensions the generic block
    promises. Does not exercise the mask or any solved value -- see
    ``test_retrofit_combined_self_pair_capacity_uprate`` for that.
    """
    model = build_model(p_retrofit_combined)

    var = model.variables["planning_retrofit_combined_power_capacity"]
    assert set(var.dims) == {
        "area",
        "retrofit_combined_pair",
        "year_inv",
        "year_ret",
        "year_dec",
    }
    assert "planning_retrofit_combined_costs" in model.variables


@pytest.fixture()
def p_retrofit_combined_value(parameters_retrofit_combined):
    """
    A self-pair setup built to exercise the retrofit mechanics with
    real solved numbers, not just declared dimensions.

    ``electric_boiler`` is the only technology ``coords`` gives
    ``combined_tech``, so "retrofit" here means the same technology
    retrofitted into itself. Read as an uprate/repowering: an existing
    asset is decommissioned early and comes back with a different
    (here, larger) rating, via ``retrofit_combined_ret_ratio=2.0``.

    Construction:
    - Investment is pinned to exactly 3.0 at ``year_inv=2020`` via
      equal min/max bounds, and to 0 at ``year_inv=2030`` (present
      only so ``year_ret=2030`` is a subset of ``year_inv``, as
      chaining requires).
    - ``year_dec`` is narrowed to ``{2030, 2050}``, excluding the
      technology's natural end of life at ``year_inv=2020``
      (``2020 + life_span 20 = 2040``). With
      ``combined_early_decommissioning=True`` this leaves ``2030`` as
      the *only* valid decommissioning year for that vintage -- early
      decommissioning is forced by the grid, not chosen for cost
      reasons.
    - The retrofit's own investment is pinned to exactly 3.0 too, which
      the source-cap constraint allows exactly: ``available`` at
      ``year_ret=2030`` is the 3.0 directly invested and decommissioned
      there, with ``eligible_share=1.0``.
    - Demand is zeroed throughout. The point of this fixture is the
      *capacity* variable, which the retrofit block ties to
      ``operation_combined_power_capacity_def`` by an equality -- it is
      forced regardless of dispatch, so no demand is needed to observe
      it.

    This proves the mask, both gather directions (leaving gathered onto
    the source, arriving onto the target -- here the same technology),
    the source cap, and the ratio's arithmetic all compose correctly
    end to end. It does NOT prove a genuine A -> B transfer is ever
    cheaper than building directly -- source and target share identical
    costs here, so that comparison is meaningless for a self-pair. That
    dimension is already covered by
    test_retrofit_conversion.py's smr -> smr_ccs tests.
    """
    p = parameters_retrofit_combined.sel(
        area=["area_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030, 2040],
        year_ret=[2030],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["combined_early_decommissioning"] = np.array(True, dtype="bool")

    p["combined_power_capacity_investment_max"] = (
        xr.full_like(p.combined_power_capacity_investment_max, 3.0)
        * xr.ones_like(p.year_inv, dtype="float64")
    ).copy()
    p["combined_power_capacity_investment_max"].loc[
        dict(year_inv=2030)
    ] = 0.0
    p["combined_power_capacity_investment_min"] = p[
        "combined_power_capacity_investment_max"
    ].copy()

    p["retrofit_combined_ret_ratio"] = xr.full_like(
        p.retrofit_combined_ret_ratio, 2.0
    )
    p["retrofit_combined_power_capacity_investment_min"] = xr.full_like(
        p.retrofit_combined_power_capacity_investment_min, 3.0
    )
    p["retrofit_combined_power_capacity_investment_max"] = xr.full_like(
        p.retrofit_combined_power_capacity_investment_max, 3.0
    )
    return p


def test_retrofit_combined_self_pair_capacity_uprate(
    p_retrofit_combined_value,
):
    """
    See ``p_retrofit_combined_value`` for the full construction.

    Before the retrofit fires (``year_op=2020``), operational capacity
    is the 3.0 directly invested. From ``year_ret=2030`` onward
    (``year_op=2040`` here), the direct vintage has decommissioned
    (its only valid ``year_dec`` is 2030) and is replaced by the
    retrofit's arriving contribution: ``ret_ratio (2.0) * transfer
    (3.0, pinned) = 6.0``.
    """
    model = build_model(p_retrofit_combined_value)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    retrofitted = float(s.planning_retrofit_combined_power_capacity.sum())
    assert retrofitted == pytest.approx(3.0, abs=1e-6)  # non-vacuous

    capacity = (s.operation_combined_power_capacity).sel(
        area="area_1", combined_tech="electric_boiler"
    )
    assert float(capacity.sel(year_op=2020)) == pytest.approx(
        3.0, abs=1e-6
    )
    assert float(capacity.sel(year_op=2040)) == pytest.approx(
        6.0, abs=1e-6
    )


@pytest.fixture()
def p_retrofit_combined_multi_vintage(parameters_retrofit_combined):
    """
    Three live transfer cells, so a bound keyed by ``year_ret`` alone can
    tell a per-vintage implementation apart from an aggregated one.

    Coordinate slice: ``year_inv=[2020, 2030, 2040]``,
    ``year_dec=[2030, 2040, 2050, 2060]``,
    ``year_op=[2020, 2030, 2040, 2050]``, ``year_ret=[2030, 2040]``.
    With ``combined_early_decommissioning=True``, host life span 20 and
    ``retrofit_combined_early_decommissioning`` left at its fixture
    default (``False``, so a transfer's ``year_dec`` is pinned by exact
    equality to ``year_ret + 20``), the transfer mask leaves exactly
    three cells alive:

    - ``a = (year_inv=2020, year_ret=2030, year_dec=2050)``
    - ``b = (year_inv=2020, year_ret=2040, year_dec=2060)``
    - ``c = (year_inv=2030, year_ret=2040, year_dec=2060)``

    ``b`` and ``c`` share ``year_ret`` and differ only in ``year_inv``,
    which is exactly the pair an implementation that summed over
    ``year_inv`` before comparing against the bound would pool.

    The host's own investment is pinned by equal min/max bounds --
    ``year_inv=2020 -> 10.0``, ``2030 -> 4.0``, ``2040 -> 0.0`` -- so
    the host planning cost is a constant of the study and cannot
    influence any choice the tests below observe (with
    ``combined_annuity_perfect_foresight`` at its fixture default of
    ``False``, ``planning_combined_costs_def`` charges the minimum
    annuity times the per-``year_inv`` total, which the pin fixes, so
    even the split across ``year_dec`` is cost-neutral). What the solver
    still chooses freely is that split, and the transfers themselves.

    The host's live ``(year_inv, year_dec)`` cells under this grid are
    ``(2020, 2030)``, ``(2020, 2040)``, ``(2030, 2040)``, ``(2030,
    2050)`` and ``(2040, 2060)``, so the pinned totals distribute as
    ``h(2020,2030) + h(2020,2040) = 10`` and ``h(2030,2040) +
    h(2030,2050) = 4``. The source-cap constraint backs each transfer
    with the host vintage decommissioning exactly at its retrofit year:
    ``a <= h(2020,2030)``, ``b <= h(2020,2040)``, ``c <= h(2030,2040)``.
    10.0 and 4.0 are deliberately generous -- large enough that the
    source cap is slack at the bound values the tests below use, so it
    is the investment bound, not the backing capacity, that binds.

    Demand is zeroed throughout: every test built on this fixture drives
    the transfer through capacity constraints alone, never through
    dispatch.
    """
    p = parameters_retrofit_combined.sel(
        area=["area_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050, 2060],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030, 2040, 2050],
        year_ret=[2030, 2040],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["combined_early_decommissioning"] = np.array(True, dtype="bool")

    host = xr.DataArray(
        [10.0, 4.0, 0.0],
        dims=["year_inv"],
        coords={"year_inv": p.year_inv.values},
    )
    p["combined_power_capacity_investment_max"] = host
    p["combined_power_capacity_investment_min"] = host.copy()
    return p


def _combined_per_vintage_at(solution, year_inv, year_ret):
    per_vintage = solution[
        "planning_retrofit_combined_power_capacity"
    ].sum("year_dec")
    return float(
        per_vintage.sel(
            area="area_1",
            retrofit_combined_pair="boiler_self",
            year_inv=year_inv,
            year_ret=year_ret,
        )
    )


def test_retrofit_combined_maximum_bound_is_per_vintage_not_aggregate(
    p_retrofit_combined_multi_vintage,
):
    """
    The maximum is keyed by ``year_ret`` alone (2030 -> 3.0, 2040 ->
    2.0), so the two ``year_ret=2040`` cells ``b`` and ``c`` -- which
    differ only in ``year_inv`` -- must each be allowed to reach 2.0
    independently (4.0 combined), not pooled against a single 2.0.

    Rather than pull the transfers up with demand (a self-pair prices
    source and target identically, so no cost comparison can favour
    retrofit here), this test pulls them up with the host module's own
    ``operation_combined_power_capacity_min_constraint``: a floor on
    *operational* capacity, which
    ``operation_combined_power_capacity_def`` satisfies from host
    planning capacity plus the retrofit's arriving contribution. With
    the host's investment pinned by the fixture, the only way to raise
    operational capacity in a year the host fleet has left is to
    retrofit, so a floor placed there is a hard, cost-independent pull
    on the transfers.

    Arithmetic, from the fixture's grid (``ret_ratio`` is 1.0, the
    fixture default, so one unit transferred is one unit arriving):

    - At ``year_op=2050``: every host vintage is gone -- ``(2020, 2030)``
      and ``(2020, 2040)`` and ``(2030, 2040)`` all decommissioned
      strictly before 2050, ``(2030, 2050)`` fails the def's strict
      ``year_op < year_dec``, and ``(2040, 2060)`` is pinned to zero.
      ``a``'s arrival window ``[2030, 2050)`` also excludes 2050. So
      operational capacity there is exactly ``b + c``. A floor of 4.0
      forces ``b + c = 4.0``, and with each capped at 2.0 that pins
      ``b = c = 2.0``.
    - At ``year_op=2040``: the only surviving host capacity is
      ``h(2030, 2050)``, and all three transfers have arrived
      (``a``'s window ``[2030, 2050)`` and ``b``/``c``'s ``[2040,
      2060)`` all cover 2040). Since ``h(2030,2040) >= c`` (the source
      cap) and ``h(2030,2040) + h(2030,2050) = 4``, operational
      capacity is at most ``a + b + c + (4 - c) = a + b + 4 <= 3 + 2 +
      4 = 9``. A floor of 9.0 therefore forces ``a = 3.0`` and
      ``b = 2.0`` with no slack anywhere.

    So the whole solution is pinned: ``a = 3.0``, ``b = 2.0``,
    ``c = 2.0``.

    What a wrong implementation produces instead: pooling the two
    ``year_ret=2040`` cells against one 2.0 cap makes ``b + c <= 2.0``,
    so the 4.0 floor at ``year_op=2050`` cannot be met at all and the
    solve comes back infeasible rather than ``ok``. Dropping the maximum
    altogether is caught too, and differently: the floors alone are
    satisfiable more cheaply than ``a=3, b=2, c=2`` (every transfer
    costs the same 15/yr over the same two-operating-year window, so the
    solver minimises ``a + b + c``, and ``a=1, b=4, c=0`` meets both
    floors at a total of 5 rather than 7), so the per-cell assertions
    below fail on value.
    """
    p = p_retrofit_combined_multi_vintage.copy(deep=True)
    p["combined_power_capacity_min"] = xr.DataArray(
        [np.nan, np.nan, 9.0, 4.0],
        dims=["year_op"],
        coords={"year_op": p.year_op.values},
    )
    p["retrofit_combined_power_capacity_investment_max"] = xr.DataArray(
        [3.0, 2.0],
        dims=["year_ret"],
        coords={"year_ret": p.year_ret.values},
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    assert _combined_per_vintage_at(s, 2020, 2030) == pytest.approx(
        3.0, abs=1e-6
    )
    assert _combined_per_vintage_at(s, 2020, 2040) == pytest.approx(
        2.0, abs=1e-6
    )
    assert _combined_per_vintage_at(s, 2030, 2040) == pytest.approx(
        2.0, abs=1e-6
    )


def test_retrofit_combined_minimum_bound_is_per_vintage_not_aggregate(
    p_retrofit_combined_multi_vintage,
):
    """
    Same three live cells as the maximum-bound test above, and the same
    reason for keying the bound by ``year_ret`` alone (2030 -> 2.0,
    2040 -> 3.0): the two ``year_ret=2040`` cells, which differ only in
    ``year_inv``, would be pooled by a wrong-axis implementation into a
    single ``>= 3.0`` requirement on their sum -- satisfiable with one
    of them left at zero.

    No capacity floor is needed here, and none is set: the investment
    minimum is a hard ``>=`` the solver must satisfy regardless of cost,
    and every incentive in this fixture pushes the transfers *down*
    (each unit transferred pays a retrofit annuity and raises the
    ``combined_fixed_cost`` charge on the extra operational capacity it
    creates), so the solver settles exactly on the bound for each live
    cell if -- and only if -- the bound is enforced per-vintage.

    The bounds are reachable: ``a >= 2.0`` and ``b >= 3.0`` draw on
    ``h(2020,2030)`` and ``h(2020,2040)``, which sum to the fixture's
    10.0, and ``c >= 3.0`` draws on ``h(2030,2040) <= 4.0``.
    """
    p = p_retrofit_combined_multi_vintage.copy(deep=True)
    p["retrofit_combined_power_capacity_investment_min"] = xr.DataArray(
        [2.0, 3.0],
        dims=["year_ret"],
        coords={"year_ret": p.year_ret.values},
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    assert _combined_per_vintage_at(s, 2020, 2030) >= 2.0 - 1e-6
    assert _combined_per_vintage_at(s, 2020, 2040) >= 3.0 - 1e-6
    assert _combined_per_vintage_at(s, 2030, 2040) >= 3.0 - 1e-6


@pytest.fixture()
def p_retrofit_combined_forced(parameters_retrofit_combined):
    """
    A slice in which retrofit is the ONLY route to capacity at
    ``year_op=2040``, with the host's own investment left free so the
    source-cap constraint has something to bite on.

    Construction:
    - ``year_dec=[2030, 2050]``, ``year_inv=[2020, 2030]``,
      ``year_op=[2020, 2030, 2040]``, ``year_ret=[2030]``. With
      ``combined_early_decommissioning=True`` and life span 20, the
      host's live ``(year_inv, year_dec)`` cells are ``(2020, 2030)``
      (early: 2030 <= end of life 2040, and 2030 <= year_inv.max()) and
      ``(2030, 2050)`` (its natural end of life). The transfer mask
      leaves exactly one cell alive, ``(year_inv=2020, year_ret=2030,
      year_dec=2050)``, backed by ``h(2020, 2030)``.
    - Host investment at ``year_inv=2030`` is pinned to 0 by equal
      min/max bounds, which is what makes the test non-vacuous: without
      it the solver could meet a capacity floor at ``year_op=2040``
      straight from that vintage (alive over ``[2030, 2050)``) and never
      retrofit at all. Investment at ``year_inv=2020`` is left free (max
      1000), so the amount of backing capacity is the solver's choice --
      which is precisely what the source cap governs.
    - ``combined_power_capacity_min`` places a floor of 4.0 on
      operational capacity at ``year_op=2040``. The ``year_inv=2020``
      vintage decommissions at 2030 and so contributes nothing there;
      with ``year_inv=2030`` pinned to zero, the floor can only be met
      by the retrofit's arriving contribution.
    - Demand is zeroed, so nothing but that floor drives the solve.
    """
    p = parameters_retrofit_combined.sel(
        area=["area_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030, 2040],
        year_ret=[2030],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["combined_early_decommissioning"] = np.array(True, dtype="bool")

    p["combined_power_capacity_investment_max"] = xr.DataArray(
        [1000.0, 0.0],
        dims=["year_inv"],
        coords={"year_inv": p.year_inv.values},
    )
    p["combined_power_capacity_investment_min"] = xr.DataArray(
        [np.nan, 0.0],
        dims=["year_inv"],
        coords={"year_inv": p.year_inv.values},
    )
    p["combined_power_capacity_min"] = xr.DataArray(
        [np.nan, np.nan, 4.0],
        dims=["year_op"],
        coords={"year_op": p.year_op.values},
    )
    return p


def _built_combined(solution):
    return float((solution.planning_combined_power_capacity).sum())


def test_retrofit_combined_cannot_exceed_the_source_capacity(
    p_retrofit_combined_forced,
):
    """
    A transfer must be backed by host capacity actually invested and
    decommissioned at the retrofit year -- see
    ``p_retrofit_combined_forced`` for the slice that makes retrofit the
    only way to meet a 4.0 capacity floor at ``year_op=2040``.

    Derivation: the floor forces the single live transfer to exactly
    4.0 (``ret_ratio`` is 1.0), and the source cap then forces
    ``h(2020, 2030) >= 4.0``. Nothing rewards building more than that
    -- host capacity costs an annuity of 100/yr plus a fixed cost of
    10/yr per unit while it is alive -- so the solver invests exactly
    4.0, and ``built_combined`` comes out at 4.0, matching the transfer.

    What the constraint is worth here, and the reversion that proves it:
    commenting out the ``retrofit_combined_power_cap_constraint`` (the
    ``leaving <= eligible_share * available`` block in ``_core.py``)
    leaves the floor satisfiable by an unbacked transfer. Retrofit is
    much the cheaper of the two -- 15/yr against the host's 100/yr --
    so with the cap gone the solver conjures the whole 4.0 out of thin
    air and builds no host capacity at all: ``built_combined`` drops to
    0.0 while ``retrofitted`` stays at 4.0, and the assertion below
    (``retrofitted <= built_combined``) fails. Restoring the file makes
    it pass again.
    """
    model = build_model(p_retrofit_combined_forced)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    retrofitted = float(
        s.planning_retrofit_combined_power_capacity.sum()
    )
    built = _built_combined(s)

    assert retrofitted > 0  # non-vacuous: retrofit is actually used
    assert retrofitted == pytest.approx(4.0, abs=1e-6)
    assert built == pytest.approx(4.0, abs=1e-6)
    assert retrofitted <= built + 1e-6


def test_retrofit_combined_pinned_by_equal_min_and_max_bounds(
    p_retrofit_combined_forced,
):
    """
    Exercises the ``==`` branch
    (``planning_retrofit_combined_power_capacity_def``), which the
    maximum/minimum tests above never touch -- their bounds always
    differ, routing through ``<=``/``>=`` exclusively.

    ``p_retrofit_combined_forced``'s own optimum is 4.0 of retrofit,
    exactly the capacity floor at ``year_op=2040`` and no more (see
    ``test_retrofit_combined_cannot_exceed_the_source_capacity``);
    every extra unit would cost a retrofit annuity plus a fixed cost on
    the capacity it creates. Pinning min and max to 6.0 -- *above* that
    natural optimum -- forces 6.0 of transfer and, with it, 6.0 of
    backing host investment (the source cap), so operational capacity
    at ``year_op=2040`` overshoots its own floor and lands at 6.0. Both
    numbers are values the optimiser would not choose on its own, so
    the equality constraint is what produces them.

    This goes strictly beyond
    ``test_retrofit_combined_self_pair_capacity_uprate``, which also
    pins by equal bounds but against a natural optimum of zero (its
    fixture has no capacity floor at all), where any positive lower
    bound would give the same answer.
    """
    p = p_retrofit_combined_forced.copy(deep=True)
    p["retrofit_combined_power_capacity_investment_min"] = xr.full_like(
        p.retrofit_combined_power_capacity_investment_min, 6.0
    )
    p["retrofit_combined_power_capacity_investment_max"] = xr.full_like(
        p.retrofit_combined_power_capacity_investment_max, 6.0
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    retrofitted = float(
        s.planning_retrofit_combined_power_capacity.sum()
    )
    assert retrofitted == pytest.approx(6.0, abs=1e-6)
    assert _built_combined(s) == pytest.approx(6.0, abs=1e-6)

    capacity = (s.operation_combined_power_capacity).sel(
        area="area_1", combined_tech="electric_boiler"
    )
    assert float(capacity.sel(year_op=2040)) == pytest.approx(
        6.0, abs=1e-6
    )


def test_combined_eligible_share_limits_the_convertible_fleet(
    p_retrofit_combined_forced,
):
    """
    ``eligible_share`` caps how much of the *source* fleet may convert
    at all -- distinct from ``ret_ratio``, an exchange rate on the
    target that derates what arrives on the other side.

    Same slice as
    ``test_retrofit_combined_cannot_exceed_the_source_capacity``, with
    ``retrofit_combined_eligible_share`` lowered to 0.3. The 4.0
    capacity floor at ``year_op=2040`` still forces the transfer to
    exactly 4.0, but the cap is now ``transfer <= 0.3 * h(2020, 2030)``,
    so the backing investment must rise to ``4.0 / 0.3 = 13.333...``.
    Nothing rewards building past that, so the solver lands on it
    exactly.

    That figure is what makes the test sensitive: dropping the
    ``eligible_share`` factor from the cap (``leaving <= available``)
    leaves the backing investment at 4.0, a factor of 10/3 away from
    the asserted value, so the assertion fails on value rather than
    passing by coincidence.
    """
    p = p_retrofit_combined_forced.copy(deep=True)
    p["retrofit_combined_eligible_share"] = xr.full_like(
        p.retrofit_combined_eligible_share, 0.3
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    retrofitted = float(
        s.planning_retrofit_combined_power_capacity.sum()
    )
    built = _built_combined(s)

    assert retrofitted > 0  # non-vacuous: retrofit is actually used
    assert retrofitted == pytest.approx(4.0, abs=1e-6)
    assert built == pytest.approx(4.0 / 0.3, abs=1e-5)


def test_combined_eligible_share_binds_by_site_not_broadcast(
    parameters_retrofit_combined,
):
    """
    ``eligible_share`` is declared on ``[area, combined_tech, year_ret,
    year_inv]`` -- its own site and technology dimensions, not the pair
    dimension the transfer variable carries. ``add_combined``
    sparsifies host capacity onto ``combined_tech_unit``, a dimension
    name that shares nothing with either, so a share multiplied into
    the cap without first being projected onto each unit's own
    ``(area, technology)`` value broadcasts across both dimensions:
    every unit ends up constrained by every site's share at once, and
    the smallest one wins everywhere.

    ``coords`` gives ``combined_tech`` a single technology, so the
    cross-product this test drives is over ``area`` rather than over
    technology (conversion's
    ``test_eligible_share_binds_by_technology_not_broadcast`` covers
    the technology axis). Two areas, two units, one share each:

    - ``area_1``: ``eligible_share = 1.0``
    - ``area_2``: ``eligible_share = 0.3``

    Both areas carry the same construction as
    ``p_retrofit_combined_forced`` (host investment free at
    ``year_inv=2020``, pinned to 0 at ``year_inv=2030``, a 4.0
    operational capacity floor at ``year_op=2040`` that only retrofit
    can meet), so each area's transfer is forced to 4.0 and its backing
    investment follows from its own share: ``4.0 / 1.0 = 4.0`` in
    ``area_1`` and ``4.0 / 0.3 = 13.333...`` in ``area_2``.

    A share that leaks across sites instead applies ``min(1.0, 0.3) =
    0.3`` to both units, dragging ``area_1``'s backing investment up to
    13.333... as well -- so the ``area_1`` assertion below is the one
    that catches it, while the ``area_2`` assertion (whose share is
    already the smallest of the two) is unaffected either way and
    serves only to show the share is genuinely doing something.
    """
    p = parameters_retrofit_combined.sel(
        area=["area_1", "area_2"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030, 2040],
        year_ret=[2030],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["combined_early_decommissioning"] = np.array(True, dtype="bool")

    p["combined_power_capacity_investment_max"] = xr.DataArray(
        [1000.0, 0.0],
        dims=["year_inv"],
        coords={"year_inv": p.year_inv.values},
    )
    p["combined_power_capacity_investment_min"] = xr.DataArray(
        [np.nan, 0.0],
        dims=["year_inv"],
        coords={"year_inv": p.year_inv.values},
    )
    p["combined_power_capacity_min"] = xr.DataArray(
        [np.nan, np.nan, 4.0],
        dims=["year_op"],
        coords={"year_op": p.year_op.values},
    )
    p["retrofit_combined_eligible_share"] = xr.DataArray(
        [1.0, 0.3], dims=["area"], coords={"area": p.area.values}
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    built = (s.planning_combined_power_capacity).sel(
        combined_tech="electric_boiler"
    )
    transferred = s.planning_retrofit_combined_power_capacity.sum(
        ["year_inv", "year_ret", "year_dec", "retrofit_combined_pair"]
    )

    assert float(transferred.sel(area="area_1")) == pytest.approx(
        4.0, abs=1e-6
    )
    assert float(transferred.sel(area="area_2")) == pytest.approx(
        4.0, abs=1e-6
    )
    assert float(built.sel(area="area_1").sum()) == pytest.approx(
        4.0, abs=1e-5
    )
    assert float(built.sel(area="area_2").sum()) == pytest.approx(
        4.0 / 0.3, abs=1e-5
    )


@pytest.fixture()
def p_retrofit_combined_chain(parameters_retrofit_combined):
    """
    A two-hop chain, which the ``electric_boiler -> electric_boiler``
    self-pair supports on its own: capacity that arrived by retrofit is
    capacity of the same technology, so the very same pair can convert
    it again at its new end of life.

    Timeline, chosen so the first hop's arrival window ends strictly
    before the study's last operating year and only the second hop's
    covers it. Grid: ``year_dec=[2030, 2050, 2070]``,
    ``year_inv=[2020, 2030, 2050]``, ``year_op=[2020, 2030, 2040,
    2050]``, ``year_ret=[2030, 2050]``, with
    ``combined_early_decommissioning=True``, host life span 20 and
    ``retrofit_combined_early_decommissioning`` at its fixture default
    of ``False`` (so each transfer's ``year_dec`` is pinned by exact
    equality to ``year_ret + 20``).

    - Hop 1 is ``(year_inv=2020, year_ret=2030, year_dec=2050)``:
      ``year_ret=2030`` is at or before the host's end of life for a
      2020 vintage (2040), which early decommissioning permits, and the
      2020 vintage's only live ``year_dec`` in this grid is 2030 -- 2050
      exceeds its 2040 end of life -- so that whole vintage
      decommissions exactly at the retrofit year and is available to
      back the transfer. Arrival window ``[2030, 2050)``: covers
      ``year_op`` 2030 and 2040, never 2050.
    - Hop 2 is ``(year_inv=2030, year_ret=2050, year_dec=2070)``:
      ``year_ret=2050`` sits exactly at the host's end of life for a
      2030 vintage (2050). Arrival window ``[2050, 2070)``: covers
      ``year_op=2050``, the study's last.
    - ``(year_inv=2020, year_ret=2050)`` is masked out -- 2050 is past
      the 2020 vintage's 2040 end of life -- so hop 1's own vintage
      cannot reach 2050 in a single hop.

    Chaining is what connects them: ``_chained_capacity`` relabels hop
    1's ``(year_ret=2030, year_dec=2050)`` as ``(year_inv=2030,
    year_ret=2050)``, exactly the cell hop 2's source cap reads.

    Three pins close every route to capacity at ``year_op=2050`` that
    does not pass through both hops:

    - ``year_inv=2030`` host investment is pinned to 0, so hop 2 has no
      *directly* invested backing and must draw on the chained term.
    - ``year_inv=2050`` host investment is pinned to 0, so the study's
      last operating year cannot simply be covered by a fresh build
      (that vintage's natural ``year_dec`` of 2070 would otherwise span
      it).
    - ``year_inv=2020`` host investment is pinned to 4.0, capping hop 1
      -- and hence, through the chain, hop 2 -- at 4.0.

    ``combined_power_capacity_min`` then places a floor of 4.0 on
    operational capacity at ``year_op=2050``, which only hop 2 can
    supply. Demand is zeroed throughout.
    """
    p = parameters_retrofit_combined.sel(
        area=["area_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2050, 2070],
        year_inv=[2020, 2030, 2050],
        year_op=[2020, 2030, 2040, 2050],
        year_ret=[2030, 2050],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["combined_early_decommissioning"] = np.array(True, dtype="bool")

    host = xr.DataArray(
        [4.0, 0.0, 0.0],
        dims=["year_inv"],
        coords={"year_inv": p.year_inv.values},
    )
    p["combined_power_capacity_investment_max"] = host
    p["combined_power_capacity_investment_min"] = host.copy()
    p["combined_power_capacity_min"] = xr.DataArray(
        [np.nan, np.nan, np.nan, 4.0],
        dims=["year_op"],
        coords={"year_op": p.year_op.values},
    )
    return p


def test_combined_capacity_acquired_by_retrofit_can_be_retrofitted_again(
    p_retrofit_combined_chain,
):
    """
    Capacity that arrived at ``electric_boiler`` through hop 1 is
    itself retrofitted again through hop 2, which is only possible if
    the source-cap constraint's right-hand side carries the chained
    term (``_chained_capacity``).

    Expected values, derived from ``p_retrofit_combined_chain``'s
    timeline rather than observed: the 4.0 floor on operational
    capacity at ``year_op=2050`` can only be met by hop 2's arriving
    contribution (hop 1's window ``[2030, 2050)`` excludes 2050, and
    every host vintage is pinned either to zero or to a 2020 vintage
    that decommissions at 2030), so hop 2 must transfer 4.0 --
    ``ret_ratio`` is 1.0, the fixture default. Hop 2's source cap
    offers exactly two terms: directly invested host capacity at
    ``(year_inv=2030, year_dec=2050)``, which the fixture pins to zero,
    and the chained term, which is hop 1's transfer. So hop 1 must also
    be 4.0 -- and it can be, since the 2020 vintage is pinned to 4.0
    and decommissions in full at ``year_ret=2030``. Both hops are
    therefore pinned at 4.0, with no slack in either direction.

    Reversion proof: dropping ``_chained_capacity`` from the source
    cap's right-hand side (returning only the directly-invested term)
    removes hop 2's only backing, forcing it to zero; the 4.0 floor at
    ``year_op=2050`` then has no way to be met and the solve comes back
    infeasible instead of ``ok``. Restoring ``_core.py`` makes it pass
    again.
    """
    model = build_model(p_retrofit_combined_chain)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    transfer = s.planning_retrofit_combined_power_capacity.sel(
        area="area_1", retrofit_combined_pair="boiler_self"
    )
    hop_one = float(transfer.sel(year_inv=2020, year_ret=2030).sum())
    hop_two = float(transfer.sel(year_inv=2030, year_ret=2050).sum())

    assert hop_two > 0  # non-vacuous: the second hop actually fires
    assert hop_one == pytest.approx(4.0, abs=1e-6)
    assert hop_two == pytest.approx(4.0, abs=1e-6)


def test_retrofit_combined_cost_enters_the_objective(
    p_retrofit_combined_value,
):
    """
    Doubling the retrofit annuity must raise the objective by exactly
    the amount transferred times the change in rate times the number of
    operating years it is charged for. Without the cost constraint and
    the totex hook, retrofit would be free and the objective would not
    move at all.

    ``p_retrofit_combined_value`` pins the transfer to exactly 3.0 by
    equal min/max bounds, so the two solves below differ in nothing but
    the annuity -- the quantity cannot shift to absorb the price
    change, which makes the difference exactly derivable rather than
    merely positive.

    Its slice has one live transfer cell, ``(year_inv=2020,
    year_ret=2030, year_dec=2050)``. With
    ``retrofit_combined_annuity_perfect_foresight`` at its fixture
    default of ``False``, the cost is charged for every ``year_op`` in
    ``[year_ret, retrofit_combined_end_of_life) = [2030, 2050)``; the
    fixture's ``year_op`` is ``{2020, 2030, 2040}``, so that is two
    operating years, 2030 and 2040. Overriding the annuity uniformly
    with ``xr.full_like`` makes the ``min`` over ``year_dec`` that the
    non-perfect-foresight branch takes equal to the override itself.

    The objective must therefore rise by exactly
    ``3.0 * (2.0 - 1.0) * 2 = 6.0``.
    """
    p = p_retrofit_combined_value.copy(deep=True)

    cheap = p.assign(
        retrofit_combined_annuity_cost=(
            xr.full_like(p.retrofit_combined_annuity_cost, 1.0)
        )
    )
    dear = p.assign(
        retrofit_combined_annuity_cost=(
            xr.full_like(p.retrofit_combined_annuity_cost, 2.0)
        )
    )

    m_cheap = build_model(cheap)
    m_cheap.solve(solver_name="highs")
    m_dear = build_model(dear)
    m_dear.solve(solver_name="highs")

    assert m_cheap.status == "ok" and m_dear.status == "ok"

    retrofit_cheap = float(
        m_cheap.solution.planning_retrofit_combined_power_capacity.sum()
    )
    retrofit_dear = float(
        m_dear.solution.planning_retrofit_combined_power_capacity.sum()
    )
    assert retrofit_cheap == pytest.approx(3.0, abs=1e-6)
    assert retrofit_dear == pytest.approx(3.0, abs=1e-6)
    assert m_dear.objective.value - m_cheap.objective.value == (
        pytest.approx(6.0, abs=1e-6)
    )


def test_combined_perfect_foresight_flag_changes_the_cost_basis(
    p_retrofit_combined_value,
):
    """
    Without the ``other`` branch of
    ``planning_retrofit_combined_costs_def``, the block would always
    charge the vintage-specific annuity no matter what
    ``retrofit_combined_annuity_perfect_foresight`` says -- the two
    builds below would then be byte-identical and their objectives
    equal.

    Built on ``p_retrofit_combined_value``, whose transfer is pinned to
    3.0 by equal min/max bounds and whose slice has exactly one live
    transfer cell, ``(year_inv=2020, year_ret=2030, year_dec=2050)``.
    Because ``retrofit_combined_early_decommissioning`` is ``False``,
    the mask pins ``year_dec`` to ``year_ret + life_span = 2050``
    exactly, so there is never a genuine choice of ``year_dec`` for the
    solver to make: the only thing the flag can change is which rate is
    charged on that one forced cell.

    The default annuity table would make the two branches coincide by
    construction -- built from ``square_array_by_diagonals(6, {0: 1/20,
    1: 1/10})``, it puts its lowest rate at exactly ``year_dec =
    year_ret + life_span``, the very cell the mask forces, so the
    ``min`` over ``year_dec`` would land on the forced cell for
    arithmetic reasons rather than because the flag does anything. So
    the table is overridden here to put the *expensive* rate (30) on
    the forced cell (``year_dec=2050``) and a *cheaper* one (15) on
    ``year_dec=2030``, a cell the mask never allows (it fails
    ``year_ret < year_dec``). ON must pay the forced cell's 30; OFF is
    free to charge the 15 the model can never physically realise --
    exactly the "pays the cheapest annuity consistent with the
    lifetime" behaviour the host module already exhibits.

    Both branches charge over the same window here (the forced
    ``year_dec`` and ``retrofit_combined_end_of_life`` are both 2050),
    i.e. ``year_op`` in ``{2030, 2040}`` -- two operating years. With
    the transfer pinned at 3.0 the objectives must differ by exactly
    ``3.0 * (30 - 15) * 2 = 90.0``.
    """
    p = p_retrofit_combined_value.copy(deep=True)
    p["retrofit_combined_annuity_cost"] = xr.full_like(
        p.retrofit_combined_annuity_cost, np.nan
    )
    p["retrofit_combined_annuity_cost"].loc[
        dict(year_ret=2030, year_dec=2050)
    ] = 30.0
    p["retrofit_combined_annuity_cost"].loc[
        dict(year_ret=2030, year_dec=2030)
    ] = 15.0

    on = p.assign(
        retrofit_combined_annuity_perfect_foresight=(
            xr.full_like(
                p.retrofit_combined_annuity_perfect_foresight, True
            )
        ).astype(bool)
    )
    off = p.assign(
        retrofit_combined_annuity_perfect_foresight=(
            xr.full_like(
                p.retrofit_combined_annuity_perfect_foresight, False
            )
        ).astype(bool)
    )

    m_on = build_model(on)
    m_on.solve(solver_name="highs")
    m_off = build_model(off)
    m_off.solve(solver_name="highs")

    assert m_on.status == "ok" and m_off.status == "ok"

    retrofit_on = float(
        m_on.solution.planning_retrofit_combined_power_capacity.sum()
    )
    retrofit_off = float(
        m_off.solution.planning_retrofit_combined_power_capacity.sum()
    )
    assert retrofit_on == pytest.approx(3.0, abs=1e-6)
    assert retrofit_off == pytest.approx(3.0, abs=1e-6)
    assert m_on.objective.value - m_off.objective.value == (
        pytest.approx(90.0, abs=1e-6)
    )


@pytest.fixture()
def p_retrofit_storage(parameters_retrofit_storage):
    return parameters_retrofit_storage.sel(
        area=["area_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030],
        year_ret=[2030, 2040],
    ).copy(deep=True)


def test_storage_retrofit_declares_both_capacity_kinds(p_retrofit_storage):
    """
    Structural: storage is the only host with two capacity kinds, so
    the generic block must emit one transfer variable per kind, each
    carrying the dimensions the block promises, plus a single pooled
    cost variable. Fails with a ``KeyError`` on either capacity
    variable if the adapter under-declares one of the two kinds.
    """
    model = build_model(p_retrofit_storage)

    for kind in ("power", "energy"):
        var = model.variables[f"planning_retrofit_storage_{kind}_capacity"]
        assert set(var.dims) == {
            "area",
            "retrofit_storage_pair",
            "year_inv",
            "year_ret",
            "year_dec",
        }
    assert "planning_retrofit_storage_costs" in model.variables


@pytest.fixture()
def p_retrofit_storage_value(parameters_retrofit_storage):
    """
    A genuine tank_methane -> tank_hydrogen transfer, built so both
    capacity kinds carry real, independently-checkable solved numbers.

    Construction (the same pinning idiom as
    ``p_retrofit_combined_value``, applied to both kinds):
    - tank_methane's own power (100) and energy (300) capacity are
      pinned at year_inv=2020 via equal min/max bounds on the HOST
      module, and to 0 at year_inv=2030 (present only so year_ret=2030
      is a subset of year_inv, as chaining requires). Every other
      storage_tech is left unconstrained; with demand zeroed, nothing
      incentivises investing in them, so they solve to 0.
    - year_dec is narrowed to {2030, 2050}, excluding tank_methane's
      natural end of life (2020 + storage_life_span 20 = 2040). With
      storage_early_decommissioning=True this leaves 2030 as the only
      valid decommissioning year for that vintage -- early
      decommissioning is forced by the grid, not chosen for cost
      reasons.
    - The retrofit's own transfer is pinned to exactly the same 100 /
      300 via equal min/max bounds on the kind-specific retrofit
      bounds, which the source cap allows exactly: ``available`` at
      year_ret=2030 is the 100 / 300 directly invested and
      decommissioned there, with eligible_share=1.0.
    - retrofit_storage_end_of_life = year_ret (2030) + life_span (20)
      = 2050, so with retrofit_storage_early_decommissioning=False
      only year_dec=2050 is a valid decommissioning year for the
      retrofit's own transfer.
    - Demand is zeroed throughout, so tank_hydrogen's own (unforced)
      investment is 0 in the optimum: any operational capacity
      observed for tank_hydrogen must come from the retrofit's
      arriving contribution.
    """
    p = parameters_retrofit_storage.sel(
        area=["area_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030, 2040],
        year_ret=[2030],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["storage_early_decommissioning"] = np.array(True, dtype="bool")

    source_capacity = {"power": 100.0, "energy": 300.0}
    for kind, value in source_capacity.items():
        bound = xr.DataArray(
            np.full((p.storage_tech.size, p.year_inv.size), np.nan),
            dims=["storage_tech", "year_inv"],
            coords={
                "storage_tech": p.storage_tech,
                "year_inv": p.year_inv,
            },
        )
        bound.loc[dict(storage_tech="tank_methane", year_inv=2020)] = value
        bound.loc[dict(storage_tech="tank_methane", year_inv=2030)] = 0.0
        p[f"storage_{kind}_capacity_investment_max"] = bound
        p[f"storage_{kind}_capacity_investment_min"] = bound.copy()

        retrofit_max = f"retrofit_storage_{kind}_capacity_investment_max"
        retrofit_min = f"retrofit_storage_{kind}_capacity_investment_min"
        p[retrofit_max] = xr.full_like(p[retrofit_max], value)
        p[retrofit_min] = xr.full_like(p[retrofit_min], value)
    return p


def test_storage_power_and_energy_convert_at_independent_ratios(
    p_retrofit_storage_value,
):
    """
    A methane cavern converted to hydrogen keeps most of its MW rating
    but loses most of its MWh: the two ratios must act separately,
    each kind scaled only by its own.

    ``ret_ratio_power=1.0`` and ``ret_ratio_energy=0.3`` derate the
    pinned 100 / 300 transfer (see ``p_retrofit_storage_value``) to
    arriving contributions of ``1.0 * 100 = 100`` (power) and
    ``0.3 * 300 = 90`` (energy) at tank_hydrogen, from year_op=2040
    onward (year_ret=2030 <= year_op, and the retrofit's own
    year_dec=2050 > year_op). Before the retrofit fires (year_op=2020)
    tank_hydrogen has no capacity of either kind: the transfer has not
    arrived yet, and tank_hydrogen's own investment is unforced -- 0,
    since demand is zeroed throughout.

    This discriminates against a module that shares one ratio across
    both kinds instead of applying each independently: with
    ret_ratio_power != ret_ratio_energy and source power (100) !=
    source energy (300), a shared-ratio bug reads the WRONG one of the
    two values below (e.g. energy landing on 100 instead of 90, or
    power on 90 instead of 100) -- confirmed by rerunning this fixture
    with both ratios forced to the same value (1.0), where the energy
    assertion changes from failing (90 != 300 * 1.0 = 300, if ratios
    were shared) to trivially passing at 300, showing the choice of
    unequal ratios and unequal source capacities is exactly what makes
    the assertions below sensitive to the two kinds being mixed up.
    """
    p = p_retrofit_storage_value.copy(deep=True)
    p["retrofit_storage_ret_ratio_power"] = xr.full_like(
        p.retrofit_storage_ret_ratio_power, 1.0
    )
    p["retrofit_storage_ret_ratio_energy"] = xr.full_like(
        p.retrofit_storage_ret_ratio_energy, 0.3
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    transferred_power = float(
        s.planning_retrofit_storage_power_capacity.sum()
    )
    transferred_energy = float(
        s.planning_retrofit_storage_energy_capacity.sum()
    )
    assert transferred_power == pytest.approx(100.0, abs=1e-6)  # pinned
    assert transferred_energy == pytest.approx(300.0, abs=1e-6)  # pinned

    power_capacity = (s.operation_storage_power_capacity).sel(
        area="area_1", storage_tech="tank_hydrogen"
    )
    energy_capacity = (s.operation_storage_energy_capacity).sel(
        area="area_1", storage_tech="tank_hydrogen"
    )

    assert float(power_capacity.sel(year_op=2020)) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(energy_capacity.sel(year_op=2020)) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(power_capacity.sel(year_op=2040)) == pytest.approx(
        100.0, abs=1e-6
    )
    assert float(energy_capacity.sel(year_op=2040)) == pytest.approx(
        90.0, abs=1e-6
    )


def _storage_bound(p, per_tech):
    """
    A ``[storage_tech, year_inv]`` bound array from a per-technology map.

    Every storage fixture below pins each technology's investment
    explicitly for every ``year_inv`` rather than leaving cells at NaN,
    so that "this technology cannot be built here" is stated in the data
    rather than inferred.
    """
    bound = xr.DataArray(
        np.full((p.storage_tech.size, p.year_inv.size), np.nan),
        dims=["storage_tech", "year_inv"],
        coords={
            "storage_tech": p.storage_tech,
            "year_inv": p.year_inv,
        },
    )
    for tech, values in per_tech.items():
        for year_inv, value in zip(p.year_inv.values, values):
            bound.loc[dict(storage_tech=tech, year_inv=year_inv)] = value
    return bound


def _storage_open_operating_caps(p, value=1000.0):
    """
    Give every storage technology a large, never-binding operating cap.

    ``add_storage`` decides which ``(area, storage_tech)`` pairs to build
    variables for from its gating parameters, and treats a pair whose
    only explicit values are non-positive as one the dense model already
    pins to zero. The fixtures below pin ``tank_hydrogen``'s own
    investment to zero (its capacity must come from retrofit, not from a
    direct build), which would otherwise drop it from the sparse index
    altogether and leave the retrofit's arriving contribution with no
    unit to land on. A positive operating cap keeps every technology
    present without constraining anything the tests care about.
    """
    caps = xr.DataArray(
        np.full(p.storage_tech.size, value),
        dims=["storage_tech"],
        coords={"storage_tech": p.storage_tech},
    )
    p["storage_power_capacity_max"] = caps
    p["storage_energy_capacity_max"] = caps.copy()
    return p


def _storage_floor(p, per_tech_year_op):
    """A ``[storage_tech, year_op]`` operating floor, NaN where unset."""
    floor = xr.DataArray(
        np.full((p.storage_tech.size, p.year_op.size), np.nan),
        dims=["storage_tech", "year_op"],
        coords={"storage_tech": p.storage_tech, "year_op": p.year_op},
    )
    for (tech, year_op), value in per_tech_year_op.items():
        floor.loc[dict(storage_tech=tech, year_op=year_op)] = value
    return floor


@pytest.fixture()
def p_retrofit_storage_multi_vintage(parameters_retrofit_storage):
    """
    Three live transfer cells per capacity kind, so a bound keyed by
    ``year_ret`` alone can tell a per-vintage implementation apart from
    an aggregated one -- the storage counterpart of
    ``p_retrofit_combined_multi_vintage``, and the same coordinate
    slice: ``year_inv=[2020, 2030, 2040]``, ``year_dec=[2030, 2040,
    2050, 2060]``, ``year_op=[2020, 2030, 2040, 2050]``,
    ``year_ret=[2030, 2040]``, with
    ``storage_early_decommissioning=True``, storage life span 20 and
    ``retrofit_storage_early_decommissioning`` at its fixture default of
    ``False``. The three cells are

    - ``a = (year_inv=2020, year_ret=2030, year_dec=2050)``
    - ``b = (year_inv=2020, year_ret=2040, year_dec=2060)``
    - ``c = (year_inv=2030, year_ret=2040, year_dec=2060)``

    ``b`` and ``c`` share ``year_ret`` and differ only in ``year_inv``.

    Unlike the combined self-pair, storage retrofits a genuine A -> B
    pair (``tank_methane -> tank_hydrogen``), which makes the reading of
    the target's operating capacity much cleaner: ``tank_hydrogen``'s
    own investment is pinned to zero at every ``year_inv``, so every
    unit of capacity it ever has arrived there by retrofit, with no host
    vintage of its own to disentangle.

    ``tank_methane``'s investment is pinned by equal min/max bounds,
    deliberately generously, so the source cap is slack at the bound
    values the tests below use and it is the investment bound that
    binds: power ``year_inv=2020 -> 10``, ``2030 -> 4``, ``2040 -> 0``;
    energy ``2020 -> 20``, ``2030 -> 8``, ``2040 -> 0``. The two kinds
    carry deliberately different numbers so a test that read one kind's
    bound while asserting on the other's variable could not pass by
    coincidence. ``battery`` is pinned to zero throughout.

    Demand is zeroed: every test built on this fixture drives the
    transfers through capacity constraints alone, never through
    dispatch.
    """
    p = parameters_retrofit_storage.sel(
        area=["area_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050, 2060],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030, 2040, 2050],
        year_ret=[2030, 2040],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["storage_early_decommissioning"] = np.array(True, dtype="bool")
    p = _storage_open_operating_caps(p)

    power = _storage_bound(
        p,
        {
            "tank_methane": [10.0, 4.0, 0.0],
            "tank_hydrogen": [0.0, 0.0, 0.0],
            "battery": [0.0, 0.0, 0.0],
        },
    )
    energy = _storage_bound(
        p,
        {
            "tank_methane": [20.0, 8.0, 0.0],
            "tank_hydrogen": [0.0, 0.0, 0.0],
            "battery": [0.0, 0.0, 0.0],
        },
    )
    p["storage_power_capacity_investment_max"] = power
    p["storage_power_capacity_investment_min"] = power.copy()
    p["storage_energy_capacity_investment_max"] = energy
    p["storage_energy_capacity_investment_min"] = energy.copy()
    return p


def _storage_per_vintage_at(solution, kind, year_inv, year_ret):
    per_vintage = solution[
        f"planning_retrofit_storage_{kind}_capacity"
    ].sum("year_dec")
    return float(
        per_vintage.sel(
            area="area_1",
            retrofit_storage_pair="methane_to_hydrogen",
            year_inv=year_inv,
            year_ret=year_ret,
        )
    )


def test_retrofit_storage_maximum_bound_is_per_vintage_not_aggregate(
    p_retrofit_storage_multi_vintage,
):
    """
    Both capacity kinds, each with its own maximum keyed by ``year_ret``
    alone -- power (2030 -> 3.0, 2040 -> 2.0) and energy (2030 -> 6.0,
    2040 -> 4.0). The two ``year_ret=2040`` cells ``b`` and ``c``, which
    differ only in ``year_inv``, must each be allowed to reach the bound
    independently rather than being pooled against a single copy of it.

    The transfers are pulled up by a floor on ``tank_hydrogen``'s
    *operating* capacity
    (``operation_storage_power/energy_capacity_min_constraint``), not by
    demand: since ``tank_hydrogen``'s own investment is pinned to zero
    (see the fixture), its operating capacity in any year is exactly the
    sum of the retrofit contributions whose arrival window covers that
    year, which makes the floor a direct, cost-independent statement
    about the transfers. ``ret_ratio_power`` and ``ret_ratio_energy``
    are both 1.0 (fixture defaults), so a unit transferred is a unit
    arriving.

    Arithmetic:

    - At ``year_op=2050``, only ``b`` and ``c`` have arrived: their
      window is ``[2040, 2060)``, while ``a``'s ``[2030, 2050)``
      excludes 2050. Floors of 4.0 (power) and 8.0 (energy) therefore
      force ``b + c`` to exactly those totals, and with each cell capped
      at 2.0 / 4.0 that pins ``b = c = 2.0`` (power) and ``b = c = 4.0``
      (energy).
    - At ``year_op=2040``, all three have arrived, so operating capacity
      is ``a + b + c``. Floors of 7.0 and 14.0 then force ``a = 7 - 4 =
      3.0`` (power) and ``a = 14 - 8 = 6.0`` (energy), exactly their own
      maxima.

    What a wrong implementation produces instead: pooling the two
    ``year_ret=2040`` cells against one copy of the bound makes
    ``b + c <= 2.0`` (power), so the 4.0 floor at ``year_op=2050``
    cannot be met at all and the solve comes back infeasible rather than
    ``ok``.
    """
    p = p_retrofit_storage_multi_vintage.copy(deep=True)
    p["storage_power_capacity_min"] = _storage_floor(
        p,
        {
            ("tank_hydrogen", 2040): 7.0,
            ("tank_hydrogen", 2050): 4.0,
        },
    )
    p["storage_energy_capacity_min"] = _storage_floor(
        p,
        {
            ("tank_hydrogen", 2040): 14.0,
            ("tank_hydrogen", 2050): 8.0,
        },
    )
    p["retrofit_storage_power_capacity_investment_max"] = xr.DataArray(
        [3.0, 2.0],
        dims=["year_ret"],
        coords={"year_ret": p.year_ret.values},
    )
    p["retrofit_storage_energy_capacity_investment_max"] = xr.DataArray(
        [6.0, 4.0],
        dims=["year_ret"],
        coords={"year_ret": p.year_ret.values},
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    for kind, (first, rest) in {
        "power": (3.0, 2.0),
        "energy": (6.0, 4.0),
    }.items():
        assert _storage_per_vintage_at(
            s, kind, 2020, 2030
        ) == pytest.approx(first, abs=1e-6)
        assert _storage_per_vintage_at(
            s, kind, 2020, 2040
        ) == pytest.approx(rest, abs=1e-6)
        assert _storage_per_vintage_at(
            s, kind, 2030, 2040
        ) == pytest.approx(rest, abs=1e-6)


def test_retrofit_storage_minimum_bound_is_per_vintage_not_aggregate(
    p_retrofit_storage_multi_vintage,
):
    """
    Same three live cells and the same rationale as the maximum-bound
    test above, with minima keyed by ``year_ret`` alone: power
    (2030 -> 2.0, 2040 -> 3.0) and energy (2030 -> 4.0, 2040 -> 6.0). A
    wrong-axis implementation pools the two ``year_ret=2040`` cells into
    a single ``>= 3.0`` (or ``>= 6.0``) requirement on their sum, which
    is satisfiable with one of them left at zero.

    No operating floor is needed here, and none is set: the investment
    minimum is a hard ``>=`` the solver must satisfy regardless of cost,
    and every transfer costs a positive annuity, so the solver settles
    exactly on the bound for each live cell if -- and only if -- the
    bound is enforced per-vintage.

    The minima are reachable under the source cap: ``a >= 2`` and
    ``b >= 3`` draw on ``h(2020,2030)`` and ``h(2020,2040)``, which sum
    to the fixture's pinned 10 (power) / 20 (energy), and ``c >= 3``
    draws on ``h(2030,2040) <= 4`` (power) / ``<= 8`` (energy).
    """
    p = p_retrofit_storage_multi_vintage.copy(deep=True)
    p["retrofit_storage_power_capacity_investment_min"] = xr.DataArray(
        [2.0, 3.0],
        dims=["year_ret"],
        coords={"year_ret": p.year_ret.values},
    )
    p["retrofit_storage_energy_capacity_investment_min"] = xr.DataArray(
        [4.0, 6.0],
        dims=["year_ret"],
        coords={"year_ret": p.year_ret.values},
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    for kind, (first, rest) in {
        "power": (2.0, 3.0),
        "energy": (4.0, 6.0),
    }.items():
        assert (
            _storage_per_vintage_at(s, kind, 2020, 2030) >= first - 1e-6
        )
        assert (
            _storage_per_vintage_at(s, kind, 2020, 2040) >= rest - 1e-6
        )
        assert (
            _storage_per_vintage_at(s, kind, 2030, 2040) >= rest - 1e-6
        )


@pytest.fixture()
def p_retrofit_storage_forced(parameters_retrofit_storage):
    """
    A slice in which retrofit is the ONLY route to ``tank_hydrogen``
    capacity at ``year_op=2040``, with ``tank_methane``'s own investment
    left free so the source-cap constraint has something to bite on.

    Construction:
    - ``year_dec=[2030, 2050]``, ``year_inv=[2020, 2030]``,
      ``year_op=[2020, 2030, 2040]``, ``year_ret=[2030]``. With
      ``storage_early_decommissioning=True`` and life span 20 the host's
      live ``(year_inv, year_dec)`` cells are ``(2020, 2030)`` (early:
      2030 is at or before the 2040 end of life, and no later than
      ``year_inv.max()``) and ``(2030, 2050)`` (its natural end of
      life), and the transfer mask leaves exactly one cell alive,
      ``(year_inv=2020, year_ret=2030, year_dec=2050)``, backed by
      ``h(2020, 2030)``.
    - ``tank_methane``'s investment at ``year_inv=2020`` is left free
      (max 1000, no minimum), so how much backing capacity exists is the
      solver's own choice -- precisely what the source cap governs --
      and is pinned to 0 at ``year_inv=2030`` so no later vintage can
      back anything.
    - ``tank_hydrogen`` and ``battery`` are pinned to zero investment
      throughout, so ``tank_hydrogen``'s operating capacity is entirely
      the retrofit's arriving contribution.
    - Floors of 4.0 (power) and 9.0 (energy) on ``tank_hydrogen``'s
      operating capacity at ``year_op=2040``. Deliberately unequal, so a
      kind mix-up cannot pass.
    - Demand is zeroed, so nothing but those floors drives the solve.
    """
    p = parameters_retrofit_storage.sel(
        area=["area_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030, 2040],
        year_ret=[2030],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["storage_early_decommissioning"] = np.array(True, dtype="bool")
    p = _storage_open_operating_caps(p)

    for kind in ("power", "energy"):
        invest_max = _storage_bound(
            p,
            {
                "tank_methane": [1000.0, 0.0],
                "tank_hydrogen": [0.0, 0.0],
                "battery": [0.0, 0.0],
            },
        )
        invest_min = _storage_bound(
            p,
            {
                "tank_methane": [np.nan, 0.0],
                "tank_hydrogen": [0.0, 0.0],
                "battery": [0.0, 0.0],
            },
        )
        p[f"storage_{kind}_capacity_investment_max"] = invest_max
        p[f"storage_{kind}_capacity_investment_min"] = invest_min

    p["storage_power_capacity_min"] = _storage_floor(
        p, {("tank_hydrogen", 2040): 4.0}
    )
    p["storage_energy_capacity_min"] = _storage_floor(
        p, {("tank_hydrogen", 2040): 9.0}
    )
    return p


def _built_storage(solution, kind, tech):
    return float(
        (solution[f"planning_storage_{kind}_capacity"])
        .sel(storage_tech=tech)
        .sum()
    )


def test_retrofit_storage_cannot_exceed_the_source_capacity(
    p_retrofit_storage_forced,
):
    """
    Each kind's transfer must be backed by ``tank_methane`` capacity of
    that same kind, actually invested and decommissioned at the retrofit
    year -- see ``p_retrofit_storage_forced`` for the slice that makes
    retrofit the only way to meet the 4.0 (power) / 9.0 (energy) floors
    at ``year_op=2040``.

    Derivation: the floors force the single live transfer to exactly
    4.0 and 9.0 (both ratios are 1.0), and the source cap then forces
    ``h(2020, 2030) >= 4.0`` for power and ``>= 9.0`` for energy.
    Nothing rewards building more -- ``tank_methane``'s annuity is
    strictly positive for both kinds -- so the solver invests exactly
    those amounts. Checking both kinds separately, against unequal
    numbers, is what makes this a statement about two independent caps
    rather than one shared cap that happens to be satisfied.

    What the constraint is worth here, and the reversion that proves it:
    commenting out the ``retrofit_storage_{kind}_cap_constraint`` (the
    ``leaving <= eligible_share * available`` block in ``_core.py``)
    leaves the floors satisfiable by an unbacked transfer. Retrofit is
    the cheaper of the two routes -- ``retrofit_storage_invest_cost_
    power`` is 300 against ``tank_methane``'s own 400, and the retrofit
    is charged over fewer operating years -- so with the cap gone the
    solver conjures the whole transfer out of thin air and builds no
    ``tank_methane`` at all: both ``built_*`` figures drop to 0.0 while
    the transfers stay at 4.0 / 9.0, failing the assertions below.
    Restoring the file makes them pass again.
    """
    model = build_model(p_retrofit_storage_forced)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    for kind, expected in (("power", 4.0), ("energy", 9.0)):
        transferred = float(
            s[f"planning_retrofit_storage_{kind}_capacity"].sum()
        )
        built = _built_storage(s, kind, "tank_methane")

        assert transferred > 0  # non-vacuous: retrofit is actually used
        assert transferred == pytest.approx(expected, abs=1e-6)
        assert built == pytest.approx(expected, abs=1e-6)
        assert transferred <= built + 1e-6


def test_retrofit_storage_pinned_by_equal_min_and_max_bounds(
    p_retrofit_storage_forced,
):
    """
    Exercises the ``==`` branch
    (``planning_retrofit_storage_{kind}_capacity_def``) for both kinds,
    which the maximum/minimum tests above never touch -- their bounds
    always differ, routing through ``<=``/``>=`` exclusively.

    ``p_retrofit_storage_forced``'s own optimum is 4.0 of power and 9.0
    of energy transferred, exactly the operating floors at
    ``year_op=2040`` and no more (see
    ``test_retrofit_storage_cannot_exceed_the_source_capacity``); every
    extra unit would pay a retrofit annuity and drag a matching unit of
    ``tank_methane`` investment along with it through the source cap.
    Pinning the two kinds to 6.0 and 12.0 -- *above* those natural
    optima, and by different margins -- forces exactly those transfers,
    and with them 6.0 and 12.0 of backing ``tank_methane`` investment.
    ``tank_hydrogen``'s operating capacity at ``year_op=2040`` then
    overshoots its own floor and lands on 6.0 / 12.0. None of those
    figures is one the optimiser would choose on its own, so the
    equality constraint is what produces them.
    """
    p = p_retrofit_storage_forced.copy(deep=True)
    pinned = {"power": 6.0, "energy": 12.0}
    for kind, value in pinned.items():
        for which in ("min", "max"):
            name = f"retrofit_storage_{kind}_capacity_investment_{which}"
            p[name] = xr.full_like(p[name], value)

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    for kind, value in pinned.items():
        transferred = float(
            s[f"planning_retrofit_storage_{kind}_capacity"].sum()
        )
        assert transferred == pytest.approx(value, abs=1e-6)
        assert _built_storage(s, kind, "tank_methane") == pytest.approx(
            value, abs=1e-6
        )

        capacity = (s[f"operation_storage_{kind}_capacity"]).sel(
            area="area_1", storage_tech="tank_hydrogen"
        )
        assert float(capacity.sel(year_op=2040)) == pytest.approx(
            value, abs=1e-6
        )


def test_storage_eligible_share_limits_the_convertible_fleet(
    p_retrofit_storage_forced,
):
    """
    ``eligible_share`` caps how much of the *source* fleet may convert
    at all -- distinct from ``ret_ratio_power`` / ``ret_ratio_energy``,
    exchange rates on the target that derate what arrives on the other
    side. Storage declares one share for the technology, shared by both
    kinds, so lowering it must move both.

    Same slice as
    ``test_retrofit_storage_cannot_exceed_the_source_capacity``, with
    ``retrofit_storage_eligible_share`` lowered to 0.4. The operating
    floors still force the transfers to exactly 4.0 (power) and 9.0
    (energy), but each cap is now ``transfer <= 0.4 * h(2020, 2030)``,
    so the backing investment must rise to ``4.0 / 0.4 = 10.0`` and
    ``9.0 / 0.4 = 22.5``. Nothing rewards building past that, so the
    solver lands on those figures exactly.

    0.4 rather than a rounder 0.5 is deliberate: with a share dropped
    from the constraint entirely (``leaving <= available``) the backing
    investment stays at 4.0 / 9.0, which is 2.5x away from the asserted
    values, so the assertions fail on value rather than passing by
    coincidence.
    """
    p = p_retrofit_storage_forced.copy(deep=True)
    p["retrofit_storage_eligible_share"] = xr.full_like(
        p.retrofit_storage_eligible_share, 0.4
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    for kind, expected in (("power", 4.0), ("energy", 9.0)):
        transferred = float(
            s[f"planning_retrofit_storage_{kind}_capacity"].sum()
        )
        built = _built_storage(s, kind, "tank_methane")

        assert transferred > 0  # non-vacuous: retrofit is actually used
        assert transferred == pytest.approx(expected, abs=1e-6)
        assert built == pytest.approx(expected / 0.4, abs=1e-5)


def test_storage_eligible_share_binds_by_technology_not_broadcast(
    p_retrofit_storage_forced,
):
    """
    ``eligible_share`` is declared on ``[area, storage_tech, year_ret,
    year_inv]`` -- its own technology dimension, not the pair dimension
    the transfer variables carry. ``add_storage`` sparsifies host
    capacity onto ``storage_tech_unit``, a dimension name that shares
    nothing with ``storage_tech``, so a share multiplied into the cap
    without first being projected onto each unit's own ``(area,
    technology)`` value broadcasts across both dimensions: every unit
    ends up constrained by every technology's share at once, and the
    smallest one wins everywhere. Every other storage retrofit test sets
    ``eligible_share`` as a 0-d scalar, where there is nothing to
    cross-product against -- this is the one that deliberately avoids
    that setup.

    The share is set to 0.4 on ``battery`` and 1.0 on everything else.
    ``battery`` is neither the source nor the target of the pair here
    (``tank_methane -> tank_hydrogen``) and is pinned to zero investment
    by the fixture, so a correctly-aligned share must leave the solve
    exactly as it is in the ``eligible_share=1`` baseline of
    ``test_retrofit_storage_cannot_exceed_the_source_capacity``:
    transfers of 4.0 / 9.0 backed by 4.0 / 9.0 of ``tank_methane``.

    A share that leaks in from an unrelated technology instead drags the
    effective cap down to ``min(1.0, 1.0, 0.4) = 0.4``, which is exactly
    the setup of
    ``test_storage_eligible_share_limits_the_convertible_fleet`` and
    would push the backing investment to 10.0 / 22.5 -- so the
    ``built`` assertions below are what catch it.
    """
    p = p_retrofit_storage_forced.copy(deep=True)
    p["retrofit_storage_eligible_share"] = xr.where(
        p.storage_tech == "battery", 0.4, 1.0
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    for kind, expected in (("power", 4.0), ("energy", 9.0)):
        transferred = float(
            s[f"planning_retrofit_storage_{kind}_capacity"].sum()
        )
        assert transferred == pytest.approx(expected, abs=1e-6)
        assert _built_storage(s, kind, "tank_methane") == pytest.approx(
            expected, abs=1e-5
        )


@pytest.fixture()
def p_retrofit_storage_chain(parameters_retrofit_storage):
    """
    A two-hop chain, on a second, self-targeting pair: pair
    ``methane_to_hydrogen`` (``tank_methane -> tank_hydrogen``, as in
    every other storage fixture here) and pair ``hydrogen_self``
    (``tank_hydrogen -> tank_hydrogen``), letting a cavern that already
    became a hydrogen store be retrofitted again -- renewed -- at its
    own end of life.

    Timeline, chosen so hop 1's arrival window ends strictly before the
    study's last operating year and only hop 2's covers it. Grid:
    ``year_dec=[2030, 2050, 2070]``, ``year_inv=[2020, 2030, 2050]``,
    ``year_op=[2020, 2030, 2040, 2050]``, ``year_ret=[2030, 2050]``,
    with ``storage_early_decommissioning=True``, host life span 20 and
    ``retrofit_storage_early_decommissioning`` at its fixture default of
    ``False`` (so each transfer's ``year_dec`` is pinned by exact
    equality to ``year_ret + 20``).

    - Hop 1 is pair 0 at ``(year_inv=2020, year_ret=2030,
      year_dec=2050)``: ``year_ret=2030`` is at or before
      ``tank_methane``'s end of life for a 2020 vintage (2040), which
      early decommissioning permits, and that vintage's only live
      ``year_dec`` in this grid is 2030 (2050 exceeds its end of life),
      so it decommissions in full exactly at the retrofit year and can
      back the transfer. Arrival window ``[2030, 2050)``: covers
      ``year_op`` 2030 and 2040, never 2050.
    - Hop 2 is pair 1 at ``(year_inv=2030, year_ret=2050,
      year_dec=2070)``: ``year_ret=2050`` sits exactly at
      ``tank_hydrogen``'s end of life for a 2030 vintage. Arrival window
      ``[2050, 2070)``: covers ``year_op=2050``, the study's last.

    Chaining is what connects them: ``_chained_capacity`` relabels hop
    1's ``(year_ret=2030, year_dec=2050)`` as ``(year_inv=2030,
    year_ret=2050)``, exactly the cell hop 2's source cap reads on the
    ``tank_hydrogen`` unit.

    Every other route into ``tank_hydrogen`` capacity at
    ``year_op=2050`` is closed by pinning investment:

    - ``tank_methane`` is pinned to 4.0 power / 9.0 energy at
      ``year_inv=2020`` and to zero at 2030 and 2050. The zero at 2030
      is what forces hop 2 onto the chained term: pair 0's own
      ``(year_inv=2030, year_ret=2050)`` cell is structurally alive
      (2050 is exactly a 2030 vintage's end of life) but has no
      directly-invested ``tank_methane`` behind it, and pair 0 targets
      ``tank_hydrogen`` so it draws on the ``tank_methane`` unit's cap,
      which is zero there.
    - ``tank_hydrogen`` is pinned to zero investment throughout, so hop
      2 has no directly-invested backing of its own and the study's last
      operating year cannot be covered by a fresh build.
    - ``battery`` is pinned to zero throughout.

    Floors of 4.0 (power) and 9.0 (energy) on ``tank_hydrogen``'s
    operating capacity at ``year_op=2050`` then force hop 2, and through
    the chain hop 1, to exactly those amounts. Demand is zeroed.
    """
    p = parameters_retrofit_storage.sel(
        area=["area_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2050, 2070],
        year_inv=[2020, 2030, 2050],
        year_op=[2020, 2030, 2040, 2050],
        year_ret=[2030, 2050],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["storage_early_decommissioning"] = np.array(True, dtype="bool")
    p = _storage_open_operating_caps(p)

    for kind, source in (("power", 4.0), ("energy", 9.0)):
        bound = _storage_bound(
            p,
            {
                "tank_methane": [source, 0.0, 0.0],
                "tank_hydrogen": [0.0, 0.0, 0.0],
                "battery": [0.0, 0.0, 0.0],
            },
        )
        p[f"storage_{kind}_capacity_investment_max"] = bound
        p[f"storage_{kind}_capacity_investment_min"] = bound.copy()

    p["storage_power_capacity_min"] = _storage_floor(
        p, {("tank_hydrogen", 2050): 4.0}
    )
    p["storage_energy_capacity_min"] = _storage_floor(
        p, {("tank_hydrogen", 2050): 9.0}
    )

    pair = np.array(["methane_to_hydrogen", "hydrogen_self"], dtype=str)
    p = p.drop_dims("retrofit_storage_pair")
    p = p.assign(
        retrofit_storage_tech_from=xr.DataArray(
            np.array(["tank_methane", "tank_hydrogen"], dtype=str),
            dims="retrofit_storage_pair",
            coords={"retrofit_storage_pair": pair},
        ),
        retrofit_storage_tech_to=xr.DataArray(
            np.array(["tank_hydrogen", "tank_hydrogen"], dtype=str),
            dims="retrofit_storage_pair",
            coords={"retrofit_storage_pair": pair},
        ),
    )
    return p


def test_storage_capacity_acquired_by_retrofit_can_be_retrofitted_again(
    p_retrofit_storage_chain,
):
    """
    Capacity that arrived at ``tank_hydrogen`` through pair 0
    (``tank_methane -> tank_hydrogen``) is itself retrofitted again
    through pair 1 (``tank_hydrogen -> tank_hydrogen``), for both
    capacity kinds. That is only possible if the source-cap
    constraint's right-hand side carries the chained term
    (``_chained_capacity``), and only if that term is built
    per-kind rather than once and reused.

    Expected values, derived from ``p_retrofit_storage_chain``'s
    timeline rather than observed: the floors of 4.0 (power) and 9.0
    (energy) on ``tank_hydrogen``'s operating capacity at
    ``year_op=2050`` can only be met by hop 2's arriving contribution
    (hop 1's window ``[2030, 2050)`` excludes 2050, and every host
    vintage is pinned either to zero or to a ``tank_methane`` vintage
    that decommissions at 2030), so hop 2 must transfer 4.0 / 9.0 --
    both ratios are 1.0, the fixture defaults. Hop 2's source cap on
    the ``tank_hydrogen`` unit offers exactly two terms: directly
    invested ``tank_hydrogen`` capacity, which the fixture pins to
    zero, and the chained term, which is hop 1's transfer. So hop 1
    must also be 4.0 / 9.0 -- and it can be, since ``tank_methane``'s
    2020 vintage is pinned to exactly those amounts and decommissions
    in full at ``year_ret=2030``. Both hops are pinned in both kinds,
    with no slack in either direction.

    Reversion proof: dropping ``_chained_capacity`` from the source
    cap's right-hand side (returning only the directly-invested term)
    removes hop 2's only backing, forcing it to zero; the floors at
    ``year_op=2050`` then have no way to be met and the solve comes
    back infeasible instead of ``ok``. Restoring ``_core.py`` makes it
    pass again.
    """
    model = build_model(p_retrofit_storage_chain)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    for kind, expected in (("power", 4.0), ("energy", 9.0)):
        transfer = s[f"planning_retrofit_storage_{kind}_capacity"].sel(
            area="area_1"
        )
        hop_one = float(
            transfer.sel(
                retrofit_storage_pair="methane_to_hydrogen",
                year_inv=2020,
                year_ret=2030,
            ).sum()
        )
        hop_two = float(
            transfer.sel(
                retrofit_storage_pair="hydrogen_self",
                year_inv=2030,
                year_ret=2050,
            ).sum()
        )

        assert hop_two > 0  # non-vacuous: the second hop actually fires
        assert hop_one == pytest.approx(expected, abs=1e-6)
        assert hop_two == pytest.approx(expected, abs=1e-6)


def test_retrofit_storage_cost_enters_the_objective(
    p_retrofit_storage_value,
):
    """
    Doubling both kinds' retrofit annuities must raise the objective by
    exactly the amounts transferred times the change in rate times the
    number of operating years they are charged for. Without the cost
    constraint and the totex hook, retrofit would be free and the
    objective would not move; with only one of the two kinds wired into
    the pooled cost variable, it would move by less than the figure
    below.

    ``p_retrofit_storage_value`` pins the transfers to exactly 100
    (power) and 300 (energy) by equal min/max bounds, so the two solves
    differ in nothing but the annuities -- the quantities cannot shift
    to absorb the price change, which makes the difference exactly
    derivable rather than merely positive.

    Its slice has one live transfer cell per kind, ``(year_inv=2020,
    year_ret=2030, year_dec=2050)``. With
    ``retrofit_storage_annuity_perfect_foresight`` at its fixture
    default of ``False``, each kind's cost is charged for every
    ``year_op`` in ``[year_ret, retrofit_storage_end_of_life) = [2030,
    2050)``; the fixture's ``year_op`` is ``{2020, 2030, 2040}``, so
    that is two operating years, 2030 and 2040. Overriding an annuity
    uniformly with ``xr.full_like`` makes the ``min`` over ``year_dec``
    that the non-perfect-foresight branch takes equal to the override
    itself.

    The objective must therefore rise by exactly
    ``(100 + 300) * (2.0 - 1.0) * 2 = 800.0``.
    """
    p = p_retrofit_storage_value.copy(deep=True)

    def _priced(rate):
        return p.assign(
            retrofit_storage_annuity_cost_power=xr.full_like(
                p.retrofit_storage_annuity_cost_power, rate
            ),
            retrofit_storage_annuity_cost_energy=xr.full_like(
                p.retrofit_storage_annuity_cost_energy, rate
            ),
        )

    m_cheap = build_model(_priced(1.0))
    m_cheap.solve(solver_name="highs")
    m_dear = build_model(_priced(2.0))
    m_dear.solve(solver_name="highs")

    assert m_cheap.status == "ok" and m_dear.status == "ok"

    for model in (m_cheap, m_dear):
        assert float(
            model.solution.planning_retrofit_storage_power_capacity.sum()
        ) == pytest.approx(100.0, abs=1e-6)
        assert float(
            model.solution.planning_retrofit_storage_energy_capacity.sum()
        ) == pytest.approx(300.0, abs=1e-6)

    assert m_dear.objective.value - m_cheap.objective.value == (
        pytest.approx(800.0, abs=1e-6)
    )


def test_storage_perfect_foresight_flag_changes_the_cost_basis(
    p_retrofit_storage_value,
):
    """
    Without the ``other`` branch of
    ``planning_retrofit_storage_costs_def``, the block would always
    charge the vintage-specific annuity no matter what
    ``retrofit_storage_annuity_perfect_foresight`` says -- the two
    builds below would then be byte-identical and their objectives
    equal.

    Built on ``p_retrofit_storage_value``, whose transfers are pinned to
    100 (power) and 300 (energy) by equal min/max bounds and whose slice
    has exactly one live transfer cell per kind, ``(year_inv=2020,
    year_ret=2030, year_dec=2050)``. Because
    ``retrofit_storage_early_decommissioning`` is ``False``, the mask
    pins ``year_dec`` to ``year_ret + life_span = 2050`` exactly, so
    there is never a genuine choice of ``year_dec`` for the solver to
    make: the only thing the flag can change is which rate is charged on
    that one forced cell.

    The default annuity tables would make the two branches coincide by
    construction -- built from ``square_array_by_diagonals(6, {0: 1/20,
    1: 1/10})``, they put their lowest rate at exactly ``year_dec =
    year_ret + life_span``, the very cell the mask forces. So both
    tables are overridden here to put the *expensive* rate on the forced
    cell (``year_dec=2050``: 30 for power, 20 for energy) and a cheaper
    one on ``year_dec=2030`` (15 and 10), a cell the mask never allows
    (it fails ``year_ret < year_dec``). ON must pay the forced cell's
    rate; OFF is free to charge the cheaper one the model can never
    physically realise.

    Both branches charge over the same window here (the forced
    ``year_dec`` and ``retrofit_storage_end_of_life`` are both 2050),
    i.e. ``year_op`` in ``{2030, 2040}`` -- two operating years. The two
    kinds are given different rate gaps (15 for power, 10 for energy) so
    that a cost definition summing only one of them cannot land on the
    right total. The objectives must differ by exactly
    ``(100 * 15 + 300 * 10) * 2 = 9000.0``.
    """
    p = p_retrofit_storage_value.copy(deep=True)
    for kind, forced, cheapest in (
        ("power", 30.0, 15.0),
        ("energy", 20.0, 10.0),
    ):
        name = f"retrofit_storage_annuity_cost_{kind}"
        p[name] = xr.full_like(p[name], np.nan)
        p[name].loc[dict(year_ret=2030, year_dec=2050)] = forced
        p[name].loc[dict(year_ret=2030, year_dec=2030)] = cheapest

    on = p.assign(
        retrofit_storage_annuity_perfect_foresight=(
            xr.full_like(p.retrofit_storage_annuity_perfect_foresight, True)
        ).astype(bool)
    )
    off = p.assign(
        retrofit_storage_annuity_perfect_foresight=(
            xr.full_like(p.retrofit_storage_annuity_perfect_foresight, False)
        ).astype(bool)
    )

    m_on = build_model(on)
    m_on.solve(solver_name="highs")
    m_off = build_model(off)
    m_off.solve(solver_name="highs")

    assert m_on.status == "ok" and m_off.status == "ok"

    for model in (m_on, m_off):
        assert float(
            model.solution.planning_retrofit_storage_power_capacity.sum()
        ) == pytest.approx(100.0, abs=1e-6)
        assert float(
            model.solution.planning_retrofit_storage_energy_capacity.sum()
        ) == pytest.approx(300.0, abs=1e-6)

    assert m_on.objective.value - m_off.objective.value == (
        pytest.approx(9000.0, abs=1e-6)
    )


@pytest.fixture()
def p_retrofit_transport(parameters_retrofit_transport):
    return parameters_retrofit_transport.sel(
        area=["area_1", "area_2"],
        link=["link_1", "link_2"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030],
        year_ret=[2030, 2040],
    ).copy(deep=True)


def test_transport_retrofit_is_link_indexed(p_retrofit_transport):
    """
    Structural: transport plans on ``link``, not ``area`` -- the
    generic block's transfer variable must inherit that from
    ``TransportRetrofitSpec.site_dim`` rather than defaulting to the
    ``area`` every other host uses.
    """
    model = build_model(p_retrofit_transport)
    var = model.variables["planning_retrofit_transport_power_capacity"]
    assert "link" in var.dims
    assert "area" not in var.dims


def test_transport_retrofit_costs_are_area_indexed(p_retrofit_transport):
    """
    Costs must reach ``area`` so they can join
    ``annualised_totex_def``, which is area-indexed -- exactly as
    ``planning_transport_costs`` is, even though transport's own
    capacity is link-indexed.
    """
    model = build_model(p_retrofit_transport)
    costs = model.variables["planning_retrofit_transport_costs"]
    assert "area" in costs.dims
    assert "link" not in costs.dims


@pytest.fixture()
def p_retrofit_transport_value(parameters_retrofit_transport):
    """
    A single link (link_1, connecting area_1 and area_2) with its
    retrofit transfer pinned to a known value, so the endpoint cost
    split can be checked against a number derived analytically rather
    than only against variable dimensions.

    Construction (the same pinning idiom as ``p_retrofit_storage_value``
    above, applied to transport's own
    ``transport_power_capacity_investment_*`` host bounds):
    - methane_pipe's own power capacity on link_1 is pinned to exactly
      10.0 at year_inv=2020 via equal min/max bounds on the HOST
      module, and to 0 at year_inv=2030 (present only so
      year_ret=2030 is a subset of year_inv, as chaining requires).
    - year_dec is narrowed to {2030, 2050}, excluding methane_pipe's
      natural end of life (2020 + transport_life_span 20 = 2040). With
      transport_early_decommissioning=True this leaves 2030 as the
      only valid decommissioning year for that vintage -- early
      decommissioning is forced by the grid, not chosen for cost
      reasons.
    - The retrofit's own transfer is pinned to exactly the same 10.0
      via equal min/max bounds on the retrofit's own investment
      bound, which the source cap allows exactly: ``available`` at
      year_ret=2030 is the 10.0 directly invested and decommissioned
      there, with eligible_share=1.0.
    - retrofit_transport_end_of_life = year_ret (2030) + life_span
      (20) = 2050, so with
      retrofit_transport_early_decommissioning=False only
      year_dec=2050 is a valid decommissioning year for the
      retrofit's own transfer.
    - Demand is zeroed throughout, so nothing but the pinned bounds
      above drives investment.
    """
    p = parameters_retrofit_transport.sel(
        area=["area_1", "area_2"],
        link=["link_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030, 2040],
        year_ret=[2030],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["transport_early_decommissioning"] = np.array(True, dtype="bool")

    value = 10.0
    bound = xr.DataArray(
        np.full((p.transport_tech.size, p.year_inv.size), np.nan),
        dims=["transport_tech", "year_inv"],
        coords={
            "transport_tech": p.transport_tech,
            "year_inv": p.year_inv,
        },
    )
    bound.loc[dict(transport_tech="methane_pipe", year_inv=2020)] = value
    bound.loc[dict(transport_tech="methane_pipe", year_inv=2030)] = 0.0
    p["transport_power_capacity_investment_max"] = bound
    p["transport_power_capacity_investment_min"] = bound.copy()

    p["retrofit_transport_power_capacity_investment_max"] = xr.full_like(
        p.retrofit_transport_power_capacity_investment_max, value
    )
    p["retrofit_transport_power_capacity_investment_min"] = xr.full_like(
        p.retrofit_transport_power_capacity_investment_min, value
    )
    return p


def test_transport_retrofit_cost_is_split_across_endpoints(
    p_retrofit_transport_value,
):
    """
    Each link's retrofit cost lands half on each endpoint area, not
    wholly on one -- see ``p_retrofit_transport_value`` for the
    pinning that forces the transfer to exactly 10.0.

    Analytical derivation of the expected per-area cost:
    - ``retrofit_transport_invest_cost`` is 300, and the shared
      fixture helper applies the annuity diagonal {0: 1/20, 1: 1/10}
      to it, giving, at year_ret=2030, values of 300/10=30.0
      (year_dec=2040) and 300/20=15.0 (year_dec=2050) across
      year_dec. With
      ``retrofit_transport_annuity_perfect_foresight=False`` the cost
      definition picks the MINIMUM of those for the year_ret=2030
      slice, 15.0, for every operating year the transfer is
      available -- independent of which year_dec is actually forced.
    - The transfer itself is pinned to 10.0 (see fixture), so the
      undivided link cost per qualifying operating year is
      10.0 * 15.0 = 150.0.
    - ``fold_costs`` halves that and assigns it to each of link_1's
      two endpoint areas (area_1 and area_2): 150.0 * 0.5 = 75.0 each.
    - The transfer is available from year_ret=2030 onward and until
      its own year_dec=2050, so the qualifying window is
      year_op in {2030, 2040} -- year_op=2020 is before year_ret and
      pays nothing.

    A ``fold_costs`` that does not split (the identity used by every
    area-sited module) would fail this differently from how it looks:
    the identity leaves the expression link-indexed, and combining a
    link-indexed expression with the area-indexed ``costs`` variable
    broadcasts the full, unhalved 150.0 onto BOTH areas instead of
    75.0 each -- see the task report for confirmation that disabling
    the split makes this exact assertion fail.
    """
    model = build_model(p_retrofit_transport_value)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    transferred = float(
        s.planning_retrofit_transport_power_capacity.sum()
    )
    assert transferred == pytest.approx(10.0, abs=1e-6)  # pinned

    costs = s.planning_retrofit_transport_costs.sum(
        "retrofit_transport_pair"
    )
    for area in ("area_1", "area_2"):
        assert float(
            costs.sel(area=area, year_op=2020)
        ) == pytest.approx(0.0, abs=1e-6)
        assert float(
            costs.sel(area=area, year_op=2030)
        ) == pytest.approx(75.0, abs=1e-6)
        assert float(
            costs.sel(area=area, year_op=2040)
        ) == pytest.approx(75.0, abs=1e-6)


def _transport_bound(p, per_tech):
    """A ``[transport_tech, year_inv]`` bound array from a per-tech map."""
    bound = xr.DataArray(
        np.full((p.transport_tech.size, p.year_inv.size), np.nan),
        dims=["transport_tech", "year_inv"],
        coords={
            "transport_tech": p.transport_tech,
            "year_inv": p.year_inv,
        },
    )
    for tech, values in per_tech.items():
        for year_inv, value in zip(p.year_inv.values, values):
            bound.loc[dict(transport_tech=tech, year_inv=year_inv)] = value
    return bound


def _transport_floor(p, per_tech_year_op):
    """A ``[transport_tech, year_op]`` operating floor, NaN where unset."""
    floor = xr.DataArray(
        np.full((p.transport_tech.size, p.year_op.size), np.nan),
        dims=["transport_tech", "year_op"],
        coords={
            "transport_tech": p.transport_tech,
            "year_op": p.year_op,
        },
    )
    for (tech, year_op), value in per_tech_year_op.items():
        floor.loc[dict(transport_tech=tech, year_op=year_op)] = value
    return floor


@pytest.fixture()
def p_retrofit_transport_multi_vintage(parameters_retrofit_transport):
    """
    Three live transfer cells, so a bound keyed by ``year_ret`` alone
    can tell a per-vintage implementation apart from an aggregated one
    -- the transport counterpart of
    ``p_retrofit_combined_multi_vintage``, on the same coordinate slice:
    ``year_inv=[2020, 2030, 2040]``, ``year_dec=[2030, 2040, 2050,
    2060]``, ``year_op=[2020, 2030, 2040, 2050]``, ``year_ret=[2030,
    2040]``, with ``transport_early_decommissioning=True``, transport
    life span 20 and ``retrofit_transport_early_decommissioning`` at its
    fixture default of ``False``. The three cells are

    - ``a = (year_inv=2020, year_ret=2030, year_dec=2050)``
    - ``b = (year_inv=2020, year_ret=2040, year_dec=2060)``
    - ``c = (year_inv=2030, year_ret=2040, year_dec=2060)``

    ``b`` and ``c`` share ``year_ret`` and differ only in ``year_inv``.

    Everything is planned on ``link_1`` alone -- transport's site
    dimension is ``link``, not ``area``, and this is the one host where
    the transfer variable and the assertions below are link-indexed.
    ``area`` still carries both of ``link_1``'s endpoints
    (``transport_area_from='area_1'``, ``transport_area_to='area_2'``)
    because the module's costs fold onto areas.

    ``methane_pipe``'s investment is pinned by equal min/max bounds,
    deliberately generously so the source cap is slack at the bound
    values the tests use and it is the investment bound that binds:
    ``year_inv=2020 -> 10``, ``2030 -> 4``, ``2040 -> 0``.
    ``big_methane_pipe`` (the pair's target) and ``power_line`` are
    pinned to zero at every ``year_inv``, so every unit of
    ``big_methane_pipe`` capacity that ever exists arrived there by
    retrofit, with no host vintage of its own to disentangle.

    Demand is zeroed: every test built on this fixture drives the
    transfer through capacity constraints alone, never through flows.
    """
    p = parameters_retrofit_transport.sel(
        area=["area_1", "area_2"],
        link=["link_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050, 2060],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030, 2040, 2050],
        year_ret=[2030, 2040],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["transport_early_decommissioning"] = np.array(True, dtype="bool")

    bound = _transport_bound(
        p,
        {
            "methane_pipe": [10.0, 4.0, 0.0],
            "big_methane_pipe": [0.0, 0.0, 0.0],
            "power_line": [0.0, 0.0, 0.0],
        },
    )
    p["transport_power_capacity_investment_max"] = bound
    p["transport_power_capacity_investment_min"] = bound.copy()
    return p


def _transport_per_vintage_at(solution, year_inv, year_ret):
    per_vintage = solution[
        "planning_retrofit_transport_power_capacity"
    ].sum("year_dec")
    return float(
        per_vintage.sel(
            link="link_1",
            retrofit_transport_pair="methane_upgrade",
            year_inv=year_inv,
            year_ret=year_ret,
        )
    )


def test_retrofit_transport_maximum_bound_is_per_vintage_not_aggregate(
    p_retrofit_transport_multi_vintage,
):
    """
    The maximum is keyed by ``year_ret`` alone (2030 -> 3.0, 2040 ->
    2.0), so the two ``year_ret=2040`` cells ``b`` and ``c`` -- which
    differ only in ``year_inv`` -- must each be allowed to reach 2.0
    independently (4.0 combined), not pooled against a single 2.0.

    The transfers are pulled up by a floor on ``big_methane_pipe``'s
    *operating* capacity
    (``operation_transport_power_capacity_min_constraint``), not by
    flows: since ``big_methane_pipe``'s own investment is pinned to
    zero (see the fixture), its operating capacity on ``link_1`` in any
    year is exactly the sum of the retrofit contributions whose arrival
    window covers that year. ``ret_ratio`` is 1.0 (fixture default), so
    a unit transferred is a unit arriving.

    Arithmetic:

    - At ``year_op=2050``, only ``b`` and ``c`` have arrived: their
      window is ``[2040, 2060)``, while ``a``'s ``[2030, 2050)``
      excludes 2050. A floor of 4.0 forces ``b + c = 4.0``, and with
      each capped at 2.0 that pins ``b = c = 2.0``.
    - At ``year_op=2040``, all three have arrived, so operating
      capacity is ``a + b + c``. A floor of 7.0 then forces
      ``a = 7 - 4 = 3.0``, exactly its own maximum.

    What a wrong implementation produces instead: pooling the two
    ``year_ret=2040`` cells against one 2.0 cap makes ``b + c <= 2.0``,
    so the 4.0 floor at ``year_op=2050`` cannot be met at all and the
    solve comes back infeasible rather than ``ok``.

    Asserting on the link-indexed transfer variable (rather than on
    anything area-indexed) is deliberate: transport is the one host
    whose capacity is planned per link while its costs fold onto areas,
    so a bound applied on the wrong index would show up here.
    """
    p = p_retrofit_transport_multi_vintage.copy(deep=True)
    p["transport_power_capacity_min"] = _transport_floor(
        p,
        {
            ("big_methane_pipe", 2040): 7.0,
            ("big_methane_pipe", 2050): 4.0,
        },
    )
    p["retrofit_transport_power_capacity_investment_max"] = xr.DataArray(
        [3.0, 2.0],
        dims=["year_ret"],
        coords={"year_ret": p.year_ret.values},
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    assert _transport_per_vintage_at(s, 2020, 2030) == pytest.approx(
        3.0, abs=1e-6
    )
    assert _transport_per_vintage_at(s, 2020, 2040) == pytest.approx(
        2.0, abs=1e-6
    )
    assert _transport_per_vintage_at(s, 2030, 2040) == pytest.approx(
        2.0, abs=1e-6
    )


def test_retrofit_transport_minimum_bound_is_per_vintage_not_aggregate(
    p_retrofit_transport_multi_vintage,
):
    """
    Same three live cells and the same rationale as the maximum-bound
    test above, with the minimum keyed by ``year_ret`` alone (2030 ->
    2.0, 2040 -> 3.0): a wrong-axis implementation pools the two
    ``year_ret=2040`` cells into a single ``>= 3.0`` requirement on
    their sum, satisfiable with one of them left at zero.

    No operating floor is needed here, and none is set: the investment
    minimum is a hard ``>=`` the solver must satisfy regardless of
    cost, and every transfer costs a positive annuity (plus a
    ``transport_fixed_cost`` charge on the operating capacity it
    creates), so the solver settles exactly on the bound for each live
    cell if -- and only if -- the bound is enforced per-vintage.

    The minima are reachable under the source cap: ``a >= 2`` and
    ``b >= 3`` draw on ``h(2020,2030)`` and ``h(2020,2040)``, which sum
    to the fixture's pinned 10, and ``c >= 3`` draws on
    ``h(2030,2040) <= 4``.
    """
    p = p_retrofit_transport_multi_vintage.copy(deep=True)
    p["retrofit_transport_power_capacity_investment_min"] = xr.DataArray(
        [2.0, 3.0],
        dims=["year_ret"],
        coords={"year_ret": p.year_ret.values},
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    assert _transport_per_vintage_at(s, 2020, 2030) >= 2.0 - 1e-6
    assert _transport_per_vintage_at(s, 2020, 2040) >= 3.0 - 1e-6
    assert _transport_per_vintage_at(s, 2030, 2040) >= 3.0 - 1e-6


@pytest.fixture()
def p_retrofit_transport_forced(parameters_retrofit_transport):
    """
    A slice in which retrofit is the ONLY route to
    ``big_methane_pipe`` capacity on ``link_1`` at ``year_op=2040``,
    with ``methane_pipe``'s own investment left free so the source-cap
    constraint has something to bite on.

    Construction:
    - ``year_dec=[2030, 2050]``, ``year_inv=[2020, 2030]``,
      ``year_op=[2020, 2030, 2040]``, ``year_ret=[2030]``. With
      ``transport_early_decommissioning=True`` and life span 20 the
      host's live ``(year_inv, year_dec)`` cells are ``(2020, 2030)``
      (early: 2030 is at or before the 2040 end of life, and no later
      than ``year_inv.max()``) and ``(2030, 2050)`` (its natural end of
      life), and the transfer mask leaves exactly one cell alive,
      ``(year_inv=2020, year_ret=2030, year_dec=2050)``, backed by
      ``h(2020, 2030)``.
    - ``methane_pipe``'s investment at ``year_inv=2020`` is left free
      (max 1000, no minimum), so how much backing capacity exists is
      the solver's own choice -- precisely what the source cap governs
      -- and is pinned to 0 at ``year_inv=2030`` so no later vintage
      can back anything.
    - ``big_methane_pipe`` and ``power_line`` are pinned to zero
      investment throughout, so ``big_methane_pipe``'s operating
      capacity is entirely the retrofit's arriving contribution.
    - ``transport_power_capacity_min`` places a floor of 4.0 on
      ``big_methane_pipe``'s operating capacity at ``year_op=2040``.
    - Demand is zeroed, so nothing but that floor drives the solve.
    """
    p = parameters_retrofit_transport.sel(
        area=["area_1", "area_2"],
        link=["link_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2050],
        year_inv=[2020, 2030],
        year_op=[2020, 2030, 2040],
        year_ret=[2030],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["transport_early_decommissioning"] = np.array(True, dtype="bool")

    p["transport_power_capacity_investment_max"] = _transport_bound(
        p,
        {
            "methane_pipe": [1000.0, 0.0],
            "big_methane_pipe": [0.0, 0.0],
            "power_line": [0.0, 0.0],
        },
    )
    p["transport_power_capacity_investment_min"] = _transport_bound(
        p,
        {
            "methane_pipe": [np.nan, 0.0],
            "big_methane_pipe": [0.0, 0.0],
            "power_line": [0.0, 0.0],
        },
    )
    p["transport_power_capacity_min"] = _transport_floor(
        p, {("big_methane_pipe", 2040): 4.0}
    )
    return p


def _built_transport(solution, tech):
    return float(
        solution.planning_transport_power_capacity.sel(
            link="link_1", transport_tech=tech
        ).sum()
    )


def test_retrofit_transport_cannot_exceed_the_source_capacity(
    p_retrofit_transport_forced,
):
    """
    A transfer must be backed by ``methane_pipe`` capacity on the same
    link, actually invested and decommissioned at the retrofit year --
    see ``p_retrofit_transport_forced`` for the slice that makes
    retrofit the only way to meet a 4.0 capacity floor at
    ``year_op=2040``.

    Derivation: the floor forces the single live transfer to exactly
    4.0 (``ret_ratio`` is 1.0), and the source cap then forces
    ``h(2020, 2030) >= 4.0`` on ``link_1``. Nothing rewards building
    more -- ``methane_pipe``'s annuity is strictly positive, and every
    unit of operating capacity also pays ``transport_fixed_cost`` --
    so the solver invests exactly 4.0.

    Both quantities are read link-indexed, which matters here: the cap
    is the one retrofit constraint that must line up the transfer with
    host capacity on transport's *site* dimension, while the module's
    costs are folded onto ``area``.

    What the constraint is worth here, and the reversion that proves
    it: commenting out the ``retrofit_transport_power_cap_constraint``
    (the ``leaving <= eligible_share * available`` block in
    ``_core.py``) leaves the floor satisfiable by an unbacked transfer.
    Retrofit is the cheaper of the two routes -- charged over fewer
    operating years than a ``methane_pipe`` vintage that must be built
    first and then converted -- so with the cap gone the solver
    conjures the whole 4.0 out of thin air and builds no
    ``methane_pipe`` at all: ``built`` drops to 0.0 while
    ``transferred`` stays at 4.0, failing the assertions below.
    Restoring the file makes them pass again.
    """
    model = build_model(p_retrofit_transport_forced)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    transferred = float(
        s.planning_retrofit_transport_power_capacity.sum()
    )
    built = _built_transport(s, "methane_pipe")

    assert transferred > 0  # non-vacuous: retrofit is actually used
    assert transferred == pytest.approx(4.0, abs=1e-6)
    assert built == pytest.approx(4.0, abs=1e-6)
    assert transferred <= built + 1e-6


def test_retrofit_transport_pinned_by_equal_min_and_max_bounds(
    p_retrofit_transport_forced,
):
    """
    Exercises the ``==`` branch
    (``planning_retrofit_transport_power_capacity_def``), which the
    maximum/minimum tests above never touch -- their bounds always
    differ, routing through ``<=``/``>=`` exclusively.

    ``p_retrofit_transport_forced``'s own optimum is 4.0 of transfer,
    exactly the capacity floor at ``year_op=2040`` and no more (see
    ``test_retrofit_transport_cannot_exceed_the_source_capacity``);
    every extra unit would pay a retrofit annuity and drag a matching
    unit of ``methane_pipe`` investment along with it through the
    source cap. Pinning min and max to 6.0 -- *above* that natural
    optimum -- forces 6.0 of transfer, 6.0 of backing investment, and
    an operating capacity for ``big_methane_pipe`` at ``year_op=2040``
    that overshoots its own floor and lands on 6.0. None of those is a
    figure the optimiser would choose on its own, so the equality
    constraint is what produces them.
    """
    p = p_retrofit_transport_forced.copy(deep=True)
    for which in ("min", "max"):
        name = f"retrofit_transport_power_capacity_investment_{which}"
        p[name] = xr.full_like(p[name], 6.0)

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    transferred = float(
        s.planning_retrofit_transport_power_capacity.sum()
    )
    assert transferred == pytest.approx(6.0, abs=1e-6)
    assert _built_transport(s, "methane_pipe") == pytest.approx(
        6.0, abs=1e-6
    )
    assert float(
        s.operation_transport_power_capacity.sel(
            link="link_1", transport_tech="big_methane_pipe", year_op=2040
        )
    ) == pytest.approx(6.0, abs=1e-6)


@pytest.fixture()
def p_retrofit_transport_chain(parameters_retrofit_transport):
    """
    A two-hop chain, on a second, self-targeting pair: pair
    ``methane_upgrade`` (``methane_pipe -> big_methane_pipe``, as in
    every other transport fixture here) and pair ``big_pipe_self``
    (``big_methane_pipe -> big_methane_pipe``), letting a pipe that has
    already been upgraded be retrofitted again -- relined -- at its own
    end of life.

    Timeline, chosen so hop 1's arrival window ends strictly before the
    study's last operating year and only hop 2's covers it. Grid:
    ``year_dec=[2030, 2050, 2070]``, ``year_inv=[2020, 2030, 2050]``,
    ``year_op=[2020, 2030, 2040, 2050]``, ``year_ret=[2030, 2050]``,
    with ``transport_early_decommissioning=True``, host life span 20
    and ``retrofit_transport_early_decommissioning`` at its fixture
    default of ``False`` (so each transfer's ``year_dec`` is pinned by
    exact equality to ``year_ret + 20``).

    - Hop 1 is pair 0 at ``(year_inv=2020, year_ret=2030,
      year_dec=2050)``: ``year_ret=2030`` is at or before
      ``methane_pipe``'s end of life for a 2020 vintage (2040), which
      early decommissioning permits, and that vintage's only live
      ``year_dec`` in this grid is 2030 (2050 exceeds its end of life),
      so it decommissions in full exactly at the retrofit year and can
      back the transfer. Arrival window ``[2030, 2050)``: covers
      ``year_op`` 2030 and 2040, never 2050.
    - Hop 2 is pair 1 at ``(year_inv=2030, year_ret=2050,
      year_dec=2070)``: ``year_ret=2050`` sits exactly at
      ``big_methane_pipe``'s end of life for a 2030 vintage. Arrival
      window ``[2050, 2070)``: covers ``year_op=2050``, the study's
      last.

    Chaining is what connects them: ``_chained_capacity`` relabels hop
    1's ``(year_ret=2030, year_dec=2050)`` as ``(year_inv=2030,
    year_ret=2050)``, exactly the cell hop 2's source cap reads for
    ``big_methane_pipe`` on ``link_1``.

    Every other route into ``big_methane_pipe`` capacity at
    ``year_op=2050`` is closed by pinning investment:

    - ``methane_pipe`` is pinned to 4.0 at ``year_inv=2020`` and to
      zero at 2030 and 2050. The zero at 2030 is what forces hop 2 onto
      the chained term: pair 0's own ``(year_inv=2030, year_ret=2050)``
      cell is structurally alive (2050 is exactly a 2030 vintage's end
      of life) but has no directly-invested ``methane_pipe`` behind it.
    - ``big_methane_pipe`` is pinned to zero investment throughout, so
      hop 2 has no directly-invested backing of its own and the
      study's last operating year cannot be covered by a fresh build.
    - ``power_line`` is pinned to zero throughout.

    A floor of 4.0 on ``big_methane_pipe``'s operating capacity at
    ``year_op=2050`` then forces hop 2, and through the chain hop 1, to
    exactly that amount. Demand is zeroed.
    """
    p = parameters_retrofit_transport.sel(
        area=["area_1", "area_2"],
        link=["link_1"],
        hour=[0],
        resource=["electricity", "heat", "hydrogen", "methane"],
        year_dec=[2030, 2050, 2070],
        year_inv=[2020, 2030, 2050],
        year_op=[2020, 2030, 2040, 2050],
        year_ret=[2030, 2050],
    ).copy(deep=True)
    p["demand"] = p.demand * 0
    p["transport_early_decommissioning"] = np.array(True, dtype="bool")

    bound = _transport_bound(
        p,
        {
            "methane_pipe": [4.0, 0.0, 0.0],
            "big_methane_pipe": [0.0, 0.0, 0.0],
            "power_line": [0.0, 0.0, 0.0],
        },
    )
    p["transport_power_capacity_investment_max"] = bound
    p["transport_power_capacity_investment_min"] = bound.copy()
    p["transport_power_capacity_min"] = _transport_floor(
        p, {("big_methane_pipe", 2050): 4.0}
    )

    pair = np.array(["methane_upgrade", "big_pipe_self"], dtype=str)
    p = p.drop_dims("retrofit_transport_pair")
    p = p.assign(
        retrofit_transport_tech_from=xr.DataArray(
            np.array(["methane_pipe", "big_methane_pipe"], dtype=str),
            dims="retrofit_transport_pair",
            coords={"retrofit_transport_pair": pair},
        ),
        retrofit_transport_tech_to=xr.DataArray(
            np.array(["big_methane_pipe", "big_methane_pipe"], dtype=str),
            dims="retrofit_transport_pair",
            coords={"retrofit_transport_pair": pair},
        ),
    )
    return p


def test_transport_capacity_acquired_by_retrofit_can_be_retrofitted_again(
    p_retrofit_transport_chain,
):
    """
    Capacity that arrived at ``big_methane_pipe`` through pair 0
    (``methane_pipe -> big_methane_pipe``) is itself retrofitted again
    through pair 1 (``big_methane_pipe -> big_methane_pipe``), which is
    only possible if the source-cap constraint's right-hand side
    carries the chained term (``_chained_capacity``) -- and, for
    transport specifically, only if that term is carried on ``link``
    rather than collapsed onto ``area``.

    Expected values, derived from ``p_retrofit_transport_chain``'s
    timeline rather than observed: the 4.0 floor on
    ``big_methane_pipe``'s operating capacity at ``year_op=2050`` can
    only be met by hop 2's arriving contribution (hop 1's window
    ``[2030, 2050)`` excludes 2050, and every host vintage is pinned
    either to zero or to a ``methane_pipe`` vintage that decommissions
    at 2030), so hop 2 must transfer 4.0 -- ``ret_ratio`` is 1.0, the
    fixture default. Hop 2's source cap offers exactly two terms:
    directly invested ``big_methane_pipe`` capacity, which the fixture
    pins to zero, and the chained term, which is hop 1's transfer. So
    hop 1 must also be 4.0 -- and it can be, since ``methane_pipe``'s
    2020 vintage is pinned to 4.0 and decommissions in full at
    ``year_ret=2030``. Both hops are pinned, with no slack in either
    direction.

    Reversion proof: dropping ``_chained_capacity`` from the source
    cap's right-hand side (returning only the directly-invested term)
    removes hop 2's only backing, forcing it to zero; the 4.0 floor at
    ``year_op=2050`` then has no way to be met and the solve comes back
    infeasible instead of ``ok``. Restoring ``_core.py`` makes it pass
    again.
    """
    model = build_model(p_retrofit_transport_chain)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    transfer = s.planning_retrofit_transport_power_capacity.sel(
        link="link_1"
    )
    hop_one = float(
        transfer.sel(
            retrofit_transport_pair="methane_upgrade",
            year_inv=2020,
            year_ret=2030,
        ).sum()
    )
    hop_two = float(
        transfer.sel(
            retrofit_transport_pair="big_pipe_self",
            year_inv=2030,
            year_ret=2050,
        ).sum()
    )

    assert hop_two > 0  # non-vacuous: the second hop actually fires
    assert hop_one == pytest.approx(4.0, abs=1e-6)
    assert hop_two == pytest.approx(4.0, abs=1e-6)


def test_retrofit_transport_cost_enters_the_objective(
    p_retrofit_transport_value,
):
    """
    Doubling the retrofit annuity must raise the objective by exactly
    the amount transferred times the change in rate times the number of
    operating years it is charged for. Without the cost constraint and
    the totex hook, retrofit would be free and the objective would not
    move at all.

    ``p_retrofit_transport_value`` pins the transfer to exactly 10.0 by
    equal min/max bounds, so the two solves differ in nothing but the
    annuity -- the quantity cannot shift to absorb the price change,
    which makes the difference exactly derivable rather than merely
    positive.

    Its slice has one live transfer cell, ``(year_inv=2020,
    year_ret=2030, year_dec=2050)`` on ``link_1``. With
    ``retrofit_transport_annuity_perfect_foresight`` at its fixture
    default of ``False``, the cost is charged for every ``year_op`` in
    ``[year_ret, retrofit_transport_end_of_life) = [2030, 2050)``; the
    fixture's ``year_op`` is ``{2020, 2030, 2040}``, so that is two
    operating years, 2030 and 2040.

    ``fold_costs`` then halves the link's cost onto each of
    ``link_1``'s two endpoint areas, so the two halves add back up to
    the undivided amount once the objective sums over ``area`` -- the
    split changes where the cost is booked, not how much of it reaches
    the objective. The objective must therefore rise by exactly
    ``10.0 * (2.0 - 1.0) * 2 * (0.5 + 0.5) = 20.0``.
    """
    p = p_retrofit_transport_value.copy(deep=True)

    cheap = p.assign(
        retrofit_transport_annuity_cost=(
            xr.full_like(p.retrofit_transport_annuity_cost, 1.0)
        )
    )
    dear = p.assign(
        retrofit_transport_annuity_cost=(
            xr.full_like(p.retrofit_transport_annuity_cost, 2.0)
        )
    )

    m_cheap = build_model(cheap)
    m_cheap.solve(solver_name="highs")
    m_dear = build_model(dear)
    m_dear.solve(solver_name="highs")

    assert m_cheap.status == "ok" and m_dear.status == "ok"

    for model in (m_cheap, m_dear):
        assert float(
            model.solution.planning_retrofit_transport_power_capacity.sum()
        ) == pytest.approx(10.0, abs=1e-6)

    assert m_dear.objective.value - m_cheap.objective.value == (
        pytest.approx(20.0, abs=1e-6)
    )


def test_transport_perfect_foresight_flag_changes_the_cost_basis(
    p_retrofit_transport_value,
):
    """
    Without the ``other`` branch of
    ``planning_retrofit_transport_costs_def``, the block would always
    charge the vintage-specific annuity no matter what
    ``retrofit_transport_annuity_perfect_foresight`` says -- the two
    builds below would then be byte-identical and their objectives
    equal.

    Built on ``p_retrofit_transport_value``, whose transfer is pinned
    to 10.0 by equal min/max bounds and whose slice has exactly one
    live transfer cell, ``(year_inv=2020, year_ret=2030,
    year_dec=2050)``. Because
    ``retrofit_transport_early_decommissioning`` is ``False``, the mask
    pins ``year_dec`` to ``year_ret + life_span = 2050`` exactly, so
    there is never a genuine choice of ``year_dec`` for the solver to
    make: the only thing the flag can change is which rate is charged
    on that one forced cell.

    The default annuity table would make the two branches coincide by
    construction -- built from ``square_array_by_diagonals(6, {0: 1/20,
    1: 1/10})``, it puts its lowest rate at exactly ``year_dec =
    year_ret + life_span``, the very cell the mask forces (this is why
    ``test_transport_retrofit_cost_is_split_across_endpoints`` sees
    15.0 either way). So the table is overridden here to put the
    *expensive* rate (30) on the forced cell (``year_dec=2050``) and a
    cheaper one (15) on ``year_dec=2030``, a cell the mask never allows
    (it fails ``year_ret < year_dec``). ON must pay the forced cell's
    30; OFF is free to charge the 15 the model can never physically
    realise.

    Both branches charge over the same window here (the forced
    ``year_dec`` and ``retrofit_transport_end_of_life`` are both 2050),
    i.e. ``year_op`` in ``{2030, 2040}`` -- two operating years, and
    the 50/50 endpoint split adds back to a whole once the objective
    sums over ``area``. With the transfer pinned at 10.0 the objectives
    must differ by exactly ``10.0 * (30 - 15) * 2 = 300.0``.
    """
    p = p_retrofit_transport_value.copy(deep=True)
    p["retrofit_transport_annuity_cost"] = xr.full_like(
        p.retrofit_transport_annuity_cost, np.nan
    )
    p["retrofit_transport_annuity_cost"].loc[
        dict(year_ret=2030, year_dec=2050)
    ] = 30.0
    p["retrofit_transport_annuity_cost"].loc[
        dict(year_ret=2030, year_dec=2030)
    ] = 15.0

    on = p.assign(
        retrofit_transport_annuity_perfect_foresight=(
            xr.full_like(
                p.retrofit_transport_annuity_perfect_foresight, True
            )
        ).astype(bool)
    )
    off = p.assign(
        retrofit_transport_annuity_perfect_foresight=(
            xr.full_like(
                p.retrofit_transport_annuity_perfect_foresight, False
            )
        ).astype(bool)
    )

    m_on = build_model(on)
    m_on.solve(solver_name="highs")
    m_off = build_model(off)
    m_off.solve(solver_name="highs")

    assert m_on.status == "ok" and m_off.status == "ok"

    for model in (m_on, m_off):
        assert float(
            model.solution.planning_retrofit_transport_power_capacity.sum()
        ) == pytest.approx(10.0, abs=1e-6)

    assert m_on.objective.value - m_off.objective.value == (
        pytest.approx(300.0, abs=1e-6)
    )
