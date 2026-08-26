"""Retrofit behaviour on the conversion module."""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model


@pytest.fixture()
def p_retrofit(parameters_retrofit_conversion):
    # year_inv includes 2040 solely so year_ret (2030, 2040) is a
    # subset of it, as chaining requires (see _check_year_ret_grid).
    # year_op tops out at 2030, so a vintage at year_inv=2040 could
    # never be operated regardless -- this is inert widening, not a
    # new investment option.
    return parameters_retrofit_conversion.sel(
        conversion_tech=["smr", "smr_ccs"],
        area=["area_1"],
        hour=[0],
        resource=["electricity", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030],
        year_ret=[2030, 2040],
    ).copy(deep=True)


def test_retrofit_block_builds_and_declares_its_variables(p_retrofit):
    model = build_model(p_retrofit)

    var = model.variables["planning_retrofit_conversion_power_capacity"]
    assert set(var.dims) == {
        "area",
        "retrofit_conversion_pair",
        "year_inv",
        "year_ret",
        "year_dec",
    }
    assert "planning_retrofit_conversion_costs" in model.variables


def test_year_ret_outside_the_year_inv_grid_is_rejected(p_retrofit):
    """
    The chained term is formed by realigning year_ret onto year_inv. A
    year_ret value absent from year_inv would make the chain silently
    evaluate to zero, so it must raise instead.
    """
    p = p_retrofit.copy(deep=True)
    p = p.assign_coords(year_ret=[2035, 2045])

    with pytest.raises(ValueError, match="year_ret"):
        build_model(p)


def test_year_ret_outside_the_year_dec_grid_is_rejected(p_retrofit):
    """
    The source-cap constraint's ``available`` term reads ``host_var.
    where(p.year_ret == p.year_dec)``: the host's own capacity that
    decommissions exactly at the retrofit year, backing the transfer.
    ``year_inv`` and ``year_dec`` are independent coordinates, so
    ``year_ret`` being a subset of ``year_inv`` (guarded above) says
    nothing about whether it is also a subset of ``year_dec`` -- this
    is realistic whenever the decommissioning grid is coarser than the
    investment grid.

    Grid: ``year_inv=[2020, 2030, 2040]`` (``p_retrofit``'s own, so the
    ``year_inv`` guard passes untouched), ``year_dec`` narrowed from
    ``p_retrofit``'s ``[2030, 2040, 2050]`` to ``[2040, 2050]`` (2030
    dropped), ``year_ret=[2030, 2040]`` (``p_retrofit``'s own,
    unchanged). ``year_ret=2030`` is absent from this narrowed
    ``year_dec``, so ``year_ret == year_dec`` holds nowhere for that
    retrofit year: every live transfer cell there would find zero
    host-capacity terms on the right-hand side of its cap constraint,
    making retrofit look structurally unprofitable rather than
    ill-specified. Confirmed against the pre-fix guard (which checks
    only ``year_ret`` against ``year_inv``): this exact grid builds
    without raising, silently accepting the ill-specified study.
    """
    p = p_retrofit.sel(year_dec=[2040, 2050]).copy(deep=True)

    with pytest.raises(ValueError, match="year_dec"):
        build_model(p)


def test_retrofit_absent_when_flag_is_off(p_retrofit):
    p = p_retrofit.copy(deep=True)
    p["retrofit_conversion"] = np.array(False, dtype="bool")
    model = build_model(p)
    assert (
        "planning_retrofit_conversion_power_capacity"
        not in model.variables
    )


@pytest.fixture()
def p_mask(parameters_retrofit_conversion):
    # year_inv includes 2040 for the same reason as in p_retrofit: it
    # is needed only so year_ret is a subset of year_inv. Both
    # early-decommissioning flags default to False here, so the extra
    # value cannot affect any assertion below (the
    # "year_dec <= year_inv.max()" branch is only reached when early
    # decommissioning is on).
    # year_dec includes 2020 for the same structural reason, now that
    # _check_year_ret_grid also requires year_ret to be a subset of
    # year_dec: without it, year_ret=2020 (used below, in cells that
    # are masked out for other reasons regardless) would have no
    # year_dec counterpart at all. The two cells that use it,
    # (year_inv=2030, year_ret=2020, year_dec=2040) and
    # (year_inv=2010, year_ret=2020, year_dec=2050), are unaffected:
    # neither reads year_dec=2020.
    return parameters_retrofit_conversion.sel(
        conversion_tech=["smr", "smr_ccs"],
        area=["area_1"],
        hour=[0],
        resource=["electricity", "hydrogen", "methane"],
        year_dec=[2020, 2030, 2040, 2050],
        year_inv=[2010, 2020, 2030, 2040],
        year_op=[2020, 2030],
        year_ret=[2020, 2030, 2040],
    ).copy(deep=True)


def test_transfer_mask_values(p_mask):
    """
    Which (year_inv, year_ret, year_dec) cells the retrofit variable
    actually allows, checked through linopy's ``labels`` (-1 marks a
    masked-out entry).

    Both the source technology (``conversion_early_decommissioning``)
    and the retrofit pair itself
    (``retrofit_conversion_early_decommissioning``) are ``False`` in
    the fixtures, so both end-of-life checks in ``_transfer_mask``
    collapse to exact equality rather than "on or before".

    From the fixtures: ``conversion_life_span`` = 20, so the host's
    ``conversion_end_of_life`` for ``smr`` (the pair's
    ``tech_from``) at a given ``year_inv`` is ``year_inv + 20``.
    ``retrofit_conversion_life_span`` = 20 too, so
    ``retrofit_conversion_end_of_life`` at a given ``year_ret`` is
    ``year_ret + 20``.

    Chosen cells:
      - ``(year_inv=2030, year_ret=2020, year_dec=2040)``: year_inv
        is not before year_ret, so a retrofit cannot precede the
        asset it converts -- masked out regardless of any end-of-life
        arithmetic.
      - ``(year_inv=2020, year_ret=2040, year_dec=2030)``: year_ret is
        not before year_dec, so the result cannot die before it is
        created -- masked out regardless of end-of-life arithmetic.
      - ``(year_inv=2010, year_ret=2020, year_dec=2050)``: year_inv <
        year_ret < year_dec all hold, and year_ret=2020 is *before*
        smr's end of life at year_inv=2010 (2010+20=2030) -- with
        early decommissioning off this must be masked out, since only
        exact equality is allowed. This is the case the ``==`` vs
        ``<=`` branch actually distinguishes.
      - ``(year_inv=2010, year_ret=2030, year_dec=2050)``: year_ret=
        2030 sits exactly at smr's end of life for year_inv=2010
        (2010+20=2030), and year_dec=2050 sits exactly at the
        result's own end of life for year_ret=2030 (2030+20=2050) --
        every gating condition holds, so this cell must be active.
    """
    model = build_model(p_mask)
    labels = model.variables[
        "planning_retrofit_conversion_power_capacity"
    ].labels

    def label_at(year_inv, year_ret, year_dec):
        return labels.sel(
            area="area_1",
            retrofit_conversion_pair="smr_to_ccs",
            year_inv=year_inv,
            year_ret=year_ret,
            year_dec=year_dec,
        ).item()

    assert label_at(2030, 2020, 2040) == -1
    assert label_at(2020, 2040, 2030) == -1
    assert label_at(2010, 2020, 2050) == -1
    assert label_at(2010, 2030, 2050) != -1


@pytest.fixture()
def p_retrofit_value(parameters_retrofit_conversion):
    """
    A variant of ``p_retrofit`` built to actually exercise the wiring
    under solve, not just structurally.

    Three departures from ``p_retrofit`` are needed, each for a
    concrete reason found while making the value tests below solvable:

    - ``ocgt`` is included alongside ``smr``/``smr_ccs``. ``smr_ccs``'s
      own ``conversion_factor`` draws a little electricity (its
      carbon-capture auxiliary load), and with only ``smr``/``smr_ccs``
      present there is no electricity producer at all, and load
      shedding is disallowed for electricity in the fixtures -- so
      ``smr_ccs`` could never actually run, no matter which technology
      supplied its capacity. ``ocgt`` (methane -> electricity) closes
      that loop using only methane, which methane load shedding
      supplies without limit.
    - ``conversion_early_decommissioning`` is ``True``, which relaxes
      the transfer mask's source-side end-of-life check from an exact
      match to "on or before" (see ``_transfer_mask``). With it left
      at the fixtures' default ``False``, every cell of the transfer
      variable in this coordinate slice is masked out -- there is no
      ``year_dec`` in range that lands exactly on the result's
      end-of-life for any unmasked ``(year_inv, year_ret)`` pair -- so
      the retrofit variable is structurally zero before cost even
      enters the picture.
    - ``smr``'s own ``conversion_power_capacity_max`` is capped at 3,
      below the 10 units of hydrogen demanded. This is what actually
      forces a choice between routes to the shortfall: build
      ``smr_ccs`` directly, or reach it by retrofitting ``smr``. Left
      uncapped, ``smr`` alone always meets the demand more cheaply
      than either route into ``smr_ccs``, and retrofit is never
      exercised regardless of how it is priced.
    """
    p = parameters_retrofit_conversion.sel(
        conversion_tech=["smr", "smr_ccs", "ocgt"],
        area=["area_1"],
        hour=[0],
        resource=["electricity", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030],
        year_ret=[2030, 2040],
    ).copy(deep=True)
    p["conversion_early_decommissioning"] = np.array(True, dtype="bool")
    p["conversion_power_capacity_max"] = (
        p.conversion_power_capacity_max
        * xr.ones_like(p.conversion_tech, dtype="float64")
    ).copy()
    p["conversion_power_capacity_max"].loc[
        dict(conversion_tech="smr")
    ] = 3.0
    # year_inv=2040 is added solely so year_ret (2030, 2040) is a
    # subset of year_inv, as chaining requires (_check_year_ret_grid).
    # Investment there is zeroed out for every technology: year_op
    # tops out at 2030, so a vintage invested at 2040 could never be
    # operated anyway, but zeroing it removes even the degenerate
    # possibility of an alternate-optimal solve.
    p["conversion_power_capacity_investment_max"] = (
        xr.full_like(p.conversion_power_capacity_investment_max, 1000.0)
        * xr.ones_like(p.conversion_tech, dtype="float64")
        * xr.ones_like(p.year_inv, dtype="float64")
    ).copy()
    p["conversion_power_capacity_investment_max"].loc[
        dict(year_inv=2040)
    ] = 0.0
    p["demand"] = p.demand * 0
    return p


def test_retrofit_capacity_reaches_the_target_technology(p_retrofit_value):
    """
    ``smr`` can supply at most 3 units of hydrogen (its operational cap),
    and demand is 10. Building ``smr_ccs`` directly is deliberately
    overpriced (``conversion_annuity_cost``, the parameter the model
    actually costs planning capacity with -- ``conversion_invest_cost``
    alone is pre-baked into a separate annuity table at fixture
    construction and is not re-derived from at solve time, so overriding
    it alone would price nothing).

    Retrofitting ``smr`` is not free capacity: the source-cap constraint
    (``retrofit_conversion_power_cap_constraint``) requires the transfer
    to be backed by ``smr`` actually invested and decommissioned at the
    retrofit year.
    In this fixture that backing investment is only reachable through an
    early decommissioning of the ``year_inv=2020`` vintage (the fixture's
    only live transfer cell decommissions ``smr`` at 2030, ten years
    before its natural 2040 end of life), and that backing investment
    counts toward ``smr``'s own 3-unit operational cap for as long as it
    is alive -- including the 2020 vintage year, when nothing else
    competes for the cap. So retrofit tops out at 3 (matching the cap,
    a hard capacity constraint independent of price), not at the full
    7-unit shortfall: 3 units reach ``smr_ccs`` by retrofit, a separate
    (naturally-timed) ``smr`` vintage supplies another 3 directly (its
    own 3-unit operational cap, met the cheap way), and the remaining
    ``10 - 3 - 3 = 4`` must be built as ``smr_ccs`` directly.
    """
    p = p_retrofit_value
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
    )
    p = p.assign(
        retrofit_conversion_annuity_cost=(
            xr.full_like(p.retrofit_conversion_annuity_cost, 0.1)
        )
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    retrofitted = float(
        s.planning_retrofit_conversion_power_capacity.sum()
    )
    direct_ccs = float(
        (s.planning_conversion_power_capacity)
        .sel(conversion_tech="smr_ccs")
        .sum()
    )
    assert retrofitted == pytest.approx(3.0, abs=1e-6)
    assert direct_ccs == pytest.approx(4.0, abs=1e-6)


def test_expensive_retrofit_is_not_used(p_retrofit_value):
    """
    Same shortfall and pricing as ``test_retrofit_capacity_reaches_the_
    target_technology`` -- demand 10, ``smr`` capped operationally at
    3, direct ``smr_ccs`` overpriced via ``conversion_annuity_cost``
    at 5000/yr -- except the retrofit annuity is raised to 1e6/yr, far
    above the 5000/yr direct-build alternative. Retrofit is still
    *structurally* available and would saturate at 3.0 if it were free
    or cheap (that test proves exactly this, at annuity 0.1); only its
    price must be what keeps it at zero here.

    This replaces an earlier version of this test that could not
    discriminate on price at all: it set demand to exactly ``smr``'s
    own cap (3), so nothing ever needed to reach ``smr_ccs``, and the
    assertion (``retrofitted == 0``) passed regardless of what the
    retrofit annuity was set to -- before this task, because retrofit
    cost was not wired into the objective at all, and even after,
    because no shortfall means retrofit buys nothing at any price. That
    version also used the banned ``p.X * 0 + value`` idiom, which is
    silently wrong wherever the underlying array holds ``nan``. Both
    problems are fixed here: the shortfall-forcing setup below gives
    retrofit something to offer, so ``retrofitted == 0`` is a genuine
    claim about price, not an artefact of there being no demand to
    fill; and the override uses ``xr.full_like``, which replaces every
    cell regardless of what it held.

    Break-even, derived: both routes reduce to "pay an annuity once,
    for the single ``year_op=2030`` the shortfall is needed at" (see
    ``test_retrofit_cost_enters_the_objective`` for why retrofit's
    window covers only that one operating year here). Direct build
    pays 5000/unit; retrofit at 1e6/unit is four orders of magnitude
    above that, leaving no room for the solver to prefer it even
    fractionally. The entire 7-unit shortfall (``10 - 3``, the same
    arithmetic as the reference test) must therefore be built directly.

    Reversion proof (recorded in the task report, not re-run here):
    removing the retrofit cost term from ``annualised_totex_def``
    makes this test fail -- retrofit reverts to 3.0, matching the
    free-retrofit optimum from
    ``test_retrofit_capacity_reaches_the_target_technology``.
    """
    p = p_retrofit_value.copy(deep=True)
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
    )
    p = p.assign(
        retrofit_conversion_annuity_cost=(
            xr.full_like(p.retrofit_conversion_annuity_cost, 1e6)
        )
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    retrofitted = float(
        s.planning_retrofit_conversion_power_capacity.sum()
    )
    direct_ccs = float(
        (s.planning_conversion_power_capacity)
        .sel(conversion_tech="smr_ccs")
        .sum()
    )
    assert retrofitted == pytest.approx(0.0, abs=1e-6)
    assert direct_ccs == pytest.approx(7.0, abs=1e-6)


@pytest.fixture()
def p_retrofit_multi_vintage(parameters_retrofit_conversion):
    """
    Same construction as ``p_retrofit_value``, widened to bring a
    second and third transfer vintage to life.

    ``p_retrofit_value``'s coordinate slice has exactly one live
    transfer cell: ``(year_inv=2020, year_ret=2030, year_dec=2050)``.
    Every other ``(year_inv, year_ret)`` combination is structurally
    alive under the mask but needs ``year_dec=2060`` to find its own
    end-of-life (``retrofit_conversion_life_span=20`` past
    ``year_ret=2040``), which ``p_retrofit_value`` excludes. Adding
    2060 brings two more to life: ``(year_inv=2020, year_ret=2040)``
    and ``(year_inv=2030, year_ret=2040)`` -- both landing on
    ``year_dec=2060``, and differing from each other only in
    ``year_inv``, and from the first vintage only in ``year_ret``.
    That covers both ways a bound could aggregate along the wrong axis.

    ``year_op=2040`` is added too: arriving capacity is only available
    for ``year_ret <= year_op < year_dec``, so without an operational
    year inside a vintage's window it stays structurally alive but is
    never actually usable by anything, which would make it useless for
    telling a per-vintage bound apart from an aggregated one.

    ``year_inv=2030`` must stay available (not just ``2020``) for a
    different reason: with a single ``year_inv`` value, every
    conversion technology's own end-of-life caps out at ``year_inv +
    life_span = 2040``, so nothing -- including ``ocgt``, which
    ``smr_ccs`` needs for its auxiliary electricity load -- could still
    be operating at ``year_op=2040``, and any positive demand there
    would be infeasible.

    Two more departures were added for the source-cap constraint
    (``retrofit_conversion_power_cap_constraint``), which requires an
    actual ``smr`` investment decommissioned at the retrofit year to
    back each transfer:

    - ``year_inv=2040`` is added so that ``year_inv=2030`` is no longer
      the latest investable vintage. The host module's own early-
      decommissioning mask only allows a vintage to retire before its
      natural end of life when some *later* investment year exists in
      the study (``year_dec <= year_inv.max()``); without ``2040``
      present, the ``(year_inv=2030, year_ret=2040)`` transfer cell
      would have no valid ``smr`` investment to back it at all, and the
      source-cap constraint would force that cell's retrofit to zero
      regardless of any bound placed on it.
    - ``smr``'s own operating cap is raised from 3.0 to 6.0. Backing
      investment counts toward this cap for as long as it is alive, and
      two of this fixture's three transfer cells -- ``(year_inv=2020,
      year_ret=2040)`` and ``(year_inv=2030, year_ret=2040)`` -- back
      onto overlapping years, so a cap of 3.0 cannot hold both open at
      once. 6.0 gives the tests below room to back every live cell
      simultaneously.
    """
    p = parameters_retrofit_conversion.sel(
        conversion_tech=["smr", "smr_ccs", "ocgt"],
        area=["area_1"],
        hour=[0],
        resource=["electricity", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050, 2060],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030, 2040],
        year_ret=[2030, 2040],
    ).copy(deep=True)
    p["conversion_early_decommissioning"] = np.array(True, dtype="bool")
    p["conversion_power_capacity_max"] = (
        p.conversion_power_capacity_max
        * xr.ones_like(p.conversion_tech, dtype="float64")
    ).copy()
    p["conversion_power_capacity_max"].loc[
        dict(conversion_tech="smr")
    ] = 6.0
    p["demand"] = p.demand * 0
    return p


def _per_vintage_at(solution, year_inv, year_ret):
    per_vintage = solution[
        "planning_retrofit_conversion_power_capacity"
    ].sum("year_dec")
    return float(
        per_vintage.sel(
            area="area_1",
            retrofit_conversion_pair="smr_to_ccs",
            year_inv=year_inv,
            year_ret=year_ret,
        )
    )


def test_retrofit_maximum_bound_applies_per_vintage_not_in_aggregate(
    p_retrofit_multi_vintage,
):
    """
    ``p_retrofit_multi_vintage`` has three live transfer cells:
    ``(year_inv=2020, year_ret=2030)``, ``(year_inv=2020,
    year_ret=2040)``, and ``(year_inv=2030, year_ret=2040)`` -- the
    last two share ``year_ret`` but differ in ``year_inv``. A maximum
    keyed only by ``year_ret`` (2030 -> 3.0, 2040 -> 2.0) lets a
    wrong-axis implementation hide in a single-vintage test: summing
    the transfer variable over ``year_inv`` before comparing would pool
    the two ``year_ret=2040`` cells against the same 2.0 cap, capping
    their *combined* total at 2.0 rather than letting each reach 2.0
    independently (4.0 combined).

    Demand is set high enough in both operational years that the solver
    wants to push every live vintage to its own cap rather than resort
    to a build directly priced via ``conversion_annuity_cost`` (the
    parameter the model actually costs planning capacity with --
    ``conversion_invest_cost`` alone is pre-baked into a separate
    annuity table at fixture construction and never re-derived from at
    solve time). Reaching each cap also requires an actual backing
    ``smr`` investment decommissioned at that cell's retrofit year (the
    source-cap constraint), which is why ``p_retrofit_multi_vintage``
    raises
    smr's own operating cap to 6.0: high enough that the source-cap
    constraint, not smr's site limit, is what keeps each cell at its
    bound. Retrofit-arrived capacity is added straight to
    ``operation_conversion_power_capacity_def`` and so carries none of
    ``smr_ccs``'s own invest annuity -- only its fixed/variable
    operating cost, unavoidable either way -- which is strictly
    cheaper than a direct build priced at 5000. So the solver always
    saturates each of the three explicit bounds (3.0 / 2.0 / 2.0)
    before touching direct build: those three assertions are pinned
    directly by the bounds, not merely observed.

    ``direct_ccs == 2.0`` takes one more step of arithmetic plus one
    cost-structure fact:

    - ``smr``'s own operating cap is exactly saturated at both
      ``year_op=2030`` and ``year_op=2040`` (``operation_conversion_
      power_capacity`` solves to 6.0 in both years), because direct
      ``smr`` output is the cheapest way to meet any of the demand, so
      the solver always wants the maximum of it.
    - The ``year_ret=2030`` cell's transfer decommissions at 2050, so
      its arrival window (``year_ret <= year_op < year_dec``) covers
      *both* remaining operational years: it contributes 3.0 to
      ``smr_ccs`` capacity at 2030 **and** at 2040. The two
      ``year_ret=2040`` cells only start arriving at 2040 (2.0 each).
      Retrofit-derived ``smr_ccs`` capacity therefore totals 3.0 at
      ``year_op=2030`` and 3.0 + 2.0 + 2.0 = 7.0 at ``year_op=2040``.
    - Net of ``smr`` (6.0) and retrofit (3.0 / 7.0), the direct-build
      shortfall is ``10 - 6 - 3 = 1.0`` at 2030 and
      ``15 - 6 - 7 = 2.0`` at 2040.
    - These fixtures cost planning capacity with non-perfect-foresight
      annuities (``conversion_annuity_perfect_foresight=False``, the
      default): ``planning_conversion_costs_def`` charges a vintage's
      annuity for every operational year within its *natural*
      end-of-life window, regardless of which ``year_dec`` is actually
      chosen for it -- so decommissioning a vintage early never
      reduces what it costs, only what it can still supply. A vintage
      invested at ``year_inv=2030`` naturally lives to 2050, already
      spanning both remaining operational years at no extra charge, so
      there is no benefit to under-building for 2030 and replacing the
      unit wholesale in 2040: that pays the same 2030-and-2040 charge
      on the retired unit while *also* funding a full-sized
      replacement, strictly more expensive. The cheapest way to cover
      a shortfall that grows from 1.0 to 2.0 is to size a single
      ``year_inv=2030`` vintage to the smaller, earlier shortfall
      (1.0) and let it persist, then top up the remaining 1.0 with a
      second vintage invested fresh at 2040. Total direct build is
      therefore forced to ``max(1.0, 2.0) = 2.0``.
    """
    p = p_retrofit_multi_vintage.copy(deep=True)
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
    p["demand"].loc[dict(resource="hydrogen", year_op=2040)] = 15.0
    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
    )
    p["retrofit_conversion_power_capacity_investment_max"] = xr.DataArray(
        [3.0, 2.0],
        dims=["year_ret"],
        coords={"year_ret": p.year_ret.values},
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    assert _per_vintage_at(s, 2020, 2030) == pytest.approx(3.0, abs=1e-6)
    assert _per_vintage_at(s, 2020, 2040) == pytest.approx(2.0, abs=1e-6)
    assert _per_vintage_at(s, 2030, 2040) == pytest.approx(2.0, abs=1e-6)

    direct_ccs = float(
        (s.planning_conversion_power_capacity)
        .sel(conversion_tech="smr_ccs")
        .sum()
    )
    assert direct_ccs == pytest.approx(2.0, abs=1e-6)


def test_retrofit_minimum_bound_applies_per_vintage_not_in_aggregate(
    p_retrofit_multi_vintage,
):
    """
    Same three live cells as the maximum-bound test above, and the
    same rationale for keying the bound by ``year_ret`` alone (2030 ->
    2.0, 2040 -> 3.0): the two ``year_ret=2040`` cells, which differ
    only in ``year_inv``, would be pooled by a wrong-axis
    implementation into a single ``>= 3.0`` requirement on their sum
    rather than ``>= 3.0`` on each -- satisfiable with one of them left
    at zero.

    No demand is needed to force this: the investment minimum is a hard
    ``>=`` constraint the solver must satisfy regardless of cost, and
    nothing here creates an incentive to invest beyond it, so the
    solver settles exactly on the bound for each live cell if -- and
    only if -- the bound is actually enforced per-vintage. Meeting each
    minimum still requires a real backing ``smr`` investment (the
    source-cap constraint), which is why ``p_retrofit_multi_vintage``
    raises smr's own operating cap to 6.0 -- otherwise two of these
    three minimums, which back onto overlapping years, could not be
    satisfied simultaneously regardless of cost.
    """
    p = p_retrofit_multi_vintage.copy(deep=True)
    p["retrofit_conversion_power_capacity_investment_min"] = xr.DataArray(
        [2.0, 3.0],
        dims=["year_ret"],
        coords={"year_ret": p.year_ret.values},
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    assert _per_vintage_at(s, 2020, 2030) >= 2.0 - 1e-6
    assert _per_vintage_at(s, 2020, 2040) >= 3.0 - 1e-6
    assert _per_vintage_at(s, 2030, 2040) >= 3.0 - 1e-6


def test_retrofit_pinned_by_equal_min_and_max_bounds(p_retrofit_value):
    """
    Exercises the ``==`` branch (``planning_retrofit_conversion_power_
    capacity_def``), which the maximum/minimum tests above never touch
    -- their bounds always differ, routing through ``<=``/``>=``
    exclusively.

    Same shortfall and pricing as ``test_retrofit_capacity_reaches_the_
    target_technology`` (10 demanded, 3 available from ``smr`` directly,
    direct ``smr_ccs`` overpriced via ``conversion_annuity_cost``), where
    the solver's own unconstrained choice is 3.0 of retrofit (the most
    the source-cap constraint allows here, see that test). Pinning min
    and max to 2.0 -- below that natural optimum -- forces exactly 2.0
    of retrofit and 5.0 of direct build, a value the optimiser would not
    pick on its own, so the constraint is what produces it.
    """
    p = p_retrofit_value.copy(deep=True)
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
    )
    p["retrofit_conversion_power_capacity_investment_min"] = xr.full_like(
        p.retrofit_conversion_power_capacity_investment_min, 2.0
    )
    p["retrofit_conversion_power_capacity_investment_max"] = xr.full_like(
        p.retrofit_conversion_power_capacity_investment_max, 2.0
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    total_retrofit = float(
        s.planning_retrofit_conversion_power_capacity.sum()
    )
    direct_ccs = float(
        (s.planning_conversion_power_capacity)
        .sel(conversion_tech="smr_ccs")
        .sum()
    )
    assert total_retrofit == pytest.approx(2.0, abs=1e-6)
    assert direct_ccs == pytest.approx(5.0, abs=1e-6)


def _built_smr(solution):
    return float(
        (solution.planning_conversion_power_capacity)
        .sel(conversion_tech="smr")
        .sum()
    )


def test_retrofit_cannot_exceed_the_source_capacity(p_retrofit_value):
    """
    Same shortfall and pricing as ``test_retrofit_capacity_reaches_the_
    target_technology``: demand 10, ``smr`` capped operationally at 3,
    ``smr_ccs`` overpriced directly via ``conversion_annuity_cost``. The
    solver uses retrofit up to 3.0 (see that test for why the source-cap
    constraint tops it out there rather than at the full 7-unit
    shortfall), all of it backed by an actual ``smr`` investment
    decommissioned at the retrofit year.

    The point of this test is the inequality, not the exact figure:
    whatever the solver invests in retrofit must not exceed what it
    actually built as ``smr``. Removing the source-cap constraint
    entirely (``retrofit_conversion_power_cap_constraint``)
    would let the solver satisfy the shortfall from an idle transfer
    variable without building any backing ``smr`` at all -- retrofit
    would jump to the full shortfall while ``built_smr`` stays at
    whatever ``smr`` capacity is still worth building for its own
    direct operation, which is not tied to the transfer amount. That
    reversion was checked by hand: with the constraint removed,
    ``retrofitted`` comes out above ``built_smr``, failing the
    assertion below.
    """
    p = p_retrofit_value
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
    )
    p = p.assign(
        retrofit_conversion_annuity_cost=(
            xr.full_like(p.retrofit_conversion_annuity_cost, 0.1)
        )
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    retrofitted = float(
        s.planning_retrofit_conversion_power_capacity.sum()
    )
    built_smr = _built_smr(s)

    assert retrofitted > 0  # non-vacuous: retrofit is actually used
    assert retrofitted <= built_smr + 1e-6


def test_eligible_share_limits_the_convertible_fleet(p_retrofit_value):
    """
    Same setup as ``test_retrofit_cannot_exceed_the_source_capacity``,
    with ``retrofit_conversion_eligible_share`` lowered to 0.3.
    ``eligible_share`` caps how much of the *source* fleet may convert
    at all -- distinct from ``ret_ratio`` (an exchange rate on the
    target, derating what arrives on the other side). Backing capacity
    still tops out at 3.0 (``smr``'s own operating cap, see
    ``test_retrofit_capacity_reaches_the_target_technology``), so with
    ``eligible_share=0.3`` only 30% of that -- 0.9 -- may actually leave
    as retrofit, and since retrofit is still strictly cheaper than
    direct build (see that test), the solver invests the full 3.0 of
    backing anyway to draw the 0.9 it is allowed.

    0.3 is deliberately far below the 0.5 one might reach for by
    default: at this fixture's numbers, ``built_smr`` totals 6.0 (3.0
    backing the transfer plus a separate 3.0 vintage built for ``smr``'s
    own direct operation, for the same reason as in that test), and the
    *unconstrained* retrofit optimum is 3.0 -- exactly ``0.5 * 6.0``. A
    test written against 0.5 would still pass by coincidence if the
    ``eligible_share`` factor were silently dropped from the constraint
    (``leaving <= available`` instead of
    ``leaving <= eligible_share * available``), since
    3.0 <= 0.5 * 6.0 holds either way. At 0.3, ``0.3 * built_smr =
    1.8`` is clearly below the un-shared optimum of 3.0, so dropping the
    factor is caught: with only the ``eligible_share`` term removed
    (the rest of the cap intact), retrofit reverts to 3.0 while
    ``built_smr`` stays at 6.0, and 3.0 <= 1.8 fails (verified in the
    reversion proof for this task).
    """
    p = p_retrofit_value
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
    )
    p = p.assign(
        retrofit_conversion_annuity_cost=(
            xr.full_like(p.retrofit_conversion_annuity_cost, 0.1)
        ),
        retrofit_conversion_eligible_share=(
            xr.full_like(p.retrofit_conversion_eligible_share, 0.3)
        ),
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    retrofitted = float(
        s.planning_retrofit_conversion_power_capacity.sum()
    )
    built_smr = _built_smr(s)

    assert retrofitted > 0  # non-vacuous: retrofit is actually used
    assert retrofitted <= 0.3 * built_smr + 1e-6


def test_retrofitted_capacity_crowds_out_greenfield_investment(
    p_retrofit_value,
):
    """
    Retrofitted and greenfield capacity share ONE budget on the target
    technology -- retrofit does not add capacity on top of ``smr_ccs``'s
    own operating limit, it consumes it. This follows from how arriving
    capacity is wired: ``add_retrofit_block`` adds ``arriving`` to
    ``operation_conversion_power_capacity_def``'s own left-hand side (an
    equality, ``operation == planning_sum + arriving``), and
    ``operation_conversion_power_capacity_max_constraint`` -- a host-
    module constraint, untouched by this task -- caps that same
    ``operation`` at ``conversion_power_capacity_max``. A binding cap on
    ``smr_ccs`` therefore forces ``planning_conversion_power_capacity``
    for ``smr_ccs`` down as retrofit rises, rather than the two summing
    past the ceiling.

    Retrofit is pinned to exactly 3.0 via equal investment min/max
    bounds -- the maximum the source-cap constraint allows in this
    fixture (``smr``'s own 3-unit operational cap is the ceiling on
    backing investment, see ``test_retrofit_capacity_reaches_the_
    target_technology``). ``smr_ccs``'s own operating cap is then also
    set to 3.0, exactly matching the pinned retrofit amount, leaving no
    headroom for greenfield build. Demand (6, hydrogen at
    ``year_op=2030``) is set to exactly what ``smr``'s own 3-unit direct
    operation plus ``smr_ccs``'s fully-retrofit-sourced 3 units can
    supply between them -- hydrogen has no other route in this fixture
    (``conversion_tech`` is restricted to ``smr``/``smr_ccs``/``ocgt``,
    and ``ocgt`` makes electricity, not hydrogen) -- so both numbers are
    forced with no slack:

    - ``arriving`` at ``year_op=2030`` equals retrofit's pinned 3.0 (its
      window, ``year_ret <= year_op < year_dec`` = ``2030 <= 2030 <
      2050``, covers it). The host's own equality then gives
      ``operation = planning_sum + 3.0``, and the 3.0 cap forces
      ``planning_sum <= 0`` -- i.e. exactly 0, the variable's own lower
      bound. Greenfield is not merely discouraged by cost here, it is
      infeasible above zero.
    - ``smr``'s own direct operation must supply the remaining
      ``6 - 3 = 3``, which is exactly its own 3-unit cap -- also forced,
      not chosen.
    - ``operation_conversion_power_capacity(smr_ccs) = 0 + 3.0 = 3.0``,
      meeting the 3.0 cap with no slack.

    Reversion checked: commenting out the line in ``_core.py`` that adds
    ``arriving`` to ``operation_conversion_power_capacity_def``'s left-
    hand side (rather than deleting the whole retrofit block) isolates
    crowding-out specifically. Retrofit stays pinned to 3.0 by its own
    bounds either way (unaffected by that line), so it remains paid-for
    but now contributes nothing towards demand; ``smr_ccs``'s own
    operating cap then constrains greenfield in isolation, and since
    demand still needs the 3 units retrofit used to cover, greenfield is
    pushed up to supply them instead -- ``planning_conversion_power_
    capacity(smr_ccs)`` comes out at 3.0, not 0, and the assertion below
    fails. (Deleting the whole retrofit block instead would drive
    retrofit itself to zero along with everything downstream of it,
    which would trivially leave ``planning_conversion_power_capacity``
    at 0 too -- a reversion that would not have caught this defect,
    which is why the narrower one is used here.)
    """
    p = p_retrofit_value.copy(deep=True)
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 6.0
    p["conversion_power_capacity_max"].loc[
        dict(conversion_tech="smr_ccs")
    ] = 3.0
    p["retrofit_conversion_power_capacity_investment_min"] = xr.full_like(
        p.retrofit_conversion_power_capacity_investment_min, 3.0
    )
    p["retrofit_conversion_power_capacity_investment_max"] = xr.full_like(
        p.retrofit_conversion_power_capacity_investment_max, 3.0
    )

    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    retrofitted = float(
        s.planning_retrofit_conversion_power_capacity.sum()
    )
    greenfield_smr_ccs = float(
        (s.planning_conversion_power_capacity)
        .sel(conversion_tech="smr_ccs")
        .sum()
    )
    operating_smr_ccs = float(
        (s.operation_conversion_power_capacity)
        .sel(conversion_tech="smr_ccs", year_op=2030)
        .sum()
    )

    assert retrofitted > 0  # non-vacuous: retrofit is actually used
    assert retrofitted == pytest.approx(3.0, abs=1e-6)
    assert greenfield_smr_ccs == pytest.approx(0.0, abs=1e-6)
    assert operating_smr_ccs <= 3.0 + 1e-6


@pytest.fixture()
def p_retrofit_chain(parameters_retrofit_conversion):
    """
    A two-hop chain on top of a second, self-targeting pair: pair
    ``smr_to_ccs`` (``smr -> smr_ccs``, as in every other fixture in
    this file) and pair ``ccs_self`` (``smr_ccs -> smr_ccs``), letting
    an asset that already became ``smr_ccs`` be retrofitted again --
    i.e. renewed -- at its own end of life. Built by widening
    ``p_retrofit_value``'s idiom (``ocgt`` for ``smr_ccs``'s auxiliary
    electricity load, ``conversion_early_decommissioning=True``,
    ``demand`` zeroed then targeted) rather than starting from
    ``p_retrofit_multi_vintage``, since nothing here needs more than
    one live vintage per pair.

    Both host and retrofit life spans are 20 (fixture constants), and
    ``retrofit_conversion_early_decommissioning`` is left at its
    fixture default of ``False``, so -- as in
    ``test_transfer_mask_values`` -- every *retrofit* decommissioning
    year is pinned by exact equality to a natural end of life; only the
    *host* technology's own decommissioning (``conversion_early_
    decommissioning``, set ``True`` here) may fall early.

    Timeline, chosen so pair 0's arrival window ends strictly before
    the study's final operating year and only pair 1's covers it:

    - ``smr`` is invested at ``year_inv=2020`` and decommissioned at
      ``year_dec=2030``: the only decommissioning year present in the
      narrowed ``year_dec`` grid that is not later than smr's natural
      end of life (``2020+20=2040``), so early decommissioning is the
      *only* option for this vintage -- it cannot instead persist to
      2040, because 2040 is absent from the grid.
    - Pair 0 fires at ``year_ret=2030`` (``<=`` smr's natural end of
      life at ``year_inv=2020``, ``2040`` -- satisfied since early
      decommissioning of the source is on), backed exactly by that
      smr investment (``available`` at ``year_inv=2020, year_ret=2030``
      is the direct term only, since nothing chains in there -- no
      pair has ``year_ret=2020``, the grid's minimum). Its own
      decommissioning year is pinned by exact equality to
      ``retrofit_conversion_end_of_life`` at ``year_ret=2030``, i.e.
      ``2030+20=2050``. Arrival window ``[2030, 2050)`` -- covers
      ``year_op`` 2030 and 2040, never 2050.
    - Pair 1 fires at ``year_inv=2030`` (pair 0's own arrival year,
      chained in via the renamed ``year_ret``), ``year_ret=2050``
      (``<=`` smr_ccs's own natural end of life for a hypothetical
      ``year_inv=2030`` vintage, ``2030+20=2050`` -- satisfied by
      equality), decommissioning pinned to ``2050+20=2070``. Arrival
      window ``[2050, 2070)`` -- covers ``year_op=2050``, the study's
      last.

    Three restrictions structurally rule out every path that would let capacity
    reach ``smr_ccs`` at ``year_op=2050`` *without* going through both
    hops:

    - ``smr``'s own investment is zeroed at every ``year_inv`` except
      2020 (``conversion_power_capacity_investment_max``), so it exists
      nowhere the chain does not need it. Without this, a second smr
      vintage at ``year_inv=2030`` could retrofit directly to
      ``smr_ccs`` at ``year_ret=2050`` (its own natural end of life),
      landing on ``year_dec=2070`` -- an arrival window
      ``[2050, 2070)`` reaching 2050 in a *single* hop.
    - ``smr_ccs``'s own investment is zeroed at ``year_inv=2030``
      (``conversion_power_capacity_investment_max``), closing the alternate
      route of directly investing in ``smr_ccs`` at 2030 to feed pair 1's
      capacity constraint.
    - ``year_ret`` is narrowed to exactly ``{2030, 2050}`` -- the two
      years the chain needs. Left at the fixture's full ``{2020, 2030,
      2040, 2050}``, pair 0 could fire at ``year_ret=2040`` instead
      (``<=`` smr's natural end of life, 2040), landing on
      ``year_dec=2060`` -- an arrival window ``[2040, 2060)`` again
      reaching 2050 directly from the *original* smr vintage, no second
      hop required.

    Direct ``smr_ccs`` investment is priced far out of reach via
    ``conversion_annuity_cost`` (mirroring every other value test in
    this file). It remains *structurally* available at
    ``year_inv=2050`` (natural ``year_dec=2070``, window
    ``[2050, 2070)``) -- deliberately: the chain must out-compete it on
    cost, not win by default because no alternative exists.

    Hydrogen has no import route (``net_import_max_yearly_energy_
    import`` is 0 for hydrogen throughout these fixtures), so the 4.0
    units of hydrogen demand placed at ``year_op=2050`` can only be
    met by production, and only ``smr``/``smr_ccs`` produce hydrogen at
    all (``ocgt`` makes electricity, needed only for ``smr_ccs``'s own
    auxiliary load). With ``smr`` itself gone from the fleet by 2030,
    the sole route left standing is the two-hop chain.
    """
    p = parameters_retrofit_conversion.sel(
        conversion_tech=["smr", "smr_ccs", "ocgt"],
        area=["area_1"],
        hour=[0],
        resource=["electricity", "hydrogen", "methane"],
        year_dec=[2030, 2050, 2070],
        year_inv=[2020, 2030, 2050],
        year_op=[2020, 2030, 2040, 2050],
        year_ret=[2030, 2050],
    ).copy(deep=True)

    p["conversion_early_decommissioning"] = np.array(True, dtype="bool")
    p["demand"] = p.demand * 0

    p["conversion_power_capacity_investment_max"] = (
        xr.full_like(p.conversion_power_capacity_investment_max, 1000.0)
        * xr.ones_like(p.conversion_tech, dtype="float64")
        * xr.ones_like(p.year_inv, dtype="float64")
    ).copy()
    p["conversion_power_capacity_investment_max"].loc[
        dict(conversion_tech="smr", year_inv=[2030, 2050])
    ] = 0.0
    p["conversion_power_capacity_investment_max"].loc[
        dict(conversion_tech="smr_ccs", year_inv=2030)
    ] = 0.0

    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
    )

    pair = np.array(["smr_to_ccs", "ccs_self"], dtype=str)
    p = p.drop_dims("retrofit_conversion_pair")
    p = p.assign(
        retrofit_conversion_tech_from=xr.DataArray(
            np.array(["smr", "smr_ccs"], dtype=str),
            dims="retrofit_conversion_pair",
            coords={"retrofit_conversion_pair": pair},
        ),
        retrofit_conversion_tech_to=xr.DataArray(
            np.array(["smr_ccs", "smr_ccs"], dtype=str),
            dims="retrofit_conversion_pair",
            coords={"retrofit_conversion_pair": pair},
        ),
    )

    p["demand"].loc[dict(resource="hydrogen", year_op=2050)] = 4.0
    return p


def test_capacity_acquired_by_retrofit_can_be_retrofitted_again(
    p_retrofit_chain,
):
    """
    Proves chaining, not just structurally but under solve: capacity
    that arrived at ``smr_ccs`` through pair 0 (``smr -> smr_ccs``) is
    itself retrofitted again through pair 1 (``smr_ccs -> smr_ccs``),
    which is only possible if the cap constraint's right-hand side
    carries the chained term added in this task.

    Expected values, derived from ``p_retrofit_chain``'s timeline and
    cost structure, not merely observed:

    - Retrofit carries no cost yet (not wired into the objective until
      a later task -- see ``test_expensive_retrofit_is_not_used``), so
      routing the 4.0-unit shortfall entirely through the chain is
      strictly cheaper than paying ``smr_ccs``'s 5000 annuity for any
      direct investment. The dominance is clear: ``smr``'s base investment
      cost is 1000, giving an annuity of 50/yr across its 20-year
      lifespan; when early-decommissioned at ``year_dec=2030`` (10 years
      early), this doubles to 100/yr. Routing 4.0 units through the chain
      costs 4 × 100 = 400/yr, far below the 5000/yr direct build cost.
      The solver has no reason to build directly at all, and every unit of
      ``smr`` invested only pays for itself if it is later needed
      downstream, so the solver invests exactly 4.0 -- no more, no less.
    - ``smr``'s entire 4.0 must decommission at 2030 (the fixture's
      only available early-decommissioning year for that vintage), so
      all 4.0 is eligible to retrofit via pair 0 at ``year_ret=2030``;
      with ``eligible_share=1`` and ``ret_ratio=1`` (fixture
      defaults), the full 4.0 transfers.
    - That 4.0 arrives at ``smr_ccs`` at year 2030 and is itself
      backed for a second retrofit: pair 1's cap at
      ``(year_inv=2030, year_ret=2050)`` is exactly this chained 4.0
      (the direct-investment term of the same cap is zero, since no
      ``smr_ccs`` is ever invested directly). With ``eligible_share=1``
      and ``ret_ratio=1`` again, pair 1 also transfers the full 4.0.
    - That second transfer is the only capacity landing on ``smr_ccs``
      at ``year_op=2050`` (pair 0's own arrival window, ``[2030,
      2050)``, excludes 2050 itself), so it must equal demand there
      exactly: 4.0.

    A production change that would make ``pair1_total`` fail: dropping
    ``_chained_capacity`` from the cap constraint's right-hand side (or
    computing it without ``ratio``, or without the coordinate reindex),
    which removes pair 1's only source of backing capacity and forces
    it to zero -- proven below by reverting that change and re-running.
    """
    p = p_retrofit_chain
    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    retrofit = s.planning_retrofit_conversion_power_capacity
    pair0_total = float(
        retrofit.sel(retrofit_conversion_pair="smr_to_ccs").sum()
    )
    pair1_total = float(
        retrofit.sel(retrofit_conversion_pair="ccs_self").sum()
    )
    direct_ccs = float(
        (s.planning_conversion_power_capacity)
        .sel(conversion_tech="smr_ccs")
        .sum()
    )

    assert pair1_total > 0  # non-vacuous: the second hop actually fires
    assert pair0_total == pytest.approx(4.0, abs=1e-6)
    assert pair1_total == pytest.approx(4.0, abs=1e-6)
    assert direct_ccs == pytest.approx(0.0, abs=1e-6)


def test_retrofit_cost_enters_the_objective(p_retrofit_value):
    """
    Doubling the retrofit annuity must raise the objective by exactly
    the amount actually retrofitted, times the change in rate. Without
    the cost constraint and the totex hook, retrofit would be free and
    the objective would not move at all.

    Built on ``p_retrofit_value`` rather than the plain ``p_retrofit``
    fixture: ``p_retrofit`` is never solved elsewhere in this file (only
    inspected structurally), and it turns out not to be solvable as-is
    -- its base ``demand`` (inherited un-zeroed from the shared
    fixtures) includes a nonzero electricity demand, and its
    ``conversion_tech`` selection (``smr``/``smr_ccs`` only) has no
    electricity producer at all, so *any* solve of it is infeasible
    regardless of retrofit. ``p_retrofit_value`` exists precisely to be
    solvable (it zeroes demand first and adds ``ocgt`` for electricity)
    and already sets up the exact tradeoff this test needs: demand of
    10 for hydrogen, ``smr`` capped at 3, and direct ``smr_ccs`` build
    priced far out of reach via ``conversion_annuity_cost`` (5000) --
    the same setup as
    ``test_retrofit_capacity_reaches_the_target_technology``, which
    establishes that retrofit saturates at exactly 3.0 there.

    At both the cheap (1.0/yr) and dear (2.0/yr) retrofit rates, 2.0/yr
    is still trivially cheaper than the 5000/yr direct-build
    alternative, so the optimal *route* is unchanged -- retrofit still
    saturates at 3.0, backed the same way. ``p_retrofit_value``'s only
    live transfer cell (``year_inv=2020, year_ret=2030,
    year_dec=2050``) has an arrival window of ``[2030, 2050)``, and
    with ``year_op`` restricted to ``{2020, 2030}`` only ``year_op=
    2030`` falls inside it, so the annuity is charged for exactly one
    operating year. The objective must therefore rise by exactly
    ``3.0 * (2.0 - 1.0) = 3.0``, not merely by some positive amount.
    """
    p = p_retrofit_value.copy(deep=True)
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
    )

    cheap = p.assign(
        retrofit_conversion_annuity_cost=(
            xr.full_like(p.retrofit_conversion_annuity_cost, 1.0)
        )
    )
    dear = p.assign(
        retrofit_conversion_annuity_cost=(
            xr.full_like(p.retrofit_conversion_annuity_cost, 2.0)
        )
    )

    m_cheap = build_model(cheap)
    m_cheap.solve(solver_name="highs")
    m_dear = build_model(dear)
    m_dear.solve(solver_name="highs")

    assert m_cheap.status == "ok" and m_dear.status == "ok"

    retrofit_cheap = float(
        m_cheap.solution.planning_retrofit_conversion_power_capacity.sum()
    )
    retrofit_dear = float(
        m_dear.solution.planning_retrofit_conversion_power_capacity.sum()
    )
    assert retrofit_cheap == pytest.approx(3.0, abs=1e-6)
    assert retrofit_dear == pytest.approx(3.0, abs=1e-6)
    assert m_dear.objective.value - m_cheap.objective.value == (
        pytest.approx(3.0, abs=1e-6)
    )


def test_retrofit_costs_variable_is_area_indexed(p_retrofit):
    model = build_model(p_retrofit)
    costs = model.variables["planning_retrofit_conversion_costs"]
    assert set(costs.dims) == {
        "area",
        "retrofit_conversion_pair",
        "year_op",
    }


def test_perfect_foresight_flag_changes_the_cost_basis(p_retrofit_value):
    """
    Without the ``other`` branch, ``planning_retrofit_conversion_costs``
    always charges the vintage-specific annuity, no matter what
    ``retrofit_conversion_annuity_perfect_foresight`` says -- the two
    builds below would then be byte-identical and the objectives equal.

    Built on ``p_retrofit_value`` (solvable, and reduces to exactly one
    live transfer cell: ``year_inv=2020, year_ret=2030, year_dec=2050``
    -- see that fixture's docstring) rather than the plain ``p_retrofit``
    used unmodified: with ``retrofit_conversion_early_decommissioning``
    left at its fixture default (``False``), ``_transfer_mask`` forces
    an *exact* match between ``year_dec`` and
    ``retrofit_conversion_end_of_life`` (``year_ret + life_span`` =
    ``2030 + 20 = 2050``), so there is never a genuine choice of
    ``year_dec`` for the solver to make either way -- the mask alone
    always pins it to 2050.

    That would leave ON and OFF trivially equal too, since the
    *default* ``retrofit_conversion_annuity_cost`` (built with
    ``square_array_by_diagonals(6, {0: 1/20, 1: 1/10})``) happens to
    put its lowest rate (``1/20``) at exactly ``year_dec = year_ret +
    life_span`` -- the same cell the mask already forces -- so
    ``min`` over ``year_dec`` would coincide with the forced cell by
    construction of the fixture, not because the flag does anything.
    To make the two branches genuinely diverge, this test overrides
    ``retrofit_conversion_annuity_cost`` so the forced cell
    (``year_dec=2050``) is the *expensive* one (30) and a cell the
    mask never allows the model to actually use
    (``year_dec=2040``) is *cheaper* (15). ON must pay the forced
    cell's rate; OFF is free to (and, per ``conversion.py``'s
    contract, does) charge the cheaper rate the model can never
    physically realise -- exactly the "pays the minimum annuity
    regardless of the vintage actually reachable" behaviour the host
    module already exhibits.

    As in ``test_retrofit_cost_enters_the_objective``, ``smr`` is
    capped at 3 and direct ``smr_ccs`` build is priced at 5000/yr, so
    retrofit is by far the cheapest route to the hydrogen shortfall
    and saturates at 3.0 regardless of whether its own rate is 30 or
    15 -- both are trivially cheaper than 5000. The only thing that
    changes between ON and OFF is which of those two rates is paid,
    on the same 3.0 units, over the single operating year
    (``year_op=2030``) inside the transfer's ``[2030, 2050)`` window.
    The objective must therefore differ by exactly
    ``3.0 * (30 - 15) = 45``, not merely be unequal.
    """
    p = p_retrofit_value.copy(deep=True)
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
    )
    p["retrofit_conversion_annuity_cost"] = xr.full_like(
        p.retrofit_conversion_annuity_cost, np.nan
    )
    p["retrofit_conversion_annuity_cost"].loc[
        dict(year_ret=2030, year_dec=2050)
    ] = 30.0
    p["retrofit_conversion_annuity_cost"].loc[
        dict(year_ret=2030, year_dec=2040)
    ] = 15.0

    on = p.assign(
        retrofit_conversion_annuity_perfect_foresight=(
            xr.full_like(
                p.retrofit_conversion_annuity_perfect_foresight, True
            )
        ).astype(bool)
    )
    off = p.assign(
        retrofit_conversion_annuity_perfect_foresight=(
            xr.full_like(
                p.retrofit_conversion_annuity_perfect_foresight, False
            )
        ).astype(bool)
    )

    m_on = build_model(on)
    m_on.solve(solver_name="highs")
    m_off = build_model(off)
    m_off.solve(solver_name="highs")

    assert m_on.status == "ok" and m_off.status == "ok"

    retrofit_on = float(
        m_on.solution.planning_retrofit_conversion_power_capacity.sum()
    )
    retrofit_off = float(
        m_off.solution.planning_retrofit_conversion_power_capacity.sum()
    )
    assert retrofit_on == pytest.approx(3.0, abs=1e-6)
    assert retrofit_off == pytest.approx(3.0, abs=1e-6)
    assert m_on.objective.value - m_off.objective.value == (
        pytest.approx(45.0, abs=1e-6)
    )


@pytest.fixture()
def p_retrofit_dispatch(parameters_retrofit_conversion):
    """
    Built from ``p_retrofit_value``'s idiom -- ``ocgt`` alongside
    ``smr``/``smr_ccs`` to close smr_ccs's auxiliary electricity loop,
    ``demand`` zeroed then targeted -- but on a fresh, wider coordinate
    slice, because every value fixture in this file up to this point
    asserts on planning capacity, never on dispatch, and none of them
    can host what dispatch coverage needs: two live ``smr`` vintages
    decommissioning in different years, so that after 150 MW retrofits
    away the other 150 MW is still there to dispatch.

    Where the surviving smr comes from (established, not assumed):
    retrofit converts capacity whose decommissioning year is exactly
    the retrofit year (``host_var.where(year_ret == year_dec)`` in
    ``_core.py``), and that vintage then drops out of ``smr``'s own
    ``operation_conversion_power_capacity_def``, which sums planning
    capacity only over ``year_inv <= year_op < year_dec`` -- excluding
    it once ``year_op`` reaches its own ``year_dec``. So ``smr``'s 300
    MW is built as two separate ``year_inv`` vintages here: 150 MW at
    ``year_inv=2010`` and 150 MW at ``year_inv=2020``. With
    ``conversion_life_span=20`` uniform across technologies in these
    fixtures, their *natural* (non-early) end-of-life years are 2030
    and 2040 respectively -- read directly off ``conversion.py``'s own
    ``planning_conversion_power_capacity`` mask (lines 219-233): with
    ``conversion_early_decommissioning`` at its fixture default of
    ``False``, that mask forces ``year_dec == conversion_end_of_life``
    exactly, i.e. ordinary scheduled retirement, one ``year_dec`` per
    ``year_inv``. The 2010 vintage's scheduled retirement (2030) lands
    exactly on the retrofit year chosen below, so it is eligible to
    back the transfer under ``_transfer_mask``'s own exact-equality
    branch (also reached only when the flag is ``False``); the 2020
    vintage's scheduled retirement (2040) is later, so it is still
    operating in 2030 once the first vintage has left.

    Finding on early decommissioning (established, not assumed): NOT
    required. The formulation being fully linear means the 300 MW does
    not have to be one indivisible block; here it is represented as
    two ordinarily-scheduled vintages rather than one prematurely-
    retired one, and ``conversion_early_decommissioning`` is left at
    its fixture default of ``False`` throughout -- confirmed by tracing
    both masks above rather than by trial. This differs from why
    ``p_retrofit_value``/``p_retrofit_multi_vintage`` turn the flag on:
    those fixtures reuse a *single* ``year_inv=2020`` vintage that must
    retire before its own natural end of life (2040) to land on their
    chosen retrofit year (2030), which is only reachable with early
    decommissioning allowed. Choosing ``year_inv`` freely instead (2010
    here) makes the backing vintage's *natural* schedule land on the
    retrofit year, sidestepping the need for the flag entirely.

    Every other cost that could confound a dispatch-preference test is
    neutralised so the only thing driving the split is
    ``conversion_variable_cost``, per the scenario:

    - ``ocgt``'s own variable, fixed and annuity costs are zeroed. It
      exists solely to satisfy smr_ccs's auxiliary electricity load
      (``conversion_factor`` draws -0.1 electricity per unit hydrogen)
      -- structurally required since electricity has no shedding or
      import route in these fixtures -- not to compete economically.
      Left at its default variable cost of 8, 0.1 unit of ocgt output
      per unit of smr_ccs output would add 0.8/unit to smr_ccs's
      marginal cost on its own, more than wiping out the 0.5/unit
      direct-cost advantage given to it below.
    - Methane's ``load_shedding_cost`` (the unlimited backstop supply
      this parameter set uses in place of an explicit import module,
      since ``parameters_dispatch_invest`` -- what ``parameters_
      retrofit_conversion`` is built from -- never merges in
      ``net_import``) is zeroed. Both ``smr`` and ``smr_ccs`` draw the
      same 1.5 units of methane per unit hydrogen, so this feedstock
      cost would otherwise be identical between them -- except for
      smr_ccs's extra 0.1-unit indirect draw through ocgt, which at
      the fixture default of 1000/unit would add 150/unit to smr_ccs's
      marginal cost, again swamping the 0.5/unit advantage.
    - ``smr``, ``smr_ccs`` and the retrofit transfer itself are all
      pinned by equal investment min/max bounds (the retrofit's alone
      is required by the scenario; the source and target capacities
      are pinned too because the scenario states them as given
      totals, not quantities for the solver to discover), so none of
      their own fixed or annuity costs can affect the dispatch
      decision either: capacity is fixed regardless of how much of it
      is actually run.

    With every other driver neutralised, the marginal cost of one unit
    of hydrogen is exactly ``conversion_variable_cost`` per technology:
    1.0 for ``smr``, 0.5 for the ``smr_ccs`` override below.
    """
    p = parameters_retrofit_conversion.sel(
        conversion_tech=["smr", "smr_ccs", "ocgt"],
        area=["area_1"],
        hour=[0],
        resource=["electricity", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2010, 2020, 2030],
        year_op=[2030],
        year_ret=[2030],
    ).copy(deep=True)

    p["conversion_variable_cost"] = xr.where(
        p.conversion_tech == "smr_ccs", 0.5, p.conversion_variable_cost
    )
    p["conversion_variable_cost"] = xr.where(
        p.conversion_tech == "ocgt", 0.0, p.conversion_variable_cost
    )
    p["conversion_fixed_cost"] = xr.where(
        p.conversion_tech == "ocgt", 0.0, p.conversion_fixed_cost
    )
    p["conversion_annuity_cost"] = xr.where(
        p.conversion_tech == "ocgt", 0.0, p.conversion_annuity_cost
    )
    p["load_shedding_cost"] = xr.where(
        p.resource == "methane", 0.0, p.load_shedding_cost
    )

    # smr: 300 MW total, split 150/150 across the two vintages
    # described above. smr_ccs: 200 MW built directly. ocgt: left
    # free (default bounds) to invest whatever it needs. Every other
    # (tech, year_inv) cell is pinned to zero so the solver cannot
    # reach the target totals any other way.
    inv_max = (
        xr.full_like(p.conversion_power_capacity_investment_max, 0.0)
        * xr.ones_like(p.conversion_tech, dtype="float64")
        * xr.ones_like(p.year_inv, dtype="float64")
    ).copy()
    inv_max.loc[dict(conversion_tech="ocgt")] = 1000.0
    inv_max.loc[dict(conversion_tech="smr", year_inv=2010)] = 150.0
    inv_max.loc[dict(conversion_tech="smr", year_inv=2020)] = 150.0
    inv_max.loc[dict(conversion_tech="smr_ccs", year_inv=2020)] = 200.0
    p["conversion_power_capacity_investment_max"] = inv_max

    inv_min = inv_max.copy()
    inv_min.loc[dict(conversion_tech="ocgt")] = 0.0
    p["conversion_power_capacity_investment_min"] = inv_min

    # Forced retrofit: 150 MW smr -> smr_ccs, pinned by equal min/max
    # bounds (exercises the "==" branch of planning_retrofit_
    # conversion_power_capacity_def). This coordinate slice has
    # exactly one live transfer cell -- (year_inv=2010, year_ret=2030,
    # year_dec=2050), backed exactly by the 150 MW smr vintage above
    # that decommissions at 2030 -- so a uniform override is
    # unambiguous.
    p["retrofit_conversion_power_capacity_investment_min"] = (
        xr.full_like(
            p.retrofit_conversion_power_capacity_investment_min, 150.0
        )
    )
    p["retrofit_conversion_power_capacity_investment_max"] = (
        xr.full_like(
            p.retrofit_conversion_power_capacity_investment_max, 150.0
        )
    )

    p["demand"] = p.demand * 0
    p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 400.0
    return p


def test_retrofitted_capacity_is_dispatched_before_the_source(
    p_retrofit_dispatch,
):
    """
    Every retrofit test up to this point stops at capacity: none
    asserts on ``operation_conversion_power``, the actual dispatched
    power, so a change severing the retrofit's contribution to
    ``operation_conversion_power_capacity_def`` (see the reversion
    proof recorded in this task's report) would leave the whole suite
    green. This test closes that gap.

    Derivation, given ``p_retrofit_dispatch``'s pinned capacities and
    neutralised costs (see that fixture's docstring):

    - ``smr_ccs``'s operating capacity at year_op=2030 is its 200 MW
      greenfield build plus the 150 MW retrofit transfer, which
      arrives there for exactly this year (arrival window
      ``[year_ret, year_dec) = [2030, 2050)`` covers 2030) --
      ``operation_conversion_power_capacity_def`` adds the two
      (``operation == planning_sum + arriving``): 200 + 150 = 350 MW.
    - ``smr``'s operating capacity at year_op=2030 is only its
      surviving 150 MW vintage (``year_inv=2020``, natural
      ``year_dec=2040``); the other 150 MW (``year_inv=2010``) is
      excluded because it decommissions exactly at 2030, and the
      def's window is ``year_inv <= year_op < year_dec`` -- strict on
      the right, so a vintage does not count toward the technology's
      own operation in the year it leaves to become the other one.
    - Both technologies have ``conversion_availability = NaN``, which
      the host module treats as full availability
      (``operation_conversion_power_max_constraint``), so each
      technology's dispatch ceiling equals its operating capacity
      exactly: 350 MW for ``smr_ccs``, 150 MW for ``smr``.
    - With every other cost driver neutralised (see the fixture),
      ``smr_ccs``'s marginal cost per unit hydrogen (0.5) is strictly
      below ``smr``'s (1.0), so a cost-minimising solve dispatches
      ``smr_ccs`` to its full 350 MW ceiling before drawing on ``smr``
      at all. The remaining ``400 - 350 = 50`` MW of demand is then
      met by ``smr``, well inside its own 150 MW ceiling -- so that
      50 MW is pinned by the demand/capacity arithmetic, not merely
      cost-preferred.

    Expected split: 350 MW from ``smr_ccs``, 50 MW from ``smr``.

    This test passed on first write, stated explicitly per this task's
    own instructions rather than moved past silently. It is not
    vacuous: the derivation above depends on the retrofit contribution
    to ``operation_conversion_power_capacity_def`` (without it,
    smr_ccs's ceiling drops to its 200 MW greenfield build alone, and
    150 + 200 = 350 MW of total capacity cannot reach the 400 MW
    demanded at all, since hydrogen has no shedding route here) -- the
    reversion proof in this task's report demonstrates the dependency
    directly by commenting out that contribution and re-running.
    Reaching the analytically-derived 350/50 split on the first
    attempt reflects that the underlying wiring already existed and
    worked (this task adds coverage only, per its brief); getting the
    fixture's cost-neutralisation right (see that fixture's docstring)
    took one iteration first -- an earlier draft, before ``ocgt`` and
    methane's load-shedding cost were neutralised, would have reversed
    the intended merit order on paper (smr_ccs's indirect auxiliary
    cost would have exceeded its direct-cost advantage), which is why
    that neutralisation is load-bearing and documented, not incidental.
    """
    p = p_retrofit_dispatch
    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    s = model.solution

    power = (s.operation_conversion_power).sel(area="area_1", year_op=2030, hour=0)

    smr_power = float(power.sel(conversion_tech="smr"))
    smr_ccs_power = float(power.sel(conversion_tech="smr_ccs"))

    assert smr_power + smr_ccs_power == pytest.approx(400.0, abs=1e-6)
    assert smr_ccs_power == pytest.approx(350.0, abs=1e-6)
    assert smr_power == pytest.approx(50.0, abs=1e-6)


def test_eligible_share_binds_by_technology_not_broadcast(p_retrofit_value):
    """
    ``eligible_share`` is declared on ``[area, conversion_tech, year_ret,
    year_inv]`` (``dataset_description.yaml``) -- its own technology
    dimension, not the pair dimension the transfer variable carries.
    ``_gather_share`` in the retrofit core projects it onto whatever
    index the source-cap constraint's ``available`` term actually uses
    before multiplying the two together, so that an unrelated
    technology's share can never leak into a pair it has nothing to do
    with regardless of how the host module itself indexes capacity.
    Every other retrofit test sets ``eligible_share`` as a 0-d scalar,
    where there is nothing to cross-product against -- this is the one
    test that deliberately avoids that setup.

    Two solves, both starting from the shortfall/pricing setup of
    ``test_retrofit_capacity_reaches_the_target_technology`` (demand 10,
    ``smr`` capped operationally at 3, direct ``smr_ccs`` overpriced via
    ``conversion_annuity_cost``, retrofit annuity cut to 0.1 so retrofit
    undercuts direct build):

    - ``eligible_share`` = 0.3 on ``ocgt``, 1.0 elsewhere. ``ocgt`` is
      neither the source nor the target of the one retrofit pair here
      (``smr -> smr_ccs``), so a correctly-aligned share must leave the
      retrofit exactly as it is in that reference test's ``eligible_
      share=1`` baseline: ``retrofitted == 3.0``, ``direct_ccs == 4.0``.
    - ``eligible_share`` = 0.3 on ``smr`` (the pair's actual ``tech_
      from``), 1.0 elsewhere. The right-hand side ``eligible_share``
      multiplies is not ``smr``'s total built capacity but only the
      vintage decommissioning exactly at ``year_ret``: the early-
      decommissioned ``year_inv=2020`` vintage, which the reference
      test pins at 3.0 (``smr``'s own operational cap, fully claimed
      there since nothing else competes for it at ``year_op=2020``).
      The cap is therefore ``0.3 * 3.0 = 0.9``, and since retrofit
      remains cheaper than direct build, the solver saturates it:
      ``retrofitted == 0.9`` exactly. This is consistent with (and
      looser than) ``test_eligible_share_limits_the_convertible_
      fleet``'s ``retrofitted <= 0.3 * built_smr`` bound, whose
      ``built_smr`` (6.0: the 3.0 backing vintage plus a separate 3.0
      vintage for ``smr``'s own direct operation) is the *total* built
      capacity rather than just the year_ret-decommissioning slice.
    """
    p_a = p_retrofit_value.copy(deep=True)
    p_a["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
    p_a["conversion_annuity_cost"] = xr.where(
        p_a.conversion_tech == "smr_ccs",
        5000.0,
        p_a.conversion_annuity_cost,
    )
    p_a = p_a.assign(
        retrofit_conversion_annuity_cost=(
            xr.full_like(p_a.retrofit_conversion_annuity_cost, 0.1)
        ),
    )
    p_a["retrofit_conversion_eligible_share"] = xr.where(
        p_a.conversion_tech == "ocgt", 0.3, 1.0
    )

    model_a = build_model(p_a)
    model_a.solve(solver_name="highs")
    assert model_a.status == "ok"
    s_a = model_a.solution

    retrofitted_a = float(
        s_a.planning_retrofit_conversion_power_capacity.sum()
    )
    direct_ccs_a = float(
        (s_a.planning_conversion_power_capacity)
        .sel(conversion_tech="smr_ccs")
        .sum()
    )
    assert retrofitted_a == pytest.approx(3.0, abs=1e-6)
    assert direct_ccs_a == pytest.approx(4.0, abs=1e-6)

    p_b = p_a.copy(deep=True)
    p_b["retrofit_conversion_eligible_share"] = xr.where(
        p_b.conversion_tech == "smr", 0.3, 1.0
    )

    model_b = build_model(p_b)
    model_b.solve(solver_name="highs")
    assert model_b.status == "ok"
    s_b = model_b.solution

    retrofitted_b = float(
        s_b.planning_retrofit_conversion_power_capacity.sum()
    )
    built_smr_b = _built_smr(s_b)
    assert retrofitted_b == pytest.approx(0.9, abs=1e-6)
    assert built_smr_b == pytest.approx(6.0, abs=1e-6)
