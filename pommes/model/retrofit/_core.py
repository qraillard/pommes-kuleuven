"""The generic retrofit block, shared by all four host modules.

Emits capacity-transfer variables, their bounds, the contribution to the
host module's operational capacity, the cap on what may leave the source
technology, and the resulting annualised costs.

Reaches host modules by name through ``model.constraints`` and
``model.variables``: no host module is modified, and where a host indexes
capacity on a sparse ``{tech_dim}_unit`` dimension, that index is
recovered from the host's own variable. The same code therefore works
unchanged whether the host is indexed densely or sparsely.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from linopy import Constraint, LinearExpression, Model, Variable
from xarray import Dataset

from pommes.model.retrofit._index import (
    gather_pairs_to_tech,
    gather_tech_to_units,
    gather_to_units,
    host_unit_dim,
)
from pommes.model.retrofit._spec import RetrofitSpec


def _gather(
    expr: Variable | LinearExpression,
    m: Model,
    p: Dataset,
    spec: RetrofitSpec,
    kind: str,
    which: str,
) -> LinearExpression:
    """
    Move a pair-indexed expression onto the host's capacity index.

    ``which`` is ``"tech_to"`` (capacity arriving) or ``"tech_from"``
    (capacity leaving). Chooses the dense or the sparse path from the
    host variable's own dimensions, which is what lets one
    implementation serve hosts indexed either way: densely by
    ``[site, tech, ...]``, or on a sparse ``{tech_dim}_unit`` dimension
    that replaces the two.
    """
    host_var = m.variables[spec.host_planning_var(kind)]
    pair_tech_names = p[spec.param(which)]
    unit_dim = host_unit_dim(host_var, spec.tech_dim)
    if unit_dim is None:
        return gather_pairs_to_tech(
            expr, p[spec.tech_dim], pair_tech_names, spec.pair_dim
        )
    return gather_to_units(
        expr,
        host_var,
        spec.site_dim,
        spec.tech_dim,
        p[spec.tech_dim],
        pair_tech_names,
        spec.pair_dim,
    )


def _gather_share(
    m: Model,
    p: Dataset,
    spec: RetrofitSpec,
    kind: str,
) -> xr.DataArray:
    """
    Move ``eligible_share`` onto the index ``leaving``/``available`` use.

    ``eligible_share`` is declared on ``[site_dim, tech_dim, year_ret,
    year_inv]`` -- a per-technology parameter, unlike the transfer
    variable it caps, which is per-pair -- regardless of how the host
    module itself indexes capacity. On the dense path that already
    matches the cap constraint's own ``[site_dim, tech_dim, ...]``
    indexing, so multiplying it in directly is correct and no gather
    is needed. On the sparse path, ``available`` carries the host's
    unit dimension instead, which shares no dimension name with
    ``eligible_share``'s ``site_dim``/``tech_dim`` pair: multiplying
    them directly would broadcast across both dimensions at once
    instead of aligning technology ``c``'s share onto unit ``c``, so
    each unit must first be projected onto its own ``(site,
    technology)`` value.
    """
    host_var = m.variables[spec.host_planning_var(kind)]
    share = p[spec.param("eligible_share")]
    unit_dim = host_unit_dim(host_var, spec.tech_dim)
    if unit_dim is None:
        return share
    return gather_tech_to_units(share, host_var, spec.site_dim, spec.tech_dim)


def add_retrofit_block(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    spec: RetrofitSpec,
) -> Model:
    """
    Add the retrofit block for one host module.

    Args:
        model: the Linopy model, with ``spec.module`` already added.
        model_parameters: the full (dense) parameter dataset.
        annualised_totex_def: totex constraint, extended with retrofit
            costs.
        spec: description of the host module.

    Returns:
        The updated model.
    """
    m = model
    p = model_parameters

    _check_year_ret_grid(p)

    site = p[spec.site_dim]
    pair = p[spec.pair_dim]
    mask = _transfer_mask(p, spec)

    for kind in spec.capacity_kinds:
        m.add_variables(
            name=f"planning_retrofit_{spec.module}_{kind}_capacity",
            lower=0,
            coords=[site, p.year_inv, pair, p.year_ret, p.year_dec],
            mask=mask,
        )

        transfer = m.variables[
            f"planning_retrofit_{spec.module}_{kind}_capacity"
        ]
        ratio = p[spec.param("ret_ratio", kind)]

        # Retrofitted capacity is available to the target technology
        # from the retrofit year until decommissioning, derated by the
        # exchange ratio.
        arriving = (
            (ratio * transfer)
            .where((p.year_ret <= p.year_op) * (p.year_op < p.year_dec))
            .sum(["year_inv", "year_dec", "year_ret"])
        )
        m.constraints[spec.host_capacity_def(kind)].lhs += _gather(
            arriving, m, p, spec, kind, "tech_to"
        )

        # Investment bounds, per (site, pair, year_ret, year_inv):
        # a vintage's total transfer, however it is later decommissioned,
        # must respect the same min/max/equal idiom the host module uses
        # for its own planning capacity.
        invest_min = p[spec.bound(kind, "min")]
        invest_max = p[spec.bound(kind, "max")]
        per_vintage = transfer.sum("year_dec")
        differ = np.not_equal(invest_max, invest_min)

        m.add_constraints(
            per_vintage <= invest_max,
            name=(
                f"planning_retrofit_{spec.module}_{kind}"
                f"_capacity_max_constraint"
            ),
            mask=np.isfinite(invest_max) * differ,
        )
        m.add_constraints(
            per_vintage >= invest_min,
            name=(
                f"planning_retrofit_{spec.module}_{kind}"
                f"_capacity_min_constraint"
            ),
            mask=np.isfinite(invest_min) * differ,
        )
        m.add_constraints(
            per_vintage == invest_min,
            name=f"planning_retrofit_{spec.module}_{kind}_capacity_def",
            mask=np.isfinite(invest_min) * np.equal(invest_max, invest_min),
        )

        # Capacity leaving the source technology cannot exceed what is
        # there to leave: the directly invested capacity of that
        # technology whose decommissioning year is the retrofit year,
        # plus (via chaining) any capacity that arrived at that same
        # technology through an earlier retrofit and has itself reached
        # its own end of life. `eligible_share` further caps how much
        # of that fleet may be converted at all -- distinct from
        # `ret_ratio`, which derates what arrives on the other side.
        host_var = m.variables[spec.host_planning_var(kind)]
        leaving = _gather(
            1 * transfer.sum("year_dec"), m, p, spec, kind, "tech_from"
        )
        available = host_var.where(p.year_ret == p.year_dec).sum(
            "year_dec"
        ) + _chained_capacity(m, p, spec, kind, transfer, ratio)
        eligible_share = _gather_share(m, p, spec, kind)
        m.add_constraints(
            leaving <= eligible_share * available,
            name=f"retrofit_{spec.module}_{kind}_cap_constraint",
            mask=(p.year_inv < p.year_ret)
            * (p.year_ret <= p[f"{spec.module}_end_of_life"]),
        )

    m.add_variables(
        name=f"planning_retrofit_{spec.module}_costs",
        lower=0,
        coords=[p.area, pair, p.year_op],
    )

    # Charged for every operating year the transferred capacity is
    # actually available (``year_ret <= year_op < year_dec``), exactly
    # the same window ``arriving`` above is gated on -- a vintage's
    # annuity is paid for as long as it delivers, regardless of when
    # (or whether) it is later decommissioned. ``costs`` above is
    # always indexed on ``area`` (never ``site_dim``), so a module
    # that plans on a different site dimension must bring its
    # per-vintage expression onto ``area`` before this constraint can
    # equate the two; ``fold_costs`` is that hook, the identity for
    # every module sited directly on ``area``.
    costs = m.variables[f"planning_retrofit_{spec.module}_costs"]
    annuity_terms = None
    for kind in spec.capacity_kinds:
        transfer = m.variables[
            f"planning_retrofit_{spec.module}_{kind}_capacity"
        ]
        annuity = p[spec.param("annuity_cost", kind)]
        term = (
            (transfer * annuity)
            .where((p.year_ret <= p.year_op) * (p.year_op < p.year_dec))
            .sum(["year_inv", "year_dec", "year_ret"])
        ).where(
            cond=p[spec.param("annuity_perfect_foresight")],
            # Mirrors the host module's own costs_def: without
            # perfect foresight the investor cannot know the eventual
            # decommissioning year, and pays the cheapest annuity
            # consistent with the asset's lifetime rather than the
            # one matching whichever year_dec is later realised.
            other=(
                (
                    transfer.sum("year_dec")
                    * annuity.min(
                        [d for d in annuity.dims if d == "year_dec"]
                    )
                )
                .where(
                    (p.year_ret <= p.year_op)
                    * (p.year_op < p[spec.param("end_of_life")])
                )
                .sum(["year_inv", "year_ret"])
            ),
        )
        annuity_terms = (
            term if annuity_terms is None else annuity_terms + term
        )

    m.add_constraints(
        -costs + spec.fold_costs(annuity_terms, p) == 0,
        name=f"planning_retrofit_{spec.module}_costs_def",
    )

    annualised_totex_def.lhs += costs.sum(spec.pair_dim)

    return m


def _check_year_ret_grid(p: Dataset) -> None:
    """
    ``year_ret`` must draw its values from both ``year_inv`` and ``year_dec``.

    Chaining realigns a retrofit's ``year_ret`` onto ``year_inv`` so an
    asset acquired by retrofit can act as the source of the next one. A
    ``year_ret`` value with no ``year_inv`` counterpart makes that
    realignment produce zero silently, which would look like "chaining
    is never worthwhile" rather than "chaining is broken".

    Independently, the source-cap constraint's ``available`` term reads
    ``host_var.where(p.year_ret == p.year_dec)``: the host's own fleet
    decommissioning exactly at the retrofit year. ``year_inv`` and
    ``year_dec`` are independent coordinates -- nothing ties them
    together -- so a ``year_ret`` value absent from ``year_dec`` is
    just as reachable as one absent from ``year_inv``, most plausibly
    whenever the decommissioning grid is coarser than the investment
    grid. That comparison then holds nowhere, leaving every transfer at
    that ``year_ret`` with no directly-invested capacity able to back
    it: not "retrofit is not worth it there" but "retrofit is
    structurally impossible there", the same silent failure mode as
    the ``year_inv`` case above.
    """
    year_ret = set(np.atleast_1d(p.year_ret.values))

    missing_inv = year_ret - set(np.atleast_1d(p.year_inv.values))
    if missing_inv:
        raise ValueError(
            f"year_ret values {sorted(missing_inv)} are absent from "
            f"year_inv {sorted(np.atleast_1d(p.year_inv.values))}. "
            f"Retrofit chaining realigns year_ret onto year_inv, so "
            f"year_ret must be a subset of year_inv."
        )

    missing_dec = year_ret - set(np.atleast_1d(p.year_dec.values))
    if missing_dec:
        raise ValueError(
            f"year_ret values {sorted(missing_dec)} are absent from "
            f"year_dec {sorted(np.atleast_1d(p.year_dec.values))}. "
            f"The source-cap constraint reads the host's own capacity "
            f"decommissioning exactly at year_ret "
            f"(host_var.where(year_ret == year_dec)), so a year_ret "
            f"value absent from year_dec leaves that vintage with no "
            f"capacity able to back a transfer at all -- year_ret must "
            f"be a subset of year_dec too."
        )


def _chained_capacity(
    m: Model,
    p: Dataset,
    spec: RetrofitSpec,
    kind: str,
    transfer: Variable,
    ratio: xr.DataArray,
) -> LinearExpression:
    """
    Capacity that arrived through an earlier retrofit, now eligible.

    An asset retrofitted into technology ``c`` at year ``i``, decom-
    missioning at ``j``, behaves from then on exactly like an asset of
    ``c`` with vintage ``i``. The grandparent's vintage is summed out:
    once the asset has become ``c``, where it came from stops mattering.

    ``ratio`` multiplies here because what arrived at ``c`` is the
    derated quantity, not the raw transfer. Omitting it would let a
    chain launder capacity back up to its pre-derating size across
    hops.

    The rename below relabels a departure year as an arrival year and
    an eventual decommissioning year as the next departure year. Its
    result is reindexed onto the constraint's own ``(year_inv,
    year_ret)`` coordinates with zero fill rather than left to default
    alignment: a renamed value with no counterpart in those grids would
    otherwise vanish silently instead of raising or erroring.
    """
    chained = (
        (ratio * transfer)
        .sum("year_inv")
        .rename({"year_ret": "year_inv", "year_dec": "year_ret"})
        .reindex(year_inv=p.year_inv, year_ret=p.year_ret, fill_value=0)
    )
    return _gather(chained, m, p, spec, kind, "tech_to")


def _transfer_mask(p: Dataset, spec: RetrofitSpec) -> xr.DataArray:
    """
    Which (site, pair, year_inv, year_ret, year_dec) transfers exist.

    A retrofit may only happen after the source was built and before the
    result is decommissioned, at the source technology's end of life
    (or earlier where the source allows early decommissioning), and the
    result must live until its own end of life (or be allowed to retire
    early).
    """
    tech = p[spec.tech_dim]
    tech_from = p[spec.param("tech_from")]

    eol_from = (
        p[f"{spec.module}_end_of_life"]
        .where(tech == tech_from, 0)
        .sum(spec.tech_dim)
    )
    early_from = (
        p[f"{spec.module}_early_decommissioning"]
        .where(tech == tech_from, False)
        .any(spec.tech_dim)
    )
    eol_ret = p[spec.param("end_of_life")]
    early_ret = p[spec.param("early_decommissioning")]

    return (
        (p.year_inv < p.year_ret)
        * (p.year_ret < p.year_dec)
        * xr.where(
            cond=early_from,
            x=p.year_ret <= eol_from,
            y=p.year_ret == eol_from,
        )
        * xr.where(
            cond=early_ret,
            x=(p.year_dec <= eol_ret)
            * np.logical_or(
                p.year_dec <= p.year_inv.max(),
                p.year_dec == eol_ret,
            ),
            y=p.year_dec == eol_ret,
        )
    )
