"""Index plumbing between retrofit pairs and host module indexes.

The retrofit variable is indexed by ``[site, pair, ...]``. Host modules
index capacity by ``[site, tech, ...]`` (dense) or by a sparse
``{tech_dim}_unit`` MultiIndex dimension that replaces the two. This
module maps between them.

Deliberately imports nothing from ``pommes.model.sparsity``: that
helper module is not guaranteed to be present everywhere this file is
used, so everything needed here is recovered from the host variable's
own coordinates instead of depending on such a helper being
importable.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr
from linopy import LinearExpression, Variable
from linopy.expressions import merge as _lp_merge


def gather_pairs_to_tech(
    expr: Variable | LinearExpression,
    tech_coord: xr.DataArray,
    pair_tech_names: xr.DataArray,
    pair_dim: str,
) -> LinearExpression:
    """
    Move a pair-indexed expression onto a technology dimension.

    Each retrofit pair names one technology (its source or its target).
    This places the pair's contribution on that technology and sums out
    the pair dimension, so the result can be added to a host constraint
    indexed by technology.

    Args:
        expr: linopy ``Variable`` or ``LinearExpression`` carrying
            ``pair_dim``.
        tech_coord: the host module's technology coordinate, e.g.
            ``p.conversion_tech``.
        pair_tech_names: per-pair technology names, e.g.
            ``p.retrofit_conversion_tech_to``.
        pair_dim: name of the pair dimension.

    Returns:
        A ``LinearExpression`` indexed by ``tech_coord``'s dimension
        instead of ``pair_dim``.
    """
    return expr.where(tech_coord == pair_tech_names).sum(pair_dim)


def host_unit_dim(host_var: Variable, tech_dim: str) -> str | None:
    """
    Name of the host variable's sparse unit dimension, if it has one.

    Some host modules are sparsified onto a ``{tech_dim}_unit``
    MultiIndex dimension in place of separate site/tech dimensions;
    others keep the dense ``[site, tech, ...]`` layout. Detecting this
    from the variable itself -- rather than assuming one layout or the
    other -- is what lets one implementation serve both.
    """
    unit_dim = f"{tech_dim}_unit"
    return unit_dim if unit_dim in host_var.dims else None


def unit_positions(
    host_var: Variable, unit_dim: str, site_dim: str, tech_dim: str
) -> tuple[pd.Index, pd.Index]:
    """
    The ``(site, tech)`` label of every unit, in unit order.

    Recovered from the MultiIndex level-name convention that
    ``to_sparse_units`` writes (``_{unit_dim}_{site_dim}`` and
    ``_{unit_dim}_{tech_dim}``), so this needs no access to the host
    module's local ``unit_index``.
    """
    index = host_var.indexes[unit_dim]
    sites = index.get_level_values(f"_{unit_dim}_{site_dim}")
    techs = index.get_level_values(f"_{unit_dim}_{tech_dim}")
    return sites, techs


def gather_tech_to_units(
    expr: xr.DataArray,
    host_var: Variable,
    site_dim: str,
    tech_dim: str,
) -> xr.DataArray:
    """
    Move a ``[site_dim, tech_dim]``-indexed array onto the unit dimension.

    Unlike ``gather_to_units``, ``expr`` already carries the host's own
    ``site_dim`` and ``tech_dim`` coordinates directly -- it is a
    per-technology parameter, not a per-pair one -- so each unit's
    value is a plain ``(site, tech)`` selection rather than a masked
    sum over a pair dimension. A parameter left at its scalar default
    carries neither dimension, in which case the same value applies to
    every unit unchanged -- selecting on a dimension ``expr`` does not
    have would raise, so only the dimensions actually present are
    selected on. Built as the same explicit per-unit loop as
    ``gather_to_units``, for the same reason: it is a plain
    ``xr.DataArray`` here rather than a linopy expression, so
    ``xr.concat`` (not ``gather_to_units``'s linopy merge) does the
    stitching.
    """
    unit_dim = f"{tech_dim}_unit"
    sites, techs = unit_positions(host_var, unit_dim, site_dim, tech_dim)

    parts = []
    for position, (site, tech) in enumerate(zip(sites, techs)):
        selector = {
            dim: label
            for dim, label in ((site_dim, site), (tech_dim, tech))
            if dim in expr.dims
        }
        selected = expr.sel(selector) if selector else expr
        selected = selected.assign_coords(
            {unit_dim: position}
        ).expand_dims(unit_dim)
        parts.append(selected)
    gathered = xr.concat(parts, dim=unit_dim)
    return gathered.assign_coords({unit_dim: host_var.indexes[unit_dim]})


def gather_to_units(
    expr: Variable | LinearExpression,
    host_var: Variable,
    site_dim: str,
    tech_dim: str,
    tech_coord: xr.DataArray,
    pair_tech_names: xr.DataArray,
    pair_dim: str,
) -> LinearExpression:
    """
    Sparse counterpart of ``gather_pairs_to_tech``.

    Produces an expression indexed by the host's unit dimension, so it
    can be added to a sparsified host constraint's ``lhs``.

    Built as an explicit per-unit loop rather than vectorised pointwise
    ``.isel`` on the linopy expression. ``pommes.model.sparsity`` records
    a confirmed case of linopy silently scrambling values under its
    groupby fast path, and the same class of risk applies to pointwise
    indexing of a LinearExpression. The loop uses only ``.sel``/``.sum``/
    ``.assign_coords``/``.expand_dims``, mirroring ``fold_to_area``,
    which is unit-tested against known values.
    """
    unit_dim = f"{tech_dim}_unit"
    sites, techs = unit_positions(host_var, unit_dim, site_dim, tech_dim)

    parts = []
    for position, (site, tech) in enumerate(zip(sites, techs)):
        selected = expr.sel({site_dim: site}).where(
            pair_tech_names == tech
        ).sum(pair_dim)
        selected = selected.assign_coords(
            {unit_dim: position}
        ).expand_dims(unit_dim)
        parts.append(selected)

    gathered = _lp_merge(parts, dim=unit_dim, cls=type(parts[0]))
    return gathered.assign_coords(
        {unit_dim: host_var.indexes[unit_dim]}
    )
