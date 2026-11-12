"""Module to write exchange related constraints."""

import numpy as np
from linopy import Constraint, Model
from xarray import Dataset

from pommes.utils import (
    OPERATING_CONDITION_WEIGHT,
    TimeBlocks,
    add_to_balances,
    annual_mask,
    annual_quantity,
    block_dim,
    block_mean,
    block_prefix,
    check_group_coord,
    condition_coords,
    condition_mask,
    expected_over_conditions,
    select_resources,
    time_blocks,
)

# Per-step parameters: Delta-weighted block means on a coarse balance (D10)
_PER_STEP = (
    "net_import_import_price",
    "net_import_export_price",
    "net_import_emission_factor",
    "net_import_total_emission_factor",
)
# Parameters read by the statements below (the dataset a group is selected on)
_READ = (
    *_PER_STEP,
    "net_import_max_yearly_energy_import",
    "net_import_max_yearly_energy_import_expected",
    "net_import_max_yearly_energy_export",
    "net_import_max_yearly_energy_export_expected",
    OPERATING_CONDITION_WEIGHT,
)


def net_import_groups(
    p: Dataset, tb: TimeBlocks
) -> list[tuple[int | None, Dataset]]:
    """
    The net_import parameters of each balance, as ``(L, p_L)`` pairs (section 6.5).

    ``L`` is None for the hourly balance. Without a coarse resource the list
    is ``[(None, p)]``, ``p`` itself and no ``sel`` (T0). Otherwise the
    hourly group is ``p`` restricted to ``tb.resources(None)`` and each coarse
    group ``p_L`` to ``tb.resources(L)``, on the net_import parameters only,
    with the four per-step parameters replaced under the same names by their
    block means on ``time_block_{L}h``: no ``hour`` is left in ``p_L`` (H5).
    Carbon reads the same groups (section 6.7).
    """
    if not tb.balance_lengths:
        return [(None, p)]
    groups = []
    for L in [None, *tb.balance_lengths]:
        res = tb.resources(L)
        if not res:
            continue
        p_g = select_resources(p, _READ, res)
        if L is not None:
            for name in _PER_STEP:
                if name in p_g:
                    # a NaN step reads as 0 in the hourly row, so in the block mean too (R8)
                    p_g[name] = block_mean(p_g[name].fillna(0), tb, L)
        groups.append((L, p_g))
    return groups


def add_net_import(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_year_normalization: float,
    operation_adequacy_constraint: Constraint,
    tb: TimeBlocks | None = None,
) -> Model:
    """
    Add energy net import components to the Linopy model.

    Including variables, constraints, and costs related to imports and exports.

    Args:
        model (linopy.Model):
            The Linopy model to which net import-related elements will be added.
        model_parameters (xarray.Dataset):
            Dataset containing energy system parameters, including net import
            constraints and cost details.
        annualised_totex_def (linopy.Constraint):
            Constraint defining annualised total expenditures (totex), which
            will be updated with net import-specific costs.
        operation_year_normalization (float):
            Normalization factor for operational year durations.
        operation_adequacy_constraint (linopy.Constraint):
            Constraint ensuring operational adequacy, enforcing supply-demand
            balance, which will be updated with net import-related
            contributions.
        tb (pommes.utils.TimeBlocks, optional):
            Block lengths and maps of the dataset, computed once by
            ``build_model`` (``time_blocks(model_parameters)`` when None).
            Without block inputs it is inactive and the statements are
            today's.

    Returns:
        linopy.Model:
            The updated Linopy model with added net import-related variables,
            costs, and constraints.

    Note:
        This function introduces the following elements into the model:

        **Variables**

        - *Operation*
            - ``operation_net_import_import``
                Represents energy imports from the rest of the world (ROW) for each
                area, hour, resource, and operational year.
            - ``operation_net_import_export``
                Represents energy exports to ROW for each area, hour, resource,
                and operational year.
            - ``operation_net_import_abs``
                Represents the absolute net import values, for each
                area, hour, resource, and operational year.
            - ``operation_net_import_net_generation``
                Represents the net energy generation from ROW, calculated for each
                area, hour, resource, and operational year.
            - ``operation_net_import_costs``
                Represents the costs associated with net imports for each area,
                resource, and operational year.

        **Constraints**

        - *Operation*
            - ``operation_net_import_import_yearly_max_constraint``
                Limits the maximum yearly energy imports from ROW based on system
                constraints.
            - ``operation_net_import_export_yearly_max_constraint``
                Limits the maximum yearly energy exports to ROW based on system
                constraints.

            - *Intermediate variables definition*
                - ``operation_net_import_abs_def``
                    Defines the absolute net import value by summing imports and exports.
                - ``operation_net_import_net_generation_def``
                    Defines the net generation from ROW as the difference between
                    imports and exports.

        - *Costs*
            - ``operation_net_import_costs_def``
                Defines operational costs for net imports and exports as a function
                of import/export prices and energy flows.

        These additions ensure that the net import mechanism functions within
        realistic constraints, enforcing energy balance while integrating
        import/export costs dynamically within the model. The model is enhanced
        to accurately capture net import behavior and cost implications.
    """
    tb = time_blocks(model_parameters) if tb is None else tb
    m = model
    p = model_parameters
    groups = net_import_groups(p, tb)

    # ------------
    # Variables
    # ------------

    # Operation - Imports & exports from rest of the world, per balance. A
    # block variable is the energy of its block (no duration in the rows).

    blocks = {}
    for L, p_g in groups:
        blocks[L] = _add_variables(m, p_g, tb, L)

    # Costs - Imports & exports

    operation_net_import_costs = m.add_variables(
        name="operation_net_import_costs",
        coords=[p.area, p.resource, p.year_op, *condition_coords(p)],
        mask=condition_mask(p),
    )

    # ------------------
    # Objective function
    # ------------------

    annualised_totex_def.lhs += expected_over_conditions(
        operation_net_import_costs.sum("resource"), p
    )

    # --------------
    # Constraints
    # --------------

    for L, p_g in groups:
        _add_rows(
            m,
            p_g,
            tb,
            L,
            blocks[L],
            operation_net_import_costs,
            operation_year_normalization,
            operation_adequacy_constraint,
        )

    return m


def _add_variables(m: Model, p: Dataset, tb: TimeBlocks, L: int | None):
    """The four variables of one balance (``L`` None: hourly, today's)."""
    prefix = block_prefix("net_import", L)
    time = p.hour if L is None else tb.duration[L][block_dim(L)]
    coords = [p.area, time, p.resource, p.year_op, *condition_coords(p)]
    mask = condition_mask(p)
    imp = m.add_variables(
        name=f"{prefix}import", lower=0, coords=coords, mask=mask
    )
    exp = m.add_variables(
        name=f"{prefix}export", lower=0, coords=coords, mask=mask
    )
    # Operation - Imports & exports intermediate variables
    abs_ = m.add_variables(name=f"{prefix}abs", lower=0, coords=coords, mask=mask)
    net = m.add_variables(name=f"{prefix}net_generation", coords=coords, mask=mask)
    return imp, exp, abs_, net


def _add_rows(
    m: Model,
    p: Dataset,
    tb: TimeBlocks,
    L: int | None,
    variables,
    costs,
    operation_year_normalization: float,
    operation_adequacy_constraint: Constraint,
) -> None:
    """
    The rows of one balance, written on its own parameters ``p`` (``L`` None: hourly).

    ``T`` is ``hour`` or ``time_block_{L}h``; the hourly statements are
    today's. The block energies sum to the year's energy, so no duration
    enters a row; the prices are block means (``net_import_groups``).
    """
    operation_net_import_import, operation_net_import_export = variables[:2]
    operation_net_import_abs, operation_net_import_net_generation = variables[2:]
    prefix = block_prefix("net_import", L)
    T = "hour" if L is None else block_dim(L)

    # Adequacy constraint

    add_to_balances(
        operation_adequacy_constraint,
        operation_net_import_net_generation,
        tb,
        src=L,
    )

    # Operation - Imports & exports

    # TODO: add max power constraint on transports with ROW

    # Operation - Imports & exports other constraints

    m.add_constraints(
        annual_quantity(
            operation_year_normalization
            * operation_net_import_import.sum(["area", T]),
            p,
            "net_import_max_yearly_energy_import_expected",
        )
        <= p.net_import_max_yearly_energy_import,
        mask=annual_mask(
            p,
            "net_import_max_yearly_energy_import_expected",
            np.isfinite(p.net_import_max_yearly_energy_import),
        ),
        name=f"{prefix}import_yearly_max_constraint",
    )

    m.add_constraints(
        annual_quantity(
            operation_year_normalization
            * operation_net_import_export.sum(["area", T]),
            p,
            "net_import_max_yearly_energy_export_expected",
        )
        <= p.net_import_max_yearly_energy_export,
        mask=annual_mask(
            p,
            "net_import_max_yearly_energy_export_expected",
            np.isfinite(p.net_import_max_yearly_energy_export),
        ),
        name=f"{prefix}export_yearly_max_constraint",
    )

    # Operation - Imports & exports intermediate variables

    m.add_constraints(
        -operation_net_import_abs
        + operation_net_import_import
        + operation_net_import_export
        == 0,
        mask=condition_mask(p),
        name=f"{prefix}abs_def",
    )

    m.add_constraints(
        -operation_net_import_net_generation
        + operation_net_import_import
        - operation_net_import_export
        == 0,
        mask=condition_mask(p),
        name=f"{prefix}net_generation_def",
    )

    # Costs - Imports & exports. Each cell of costs is defined once across
    # the groups: the partner is restricted to the group's resources, in the
    # group's order (H1).

    res = list(p.indexes["resource"])
    partner = costs if res == list(costs.indexes["resource"]) else costs.sel(resource=res)
    m.add_constraints(
        -partner
        # Normalised variable cots with the duration of the operation periods
        # to be consistent with annualised costs
        + operation_year_normalization
        * (
            +p.net_import_import_price * operation_net_import_import
            - p.net_import_export_price * operation_net_import_export
        ).sum(T)
        == 0,
        mask=condition_mask(p),
        name=f"{prefix}costs_def",
    )
    check_group_coord(
        m.constraints[f"{prefix}costs_def"], p.indexes["resource"], "resource"
    )
