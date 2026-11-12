"""Module to write in model combined related constraints."""

import itertools

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset

from pommes.model.annuity import annuity_full_life, full_life_window
from pommes.utils import (
    TimeBlocks,
    add_to_balances,
    block_dim,
    block_prefix,
    check_group_coord,
    check_partition,
    condition_coords,
    condition_mask,
    expected_over_conditions,
    time_blocks,
)


def combined_groups(
    p: Dataset, tb: TimeBlocks, keep: list[str] | None = None
) -> list[tuple[int | None, Dataset, pd.Index | None]]:
    """
    The groups ``(L, p_g, idx_g)`` of the combined module (section 5.1).

    ``L`` is None for the hourly technologies, else a block length (ascending);
    a group without technology is left out. Dense: ``p_g`` is
    ``p.sel(combined_tech=techs_g)`` and ``idx_g`` is None (no unit
    dimension). Without a block length on a combined technology (T0 included)
    there is one group: ``(None, p, None)``, ``p`` itself. ``keep`` (the
    ``combined_*`` names a reader needs) only matters in the sparse engine,
    where it is applied before the gather.
    """
    lengths = tb.tech_length["combined"].values
    if not (tb.active and np.isfinite(lengths).any()):
        return [(None, p, None)]
    keys = [None, *sorted({int(x) for x in lengths[np.isfinite(lengths)]})]
    parts = check_partition(
        {L: pd.Index(tb.techs("combined", L)) for L in keys},
        p.indexes["combined_tech"],
    )
    return [
        (L, p.sel(combined_tech=list(part)), None) for L, part in parts.items()
    ]


def _net_generation_mask(p: Dataset):
    """Where the net generation of a group exists (a nonzero factor in some mode)."""
    return condition_mask(
        p,
        (np.isfinite(p.combined_factor) * (p.combined_factor != 0)).any(
            [dim for dim in ["mode"] if dim in p.combined_factor.dims]
        ),
    )


def _group_variables(m: Model, p: Dataset, tb: TimeBlocks, L: int | None):
    """Power and net generation of a group, on its own technologies and time axis."""
    prefix = block_prefix("combined", L)
    time = p.hour if L is None else tb.duration[L][block_dim(L)]
    power = m.add_variables(
        name=f"{prefix}power",
        lower=0,
        coords=[
            p.area,
            p.combined_tech,
            p.mode,
            time,
            p.year_op,
            *condition_coords(p),
        ],
        mask=condition_mask(p),
    )
    net_generation = m.add_variables(
        name=f"{prefix}net_generation",
        coords=[
            p.area,
            p.combined_tech,
            time,
            p.resource,
            p.year_op,
            *condition_coords(p),
        ],
        mask=_net_generation_mask(p),
    )
    return power, net_generation


class _CombinedRows:
    """
    The operational rows of one combined group (sections 5.3, 6.2, 6.3).

    ``L`` is None for the hourly group, whose rows are today's; else the rows
    are the hourly rows on the block axis ``T = time_block_{L}h``: the step
    ``D`` is the block duration, the ramp divisor ``D0`` the duration of its
    first step, and ``first`` replaces ``hour != hour[0]``. ``capacity`` and
    ``costs`` are the year-level variables restricted to the group;
    ``p_g`` supplies every parameter (H7). After each row of a split module
    the group's coordinate is checked (H1/H2).
    """

    def __init__(
        self, m, p_g, idx_g, tb, L, power, net_generation, capacity, costs, norm
    ):
        self.m, self.p, self.power = m, p_g, power
        self.net_generation, self.capacity = net_generation, capacity
        self.costs, self.norm = costs, norm
        self.prefix = block_prefix("combined", L)
        if L is None:
            self.T = "hour"
            self.D = self.D0 = p_g.time_step_duration
            self.first = p_g.hour != p_g.hour[0]
        else:
            self.T = block_dim(L)
            self.D, self.D0 = tb.duration[L], tb.first_step_duration[L]
            self.first = tb.duration[L][self.T] != 0
        self.guard = tb.active
        self.dim, self.ref = "combined_tech", p_g.indexes["combined_tech"]

    def add(self, constraint, name, mask=None):
        """Add the row ``prefix + name``; check its coordinate in a split module."""
        con = self.m.add_constraints(
            constraint, name=f"{self.prefix}{name}", mask=mask
        )
        if self.guard:
            check_group_coord(con, self.ref, self.dim)
        return con

    def limits(self):
        """Power bounds, undefined modes, constant operation, ramps, mode-pair ramps."""
        p, power, capacity = self.p, self.power, self.capacity
        T, first = self.T, self.first
        self.add(
            power.sum("mode") - capacity * p.combined_max_load <= 0,
            "power_max_constraint",
            mask=condition_mask(p),
        )
        self.add(
            power.sum("mode") - capacity * p.combined_min_load >= 0,
            "power_min_constraint",
            mask=condition_mask(p),
        )
        self.add(
            power == 0,
            # if combined_factor is not defined for a given mode, then the
            # technology cannot operate under this mode
            "power_undefined_mode",
            mask=condition_mask(p, (p.combined_factor == 0).all(dim="resource")),
        )

        # TODO: must run in combined
        # m.add_constraints(
        #     operation_combined_power - operation_combined_power_capacity * p.combined_must_run
        #     == 0,
        #     name="operation_combined_must_run_constraint_equality",
        #     mask=(p.combined_must_run.sum("mode") == 1),
        # )
        #
        # m.add_constraints(
        #     operation_combined_power - operation_combined_power_capacity * p.combined_must_run >= 0,
        #     name="operation_combined_must_run_constraint_inequality",
        #     mask=(0 < p.combined_must_run) * (p.combined_must_run < 1),
        # )

        # Operation - Combined ramping (no unit commitment: bounds are
        # expressed directly against installed capacity, as there is no on/off
        # state). Ramp applies to the total across modes, not each mode
        # independently (else a 2-mode tech's aggregate output could ramp ~2x
        # too fast).

        combined_ramp_up = p.combined_ramp_up.min(
            [dim for dim in ["mode"] if dim in p.combined_ramp_up.dims]
        )
        combined_ramp_down = p.combined_ramp_down.min(
            [dim for dim in ["mode"] if dim in p.combined_ramp_down.dims]
        )

        combined_mode_ramp_up = p.combined_mode_ramp_up.min(
            [dim for dim in ["mode"] if dim in p.combined_mode_ramp_up.dims]
        )
        combined_mode_ramp_down = p.combined_mode_ramp_down.min(
            [dim for dim in ["mode"] if dim in p.combined_mode_ramp_down.dims]
        )

        # Constant operation (P_t = P_(t-1)) where the ramps are 0: equalities
        # instead of the ramp inequalities, which are removed from the masks
        # below. Aggregate ramps both 0: one equality on the sum over modes.
        # If the mode ramps are both 0 too, one equality per mode instead (it
        # fixes every mode, so the aggregate equality and the pairwise mode
        # rows of these units are redundant and left out). A single 0 keeps
        # the inequalities. The gates read the group's own parameters.
        aggregate_constant = (combined_ramp_up == 0) * (combined_ramp_down == 0)
        mode_constant = (
            aggregate_constant
            * (combined_mode_ramp_up == 0)
            * (combined_mode_ramp_down == 0)
        )

        if (aggregate_constant * ~mode_constant).any():
            self.add(
                power.sum("mode") - power.sum("mode").shift({T: 1}) == 0,
                "constant_operation_constraint",
                mask=condition_mask(
                    p, aggregate_constant * ~mode_constant * first
                ),
            )

        if mode_constant.any():
            self.add(
                power - power.shift({T: 1}) == 0,
                "mode_constant_operation_constraint",
                mask=condition_mask(p, mode_constant * first),
            )

        self.add(
            (power.sum("mode") - power.sum("mode").shift({T: 1})) / self.D0
            - combined_ramp_up * capacity
            <= 0,
            "ramp_up_constraint",
            mask=condition_mask(
                p,
                np.isfinite(combined_ramp_up) * ~aggregate_constant * first,
            ),
        )

        self.add(
            (power.sum("mode").shift({T: 1}) - power.sum("mode")) / self.D0
            - combined_ramp_down * capacity
            <= 0,
            "ramp_down_constraint",
            mask=condition_mask(
                p,
                np.isfinite(combined_ramp_down) * ~aggregate_constant * first,
            ),
        )

        # Optional, opt-in: limits how fast the split between modes shifts,
        # independent of the aggregate ramp above. Off by default (nan).
        # Applied pairwise across all modes -- no new variable: power(mode_a)
        # - power(mode_b) is a linear combination of the existing power
        # variable.

        for mode_a, mode_b in itertools.combinations(p.mode.values, 2):
            combined_mode_balance = power.sel(mode=mode_a) - power.sel(
                mode=mode_b
            )

            self.add(
                (combined_mode_balance - combined_mode_balance.shift({T: 1}))
                / self.D0
                - combined_mode_ramp_up * capacity
                <= 0,
                f"mode_ramp_up_constraint_{mode_a}_{mode_b}",
                mask=condition_mask(
                    p,
                    np.isfinite(combined_mode_ramp_up)
                    * ~mode_constant
                    * first,
                ),
            )

            self.add(
                (combined_mode_balance.shift({T: 1}) - combined_mode_balance)
                / self.D0
                - combined_mode_ramp_down * capacity
                <= 0,
                f"mode_ramp_down_constraint_{mode_a}_{mode_b}",
                mask=condition_mask(
                    p,
                    np.isfinite(combined_mode_ramp_down)
                    * ~mode_constant
                    * first,
                ),
            )

        # TODO: Ramp constraints for lag > 1 not implemented yet and for
        #     storage also

    def net_generation_def(self):
        """Net generation (energy per step or per block) from power and factor."""
        p = self.p
        self.add(
            -self.net_generation
            + self.D * (p.combined_factor * self.power).sum(["mode"])
            == 0,
            "net_generation_def",
            mask=_net_generation_mask(p),
        )

    def costs_def(self):
        """Operational costs: variable cost on the energy, fixed cost on the capacity."""
        p = self.p
        self.add(
            -self.costs
            + self.norm
            * (
                p.combined_variable_cost * (self.D * self.power).sum(self.T)
            ).sum("mode")
            + p.combined_fixed_cost * self.capacity
            == 0,
            "costs_def",
            mask=condition_mask(p),
        )


def add_combined(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_year_normalization: float,
    operation_adequacy_constraint: Constraint,
    tb: TimeBlocks | None = None,
) -> Model:
    """
    Add combined-related components to the Linopy model.

    Including variables, costs, and constraints.

    This module mirrors the structure of the process module (see
    :func:`pommes.model.process.add_process`), covering load-band,
    ramping, and capacity-bound constraints, but purposefully excludes
    unit commitment: there are no integer/boolean variables (no on/off
    state, startup/shutdown, or number-of-units variables), so combined
    technologies are always dispatched continuously.

    Args:
        model (linopy.Model):
            The Linopy model to which combined-related elements will be added.
        model_parameters (xarray.Dataset):
            Dataset containing energy system parameters, including combined
            technology details, load bounds, ramping rates, and production
            limits.
        annualised_totex_def (linopy.Constraint):
            Expression representing the annualised total expenditure (totex),
            which will be updated with combined-specific costs.
        operation_year_normalization (float | numpy.ndarray):
            Normalization factor for operational year durations.
        operation_adequacy_constraint (linopy.Constraint):
            Constraint ensuring operational adequacy, enforcing supply-demand
            balance, which will be updated with combined-related contributions.
        tb (pommes.utils.TimeBlocks, optional):
            Block lengths and maps of the dataset, computed once by
            ``build_model`` (``time_blocks(model_parameters)`` when None).
            Without block inputs it is inactive and the statements are
            today's.

    Returns:
        linopy.Model:
            The updated Linopy model with added combined-related variables,
            costs, and constraints.

    Example:
        This function introduces the following key elements to the model:

        Variables:
        - ``operation_combined_power_capacity``: Operational power capacity
          for each combined technology in each area and operational year.
        - ``operation_combined_power``: Operational power output for each
          combined technology by area, mode, hour, and operational year.
        - ``operation_combined_net_generation``: Net generation of power
          from combined technologies.
        - ``planning_combined_power_capacity``: Planned power capacity
          investments for each combined technology.

        Constraints:
        - Power output limits between a minimum and maximum load factor of
          the installed capacity (no on/off commitment).
        - Ramp-up and ramp-down constraints based on installed capacity.
        - Planning and operational capacity constraints.
        - Net generation definitions ensuring proper accounting.
        - Cost constraints considering fixed and variable costs.

    Note:
        This function introduces the following elements into the model:

        **Variables**

        - *Operation*
            - ``operation_combined_power_capacity``
                Represents the operational power capacity for each combined
                technology by area and operational year.
            - ``operation_combined_power``
                Represents the operational power output for each combined
                technology by area, mode, hour, and operational year.

            - *Intermediate variables*
                - ``operation_combined_net_generation``
                    Represents the net generation from combined technologies
                    calculated for each area, technology, hour, resource, and
                    operational year.

        - *Planning*
            - ``planning_combined_power_capacity``
                Represents the planned power capacity for each combined
                technology by area, decision year, and investment year.

        - *Costs (intermediate variables)*:
            - ``operation_combined_costs``
                Represents the operational costs associated with combined
                technologies for each area, technology, and operational year.
            - ``planning_combined_costs``
                Represents the planning costs associated with combined
                technologies for each area, technology, and operational year.

        **Constraints**

        - *Operation*
            - ``operation_combined_power_max_constraint``
                Limits operational power to the installed power capacity
                scaled by ``combined_max_load``.
            - ``operation_combined_power_min_constraint``
                Enforces operational power to stay above the installed power
                capacity scaled by ``combined_min_load``.
            - ``operation_combined_power_undefined_mode``
                Forces power to zero for modes that are not defined (i.e.
                ``combined_factor`` is zero for every resource in that mode).
            - ``operation_combined_ramp_up_constraint``
                Constrains the ramp-up rate of a combined technology's total
                output (summed across mode), based on the installed power
                capacity. Not written where the aggregate ramp up and ramp
                down are both 0 (see the constant operation constraints).
            - ``operation_combined_ramp_down_constraint``
                Constrains the ramp-down rate of a combined technology's total
                output (summed across mode), based on the installed power
                capacity. Not written where the aggregate ramp up and ramp
                down are both 0.
            - ``operation_combined_mode_ramp_up_constraint``
                Optional, opt-in (``combined_mode_ramp_up``, default nan/off):
                limits how fast the split between each pair of modes can
                shift (a fuel/pathway-shift rate), independent of the
                aggregate ramp above. Not written where the aggregate ramps
                and the mode ramps are all 0.
            - ``operation_combined_mode_ramp_down_constraint``
                Optional, opt-in (``combined_mode_ramp_down``, default
                nan/off): the opposite-direction counterpart of
                ``operation_combined_mode_ramp_up_constraint``.
            - ``operation_combined_constant_operation_constraint``
                Holds the total output (summed across mode) constant from one
                time step to the next, ``P_t = P_(t-1)``, where the aggregate
                ``combined_ramp_up`` and ``combined_ramp_down`` (after the
                minimum over mode) are both 0. One equality replaces the two
                aggregate ramp rows. Not written where the mode ramps are 0
                too (next constraint).
            - ``operation_combined_mode_constant_operation_constraint``
                Holds the output of each mode constant, ``P_t = P_(t-1)``,
                where the aggregate ramps and ``combined_mode_ramp_up`` and
                ``combined_mode_ramp_down`` (after the minimum over mode) are
                all 0. It replaces the aggregate equality and the pairwise
                mode ramp rows of these technologies: a constant sum and
                constant pairwise differences fix every mode. Any other
                combination of ramp values keeps the ramp inequalities.

            - *Operational capacity*
                - ``operation_combined_power_capacity_max_constraint``
                    Sets an upper limit on the operation power capacity for
                    each combined technology.
                - ``operation_combined_power_capacity_min_constraint``
                    Sets a lower limit on the operation power capacity for
                    each combined technology.
                - ``operation_combined_power_capacity_def``
                    Defines the operational power capacity based on the
                    planned investments over the years.

            - *Intermediate variables definition*
                - ``operation_combined_net_generation_def``
                    Relates net generation to power output and conversion
                    factor, ensuring consistent accounting.

        - *Planning*
            - ``planning_combined_power_capacity_min_constraint``
                Sets a lower limit on planned power capacity investments
                for combined technologies.
            - ``planning_combined_power_capacity_max_constraint``
                Sets an upper limit on planned power capacity investments for
                combined technologies.
            - ``planning_combined_power_capacity_def``
                Ensures planned power capacity matches the minimum requirement
                when upper and lower limits are equal.

        - *Costs*
            - ``operation_combined_costs_def``
                Defines operational costs as a function of variable costs,
                power output, and fixed costs.
            - ``planning_combined_costs_def``
                Defines planning costs, accounting for investment costs spread
                over operational periods, with optional perfect foresight.

        These additions ensure that the combined technologies operate within
        feasible and efficient limits, respecting load-band, ramping
        capabilities, and power capacity limits, without resorting to
        integer or boolean unit-commitment variables. The model is thereby
        enhanced to accurately simulate combined energy behavior and costs.
    """
    tb = time_blocks(model_parameters) if tb is None else tb
    m = model
    p = model_parameters

    # Today's statements on p itself unless some combined tech has a block
    # length: then the hourly techs and each block length form a group.
    groups = combined_groups(p, tb)
    blocked = groups[0][1] is not p
    hourly = [g for g in groups if g[0] is None]
    blocks = [g for g in groups if g[0] is not None]

    def rows_of(L, p_g, idx_g, variables):
        """The row object of a group, its partners restricted to its technologies (H1)."""

        def restrict(var):
            return (
                var.sel(combined_tech=p_g.indexes["combined_tech"])
                if blocked
                else var
            )

        return _CombinedRows(
            m,
            p_g,
            idx_g,
            tb,
            L,
            *variables,
            restrict(operation_combined_power_capacity),
            restrict(operation_combined_costs),
            operation_year_normalization,
        )

    # ------------
    # Variables
    # ------------

    # Operation - Combined

    operation_combined_power_capacity = m.add_variables(
        name="operation_combined_power_capacity",
        lower=0,
        coords=[p.area, p.combined_tech, p.year_op],
    )

    # Operation - Combined power and intermediate variables (the hourly
    # group's, in today's order)

    if hourly:
        _, p_h, idx_h = hourly[0]
        hourly_variables = _group_variables(m, p_h, tb, None)

    # Planning - Combined

    planning_combined_power_capacity = m.add_variables(
        name="planning_combined_power_capacity",
        lower=0,
        coords=[p.area, p.combined_tech, p.year_dec, p.year_inv],
        mask=xr.where(
            cond=p.combined_early_decommissioning,
            x=(p.year_inv < p.year_dec)
            * (p.year_dec <= p.combined_end_of_life)
            * np.logical_or(
                p.year_dec <= p.year_inv.max(),
                p.year_dec == p.combined_end_of_life,
            ),
            y=p.year_dec == p.combined_end_of_life,
        ),
    )

    # Costs - Combined

    operation_combined_costs = m.add_variables(
        name="operation_combined_costs",
        coords=[p.area, p.combined_tech, p.year_op, *condition_coords(p)],
        mask=condition_mask(p),
    )

    planning_combined_costs = m.add_variables(
        name="planning_combined_costs",
        coords=[p.area, p.combined_tech, p.year_op],
    )

    if hourly:
        hourly_rows = rows_of(None, p_h, idx_h, hourly_variables)

    # ------------------
    # Objective function
    # ------------------

    annualised_totex_def.lhs += expected_over_conditions(
        operation_combined_costs.sum("combined_tech"), p
    ) + planning_combined_costs.sum("combined_tech")

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    if hourly:
        add_to_balances(
            operation_adequacy_constraint,
            hourly_variables[1].sum("combined_tech"),
            tb,
        )

    # Operation - Combined (power bounds, ramps)

    if hourly:
        hourly_rows.limits()

    # Operation capacity

    m.add_constraints(
        operation_combined_power_capacity <= p.combined_power_capacity_max,
        name="operation_combined_power_capacity_max_constraint",
        mask=np.isfinite(p.combined_power_capacity_max),
    )

    m.add_constraints(
        operation_combined_power_capacity >= p.combined_power_capacity_min,
        name="operation_combined_power_capacity_min_constraint",
        mask=np.isfinite(p.combined_power_capacity_min),
    )

    # Operation - Combined intermediate variables

    m.add_constraints(
        -operation_combined_power_capacity
        + planning_combined_power_capacity.where(
            (p.year_inv <= p.year_op) * (p.year_op < p.year_dec)
        ).sum(["year_dec", "year_inv"])
        == 0,
        name="operation_combined_power_capacity_def",
    )

    if hourly:
        hourly_rows.net_generation_def()

    # Planning - Combined

    m.add_constraints(
        planning_combined_power_capacity.sum("year_dec")
        <= p.combined_power_capacity_investment_max,
        name="planning_combined_power_capacity_max_constraint",
        mask=np.isfinite(p.combined_power_capacity_investment_max)
        * np.not_equal(
            p.combined_power_capacity_investment_max,
            p.combined_power_capacity_investment_min,
        ),
    )

    m.add_constraints(
        planning_combined_power_capacity.sum("year_dec")
        >= p.combined_power_capacity_investment_min,
        name="planning_combined_power_capacity_min_constraint",
        mask=np.isfinite(p.combined_power_capacity_investment_min)
        * np.not_equal(
            p.combined_power_capacity_investment_max,
            p.combined_power_capacity_investment_min,
        ),
    )

    m.add_constraints(
        planning_combined_power_capacity.sum("year_dec")
        == p.combined_power_capacity_investment_min,
        name="planning_combined_power_capacity_def",
        mask=np.isfinite(p.combined_power_capacity_investment_min)
        * np.equal(
            p.combined_power_capacity_investment_max,
            p.combined_power_capacity_investment_min,
        ),
    )

    # Costs - Combined

    if hourly:
        hourly_rows.costs_def()

    if annuity_full_life(p, "combined_annuity_cost"):
        # Annuity fixed at investment, paid over the vintage's full life
        # whatever the decommissioning year (sunk cost).
        annuity_terms = (
            (
                planning_combined_power_capacity.sum("year_dec")
                * p.combined_annuity_cost
            )
            .where(
                full_life_window(p.year_inv, p.combined_life_span, p.year_op)
            )
            .sum("year_inv")
        )
    else:
        # The flag has a year_inv dimension: choose the annuity per vintage,
        # then sum over the vintages (summing first and choosing afterwards
        # gives the expression a year_inv dimension back and writes the cost
        # row once per vintage).
        perfect_foresight = (
            (planning_combined_power_capacity * p.combined_annuity_cost)
            .where((p.year_inv <= p.year_op) * (p.year_op < p.year_dec))
            .sum("year_dec")
        )
        cheapest = (
            planning_combined_power_capacity.sum("year_dec")
            * p.combined_annuity_cost.min(
                [
                    dim
                    for dim in p.combined_annuity_cost.dims
                    if dim == "year_dec"
                ]
            )
        ).where(
            (p.year_inv <= p.year_op) * (p.year_op < p.combined_end_of_life)
        )
        annuity_terms = perfect_foresight.where(
            cond=p.combined_annuity_perfect_foresight, other=cheapest
        ).sum("year_inv")

    m.add_constraints(
        -planning_combined_costs + annuity_terms == 0,
        name="planning_combined_costs_def",
    )

    # Block groups, after the module's last statement: their labels follow
    # every hourly label, and each row is the hourly row on the block axis.

    for L, p_g, idx_g in blocks:
        variables = _group_variables(m, p_g, tb, L)
        add_to_balances(
            operation_adequacy_constraint,
            variables[1].sum("combined_tech"),
            tb,
            src=L,
        )
        rows = rows_of(L, p_g, idx_g, variables)
        rows.limits()
        rows.net_generation_def()
        rows.costs_def()

    return m
