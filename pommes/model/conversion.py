"""Module adding conversion technologies ot the linopy Model."""

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset

from pommes.model.annuity import annuity_full_life, full_life_window
from pommes.utils import (
    TimeBlocks,
    add_to_balances,
    annual_mask,
    annual_quantity,
    block_dim,
    block_prefix,
    check_axis,
    check_block_daily_cap,
    check_group,
    check_group_coord,
    check_partition,
    condition_coords,
    condition_mask,
    conversion_block_availability,
    day_of_blocks,
    expected_over_conditions,
    longest_day_in_blocks,
    time_blocks,
)


def conversion_groups(
    p: Dataset, tb: TimeBlocks, keep: list[str] | None = None
) -> list[tuple[int | None, Dataset, pd.Index | None]]:
    """
    The groups ``(L, p_g, idx_g)`` of the conversion module (section 5.1).

    ``L`` is None for the hourly technologies, else a block length (ascending);
    a group without technology is left out. Dense: ``p_g`` is
    ``p.sel(conversion_tech=techs_g)`` and ``idx_g`` is None (no unit
    dimension). A block group holds ``conversion_block_availability`` (section
    6.1, computed on the full ``p``) and no ``conversion_availability``.
    Without a block length on a conversion technology (T0 included) there is
    one group: ``(None, p, None)``, ``p`` itself. ``keep`` (the
    ``conversion_*`` names a reader needs) only matters in the sparse engine,
    where it is applied before the gather.
    """
    lengths = tb.tech_length["conversion"].values
    if not (tb.active and np.isfinite(lengths).any()):
        return [(None, p, None)]
    keys = [None, *sorted({int(x) for x in lengths[np.isfinite(lengths)]})]
    parts = check_partition(
        {L: pd.Index(tb.techs("conversion", L)) for L in keys},
        p.indexes["conversion_tech"],
    )
    out = []
    for L, part in parts.items():
        if L is None:
            p_g = p.sel(conversion_tech=list(part))
        else:  # no copy of the hourly availability: the group holds its block value
            p_g = p.drop_vars("conversion_availability", errors="ignore").sel(
                conversion_tech=list(part)
            )
            p_g = p_g.assign(
                conversion_block_availability=conversion_block_availability(
                    p, tb, L, list(part)
                )
            )
        out.append((L, p_g, None))
    return out


def _net_generation_mask(p: Dataset):
    """Where the net generation of a group exists (a nonzero factor)."""
    return condition_mask(
        p, np.isfinite(p.conversion_factor) * (p.conversion_factor != 0)
    )


def _group_variables(m: Model, p: Dataset, tb: TimeBlocks, L: int | None):
    """Power and net generation of a group, on its own technologies and time axis."""
    prefix = block_prefix("conversion", L)
    time = p.hour if L is None else tb.duration[L][block_dim(L)]
    power = m.add_variables(
        name=f"{prefix}power",
        lower=0,
        coords=[
            p.area,
            p.conversion_tech,
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
            p.conversion_tech,
            time,
            p.resource,
            p.year_op,
            *condition_coords(p),
        ],
        mask=_net_generation_mask(p),
    )
    return power, net_generation


class _ConversionRows:
    """
    The operational rows of one conversion group (sections 5.3, 6.1, 8.5).

    ``L`` is None for the hourly group, whose rows are today's; else the rows
    are the hourly rows on the block axis ``T = time_block_{L}h``: the step
    ``D`` is the block duration, the ramp divisor ``D0`` the duration of its
    first step, ``first`` replaces ``hour != hour[0]`` and the availability
    is ``conversion_block_availability`` (the rule's block value, D9').
    ``capacity`` and ``costs`` are the year-level variables restricted to the
    group; ``p_g`` supplies every parameter (H7). After each row of a split
    module the group's coordinate is checked (H1/H2).
    """

    def __init__(
        self, m, p_g, idx_g, tb, L, power, net_generation, capacity, costs, norm
    ):
        self.m, self.p, self.power = m, p_g, power
        self.net_generation, self.capacity = net_generation, capacity
        self.costs, self.norm, self.tb, self.L = costs, norm, tb, L
        self.prefix = block_prefix("conversion", L)
        if L is None:
            self.T = "hour"
            self.D = self.D0 = p_g.time_step_duration
            self.first = p_g.hour != p_g.hour[0]
        else:
            self.T = block_dim(L)
            self.D, self.D0 = tb.duration[L], tb.first_step_duration[L]
            self.first = tb.duration[L][self.T] != 0
        self.guard = tb.active
        self.dim, self.ref = "conversion_tech", p_g.indexes["conversion_tech"]

    @property
    def availability(self):
        """Raw availability of the hourly group, the block value of a block group."""
        p = self.p
        return (
            p.conversion_availability
            if self.L is None
            else p.conversion_block_availability
        )

    def add(self, constraint, name, mask=None):
        """Add the row ``prefix + name``; check its coordinate in a split module."""
        con = self.m.add_constraints(
            constraint, name=f"{self.prefix}{name}", mask=mask
        )
        if self.guard:
            check_group_coord(con, self.ref, self.dim)
        return con

    def limits(self):
        """Power bounds, must-run, constant operation, ramps and production caps."""
        p, power, capacity, norm = self.p, self.power, self.capacity, self.norm
        T, D, first, a = self.T, self.D, self.first, self.availability
        self.add(
            power - capacity * xr.where(cond=np.isfinite(a), x=a, y=1) <= 0,
            "power_max_constraint",
            mask=condition_mask(
                p,
                np.logical_or(
                    np.isnan(p.conversion_must_run),
                    p.conversion_must_run < 1,
                ),
            ),
        )

        self.add(
            power - capacity * a == 0,
            "must_run_constraint_equality",
            mask=condition_mask(p, p.conversion_must_run == 1),
        )

        self.add(
            power
            - capacity * xr.where(np.isfinite(a), a, 1) * p.conversion_must_run
            >= 0,
            "must_run_constraint_inequality",
            mask=condition_mask(
                p, (0 < p.conversion_must_run) * (p.conversion_must_run < 1)
            ),
        )

        # Operation - Conversion unit commitment

        # Ramp up and ramp down both 0: the power is constant, P_t - P_(t-1) = 0.
        # One equality instead of the two inequalities (which are removed from
        # the masks of the two ramp constraints below). A single 0 keeps them.
        constant_operation = (p.conversion_ramp_up == 0) * (
            p.conversion_ramp_down == 0
        )

        if constant_operation.any():
            self.add(
                power - power.shift({T: 1}) == 0,
                "constant_operation_constraint",
                mask=condition_mask(p, constant_operation * first),
            )

        self.add(
            -p.conversion_ramp_up * capacity
            + (power - power.shift({T: 1})) / self.D0
            <= 0,
            "ramp_up_constraint",
            mask=condition_mask(
                p,
                np.isfinite(p.conversion_ramp_up) * ~constant_operation * first,
            ),
        )

        # TODO : implement ramp based on available capacity, not all capacity
        self.add(
            -p.conversion_ramp_down * capacity
            + (power.shift({T: 1}) - power) / self.D0
            <= 0,
            "ramp_down_constraint",
            mask=condition_mask(
                p,
                np.isfinite(p.conversion_ramp_down)
                * ~constant_operation
                * first,
            ),
        )

        # TODO: Ramp constraints fo lag > 1 not implemented yet and for storage
        #     also

        # Operation - Conversion other constraints

        self.add(
            annual_quantity(
                +norm * (D * power).sum([T]),
                p,
                "conversion_max_yearly_production_expected",
            )
            <= p.conversion_max_yearly_production,
            "yearly_production_max_constraint",
            mask=annual_mask(
                p,
                "conversion_max_yearly_production_expected",
                np.isfinite(p.conversion_max_yearly_production),
            ),
        )

        # The year's output at most conversion_max_capacity_factor times the
        # installed capacity over the year (refuelling and maintenance of a
        # nuclear fleet, seasonal availability). Unlike
        # conversion_max_yearly_production, a fixed energy, it follows the
        # capacity the model builds.
        self.add(
            norm * (D * power).sum([T])
            <= p.conversion_max_capacity_factor
            * p.operation_year_duration
            * capacity,
            "capacity_factor_max_constraint",
            mask=condition_mask(p, np.isfinite(p.conversion_max_capacity_factor)),
        )

        # Production summed over the operation years of the run, each standing for
        # operation_year_weight calendar years (a CO2 storage site's total capacity). In a myopic
        # run, give the capacity the earlier runs left. Optional: datasets built without
        # check_inputs may not have the parameter.
        if (
            "conversion_max_cumulative_production" in p
            and np.isfinite(p.conversion_max_cumulative_production).any()
        ):
            if "operation_year_weight" not in p or not np.isfinite(p.operation_year_weight).all():
                raise ValueError(
                    "conversion_max_cumulative_production needs operation_year_weight "
                    "(calendar years per operation year) for every year_op"
                )
            self.add(
                (
                    norm
                    * p.operation_year_weight
                    * expected_over_conditions((D * power).sum([T]), p)
                ).sum("year_op")
                <= p.conversion_max_cumulative_production,
                "cumulative_production_max_constraint",
                mask=np.isfinite(p.conversion_max_cumulative_production),
            )

    def daily(self):
        """
        Maximum daily production (for eDSR specifically), only when a limit is set.

        A time step belongs to day floor(start time / operation_day_duration), its
        start time being the cumulative duration of the steps before it, so steps
        longer than 1 h are handled; a last, shorter day is kept. A block group
        nests with the day (R6): blocks inside days sum their blocks per day; days
        inside blocks give one row per block on the longest day it holds (section 8.5).
        """
        p, power, norm = self.p, self.power, self.norm
        if not np.isfinite(p.conversion_max_daily_production).any():
            return
        mask = condition_mask(p, np.isfinite(p.conversion_max_daily_production))
        if self.L is None:
            start_time = p.time_step_duration.cumsum("hour") - p.time_step_duration
            day = (start_time // p.operation_day_duration).astype("int64").rename("day")
            self.add(
                +norm * (p.time_step_duration * power).groupby(day).sum()
                <= p.conversion_max_daily_production,
                "daily_production_max_constraint",
                mask=mask,
            )
            return
        L, T, tb = self.L, self.T, self.tb
        d = int(p.operation_day_duration)
        if d % L == 0:  # blocks inside days
            day = day_of_blocks(tb, L, d)
            check_group(day, T)
            expr = self.D * power
            check_axis(expr, T, tb.duration[L].indexes[T])
            self.add(
                +norm * expr.groupby(day).sum()
                <= p.conversion_max_daily_production,
                "daily_production_max_constraint",
                mask=mask,
            )
        else:  # days inside blocks (R6: L is a multiple of d)
            self.add(
                +norm * longest_day_in_blocks(tb, L, d) * power
                <= p.conversion_max_daily_production,
                "daily_production_max_constraint",
                mask=mask,
            )

    def net_generation_def(self):
        """Net generation (energy per step or per block) from power and factor."""
        p = self.p
        self.add(
            -self.net_generation + self.D * (p.conversion_factor * self.power)
            == 0,
            "net_generation_def",
            mask=_net_generation_mask(p),
        )

    def costs_def(self):
        """Operational costs: variable cost on the energy, fixed cost on the capacity."""
        p = self.p
        self.add(
            -self.costs
            + (
                p.conversion_variable_cost
                * self.norm
                * (self.D * self.power).sum(self.T)
            )
            + (p.conversion_fixed_cost * self.capacity)
            == 0,
            "costs_def",
            mask=condition_mask(p),
        )


def add_conversion(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_year_normalization: float,
    operation_adequacy_constraint: Constraint,
    tb: TimeBlocks | None = None,
) -> Model:
    """
    Add conversion-related components to the Linopy model.

    Including variables, costs, and constraints.

    Args:
        model (linopy.Model):
            The Linopy model to which conversion-related elements will be
            added.
        model_parameters (xarray.Dataset):
            Dataset containing energy system parameters, including conversion
            technology details, availability, ramping rates, and production
            limits.
        annualised_totex_def (linopy.Constraint):
            Constraint defining the annualised total expenditure (totex),
            which will be updated with conversion-specific costs.
        operation_year_normalization (float):
            Normalization factor for operational year durations.
        operation_adequacy_constraint (linopy.Constraint):
            Constraint defining operational adequacy, ensuring supply meets
            demand. This will be updated with conversion-related contributions.
        tb (pommes.utils.TimeBlocks, optional):
            Block lengths and maps of the dataset, computed once by
            ``build_model`` (``time_blocks(model_parameters)`` when None).
            Without block inputs it is inactive and the statements are
            today's.

    Returns:
        linopy.Model:
            The updated Linopy model with added conversion-related variables,
            costs, and constraints.

    Example:
        This function introduces the following key elements to the model:

        Variables:
        - ``operation_conversion_power_capacity``: Operational power capacity
          for each conversion technology in each area and operational year.
        - ``operation_conversion_power``: Operational power output for each
          conversion technology by area, hour, and operational year.
        - ``operation_conversion_net_generation``: Net generation of power
          from conversion technologies.
        - ``planning_conversion_power_capacity``: Planned power capacity
          investments for each conversion technology.

        Constraints:
        - Operational power limits based on availability.
        - Ramp-up and ramp-down constraints for smooth power output changes.
        - Production and capacity limits for yearly operations.
        - Definitions for intermediate variables like net generation.
        - Cost constraints accounting for variable and fixed costs.

    Note:
        This function introduces the following elements into the model:

        **Variables**

        - *Operation*
            - ``operation_conversion_power_capacity``
                Represents the operational power capacity for each conversion
                technology in each area and operational year.
            - ``operation_conversion_power``
                Represents the operational power output for each conversion
                technology by area, hour, and operational year.

            - *Intermediate variables*
                - ``operation_conversion_net_generation``
                    Represents the net generation of power from conversion
                    technologies, calculated for each area, technology, hour,
                    resource, and operational year.

        - *Planning*
            - ``planning_conversion_power_capacity``
                Represents the planned power capacity for each conversion
                technology by area, decision year, and investment year.

        - *Costs (intermediate variables)*:
            - ``operation_conversion_costs``
                Represents the operational costs associated with the conversion
                technologies for each area, technology, and operational year.
            - ``planning_conversion_costs``
                Represents the planning costs associated with the conversion
                technologies for each area, technology, and operational year.

        **Constraints**

            - *Operation*
                - ``operation_conversion_power_max_constraint``
                    Limits operational power to the available power capacity
                    adjusted by availability factors.
                - ``operation_conversion_must_run_constraint_equality``
                    Ensures specific technologies operate at their must-run levels,
                    where applicable.
                - ``operation_conversion_must_run_constraint_inequality``
                    Enforces partial must-run conditions for certain technologies
                    based on availability.
                - ``operation_conversion_ramp_up_constraint``
                    Constrains the ramp-up rate of conversion technologies,
                    ensuring smooth increases in power output (not written
                    where ramp up and ramp down are both 0, see the constant
                    operation constraint).
                - ``operation_conversion_ramp_down_constraint``
                    Constrains the ramp-down rate of conversion technologies,
                    ensuring smooth decreases in power output (not written
                    where ramp up and ramp down are both 0).
                - ``operation_conversion_constant_operation_constraint``
                    Holds the power constant from one time step to the next
                    (``P_t = P_(t-1)``) where ``conversion_ramp_up`` and
                    ``conversion_ramp_down`` are both 0. One equality replaces
                    the two ramp rows of these technologies. A single 0 (for
                    example ramp up 0 and ramp down NaN, "never increase") and
                    any other value keep the ramp inequalities.
                - ``operation_conversion_yearly_production_max_constraint``
                    Limits the yearly production of conversion technologies
                    to a maximum value.
                - ``operation_conversion_cumulative_production_max_constraint``
                    Limits the production summed over the operation years,
                    each weighted by ``operation_year_weight`` calendar years
                    (only where ``conversion_max_cumulative_production`` is set).
                - ``operation_conversion_capacity_factor_max_constraint``
                    Limits the yearly production to
                    ``conversion_max_capacity_factor`` times the installed
                    capacity over the year, so the cap follows the capacity
                    built.

                - *Operational capacity*
                    - ``operation_conversion_power_capacity_max_constraint``
                        Sets an upper limit on the operation power capacity for
                        each conversion technology.
                    - ``operation_conversion_power_capacity_min_constraint``
                        Sets a lower limit on the operation power capacity for
                        each conversion technology.

                - *Intermediate variables definition*
                    - ``operation_conversion_power_capacity_def``
                        Defines the operational power capacity based on the
                        planned investments over the years.
                    - ``operation_conversion_net_generation_def``
                        Relates net generation to the power output and conversion
                        factor, ensuring consistent accounting.
                        :eq:`equation conversion net generation <conv-net-gen-def>`

            - *Planning*
                - ``planning_conversion_power_capacity_max_constraint``
                    Sets an upper limit on planned power capacity investments
                    for conversion technologies.
                - ``planning_conversion_power_capacity_min_constraint``
                    Sets a lower limit on planned power capacity investments
                    for conversion technologies.
                - ``planning_conversion_power_capacity_def``
                    Ensures planned capacity matches the minimum requirement
                    when upper and lower limits are equal.

            - *Costs*
                - ``operation_conversion_costs_def``
                    Defines operational costs as a function of variable costs,
                    power output, and fixed costs.
                - ``planning_conversion_costs_def``
                    Defines planning costs, accounting for investment costs spread
                    over operational periods, with optional perfect foresight.

        These additions ensure that the conversion technologies operate within
        feasible and efficient limits, respecting availability,
        must-run requirements, ramping capabilities, and power capacity limits.
        The model is thereby enhanced to accurately simulate
        energy conversion behavior and costs.
    """
    tb = time_blocks(model_parameters) if tb is None else tb
    m = model
    p = model_parameters

    # R6 on the full dataset, before any selection (build-time twin of check_inputs)
    check_block_daily_cap(p, tb)

    # Today's statements on p itself unless some conversion tech has a block
    # length: then the hourly techs and each block length form a group.
    groups = conversion_groups(p, tb)
    blocked = groups[0][1] is not p
    hourly = [g for g in groups if g[0] is None]
    blocks = [g for g in groups if g[0] is not None]

    def rows_of(L, p_g, idx_g, variables):
        """The row object of a group, its partners restricted to its technologies (H1)."""

        def restrict(var):
            return (
                var.sel(conversion_tech=p_g.indexes["conversion_tech"])
                if blocked
                else var
            )

        return _ConversionRows(
            m,
            p_g,
            idx_g,
            tb,
            L,
            *variables,
            restrict(operation_conversion_power_capacity),
            restrict(operation_conversion_costs),
            operation_year_normalization,
        )

    # ------------
    # Variables
    # ------------

    # Operation - Conversion

    operation_conversion_power_capacity = m.add_variables(
        name="operation_conversion_power_capacity",
        lower=0,
        coords=[p.area, p.conversion_tech, p.year_op],
    )

    # Operation - Conversion power and intermediate variables (the hourly
    # group's, in today's order)

    if hourly:
        _, p_h, idx_h = hourly[0]
        hourly_variables = _group_variables(m, p_h, tb, None)

    # Planning - Conversion

    planning_conversion_power_capacity = m.add_variables(
        name="planning_conversion_power_capacity",
        lower=0,
        coords=[p.area, p.conversion_tech, p.year_dec, p.year_inv],
        mask=xr.where(
            cond=p.conversion_early_decommissioning,
            x=(p.year_inv < p.year_dec)
            * (p.year_dec <= p.conversion_end_of_life)
            * np.logical_or(
                p.year_dec <= p.year_inv.max(),
                p.year_dec == p.conversion_end_of_life,
            ),
            y=p.year_dec == p.conversion_end_of_life,
        ),
    )

    # Costs - Conversion

    operation_conversion_costs = m.add_variables(
        name="operation_conversion_costs",
        coords=[p.area, p.conversion_tech, p.year_op, *condition_coords(p)],
        mask=condition_mask(p),
    )

    planning_conversion_costs = m.add_variables(
        name="planning_conversion_costs",
        coords=[p.area, p.conversion_tech, p.year_op],
    )

    if hourly:
        hourly_rows = rows_of(None, p_h, idx_h, hourly_variables)

    # ------------------
    # Objective function
    # ------------------

    annualised_totex_def.lhs += expected_over_conditions(
        operation_conversion_costs.sum("conversion_tech"), p
    ) + planning_conversion_costs.sum("conversion_tech")

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    if hourly:
        add_to_balances(
            operation_adequacy_constraint,
            hourly_variables[1].sum("conversion_tech"),
            tb,
        )

    # Operation - Conversion (power bounds, must-run, ramps, production caps)

    if hourly:
        hourly_rows.limits()

    m.add_constraints(
        operation_conversion_power_capacity <= p.conversion_power_capacity_max,
        name="operation_conversion_power_capacity_max_constraint",
        mask=np.isfinite(p.conversion_power_capacity_max),
    )

    m.add_constraints(
        operation_conversion_power_capacity >= p.conversion_power_capacity_min,
        name="operation_conversion_power_capacity_min_constraint",
        mask=np.isfinite(p.conversion_power_capacity_min),
    )

    # Operation - maximum daily production

    if hourly:
        hourly_rows.daily()

    # Operation - Conversion intermediate variables

    m.add_constraints(
        -operation_conversion_power_capacity
        + planning_conversion_power_capacity.where(
            (p.year_inv <= p.year_op) * (p.year_op < p.year_dec)
        ).sum(["year_dec", "year_inv"])
        == 0,
        name="operation_conversion_power_capacity_def",
    )

    if hourly:
        hourly_rows.net_generation_def()

    # Planning - Conversion

    m.add_constraints(
        planning_conversion_power_capacity.sum("year_dec")
        <= p.conversion_power_capacity_investment_max,
        name="planning_conversion_power_capacity_max_constraint",
        mask=np.isfinite(p.conversion_power_capacity_investment_max)
        * np.not_equal(
            p.conversion_power_capacity_investment_max,
            p.conversion_power_capacity_investment_min,
        ),
    )

    m.add_constraints(
        planning_conversion_power_capacity.sum("year_dec")
        >= p.conversion_power_capacity_investment_min,
        name="planning_conversion_power_capacity_min_constraint",
        mask=np.isfinite(p.conversion_power_capacity_investment_min)
        * np.not_equal(
            p.conversion_power_capacity_investment_max,
            p.conversion_power_capacity_investment_min,
        ),
    )

    m.add_constraints(
        planning_conversion_power_capacity.sum("year_dec")
        == p.conversion_power_capacity_investment_min,
        name="planning_conversion_power_capacity_def",
        mask=np.isfinite(p.conversion_power_capacity_investment_min)
        * np.equal(
            p.conversion_power_capacity_investment_max,
            p.conversion_power_capacity_investment_min,
        ),
    )

    # Costs - Conversion

    if hourly:
        hourly_rows.costs_def()

    if annuity_full_life(p, "conversion_annuity_cost"):
        # Annuity fixed at investment, paid over the vintage's full life
        # whatever the decommissioning year (sunk cost).
        annuity_terms = (
            (
                planning_conversion_power_capacity.sum("year_dec")
                * p.conversion_annuity_cost
            )
            .where(
                full_life_window(p.year_inv, p.conversion_life_span, p.year_op)
            )
            .sum("year_inv")
        )
    else:
        # The flag has a year_inv dimension: choose the annuity per vintage,
        # then sum over the vintages (summing first and choosing afterwards
        # gives the expression a year_inv dimension back and writes the cost
        # row once per vintage).
        perfect_foresight = (
            (planning_conversion_power_capacity * p.conversion_annuity_cost)
            .where((p.year_inv <= p.year_op) * (p.year_op < p.year_dec))
            .sum("year_dec")
        )
        cheapest = (
            planning_conversion_power_capacity.sum("year_dec")
            * p.conversion_annuity_cost.min(
                [
                    dim
                    for dim in p.conversion_annuity_cost.dims
                    if dim == "year_dec"
                ]
            )
        ).where(
            (p.year_inv <= p.year_op) * (p.year_op < p.conversion_end_of_life)
        )
        annuity_terms = perfect_foresight.where(
            cond=p.conversion_annuity_perfect_foresight, other=cheapest
        ).sum("year_inv")

    m.add_constraints(
        -planning_conversion_costs + annuity_terms == 0,
        name="planning_conversion_costs_def",
    )

    # Block groups, after the module's last statement: their labels follow
    # every hourly label, and each row is the hourly row on the block axis.

    for L, p_g, idx_g in blocks:
        variables = _group_variables(m, p_g, tb, L)
        add_to_balances(
            operation_adequacy_constraint,
            variables[1].sum("conversion_tech"),
            tb,
            src=L,
        )
        rows = rows_of(L, p_g, idx_g, variables)
        rows.limits()
        rows.daily()
        rows.net_generation_def()
        rows.costs_def()

    return m
