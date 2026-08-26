"""Module to write in model combined related constraints."""

import itertools

import numpy as np
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset


def add_combined(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_year_normalization: float,
    operation_adequacy_constraint: Constraint,
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
                capacity.
            - ``operation_combined_ramp_down_constraint``
                Constrains the ramp-down rate of a combined technology's total
                output (summed across mode), based on the installed power
                capacity.
            - ``operation_combined_mode_ramp_up_constraint``
                Optional, opt-in (``combined_mode_ramp_up``, default nan/off):
                limits how fast the split between each pair of modes can
                shift (a fuel/pathway-shift rate), independent of the
                aggregate ramp above.
            - ``operation_combined_mode_ramp_down_constraint``
                Optional, opt-in (``combined_mode_ramp_down``, default
                nan/off): the opposite-direction counterpart of
                ``operation_combined_mode_ramp_up_constraint``.

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
    m = model
    p = model_parameters

    # ------------
    # Variables
    # ------------

    # Operation - Combined

    operation_combined_power_capacity = m.add_variables(
        name="operation_combined_power_capacity",
        lower=0,
        coords=[p.area, p.combined_tech, p.year_op],
    )

    operation_combined_power = m.add_variables(
        name="operation_combined_power",
        lower=0,
        coords=[p.area, p.combined_tech, p.mode, p.hour, p.year_op],
    )

    # Operation - Combined intermediate variables

    operation_combined_net_generation = m.add_variables(
        name="operation_combined_net_generation",
        coords=[p.area, p.combined_tech, p.hour, p.resource, p.year_op],
        mask=(np.isfinite(p.combined_factor) * (p.combined_factor != 0)).any(
            [dim for dim in ["mode"] if dim in p.combined_factor.dims]
        ),
    )

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
        coords=[p.area, p.combined_tech, p.year_op],
    )

    planning_combined_costs = m.add_variables(
        name="planning_combined_costs",
        coords=[p.area, p.combined_tech, p.year_op],
    )

    # ------------------
    # Objective function
    # ------------------

    annualised_totex_def.lhs += operation_combined_costs.sum(
        "combined_tech"
    ) + planning_combined_costs.sum("combined_tech")

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    operation_adequacy_constraint.lhs += operation_combined_net_generation.sum(
        "combined_tech"
    )

    # Operation - Combined

    m.add_constraints(
        operation_combined_power.sum("mode")
        - operation_combined_power_capacity * p.combined_max_load
        <= 0,
        name="operation_combined_power_max_constraint",
    )

    m.add_constraints(
        operation_combined_power.sum("mode")
        - operation_combined_power_capacity * p.combined_min_load
        >= 0,
        name="operation_combined_power_min_constraint",
    )

    m.add_constraints(
        operation_combined_power == 0,
        # if combined_factor is not defined for a given mode, then the
        # technology cannot operate under this mode
        name="operation_combined_power_undefined_mode",
        mask=(p.combined_factor == 0).all(dim="resource"),
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

    # Operation - Combined ramping (no unit commitment: bounds are expressed
    # directly against installed capacity, as there is no on/off state).
    # Ramp applies to the total across modes, not each mode independently
    # (else a 2-mode tech's aggregate output could ramp ~2x too fast).

    combined_ramp_up = p.combined_ramp_up.min(
        [dim for dim in ["mode"] if dim in p.combined_ramp_up.dims]
    )
    combined_ramp_down = p.combined_ramp_down.min(
        [dim for dim in ["mode"] if dim in p.combined_ramp_down.dims]
    )

    m.add_constraints(
        (
            operation_combined_power.sum("mode")
            - operation_combined_power.sum("mode").shift(hour=1)
        )
        / p.time_step_duration
        - combined_ramp_up * operation_combined_power_capacity
        <= 0,
        name="operation_combined_ramp_up_constraint",
        mask=np.isfinite(combined_ramp_up) * (p.hour != p.hour[0]),
    )

    m.add_constraints(
        (
            operation_combined_power.sum("mode").shift(hour=1)
            - operation_combined_power.sum("mode")
        )
        / p.time_step_duration
        - combined_ramp_down * operation_combined_power_capacity
        <= 0,
        name="operation_combined_ramp_down_constraint",
        mask=np.isfinite(combined_ramp_down) * (p.hour != p.hour[0]),
    )

    # Optional, opt-in: limits how fast the split between modes shifts,
    # independent of the aggregate ramp above. Off by default (nan). Applied
    # pairwise across all modes -- no new variable: power(mode_a) -
    # power(mode_b) is a linear combination of the existing power variable.

    combined_mode_ramp_up = p.combined_mode_ramp_up.min(
        [dim for dim in ["mode"] if dim in p.combined_mode_ramp_up.dims]
    )
    combined_mode_ramp_down = p.combined_mode_ramp_down.min(
        [dim for dim in ["mode"] if dim in p.combined_mode_ramp_down.dims]
    )

    for mode_a, mode_b in itertools.combinations(p.mode.values, 2):
        combined_mode_balance = operation_combined_power.sel(
            mode=mode_a
        ) - operation_combined_power.sel(mode=mode_b)

        m.add_constraints(
            (
                combined_mode_balance
                - combined_mode_balance.shift(hour=1)
            )
            / p.time_step_duration
            - combined_mode_ramp_up * operation_combined_power_capacity
            <= 0,
            name=f"operation_combined_mode_ramp_up_constraint_{mode_a}_{mode_b}",
            mask=np.isfinite(combined_mode_ramp_up) * (p.hour != p.hour[0]),
        )

        m.add_constraints(
            (
                combined_mode_balance.shift(hour=1)
                - combined_mode_balance
            )
            / p.time_step_duration
            - combined_mode_ramp_down * operation_combined_power_capacity
            <= 0,
            name=f"operation_combined_mode_ramp_down_constraint_{mode_a}_{mode_b}",
            mask=np.isfinite(combined_mode_ramp_down) * (p.hour != p.hour[0]),
        )

    # TODO: Ramp constraints for lag > 1 not implemented yet and for storage
    #     also

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

    m.add_constraints(
        -operation_combined_net_generation
        + p.time_step_duration
        * (p.combined_factor * operation_combined_power).sum(["mode"])
        == 0,
        name="operation_combined_net_generation_def",
        mask=(np.isfinite(p.combined_factor) * (p.combined_factor != 0)).any(
            [dim for dim in ["mode"] if dim in p.combined_factor.dims]
        ),
    )

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

    m.add_constraints(
        -operation_combined_costs
        + operation_year_normalization
        * (
            p.combined_variable_cost
            * (p.time_step_duration * operation_combined_power).sum("hour")
        ).sum("mode")
        + p.combined_fixed_cost * operation_combined_power_capacity
        == 0,
        name="operation_combined_costs_def",
    )

    m.add_constraints(
        -planning_combined_costs
        + (
            (planning_combined_power_capacity * p.combined_annuity_cost)
            .where((p.year_inv <= p.year_op) * (p.year_op < p.year_dec))
            .sum(["year_dec", "year_inv"])
        ).where(
            cond=p.combined_annuity_perfect_foresight,
            other=(
                (
                    planning_combined_power_capacity.sum("year_dec")
                    * p.combined_annuity_cost.min(
                        [
                            dim
                            for dim in p.combined_annuity_cost.dims
                            if dim == "year_dec"
                        ]
                    )
                )
                .where(
                    (p.year_inv <= p.year_op)
                    * (p.year_op < p.combined_end_of_life)
                )
                .sum(["year_inv"])
            ),
        )
        == 0,
        name="planning_combined_costs_def",
    )

    return m
