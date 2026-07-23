"""Module adding conversion technologies ot the linopy Model."""

import numpy as np
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset


def add_conversion(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_year_normalization: float,
    operation_adequacy_constraint: Constraint,
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
                    ensuring smooth increases in power output.
                - ``operation_conversion_ramp_down_constraint``
                    Constrains the ramp-down rate of conversion technologies,
                    ensuring smooth decreases in power output.
                - ``operation_conversion_yearly_production_max_constraint``
                    Limits the yearly production of conversion technologies
                    to a maximum value.

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
    m = model
    p = model_parameters

    # ------------
    # Variables
    # ------------

    # Operation - Conversion

    operation_conversion_power_capacity = m.add_variables(
        name="operation_conversion_power_capacity",
        lower=0,
        coords=[p.area, p.conversion_tech, p.year_op],
    )

    operation_conversion_power = m.add_variables(
        name="operation_conversion_power",
        lower=0,
        coords=[p.area, p.conversion_tech, p.hour, p.year_op],
    )

    # Operation - Conversion intermediate variables

    operation_conversion_net_generation = m.add_variables(
        name="operation_conversion_net_generation",
        coords=[p.area, p.conversion_tech, p.hour, p.resource, p.year_op],
        mask=np.isfinite(p.conversion_factor) * (p.conversion_factor != 0),
    )

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
        coords=[p.area, p.conversion_tech, p.year_op],
    )

    planning_conversion_costs = m.add_variables(
        name="planning_conversion_costs",
        coords=[p.area, p.conversion_tech, p.year_op],
    )

    # ------------------
    # Objective function
    # ------------------

    annualised_totex_def.lhs += operation_conversion_costs.sum(
        "conversion_tech"
    ) + planning_conversion_costs.sum("conversion_tech")

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    operation_adequacy_constraint.lhs += (
        operation_conversion_net_generation.sum("conversion_tech")
    )

    # Operation - Conversion

    m.add_constraints(
        operation_conversion_power
        - operation_conversion_power_capacity
        * xr.where(
            cond=np.isfinite(p.conversion_availability),
            x=p.conversion_availability,
            y=1,
        )
        <= 0,
        name="operation_conversion_power_max_constraint",
        mask=np.logical_or(
            np.isnan(p.conversion_must_run),
            p.conversion_must_run < 1,
        ),
    )

    m.add_constraints(
        operation_conversion_power
        - operation_conversion_power_capacity * p.conversion_availability
        == 0,
        name="operation_conversion_must_run_constraint_equality",
        mask=p.conversion_must_run == 1,
    )

    m.add_constraints(
        operation_conversion_power
        - operation_conversion_power_capacity
        * xr.where(
            np.isfinite(p.conversion_availability),
            p.conversion_availability,
            1,
        )
        * p.conversion_must_run
        >= 0,
        name="operation_conversion_must_run_constraint_inequality",
        mask=(0 < p.conversion_must_run) * (p.conversion_must_run < 1),
    )

    # Operation - Conversion unit commitment

    m.add_constraints(
        -p.conversion_ramp_up * operation_conversion_power_capacity
        + (
            operation_conversion_power
            - operation_conversion_power.shift(hour=1)
        )
        / p.time_step_duration
        <= 0,
        name="operation_conversion_ramp_up_constraint",
        mask=np.isfinite(p.conversion_ramp_up) * (p.hour != p.hour[0]),
    )

    # TODO : implement ramp based on available capacity, not all capacity
    m.add_constraints(
        -p.conversion_ramp_down * operation_conversion_power_capacity
        + (
            operation_conversion_power.shift(hour=1)
            - operation_conversion_power
        )
        / p.time_step_duration
        <= 0,
        name="operation_conversion_ramp_down_constraint",
        mask=np.isfinite(p.conversion_ramp_down) * (p.hour != p.hour[0]),
    )

    # TODO: Ramp constraints fo lag > 1 not implemented yet and for storage
    #     also

    # Operation - Conversion other constraints

    m.add_constraints(
        +operation_year_normalization
        * (p.time_step_duration * operation_conversion_power).sum(["hour"])
        <= p.conversion_max_yearly_production,
        name="operation_conversion_yearly_production_max_constraint",
        mask=np.isfinite(p.conversion_max_yearly_production),
    )

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

    # Operation - maximum daily production (for eDSR specifically)
    hour_range_list = []
    for hr in list(range(p.hour.values.min(), p.hour.values.max() + 1, p.operation_day_duration.values)):
        if hr + p.operation_day_duration.values<p.hour.values.max():
            hour_range_list.append(list(range(hr, hr + p.operation_day_duration.values)))
        else:
            hour_range_list.append(list(range(hr, p.hour.values.max() + 1)))
    for hr_range in hour_range_list:
        m.add_constraints(
            +operation_year_normalization
            * (p.time_step_duration * operation_conversion_power.sel(hour=hr_range)).sum(["hour"])
            <= p.conversion_max_daily_production,
            name=f"operation_conversion_daily_production_max_constraint_{hr_range[0]}",
            mask=np.isfinite(p.conversion_max_daily_production),
        )


    # Operation - Conversion intermediate variables

    m.add_constraints(
        -operation_conversion_power_capacity
        + planning_conversion_power_capacity.where(
            (p.year_inv <= p.year_op) * (p.year_op < p.year_dec)
        ).sum(["year_dec", "year_inv"])
        == 0,
        name="operation_conversion_power_capacity_def",
    )

    m.add_constraints(
        -operation_conversion_net_generation
        + p.time_step_duration
        * (p.conversion_factor * operation_conversion_power)
        == 0,
        name="operation_conversion_net_generation_def",
        mask=np.isfinite(p.conversion_factor) * (p.conversion_factor != 0),
    )

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

    m.add_constraints(
        -operation_conversion_costs
        + (
            p.conversion_variable_cost
            * operation_year_normalization
            * (p.time_step_duration * operation_conversion_power).sum("hour")
        )
        + (p.conversion_fixed_cost * operation_conversion_power_capacity)
        == 0,
        name="operation_conversion_costs_def",
    )

    m.add_constraints(
        -planning_conversion_costs
        + (
            (planning_conversion_power_capacity * p.conversion_annuity_cost)
            .where((p.year_inv <= p.year_op) * (p.year_op < p.year_dec))
            .sum(["year_dec", "year_inv"])
        ).where(
            cond=p.conversion_annuity_perfect_foresight,
            other=(
                (
                    planning_conversion_power_capacity.sum("year_dec")
                    * p.conversion_annuity_cost.min(
                        [
                            dim
                            for dim in p.conversion_annuity_cost.dims
                            if dim == "year_dec"
                        ]
                    )
                )
                .where(
                    (p.year_inv <= p.year_op)
                    * (p.year_op < p.conversion_end_of_life)
                )
                .sum(["year_inv"])
            ),
        )
        == 0,
        name="planning_conversion_costs_def",
    )

    return m
