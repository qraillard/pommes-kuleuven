"""Module to write in model process related constraints."""

import itertools

import numpy as np
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset


def add_process(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_year_normalization: float,
    operation_adequacy_constraint: Constraint,
) -> Model:
    """
    Add process-related components to the Linopy model.

    Including variables, costs, and constraints.

    Args:
        model (linopy.Model):
            The Linopy model to which process-related elements will be added.
        model_parameters (xarray.Dataset):
            Dataset containing energy system parameters, including process
            technology details, availability, ramping rates, and production
            limits.
        annualised_totex_def (linopy.Constraint):
            Expression representing the annualised total expenditure (totex),
            which will be updated with process-specific costs.
        operation_year_normalization (float | numpy.ndarray):
            Normalization factor for operational year durations.
        operation_adequacy_constraint (linopy.Constraint):
            Constraint ensuring operational adequacy, enforcing supply-demand
            balance, which will be updated with process-related contributions.

    Returns:
        linopy.Model:
            The updated Linopy model with added process-related variables,
            costs, and constraints.

    Example:
        This function introduces the following key elements to the model:

        Variables:
        - ``operation_process_power_capacity``: Operational power capacity
          for each process technology in each area and operational year.
        - ``operation_process_power``: Operational power output for each
          process technology by area, mode, hour, and operational year.
        - ``operation_process_net_generation``: Net generation of power
          from process technologies.
        - ``planning_process_power_capacity``: Planned power capacity
          investments for each process technology.

        Constraints:
        - Power output limits adjusted for availability.
        - Planning and operational capacity constraints.
        - Net generation definitions ensuring proper accounting.
        - Cost constraints considering fixed and variable costs.

    Note:
        This function introduces the following elements into the model:

        **Variables**

        - *Operation*
            - ``operation_process_power_capacity``
                Represents the operational power capacity for each process
                technology by area and operational year.
            - ``operation_process_power``
                Represents the operational power output for each process
                technology by area, mode, hour, and operational year.

            - *Intermediate variables*
                - ``operation_process_net_generation``
                    Represents the net generation from process technologies
                    calculated for each area, technology, hour, resource, and
                    operational year.

        - *Planning*
            - ``planning_process_power_capacity``
                Represents the planned power capacity for each process
                technology by area, decision year, and investment year.

        - *Costs (intermediate variables)*:
            - ``operation_process_costs``
                Represents the operational costs associated with process
                technologies for each area, technology, and operational year.
            - ``planning_process_costs``
                Represents the planning costs associated with process
                technologies for each area, technology, and operational year.

        **Constraints**

        - *Operation*
            - ``operation_process_power_max_constraint``
                Limits operational power to the available power capacity
                adjusted by availability factors.

            - *Operational capacity*
                - ``operation_process_power_capacity_def``
                    Defines the operational power capacity based on the
                    planned investments over the years.

            - *Intermediate variables definition*
                - ``operation_process_net_generation_def``
                    Relates net generation to power output and conversion
                    factor, ensuring consistent accounting.

        - *Planning*
            - ``planning_process_power_capacity_min_constraint``
                Sets a lower limit on planned power capacity investments
                for process technologies.
            - ``planning_process_power_capacity_max_constraint``
                Sets an upper limit on planned power capacity investments for
                process technologies.
            - ``planning_process_power_capacity_def``
                Ensures planned power capacity matches the minimum requirement
                when upper and lower limits are equal.

        - *Costs*
            - ``operation_process_costs_def``
                Defines operational costs as a function of variable costs,
                power output, and fixed costs.
            - ``planning_process_costs_def``
                Defines planning costs, accounting for investment costs spread
                over operational periods, with optional perfect foresight.

        These additions ensure that the process technologies operate within
        feasible and efficient limits, respecting capacity constraints, ramping
        capabilities, and power output limits. The model is thereby enhanced
        to accurately simulate process energy behavior and costs.
    """
    m = model
    p = model_parameters

    bigM=1e1*max(np.nan_to_num(p.process_power_capacity_max.values).sum(),np.nan_to_num(p.process_power_capacity_min.values).sum(),np.nan_to_num(p.process_power_capacity_investment_max.values).sum(),np.nan_to_num(p.process_power_capacity_investment_min.values).sum())

    if np.isnan(bigM):
        bigM=1e9
    min_downtime=p.process_minimum_shutdown_time
    min_uptime = p.process_minimum_startup_time
    for dim in ["area","year_op","process_tech"]:
        if dim not in min_downtime.dims:
            min_downtime = min_downtime.expand_dims(dim={dim: p[dim]})
        if dim not in min_uptime.dims:
            min_uptime = min_uptime.expand_dims(dim={dim: p[dim]})
    # ------------
    # Variables
    # ------------

    # Operation - process

    operation_process_power_capacity = m.add_variables(
        name="operation_process_power_capacity",
        lower=0,
        coords=[p.area, p.process_tech, p.year_op],
    )

    operation_process_power = m.add_variables(
        name="operation_process_power",
        lower=0,
        coords=[p.area, p.process_tech, p.mode, p.hour, p.year_op],
    )

    # Operation - process intermediate variables

    operation_process_net_generation = m.add_variables(
        name="operation_process_net_generation",
        coords=[p.area, p.process_tech, p.hour, p.resource, p.year_op],
        mask=(np.isfinite(p.process_factor) * (p.process_factor != 0)).any(
            [dim for dim in ["mode"] if dim in p.process_factor.dims]
        ),
    )

    # Operation - Process state variables

    operation_process_state = m.add_variables(
        name="operation_process_state",
        coords=[p.area, p.process_tech, p.hour, p.year_op],
        # binary=True
        integer=True,
        lower=0
    )

    operation_process_startup= m.add_variables(
        name="operation_process_startup",
        coords=[p.area, p.process_tech, p.hour, p.year_op],
        # binary=True
        integer=True,
        lower=0
    )

    operation_process_shutdown = m.add_variables(
        name="operation_process_shutdown",
        coords=[p.area, p.process_tech, p.hour, p.year_op],
        # binary=True
        integer=True,
        lower=0
    )



    operation_process_nb_units = m.add_variables(
        name="operation_process_nb_units",
        coords=[p.area, p.process_tech, p.year_op],
        integer=True,
        lower=0
    )

    # Planning - process

    planning_process_power_capacity = m.add_variables(
        name="planning_process_power_capacity",
        lower=0,
        coords=[p.area, p.process_tech, p.year_dec, p.year_inv],
        mask=xr.where(
            cond=p.process_early_decommissioning,
            x=(p.year_inv < p.year_dec)
            * (p.year_dec <= p.process_end_of_life)
            * np.logical_or(
                p.year_dec <= p.year_inv.max(),
                p.year_dec == p.process_end_of_life,
            ),
            y=p.year_dec == p.process_end_of_life,
        ),
    )

    # Costs - process

    operation_process_costs = m.add_variables(
        name="operation_process_costs",
        coords=[p.area, p.process_tech, p.year_op],
    )

    planning_process_costs = m.add_variables(
        name="planning_process_costs",
        coords=[p.area,p.process_tech, p.year_op],
    )

    # ------------------
    # Objective function
    # ------------------

    annualised_totex_def.lhs += operation_process_costs.sum(
        "process_tech"
    ) + planning_process_costs.sum("process_tech")

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    operation_adequacy_constraint.lhs += operation_process_net_generation.sum(
        "process_tech"
    )

    # Operation - process

    m.add_constraints(
        operation_process_power.sum("mode")
        - operation_process_state * p.process_unit_size * p.process_max_load
        <= 0,
        name="operation_process_power_max_constraint",
        mask=np.isfinite(p.process_unit_size) * (
                    p.process_unit_size > 0),
    )

    m.add_constraints(
        operation_process_power.sum("mode")
        - operation_process_power_capacity * p.process_max_load
        <= 0,
        name="operation_process_power_max_constraint_no_unit_size",
        mask=np.isfinite(p.process_unit_size) * (
                p.process_unit_size <= 0),
    )

    m.add_constraints(
        operation_process_power.sum("mode")
        - bigM * operation_process_state
        <= 0,
        name="operation_process_power_max_bigM_constraint", #particularly relevant if no unit size
    )

    m.add_constraints(
        operation_process_power.sum("mode")
        - operation_process_state * p.process_unit_size * p.process_min_load
        >= 0,

        name="operation_process_power_min_constraint",
        mask=np.isfinite(p.process_unit_size) * (
                p.process_unit_size > 0),
    )

    m.add_constraints(
        operation_process_power.sum("mode")
        - operation_process_power_capacity * p.process_min_load - bigM * operation_process_state
        >= -bigM,

        name="operation_process_power_min_constraint_no_unit_size",
        mask=np.isfinite(p.process_unit_size) * (
                p.process_unit_size <= 0),
    )

    # Operation - process unit commitment
    # Ramp applies to the total across modes, not each mode independently
    # (else a 2-mode tech's aggregate output could ramp ~2x too fast).

    process_ramp_up = p.process_ramp_up.min(
        [dim for dim in ["mode"] if dim in p.process_ramp_up.dims]
    )
    process_ramp_down = p.process_ramp_down.min(
        [dim for dim in ["mode"] if dim in p.process_ramp_down.dims]
    )

    m.add_constraints(

        (operation_process_power.sum("mode")
        - operation_process_power.sum("mode").shift(hour=1))/p.time_step_duration
        - process_ramp_up * p.process_unit_size * operation_process_state
        <= 0,
        name="operation_process_ramp_up_constraint",
        mask=np.isfinite(process_ramp_up) * (p.hour != p.hour[0]) * np.isfinite(p.process_unit_size)* (p.process_unit_size > 0),
    )

    m.add_constraints(

        (operation_process_power.sum("mode")
         - operation_process_power.sum("mode").shift(hour=1)) / p.time_step_duration
        -process_ramp_up * operation_process_power_capacity
        <= 0,
        name="operation_process_ramp_up_constraint_no_unit_size",
        mask=np.isfinite(process_ramp_up) * (p.hour != p.hour[0]) * np.isfinite(p.process_unit_size) * (
                    p.process_unit_size <= 0),
    )

    m.add_constraints(

        (operation_process_power.sum("mode").shift(hour=1)
        - operation_process_power.sum("mode"))/p.time_step_duration
        - process_ramp_down * p.process_unit_size * operation_process_state
        <= 0,
        name="operation_process_ramp_down_constraint",
        mask=np.isfinite(process_ramp_down) * (p.hour != p.hour[0])* np.isfinite(p.process_unit_size)* (p.process_unit_size > 0),
    )

    m.add_constraints(

        (operation_process_power.sum("mode").shift(hour=1)
         - operation_process_power.sum("mode")) / p.time_step_duration
        - process_ramp_down * operation_process_power_capacity
        <= 0,
        name="operation_process_ramp_down_constraint_no_unit_size",
        mask=np.isfinite(process_ramp_down) * (p.hour != p.hour[0]) * np.isfinite(p.process_unit_size) * (
                    p.process_unit_size <= 0),
    )

    # Optional, opt-in: limits how fast the split between modes shifts,
    # independent of the aggregate ramp above. Off by default (nan). Applied
    # pairwise across all modes -- no new variable: power(mode_a) -
    # power(mode_b) is a linear combination of the existing power variable.

    process_mode_ramp_up = p.process_mode_ramp_up.min(
        [dim for dim in ["mode"] if dim in p.process_mode_ramp_up.dims]
    )
    process_mode_ramp_down = p.process_mode_ramp_down.min(
        [dim for dim in ["mode"] if dim in p.process_mode_ramp_down.dims]
    )

    for mode_a, mode_b in itertools.combinations(p.mode.values, 2):
        process_mode_balance = operation_process_power.sel(
            mode=mode_a
        ) - operation_process_power.sel(mode=mode_b)

        m.add_constraints(
            (process_mode_balance - process_mode_balance.shift(hour=1)) / p.time_step_duration
            - process_mode_ramp_up * p.process_unit_size * operation_process_state
            <= 0,
            name=f"operation_process_mode_ramp_up_constraint_{mode_a}_{mode_b}",
            mask=np.isfinite(process_mode_ramp_up) * (p.hour != p.hour[0]) * np.isfinite(p.process_unit_size) * (p.process_unit_size > 0),
        )

        m.add_constraints(
            (process_mode_balance - process_mode_balance.shift(hour=1)) / p.time_step_duration
            - process_mode_ramp_up * operation_process_power_capacity
            <= 0,
            name=f"operation_process_mode_ramp_up_constraint_no_unit_size_{mode_a}_{mode_b}",
            mask=np.isfinite(process_mode_ramp_up) * (p.hour != p.hour[0]) * np.isfinite(p.process_unit_size) * (p.process_unit_size <= 0),
        )

        m.add_constraints(
            (process_mode_balance.shift(hour=1) - process_mode_balance) / p.time_step_duration
            - process_mode_ramp_down * p.process_unit_size * operation_process_state
            <= 0,
            name=f"operation_process_mode_ramp_down_constraint_{mode_a}_{mode_b}",
            mask=np.isfinite(process_mode_ramp_down) * (p.hour != p.hour[0]) * np.isfinite(p.process_unit_size) * (p.process_unit_size > 0),
        )

        m.add_constraints(
            (process_mode_balance.shift(hour=1) - process_mode_balance) / p.time_step_duration
            - process_mode_ramp_down * operation_process_power_capacity
            <= 0,
            name=f"operation_process_mode_ramp_down_constraint_no_unit_size_{mode_a}_{mode_b}",
            mask=np.isfinite(process_mode_ramp_down) * (p.hour != p.hour[0]) * np.isfinite(p.process_unit_size) * (p.process_unit_size <= 0),
        )

    m.add_constraints(operation_process_state - operation_process_nb_units <= 0,
                      name="operation_process_state_nb_units", mask=np.isfinite(p.process_unit_size) * (p.process_unit_size > 0 ) * (p.process_no_shutdown == False))

    m.add_constraints(operation_process_state - operation_process_nb_units == 0,
                      name="operation_process_no_shutdown_constraint",
                      mask=np.isfinite(p.process_unit_size) * (p.process_unit_size > 0) * (p.process_no_shutdown == True))


    m.add_constraints(operation_process_state - 1 == 0,
                      name="operation_process_no_shutdown_constraint_no_unit_size",
                      mask=np.isfinite(p.process_unit_size) * (p.process_unit_size == 0) * (p.process_no_shutdown == True))

    m.add_constraints(operation_process_startup - operation_process_nb_units <= 0,
                      name="operation_process_startup_nb_units_constraint",
                      mask=np.isfinite(p.process_unit_size) * (p.process_unit_size > 0))

    m.add_constraints(operation_process_shutdown - operation_process_state <= 0, #cannot stop more units than the ones already on
                      name="operation_process_shutdown_nb_units_constraint",
                      mask=np.isfinite(p.process_unit_size) * (p.process_unit_size > 0))

    m.add_constraints(operation_process_nb_units <= 1, #if no process unit size specified, only 1 unit can be installed
                      name="operation_process_nb_units_max_if_no_unit_size_constraint",
                      mask=np.isfinite(p.process_unit_size)* (p.process_unit_size <=0))

    m.add_constraints(operation_process_power==0, #if process_factor not defined for a given mode, then the technology cannot operate under this mode
                      name="operation_process_power_undefined_mode",
                      mask=(p.process_factor == 0).all(dim="resource"))

    # Operation capacity
    m.add_constraints(
        operation_process_power_capacity- operation_process_nb_units * p.process_unit_size ==0,
        name="operation_process_power_nb_units_constraint", mask=np.isfinite(p.process_unit_size) * (p.process_unit_size > 0),
    )

    m.add_constraints(
        operation_process_power_capacity <= p.process_power_capacity_max,
        name="operation_process_power_capacity_max_constraint",
        mask=np.isfinite(p.process_power_capacity_max),
    )

    m.add_constraints(
        operation_process_power_capacity >= p.process_power_capacity_min,
        name="operation_process_power_capacity_min_constraint",
        mask=np.isfinite(p.process_power_capacity_min),
    )

    # Operation - process intermediate variables

    m.add_constraints(
        -operation_process_power_capacity
        + planning_process_power_capacity.where(
            (p.year_inv <= p.year_op) * (p.year_op < p.year_dec)
        ).sum(["year_dec", "year_inv"])
        == 0,
        name="operation_process_power_capacity_def",
    )

    m.add_constraints(
        -operation_process_net_generation
        + p.time_step_duration
        * (p.process_factor * operation_process_power).sum(["mode"])
        == 0,
        name="operation_process_net_generation_def",
        mask=(np.isfinite(p.process_factor) * (p.process_factor != 0)).any(
            [dim for dim in ["mode"] if dim in p.process_factor.dims]
        ),
    )

    m.add_constraints(operation_process_startup+operation_process_shutdown-operation_process_nb_units<=0,
                      name="operation_process_startupshutdown_def",)

    m.add_constraints(operation_process_startup - operation_process_shutdown
                      - (operation_process_state - operation_process_state.shift(hour=1))
        == 0 ,
                      name="operation_process_state_def",
                      mask= (p.hour != p.hour[0]))



    uptime_mask = min_uptime > 1
    downtime_mask = min_downtime > 1

    # Apply uptime constraints
    for uptime in np.unique(min_uptime.values):
        if uptime > 1:
            # Create a rolling sum over the hours for each area, process_tech, and year_op
            rolling_sum = operation_process_startup.rolling(hour=int(uptime), min_periods=1).sum()
            uptime_constraints = rolling_sum - operation_process_state
            uptime_constraints = uptime_constraints.where(uptime_mask, drop=True)
            m.add_constraints(uptime_constraints <= 0, name=f"minimum_uptime_constraint_{uptime}")

    # Apply downtime constraints
    for downtime in np.unique(min_downtime.values):
        if downtime > 1:
            # Create a rolling sum over the hours for each area, process_tech, and year_op
            rolling_sum = operation_process_shutdown.rolling(hour=int(downtime), min_periods=1).sum()
            downtime_constraints = rolling_sum + operation_process_state
            downtime_constraints = downtime_constraints.where(downtime_mask, drop=True)
            m.add_constraints(downtime_constraints <= operation_process_nb_units,name=f"minimum_downtime_constraint_{downtime}")

    #TODO: apply maximum uptime and downtime constraints - need to create 2 new input parameters

    # Planning - process

    m.add_constraints(
        planning_process_power_capacity.sum("year_dec")
        <= p.process_power_capacity_investment_max,
        name="planning_process_power_capacity_max_constraint",
        mask=np.isfinite(p.process_power_capacity_investment_max)
        * np.not_equal(
            p.process_power_capacity_investment_max,
            p.process_power_capacity_investment_min,
        ),
    )

    m.add_constraints(
        planning_process_power_capacity.sum("year_dec")
        >= p.process_power_capacity_investment_min,
        name="planning_process_power_capacity_min_constraint",
        mask=np.isfinite(p.process_power_capacity_investment_min)
        * np.not_equal(
            p.process_power_capacity_investment_max,
            p.process_power_capacity_investment_min,
        ),
    )

    m.add_constraints(
        planning_process_power_capacity.sum("year_dec")
        == p.process_power_capacity_investment_min,
        name="planning_process_power_capacity_def",
        mask=np.isfinite(p.process_power_capacity_investment_min)
        * np.equal(
            p.process_power_capacity_investment_max,
            p.process_power_capacity_investment_min,
        ),
    )

    # Costs - process

    m.add_constraints(
        -operation_process_costs
        + operation_year_normalization
        * (
            p.process_variable_cost
            * (p.time_step_duration * operation_process_power).sum("hour")
        ).sum("mode")
        + p.process_fixed_cost * operation_process_power_capacity
        + p.process_startup_cost* operation_process_startup.sum("hour")
        + p.process_shutdown_cost * operation_process_shutdown.sum("hour")
        == 0,
        name="operation_process_costs_def",
    )

    m.add_constraints(
        -planning_process_costs
        + (
            (planning_process_power_capacity * p.process_annuity_cost)
            .where((p.year_inv <= p.year_op) * (p.year_op < p.year_dec))
            .sum(["year_dec", "year_inv"])
        ).where(
            cond=p.process_annuity_perfect_foresight,
            other=(
                (
                    planning_process_power_capacity.sum("year_dec")
                    * p.process_annuity_cost.min(
                        [
                            dim
                            for dim in p.process_annuity_cost.dims
                            if dim == "year_dec"
                        ]
                    )
                )
                .where(
                    (p.year_inv <= p.year_op)
                    * (p.year_op < p.process_end_of_life)
                )
                .sum(["year_inv"])
            ),
        )
        == 0,
        name="planning_process_costs_def",
    )

    return m
