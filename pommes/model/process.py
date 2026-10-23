"""Module to write in model process related constraints."""

import itertools
import warnings

import numpy as np
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset

from pommes.model.annuity import annuity_full_life, full_life_window


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

    # TODO (time steps): the process module is only valid at 1 h steps. Minimum start-up / shut-down
    #  times (windows below) and ramps are counted in time steps, not in hours; making them hours
    #  needs ceil(hours / time_step_duration) steps and a uniform time_step_duration.
    if bool((p.time_step_duration != 1).any()):
        warnings.warn(
            "The process module is only valid with time steps of 1 hour (time_step_duration = 1): "
            "the minimum start-up and shut-down times and the ramps of process units are counted "
            "in time steps, not in hours. The simulation goes on, but these constraints are "
            "not those of the hourly model.",
            UserWarning,
            stacklevel=2,
        )

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

    # Clustered unit commitment: the units on in both steps (state - startup) ramp at most ramp each, a
    # started unit enters at most at startup_capability = max(min_load, ramp_up) and a stopped one leaves
    # from at most shutdown_capability = max(min_load, ramp_down), and either moves the output by at
    # least its min load, so a start-up/shut-down pair buys no extra ramp.
    process_running = operation_process_state - operation_process_startup
    startup_capability = np.fmax(p.process_min_load, process_ramp_up)
    shutdown_capability = np.fmax(p.process_min_load, process_ramp_down)
    sized = np.isfinite(p.process_unit_size) * (p.process_unit_size > 0)
    # A tech without unit size is one on/off block whose ramp is bounded by its capacity, not by its
    # state: a start-up (shut-down) may move it by up to that capability of its capacity bound, else a
    # ramp below min_load would forbid it ever to start or stop.
    capacity_bound = p.process_power_capacity_max.where(np.isfinite(p.process_power_capacity_max), bigM)

    m.add_constraints(

        (operation_process_power.sum("mode")
        - operation_process_power.sum("mode").shift(hour=1))/p.time_step_duration
        - process_ramp_up * p.process_unit_size * process_running
        - startup_capability * p.process_unit_size * operation_process_startup
        + p.process_min_load * p.process_unit_size * operation_process_shutdown
        <= 0,
        name="operation_process_ramp_up_constraint",
        mask=np.isfinite(process_ramp_up) * (p.hour != p.hour[0]) * np.isfinite(p.process_unit_size)* (p.process_unit_size > 0),
    )

    m.add_constraints(
        operation_process_power.sum("mode")
        - p.process_max_load * p.process_unit_size * process_running
        - startup_capability * p.process_unit_size * operation_process_startup
        <= 0,
        name="operation_process_startup_capability_constraint",
        mask=np.isfinite(process_ramp_up) * sized,
    )

    m.add_constraints(
        operation_process_power.sum("mode").shift(hour=1)
        - p.process_max_load * p.process_unit_size * process_running
        - shutdown_capability * p.process_unit_size * operation_process_shutdown
        <= 0,
        name="operation_process_shutdown_capability_constraint",
        mask=np.isfinite(process_ramp_down) * (p.hour != p.hour[0]) * sized,
    )

    m.add_constraints(

        (operation_process_power.sum("mode")
         - operation_process_power.sum("mode").shift(hour=1)) / p.time_step_duration
        -process_ramp_up * operation_process_power_capacity
        - startup_capability * capacity_bound * operation_process_startup
        <= 0,
        name="operation_process_ramp_up_constraint_no_unit_size",
        mask=np.isfinite(process_ramp_up) * (p.hour != p.hour[0]) * np.isfinite(p.process_unit_size) * (
                    p.process_unit_size <= 0),
    )

    m.add_constraints(

        (operation_process_power.sum("mode").shift(hour=1)
        - operation_process_power.sum("mode"))/p.time_step_duration
        - process_ramp_down * p.process_unit_size * process_running
        - shutdown_capability * p.process_unit_size * operation_process_shutdown
        + p.process_min_load * p.process_unit_size * operation_process_startup
        <= 0,
        name="operation_process_ramp_down_constraint",
        mask=np.isfinite(process_ramp_down) * (p.hour != p.hour[0])* np.isfinite(p.process_unit_size)* (p.process_unit_size > 0),
    )

    m.add_constraints(

        (operation_process_power.sum("mode").shift(hour=1)
         - operation_process_power.sum("mode")) / p.time_step_duration
        - process_ramp_down * operation_process_power_capacity
        - shutdown_capability * capacity_bound * operation_process_shutdown
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
    if bool(((np.isfinite(process_mode_ramp_up) | np.isfinite(process_mode_ramp_down))
             * sized * (p.process_no_shutdown == False)).any()):
        warnings.warn(
            "process_mode_ramp_* is set on units that can stop: the mode ramps have no start-up or "
            "shut-down terms, so such a unit can only stop with power(mode_a) == power(mode_b) and only "
            "start with |power(mode_a) - power(mode_b)| within the mode ramp.",
            UserWarning,
            stacklevel=2,
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

    # state is the state after the step (state_def), so the units that can stop are those on before it
    m.add_constraints(operation_process_shutdown - operation_process_state.shift(hour=1) <= 0, #cannot stop more units than the ones already on
                      name="operation_process_shutdown_nb_units_constraint",
                      mask=np.isfinite(p.process_unit_size) * (p.process_unit_size > 0) * (p.hour != p.hour[0]))

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



    # Minimum start-up / shut-down times. Only the units whose constraints are not empty are
    # constrained: no_shutdown == False (with no_shutdown the state is fixed, the windows are empty)
    # and a finite unit size (a NaN size has no power constraint tied to the state). A unit size of 0
    # is kept: the state is then the on/off of the whole plant (bigM constraints) and the windows
    # bind when process_min_load > 0. Each distinct minimum time v applies only to the
    # units whose minimum time is v. The window sum over the last v steps (rolling, min_periods=1:
    # nothing before the first step, no wrap-around) is written with a cumulative count C of the
    # start-ups (resp. shut-downs), C(t) = C(t-1) + startup(t): sum_{t-v < s <= t} startup(s) = C(t) - C(t-v)
    # (= C(t) for t < v), so each row has 3 terms instead of v + 1.
    has_onoff = np.isfinite(p.process_unit_size) * (p.process_no_shutdown == False)
    uptime_active = has_onoff * (min_uptime > 1)
    downtime_active = has_onoff * (min_downtime > 1)

    if bool(uptime_active.any()):
        operation_process_startup_cumulative = m.add_variables(
            name="operation_process_startup_cumulative",
            coords=[p.area, p.process_tech, p.hour, p.year_op],
            lower=0,
            mask=uptime_active,
        )
        m.add_constraints(
            operation_process_startup_cumulative
            - operation_process_startup_cumulative.shift(hour=1)
            - operation_process_startup
            == 0,
            name="operation_process_startup_cumulative_def",
            mask=uptime_active,
        )
        for uptime in np.unique(min_uptime.where(uptime_active).values):
            if uptime > 1:  # NaN (not an active unit) is False
                m.add_constraints(
                    operation_process_startup_cumulative
                    - operation_process_startup_cumulative.shift(hour=int(uptime))
                    - operation_process_state
                    <= 0,
                    name=f"minimum_uptime_constraint_{uptime}",
                    mask=uptime_active * (min_uptime == uptime),
                )

    if bool(downtime_active.any()):
        operation_process_shutdown_cumulative = m.add_variables(
            name="operation_process_shutdown_cumulative",
            coords=[p.area, p.process_tech, p.hour, p.year_op],
            lower=0,
            mask=downtime_active,
        )
        m.add_constraints(
            operation_process_shutdown_cumulative
            - operation_process_shutdown_cumulative.shift(hour=1)
            - operation_process_shutdown
            == 0,
            name="operation_process_shutdown_cumulative_def",
            mask=downtime_active,
        )
        for downtime in np.unique(min_downtime.where(downtime_active).values):
            if downtime > 1:
                m.add_constraints(
                    operation_process_shutdown_cumulative
                    - operation_process_shutdown_cumulative.shift(hour=int(downtime))
                    + operation_process_state
                    - operation_process_nb_units
                    <= 0,
                    name=f"minimum_downtime_constraint_{downtime}",
                    mask=downtime_active * (min_downtime == downtime),
                )

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

    # TODO (start-up costs): process_startup_cost and process_shutdown_cost are not multiplied by
    # operation_year_normalization, unlike the variable cost, so on a horizon shorter than a year a start
    # weighs 8760/len(hours) times too little against energy costs. Scale them like the variable cost.
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

    if annuity_full_life(p, "process_annuity_cost"):
        # Annuity fixed at investment, paid over the vintage's full life
        # whatever the decommissioning year (sunk cost).
        annuity_terms = (
            (
                planning_process_power_capacity.sum("year_dec")
                * p.process_annuity_cost
            )
            .where(
                full_life_window(p.year_inv, p.process_life_span, p.year_op)
            )
            .sum("year_inv")
        )
    else:
        # The flag has a year_inv dimension: choose the annuity per vintage,
        # then sum over the vintages (summing first and choosing afterwards
        # gives the expression a year_inv dimension back and writes the cost
        # row once per vintage).
        perfect_foresight = (
            (planning_process_power_capacity * p.process_annuity_cost)
            .where((p.year_inv <= p.year_op) * (p.year_op < p.year_dec))
            .sum("year_dec")
        )
        cheapest = (
            planning_process_power_capacity.sum("year_dec")
            * p.process_annuity_cost.min(
                [
                    dim
                    for dim in p.process_annuity_cost.dims
                    if dim == "year_dec"
                ]
            )
        ).where(
            (p.year_inv <= p.year_op) * (p.year_op < p.process_end_of_life)
        )
        annuity_terms = perfect_foresight.where(
            cond=p.process_annuity_perfect_foresight, other=cheapest
        ).sum("year_inv")

    m.add_constraints(
        -planning_process_costs + annuity_terms == 0,
        name="planning_process_costs_def",
    )

    return m
