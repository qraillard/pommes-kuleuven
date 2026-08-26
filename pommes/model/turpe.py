"""Turpe related content. WIP."""

import linopy
from linopy import Constraint, Model
from xarray import Dataset


def add_turpe(
    model: Model, model_parameters: Dataset, annualised_totex_def: Constraint
) -> Model:
    """
    Add TURPE-related components to the Linopy model.

    Including variables, constraints, and costs associated with network
    tariffs.

    Args:
        model (linopy.Model):
            The Linopy model to which TURPE-related elements will be added.
        model_parameters (xarray.Dataset):
            Dataset containing energy system parameters, including TURPE tariff
            structures and network cost factors.
        annualised_totex_def (linopy.Constraint):
            Constraint defining annualised total expenditures (totex), which
            will be updated with TURPE-specific costs.

    Returns:
        linopy.Model:
            The updated Linopy model with added TURPE-related variables,
            costs, and constraints.

    Note:
        This function introduces the following elements into the model:

        **Variables**

        - *Operation*
            - ``operation_turpe_contract_power``
                Represents the contracted power under TURPE tariffs,
                defined per area, hour type, and operational year.
            - ``operation_turpe_variable_costs``
                Represents the variable costs associated with network tariffs
                for each area and operational year.
            - ``operation_turpe_fixed_costs``
                Represents the fixed costs associated with network tariffs
                for each area and operational year.

        **Constraints**

        - *Operation*
            - ``operation_turpe_contract_power_max_constraint``
                Ensures that the contracted power level is sufficient to meet
                the net electricity imports defined by the TURPE calendar.
            - ``operation_turpe_increasing_contract_power_constraint``
                Ensures that the contracted power does not decrease over time.

        - *Costs*
            - ``operation_turpe_variable_costs_def``
                Defines variable TURPE costs as a function of net imports and
                TURPE tariff rates.
            - ``operation_turpe_fixed_costs_def``
                Defines fixed TURPE costs as a function of contracted power
                and fixed tariff components.

        These additions ensure that network tariff costs are accurately
        modeled, incorporating both fixed and variable cost components into
        the overall energy system optimization.
    """
    m = model
    p = model_parameters
    v = m.variables

    # ------------
    # Variables
    # ------------

    # Operation - TURPE

    operation_turpe_contract_power = m.add_variables(
        name="operation_turpe_contract_power",
        lower=0,
        coords=[p.area, p.hour_type, p.year_op],
    )
    # TODO: add differentiate contract power for injection ans withdrawal?

    # Costs - TURPE

    operation_turpe_variable_costs = m.add_variables(
        name="operation_turpe_variable_costs",
        lower=0,
        coords=[p.area, p.year_op],
    )

    operation_turpe_fixed_costs = m.add_variables(
        name="operation_turpe_fixed_costs", lower=0, coords=[p.area, p.year_op]
    )

    # ------------------
    # Objective function
    # ------------------

    annualised_totex_def.lhs += (
        operation_turpe_variable_costs + operation_turpe_fixed_costs
    )

    # --------------
    # Constraints
    # --------------

    # Operation - TURPE

    m.add_constraints(
        linopy.expressions.merge(
            [
                v.operation_net_import_abs.sel(resource="electricity").where(
                    p.turpe_calendar == hour_type
                )
                - operation_turpe_contract_power.sel(hour_type=hour_type)
                for hour_type in p.hour_type
            ],
            dim="hour_type",
        )
        <= 0,
        name="operation_turpe_contract_power_max_constraint",
    )

    m.add_constraints(
        operation_turpe_contract_power
        - operation_turpe_contract_power.shift(hour_type=1)
        >= 0,
        name="operation_turpe_increasing_contract_power_constraint",
    )

    # Costs - TURPE

    m.add_constraints(
        (
            linopy.expressions.merge(
                [
                    v.operation_net_import_abs.sel(
                        resource="electricity"
                    ).where(p.turpe_calendar == hour_type)
                    * p.turpe_variable_cost.sel(hour_type=hour_type)
                    for hour_type in p.hour_type
                ],
                dim="hour",
            )
            # *p.time_step_duration
            # TODO Check the issue with the index hour
            #  (here problem of dimension due to a broadcast of the index
            #  by the merge function)
        ).sum("hour")
        - operation_turpe_variable_costs
        == 0,
        name="operation_turpe_variable_costs_def",
    )

    m.add_constraints(
        (
            (
                operation_turpe_contract_power
                - operation_turpe_contract_power.shift(hour_type=1)
            )
            * p.turpe_fixed_cost
        ).sum("hour_type")
        - operation_turpe_fixed_costs
        == 0,
        name="operation_turpe_fixed_costs_def",
    )

    return m
