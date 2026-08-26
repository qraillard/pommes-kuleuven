"""Module to write exchange related constraints."""

import numpy as np
from linopy import Constraint, Model
from xarray import Dataset


def add_net_import(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_year_normalization: float,
    operation_adequacy_constraint: Constraint,
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
    m = model
    p = model_parameters

    # ------------
    # Variables
    # ------------

    # Operation - Imports & exports from rest of the world

    operation_net_import_import = m.add_variables(
        name="operation_net_import_import",
        lower=0,
        coords=[p.area, p.hour, p.resource, p.year_op],
    )

    operation_net_import_export = m.add_variables(
        name="operation_net_import_export",
        lower=0,
        coords=[p.area, p.hour, p.resource, p.year_op],
    )

    # Operation - Imports & exports intermediate variables

    operation_net_import_abs = m.add_variables(
        name="operation_net_import_abs",
        lower=0,
        coords=[p.area, p.hour, p.resource, p.year_op],
    )

    operation_net_import_net_generation = m.add_variables(
        name="operation_net_import_net_generation",
        coords=[p.area, p.hour, p.resource, p.year_op],
    )

    # Costs - Imports & exports

    operation_net_import_costs = m.add_variables(
        name="operation_net_import_costs",
        coords=[p.area, p.resource, p.year_op],
    )

    # ------------------
    # Objective function
    # ------------------

    annualised_totex_def.lhs += operation_net_import_costs.sum("resource")

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    operation_adequacy_constraint.lhs += operation_net_import_net_generation

    # Operation - Imports & exports

    # TODO: add max power constraint on transports with ROW

    # Operation - Imports & exports other constraints

    m.add_constraints(
        operation_year_normalization
        * operation_net_import_import.sum(["area", "hour"])
        <= p.net_import_max_yearly_energy_import,
        mask=np.isfinite(p.net_import_max_yearly_energy_import),
        name="operation_net_import_import_yearly_max_constraint",
    )

    m.add_constraints(
        operation_year_normalization
        * operation_net_import_export.sum(["area", "hour"])
        <= p.net_import_max_yearly_energy_export,
        mask=np.isfinite(p.net_import_max_yearly_energy_export),
        name="operation_net_import_export_yearly_max_constraint",
    )

    # Operation - Imports & exports intermediate variables

    m.add_constraints(
        -operation_net_import_abs
        + operation_net_import_import
        + operation_net_import_export
        == 0,
        name="operation_net_import_abs_def",
    )

    m.add_constraints(
        -operation_net_import_net_generation
        + operation_net_import_import
        - operation_net_import_export
        == 0,
        name="operation_net_import_net_generation_def",
    )

    # Costs - Imports & exports

    m.add_constraints(
        -operation_net_import_costs
        # Normalised variable cots with the duration of the operation periods
        # to be consistent with annualised costs
        + operation_year_normalization
        * (
            +p.net_import_import_price * operation_net_import_import
            - p.net_import_export_price * operation_net_import_export
        ).sum("hour")
        == 0,
        name="operation_net_import_costs_def",
    )

    return m
