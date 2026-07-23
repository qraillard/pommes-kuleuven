"""Module related to storage components."""

# TODO : check year normalisation and constraints

import numpy as np
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset


def add_storage(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_adequacy_constraint: Constraint,
) -> Model:
    """
    Add storage-related components to the Linopy model.

    Including variables, constraints, and costs related to energy storage.

    Args:
        model (linopy.Model):
            The Linopy model to which storage-related elements will be added.
        model_parameters (xarray.Dataset):
            Dataset containing energy system parameters, including storage
            capacity limits, efficiency factors, and costs.
        annualised_totex_def (linopy.Constraint):
            Constraint defining annualised total expenditures (totex), which
            will be updated with storage-specific costs.
        operation_adequacy_constraint (linopy.Constraint):
            Constraint ensuring operational adequacy, enforcing supply-demand
            balance, which will be updated with storage-related contributions.

    Returns:
        linopy.Model:
            The updated Linopy model with added storage-related variables,
            costs, and constraints.

    Note:
        This function introduces the following elements into the model:

        **Variables**

        - *Operation*
            - ``operation_storage_energy_capacity``
                Represents the operational energy capacity of storage for each
                area, technology, and operational year.
            - ``operation_storage_power_capacity``
                Represents the operational power capacity of storage for each
                area, technology, and operational year.
            - ``operation_storage_level``
                Represents the state of charge of the storage system over time.
            - ``operation_storage_power_in``
                Represents power input to the storage system at each time step.
            - ``operation_storage_power_out``
                Represents power output from the storage system at each time
                step.
            - ``operation_storage_net_generation``
                Represents the net energy flow from storage, accounting for
                charging, discharging, and retention losses.

        - *Planning*
            - ``planning_storage_energy_capacity``
                Represents the planned energy capacity investments in storage.
            - ``planning_storage_power_capacity``
                Represents the planned power capacity investments in storage.

        - *Costs*
            - ``operation_storage_costs``
                Represents operational costs associated with storage technologies.
            - ``planning_storage_costs``
                Represents planning costs associated with storage investments.

        **Constraints**

        - *Operation*
            - ``operation_storage_level_def``
                Defines the evolution of the storage level over time.
            - ``operation_storage_power_in_max_constraint``
                Limits the maximum charging power based on capacity.
            - ``operation_storage_power_out_max_constraint``
                Limits the maximum discharging power based on capacity.
            - Ramp-up and ramp-down constraints for smooth power output changes.
            - ``operation_storage_level_max_constraint``
                Ensures storage level does not exceed its maximum capacity.

        - *Planning*
            - ``planning_storage_power_capacity_min_constraint``
                Enforces a lower limit on planned storage power investments.
            - ``planning_storage_power_capacity_max_constraint``
                Enforces an upper limit on planned storage power investments.
            - ``planning_storage_energy_capacity_min_constraint``
                Enforces a lower limit on planned storage energy investments.
            - ``planning_storage_energy_capacity_max_constraint``
                Enforces an upper limit on planned storage energy investments.
            - ``planning_storage_energy_power_ratio_constraint``
                Ties planned energy capacity to planned power capacity by a
                fixed ratio (``storage_energy_power_ratio``, e.g. 4 for a 4h
                battery), only where that ratio is configured (not NaN) --
                otherwise power and energy capacities evolve independently.

        - *Costs*
            - ``operation_storage_costs_def``
                Defines operational storage costs based on fixed costs.
            - ``planning_storage_costs_def``
                Defines planning costs for storage investments, considering
                annuities.

        These additions ensure that storage technologies operate efficiently
        and within their technical and economic limits, allowing the model to
        simulate energy storage behavior realistically.
    """
    m = model
    p = model_parameters

    # ------------
    # Variables
    # ------------

    # Operation - Storage

    operation_storage_energy_capacity = m.add_variables(
        name="operation_storage_energy_capacity",
        lower=0,
        coords=[p.area, p.storage_tech, p.year_op],
    )

    operation_storage_power_capacity = m.add_variables(
        name="operation_storage_power_capacity",
        lower=0,
        coords=[p.area, p.storage_tech, p.year_op],
        # TODO: Variable P_in and P_out power?
    )

    operation_storage_level = m.add_variables(
        name="operation_storage_level",
        lower=0,
        coords=[p.area, p.hour, p.storage_tech, p.year_op],
    )

    operation_storage_power_in = m.add_variables(
        name="operation_storage_power_in",
        lower=0,
        coords=[p.area, p.hour, p.storage_tech, p.year_op],
    )

    operation_storage_power_out = m.add_variables(
        name="operation_storage_power_out",
        lower=0,
        coords=[p.area, p.hour, p.storage_tech, p.year_op],
    )

    # Operation - Storage intermediate variables

    operation_storage_net_generation = m.add_variables(
        name="operation_storage_net_generation",
        coords=[p.area, p.hour, p.storage_tech, p.resource, p.year_op],
        mask=np.isfinite(p.storage_factor_in) * (p.storage_factor_in != 0)
        + np.isfinite(p.storage_factor_out) * (p.storage_factor_out != 0)
        + np.isfinite(p.storage_factor_keep) * (p.storage_factor_keep != 0)
        > 0,
    )

    # Planning - Storage

    planning_storage_energy_capacity = m.add_variables(
        name="planning_storage_energy_capacity",
        lower=0,
        coords=[p.area, p.storage_tech, p.year_dec, p.year_inv],
        mask=xr.where(
            cond=p.storage_early_decommissioning,
            x=(p.year_inv < p.year_dec)
            * (p.year_dec <= p.storage_end_of_life)
            * np.logical_or(
                p.year_dec <= p.year_inv.max(),
                p.year_dec == p.storage_end_of_life,
            ),
            y=p.year_dec == p.storage_end_of_life,
        ),
    )

    planning_storage_power_capacity = m.add_variables(
        name="planning_storage_power_capacity",
        lower=0,
        coords=[p.area, p.storage_tech, p.year_dec, p.year_inv],
        mask=xr.where(
            cond=p.storage_early_decommissioning,
            x=(p.year_inv < p.year_dec)
            * (p.year_dec <= p.storage_end_of_life)
            * np.logical_or(
                p.year_dec <= p.year_inv.max(),
                p.year_dec == p.storage_end_of_life,
            ),
            y=p.year_dec == p.storage_end_of_life,
        ),
    )

    # Costs - Storage

    operation_storage_costs = m.add_variables(
        name="operation_storage_costs",
        lower=0,
        coords=[p.area, p.storage_tech, p.year_op],
    )

    planning_storage_costs = m.add_variables(
        name="planning_storage_costs",
        lower=0,
        coords=[p.area, p.storage_tech, p.year_op],
    )

    # ------------------
    # Objective function
    # ------------------

    annualised_totex_def.lhs += operation_storage_costs.sum(
        "storage_tech"
    ) + planning_storage_costs.sum("storage_tech")

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    operation_adequacy_constraint.lhs += operation_storage_net_generation.sum(
        "storage_tech"
    )

    # Operation - Storage

    m.add_constraints(
        -operation_storage_level
        # It is assumed that the storage dissipation is defined per hour
        + operation_storage_level.roll(hour=1)
        * (1 - p.storage_dissipation) ** p.time_step_duration
        + (operation_storage_power_in - operation_storage_power_out)
        * p.time_step_duration
        == 0,
        name="operation_storage_level_def",
    )

    m.add_constraints(
        operation_storage_power_in - operation_storage_power_capacity <= 0,
        name="operation_storage_power_in_max_constraint",
    )

    m.add_constraints(
        operation_storage_power_out - operation_storage_power_capacity <= 0,
        name="operation_storage_power_out_max_constraint",
    )

    m.add_constraints(
        operation_storage_level - operation_storage_energy_capacity <= 0,
        name="operation_storage_level_max_constraint",
    )

    m.add_constraints(
        -p.storage_charge_ramp * operation_storage_power_capacity
        + (
                operation_storage_power_in
                - operation_storage_power_in.shift(hour=1)
        )
        / p.time_step_duration
        <= 0,
        name="operation_storage_charge_ramp_constraint",
        mask=np.isfinite(p.storage_charge_ramp) * (p.hour != p.hour[0]),
    )

    m.add_constraints(
        -p.storage_discharge_ramp * operation_storage_power_capacity
        + (
                operation_storage_power_out
                - operation_storage_power_out.shift(hour=1)
        )
        / p.time_step_duration
        <= 0,
        name="operation_storage_discharge_ramp_constraint",
        mask=np.isfinite(p.storage_discharge_ramp) * (p.hour != p.hour[0]),
    )

    #TODO: storage_level ramping - do I apply a ramp based on the remaining volume level or on the volume capacity ? Do I allow for both ?

    # Operation - Storage intermediate variables

    m.add_constraints(
        -operation_storage_power_capacity
        + planning_storage_power_capacity.where(
            (p.year_inv <= p.year_op) * (p.year_op < p.year_dec)
        ).sum(["year_dec", "year_inv"])
        == 0,
        name="operation_storage_power_capacity_def",
    )

    m.add_constraints(
        -operation_storage_energy_capacity
        + planning_storage_energy_capacity.where(
            (p.year_inv <= p.year_op) * (p.year_op < p.year_dec)
        ).sum(["year_dec", "year_inv"])
        == 0,
        name="operation_storage_energy_capacity_def",
    )

    m.add_constraints(
        -operation_storage_net_generation
        + p.time_step_duration
        * (
            operation_storage_power_in * p.storage_factor_in
            + operation_storage_level * p.storage_factor_keep
            + operation_storage_power_out * p.storage_factor_out
        )
        == 0,
        name="operation_storage_net_generation_def",
        mask=np.isfinite(p.storage_factor_in) * (p.storage_factor_in != 0)
        + np.isfinite(p.storage_factor_out) * (p.storage_factor_out != 0)
        + np.isfinite(p.storage_factor_keep) * (p.storage_factor_keep != 0)
        > 0,
    )

    # Planning - Storage

    m.add_constraints(
        planning_storage_power_capacity.sum("year_dec")
        >= p.storage_power_capacity_investment_min,
        name="planning_storage_power_capacity_min_constraint",
        mask=np.isfinite(p.storage_power_capacity_investment_min)
        * np.not_equal(
            p.storage_power_capacity_investment_min,
            p.storage_power_capacity_investment_max,
        ),
    )

    m.add_constraints(
        planning_storage_power_capacity.sum("year_dec")
        <= p.storage_power_capacity_investment_max,
        name="planning_storage_power_capacity_max_constraint",
        mask=np.isfinite(p.storage_power_capacity_investment_max)
        * np.not_equal(
            p.storage_power_capacity_investment_min,
            p.storage_power_capacity_investment_max,
        ),
    )

    m.add_constraints(
        planning_storage_power_capacity.sum("year_dec")
        == p.storage_power_capacity_investment_max,
        name="planning_storage_power_capacity_def",
        mask=np.isfinite(p.storage_power_capacity_investment_max)
        * np.equal(
            p.storage_power_capacity_investment_min,
            p.storage_power_capacity_investment_max,
        ),
    )

    m.add_constraints(
        planning_storage_energy_capacity.sum("year_dec")
        >= p.storage_energy_capacity_investment_min,
        name="planning_storage_energy_capacity_min_constraint",
        mask=np.isfinite(p.storage_energy_capacity_investment_min)
        * np.not_equal(
            p.storage_energy_capacity_investment_min,
            p.storage_energy_capacity_investment_max,
        ),
    )

    m.add_constraints(
        planning_storage_energy_capacity.sum("year_dec")
        <= p.storage_energy_capacity_investment_max,
        name="planning_storage_energy_capacity_max_constraint",
        mask=np.isfinite(p.storage_energy_capacity_investment_max)
        * np.not_equal(
            p.storage_energy_capacity_investment_min,
            p.storage_energy_capacity_investment_max,
        ),
    )

    m.add_constraints(
        planning_storage_energy_capacity.sum("year_dec")
        == p.storage_energy_capacity_investment_max,
        name="planning_storage_energy_capacity_def",
        mask=np.isfinite(p.storage_energy_capacity_investment_max)
        * np.equal(
            p.storage_energy_capacity_investment_min,
            p.storage_energy_capacity_investment_max,
        ),
    )

    # Ties energy capacity to power capacity by a fixed ratio (e.g. 4 for
    # a 4h battery: 1 MW power -> 4 MWh energy), only where configured --
    # power and energy are otherwise free to evolve independently.
    # storage_energy_power_ratio has no year_inv dimension, so this holds
    # per investment vintage and (being linear) automatically also holds
    # at the operation level once summed by operation_storage_*_capacity_def.
    m.add_constraints(
        planning_storage_energy_capacity
        - p.storage_energy_power_ratio * planning_storage_power_capacity
        == 0,
        name="planning_storage_energy_power_ratio_constraint",
        mask=np.isfinite(p.storage_energy_power_ratio),
    )

    # Costs - Storage

    m.add_constraints(
        -operation_storage_costs
        # No variable costs in the model for storage
        + (p.storage_fixed_cost_power * operation_storage_power_capacity)
        + (p.storage_fixed_cost_energy * operation_storage_energy_capacity)
        == 0,
        name="operation_storage_costs_def",
    )

    m.add_constraints(
        -planning_storage_costs
        + (
            (
                planning_storage_power_capacity * p.storage_annuity_cost_power
                + planning_storage_energy_capacity
                * p.storage_annuity_cost_energy
            )
            .where((p.year_inv <= p.year_op) * (p.year_op < p.year_dec))
            .sum(["year_dec", "year_inv"])
        ).where(
            cond=p.storage_annuity_perfect_foresight,
            other=(
                (
                    planning_storage_power_capacity.sum("year_dec")
                    * p.storage_annuity_cost_power.min(
                        [
                            dim
                            for dim in p.storage_annuity_cost_power.dims
                            if dim == "year_dec"
                        ]
                    )
                    + planning_storage_energy_capacity.sum("year_dec")
                    * p.storage_annuity_cost_energy.min(
                        [
                            dim
                            for dim in p.storage_annuity_cost_energy.dims
                            if dim == "year_dec"
                        ]
                    )
                )
                .where(
                    (p.year_inv <= p.year_op)
                    * (p.year_op < p.storage_end_of_life)
                )
                .sum(["year_inv"])
            ),
        )
        == 0,
        name="planning_storage_costs_def",
    )

    return m
