"""Module for carbon related constraints."""

import numpy as np
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset

from pommes.model.combined import combined_groups
from pommes.model.conversion import conversion_groups
from pommes.model.net_import import net_import_groups
from pommes.utils import (
    TimeBlocks,
    annual_mask,
    annual_quantity,
    block_prefix,
    change_resolution,
    condition_coords,
    condition_mask,
    expected_over_conditions,
    time_blocks,
)


def _add_emissions(defs, expr, tb, L, hourly_scale=None):
    """
    Add ``expr`` (tCO2 per step of a group on resolution ``L``) to the emission rows ``defs``.

    The hourly group (``L`` None) adds today's ``expr * hourly_scale`` with ``+=``;
    a block group adds the block energy broadcast to the hours by
    ``change_resolution``, joined to the rows by label (``join="left"``).
    """
    for con in defs:
        if L is None:
            con.lhs += expr if hourly_scale is None else expr * hourly_scale
        else:
            con.lhs = con.lhs.add(change_resolution(expr, tb, L, None), join="left")


def add_carbon(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_year_normalization: float,
    tb: TimeBlocks | None = None,
) -> Model:
    """
    Add carbon-related components to the Linopy model.

    Including variables, constraints, and costs related to carbon emissions
    and taxation.

    Args:
        model (linopy.Model):
            The Linopy model to which carbon-related elements will be added.
        model_parameters (xarray.Dataset):
            Dataset containing energy system parameters, including carbon
            emission factors and cost details.
        annualised_totex_def (linopy.Constraint):
            Constraint defining annualised total expenditures (totex), which
            will be updated with carbon-specific costs.
        operation_year_normalization (float):
            Normalization factor for operational year durations.
        tb (pommes.utils.TimeBlocks, optional):
            Block lengths and maps of the dataset, computed once by
            ``build_model`` (``time_blocks(model_parameters)`` when None).
            The emissions of block technology groups and of coarse
            net_import balances are broadcast to the hours.

    Returns:
        linopy.Model:
            The updated Linopy model with added carbon-related variables,
            costs, and constraints.

    Note:
        This function introduces the following elements into the model:

        **Variables**

        - *Operation*
            - ``operation_carbon_emissions``
                Represents carbon emissions at the operational level for each
                area, hour, and operational year.
            - ``operation_total_carbon_emissions``
                Represents total carbon emissions over time for each area,
                hour, and operational year.
            - ``operation_carbon_costs``
                Represents the cost associated with carbon emissions for each
                area and operational year.
            - ``operation_carbon_tax_group_costs``
                Represents the cost of the carbon tax charged on the net
                emissions of a group of areas, for each carbon tax group and
                operational year. Only built when ``carbon_tax_group``
                membership data (``carbon_tax_group_areas``) is present in
                the dataset.

        **Constraints**

        - *Operation*
            - ``operation_carbon_goal_constraint``
                Limits the total carbon emissions to a predefined carbon goal.
            - ``operation_carbon_goal_area_constraint``
                Limits each area's own total carbon emissions to a
                per-area carbon goal (``carbon_goal_area``), only where
                configured (not NaN) -- in addition to, not instead of,
                the system-wide ``carbon_goal`` above.
            - ``operation_carbon_goal_group_constraint``
                Limits the aggregated total carbon emissions of a group of
                areas (e.g. a country made up of several modelled areas) to
                a per-group carbon goal (``carbon_goal_group_max``), only
                where ``carbon_goal_group`` membership data
                (``carbon_goal_group_areas``) is present in the dataset.
            - ``operation_carbon_emissions_def``
                Defines carbon emissions based on power production and emission
                factors.
            - ``operation_total_carbon_emissions_def``
                Ensures total carbon emissions correctly aggregate emissions
                over time.

        - *Costs*
            - ``operation_carbon_costs_def``
                Defines carbon costs as a function of carbon tax and emissions.
            - ``operation_carbon_tax_group_costs_def``
                Defines the carbon tax cost of a group of areas as
                ``max(0, carbon_tax_group_rate x net emissions of its member
                areas)``: removals of one member offset the emissions of the
                others inside the group, and a group with negative net
                emissions pays nothing and is not paid for its removals.
                Only built when ``carbon_tax_group`` is in the dataset.

        **Carbon tax per area or per group of areas**

        By default the tax is charged per area (``carbon_tax``): each area
        pays ``max(0, tax x its own net emissions)``. When a study supplies
        the ``carbon_tax_group`` dimension, the tax can instead be charged on
        a group of areas (``carbon_tax_group_areas`` is the boolean
        area x group membership, ``carbon_tax_group_rate`` the tax of the
        group in each operational year [currency per tCO2]); the group pays
        ``max(0, rate x sum of the net emissions of its members)``. The
        per-area tax is masked for the member areas: an area that belongs to
        a group whose rate is set (not NaN) in an operational year does not
        also pay ``carbon_tax`` in that year, so nothing is charged twice. A
        NaN group rate means "no group tax": the member areas then keep their
        own ``carbon_tax``. An area in several groups is taxed in each of
        them. The group costs enter the objective with the same discount
        factor as the per-area costs, but not ``annualised_totex`` (which is
        per area); read them in ``operation_carbon_tax_group_costs``.

        These additions ensure that carbon constraints and costs are
        appropriately modeled, allowing for emissions regulation within the
        energy system. The model is enhanced to accurately capture carbon
        emissions behavior and its economic impact.
    """
    m = model
    p = model_parameters
    v = m.variables
    tb = time_blocks(p) if tb is None else tb

    # ------------
    # Variables
    # ------------

    # Operation - Carbon

    operation_carbon_emissions = m.add_variables(
        name="operation_carbon_emissions",
        coords=[p.area, p.hour, p.year_op, *condition_coords(p)],
        mask=condition_mask(p),
    )

    operation_total_carbon_emissions = m.add_variables(
        name="operation_total_carbon_emissions",
        coords=[p.area, p.hour, p.year_op, *condition_coords(p)],
        mask=condition_mask(p),
    )

    # Costs - Carbon

    # Carbon costs = max(0, tax x net emissions) per area and year: a net-negative area (biogenic
    # credit with capture, DAC) pays nothing and is not paid for its removals. Lower bound 0 here,
    # an inequality below, the objective makes it tight.
    operation_carbon_costs = m.add_variables(
        name="operation_carbon_costs",
        lower=0,
        coords=[p.area, p.year_op, *condition_coords(p)],
        mask=condition_mask(p),
    )

    # ------------------
    # Objective function
    # ------------------

    annualised_totex_def.lhs += expected_over_conditions(operation_carbon_costs, p)

    # --------------
    # Constraints
    # --------------

    # Operation - Carbon

    m.add_constraints(
        # Warning, timestep_duration is used here and for the total emission
        # as there is a "duplication"
        # of the variables operation_total_carbon_emissions and
        # operation_carbon_emissions
        annual_quantity(
            +operation_year_normalization
            * operation_total_carbon_emissions.sum(["area", "hour"]),
            p,
            "carbon_goal_expected",
        )
        <= p.carbon_goal,
        mask=annual_mask(p, "carbon_goal_expected", np.isfinite(p.carbon_goal)),
        name="operation_carbon_goal_constraint",
    )

    # carbon_tax is already area-indexed (see operation_carbon_costs_def
    # below) -- carbon_goal_area is the equivalent per-area limit for the
    # system-wide carbon_goal above, only where configured (not NaN).
    m.add_constraints(
        annual_quantity(
            +operation_year_normalization
            * operation_total_carbon_emissions.sum("hour"),
            p,
            "carbon_goal_area_expected",
        )
        <= p.carbon_goal_area,
        mask=annual_mask(
            p, "carbon_goal_area_expected", np.isfinite(p.carbon_goal_area)
        ),
        name="operation_carbon_goal_area_constraint",
    )

    # General goals for a group of areas (e.g. a country made up of several
    # modelled areas) that, aggregated, must follow a common bound.
    # carbon_goal_group_areas is a boolean (area, carbon_goal_group)
    # membership matrix (an area may belong to more than one group); only
    # built when a study actually supplies the carbon_goal_group dimension
    # -- a study that never defines it gets no constraint at all, rather
    # than relying on a dimensionless default to broadcast to a no-op.
    if "carbon_goal_group" in p.dims:
        group_total_emissions = (
            operation_total_carbon_emissions.sum("hour") * p.carbon_goal_group_areas
        ).sum("area")
        m.add_constraints(
            annual_quantity(
                +operation_year_normalization * group_total_emissions,
                p,
                "carbon_goal_group_expected",
            )
            <= p.carbon_goal_group_max,
            mask=annual_mask(
                p,
                "carbon_goal_group_expected",
                np.isfinite(p.carbon_goal_group_max),
            ),
            name="operation_carbon_goal_group_constraint",
        )

    # Operation - Carbon intermediate variables

    operation_carbon_emissions_def = m.add_constraints(
        -operation_carbon_emissions
        == 0,
        mask=condition_mask(p),
        name="operation_carbon_emissions_def",
    )

    operation_total_carbon_emissions_def = m.add_constraints(
        -operation_total_carbon_emissions
        == 0,
        mask=condition_mask(p),
        name="operation_total_carbon_emissions_def",
    )

    both = [operation_carbon_emissions_def, operation_total_carbon_emissions_def]

    # Technology groups: the hourly one as today, a block group as tCO2 per
    # block (power x factor x block duration) broadcast to its hours
    for module, groups_of, tech_dims in (
        ("conversion", conversion_groups, ["conversion_tech"]),
        ("combined", combined_groups, ["combined_tech", "mode"]),
    ):
        if module not in p.keys() or not p[module]:
            continue
        for L, p_g, _ in groups_of(p, tb, keep=[f"{module}_emission_factor"]):
            power = v[f"{block_prefix(module, L)}power"]
            emissions = (power * p_g[f"{module}_emission_factor"]).sum(tech_dims)
            if L is not None:
                emissions = emissions * tb.duration[L]
            _add_emissions(both, emissions, tb, L, p.time_step_duration)

    if "process" in p.keys() and p.process:
        operation_carbon_emissions_def.lhs += (
            v.operation_process_power * p.process_emission_factor
        ).sum(["process_tech", "mode"]) * p.time_step_duration
        operation_total_carbon_emissions_def.lhs += (
            v.operation_process_power * p.process_emission_factor
        ).sum(["process_tech", "mode"]) * p.time_step_duration

    if "net_import" in p.keys() and p.net_import:
        # Imports are energies per time step (they enter the adequacy balance as they are), so
        # unlike the power-based terms above they are not multiplied by time_step_duration,
        # and a coarse balance's block imports (energies per block) are broadcast as they are.
        for L, p_g in net_import_groups(p, tb):
            imports = v[f"{block_prefix('net_import', L)}import"]
            _add_emissions(
                [operation_carbon_emissions_def],
                (imports * p_g.net_import_emission_factor).sum(["resource"]),
                tb,
                L,
            )
            _add_emissions(
                [operation_total_carbon_emissions_def],
                (imports * p_g.net_import_total_emission_factor).sum(["resource"]),
                tb,
                L,
            )

    # Costs - Carbon

    carbon_tax_area = xr.where(
        cond=np.isfinite(p.carbon_tax),
        x=p.carbon_tax,
        y=0,
    )
    if "carbon_tax_group" in p.dims:
        # A member area of a carbon tax group is taxed through the group,
        # not also on its own, in the years where the group has a rate.
        taxed_by_group = (
            p.carbon_tax_group_areas * np.isfinite(p.carbon_tax_group_rate)
        ).any("carbon_tax_group")
        carbon_tax_area = xr.where(
            cond=taxed_by_group, x=0, y=carbon_tax_area
        )

    m.add_constraints(
        -operation_carbon_costs
        # Warning, timestep_duration is used here and for the total costs as
        # there is a "duplication"
        # of the variables operation_total_carbon_emissions and
        # operation_carbon_emissions
        + operation_year_normalization
        * operation_carbon_emissions.sum("hour")
        * carbon_tax_area
        <= 0,
        mask=condition_mask(p),
        name="operation_carbon_costs_def",
    )

    # Carbon tax on groups of areas (e.g. a country made up of several
    # modelled areas, or all areas together): the group pays the tax on the
    # net emissions of all its members, so that a removal in one member
    # offsets the emissions of the others. Same shape as the per-area cost:
    # cost >= rate x net emissions, cost >= 0, tight at the optimum because
    # the cost is minimised. Pure LP. Only built when a study supplies the
    # carbon_tax_group dimension.
    if "carbon_tax_group" in p.dims:
        operation_carbon_tax_group_costs = m.add_variables(
            name="operation_carbon_tax_group_costs",
            lower=0,
            coords=[p.carbon_tax_group, p.year_op, *condition_coords(p)],
            mask=condition_mask(p),
        )

        # Same discounting as the per-area costs, which reach the objective
        # through annualised_totex (per area); a group is not an area.
        m.objective += (
            p.discount_factor
            * expected_over_conditions(
                operation_carbon_tax_group_costs.sum("carbon_tax_group"), p
            )
        ).sum("year_op")

        group_emissions = (
            operation_carbon_emissions.sum("hour") * p.carbon_tax_group_areas
        ).sum("area")
        m.add_constraints(
            -operation_carbon_tax_group_costs
            + operation_year_normalization
            * group_emissions
            * xr.where(
                cond=np.isfinite(p.carbon_tax_group_rate),
                x=p.carbon_tax_group_rate,
                y=0,
            )
            <= 0,
            mask=condition_mask(p),
            name="operation_carbon_tax_group_costs_def",
        )

    return m
