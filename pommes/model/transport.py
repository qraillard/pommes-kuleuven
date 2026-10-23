"""Module related to inter-zone resource transport."""

import numpy as np
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset

from pommes.model.annuity import annuity_full_life, full_life_window


def add_transport(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_adequacy_constraint: Constraint,
) -> Model:
    """
    Add transport-related components to the Linopy model.

    Including variables, constraints, and costs related to energy transport
    between areas.

    Args:
        model (linopy.Model):
            The Linopy model to which transport-related elements will be added.
        model_parameters (xarray.Dataset):
            Dataset containing energy system parameters, including transport
            capacity limits, network topology, and costs.
        annualised_totex_def (linopy.Constraint):
            Constraint defining annualised total expenditures (totex), which
            will be updated with transport-specific costs.
        operation_adequacy_constraint (linopy.Constraint):
            Constraint ensuring operational adequacy, enforcing supply-demand
            balance, which will be updated with transport-related contributions.

    Returns:
        linopy.Model:
            The updated Linopy model with added transport-related variables,
            costs, and constraints.

    Note:
        A (link, transport_tech) pair exists where ``transport_resource`` is a
        resource of the model. The flows, capacities and rows below are created
        for these pairs only (the others are NaN in a solution, as masked
        variables are): a study with one tech per resource, each on its own
        links, does not build the full link x tech grid.

        This function introduces the following elements into the model:

        **Variables**

        - *Operation*
            - ``operation_transport_power_capacity``
                Represents the operational power capacity for transport
                technologies on each link and operational year.
            - ``operation_transport_power``
                Represents the power flow for transport technologies by link,
                hour, and operational year.
            - ``operation_transport_net_generation``
                Represents the net energy transferred between areas, accounting
                for inflows and outflows.
            - ``operation_transport_power_in``
                Represents the power injected into the link at its
                ``transport_area_from`` end. Equal to
                ``operation_transport_power`` (the flow withdrawn at
                ``transport_area_to``) unless ``transport_linepack_capacity``
                is configured for that link/tech, in which case the two may
                differ while the linepack buffer charges/discharges.
            - ``operation_transport_linepack_level``
                Represents the pipe's own gas/volume buffer level, acting as
                a store that decouples the injected and withdrawn flows.
                Only meaningful where ``transport_linepack_capacity`` is
                configured (not NaN).
            - ``operation_transport_costs``
                Represents operational costs associated with transport
                technologies for each area and operational year.

        - *Planning*
            - ``planning_transport_power_capacity``
                Represents the planned power capacity investments in transport
                infrastructure.
            - ``planning_transport_costs``
                Represents planning costs associated with transport investments.

        **Constraints**

        - *Operation*
            - ``operation_transport_power_max_constraint``
                Limits operational power flow to available capacity.
            - ``operation_transport_power_in_def``
                Where linepack isn't configured, sets
                ``operation_transport_power`` (withdrawn) to
                ``transport_efficiency`` times
                ``operation_transport_power_in`` (injected); at the
                default efficiency of 1 the two are equal.
            - ``operation_transport_linepack_level_def``
                Defines the linepack buffer level's evolution (cyclic over
                the year) from the difference between injected and
                withdrawn flow, where configured.
            - ``operation_transport_linepack_level_max_constraint``
                Bounds the linepack level by ``transport_linepack_capacity``.
            - ``operation_transport_power_in_max_constraint``
                Bounds ``operation_transport_power_in`` by the same
                capacity/availability envelope as the outflow.
            - ``operation_transport_net_generation_def``
                Defines net energy flow for transport between different areas.

        - *Planning*
            - ``planning_transport_power_capacity_min_constraint``
                Enforces a lower limit on planned transport power investments.
            - ``planning_transport_power_capacity_max_constraint``
                Enforces an upper limit on planned transport power investments.

        - *Costs*
            - ``operation_transport_costs_def``
                Defines operational transport costs: fixed costs on the
                capacity, and hurdle costs on the flow weighted by time step
                and year normalisation like other variable costs.
            - ``planning_transport_costs_def``
                Defines planning costs for transport investments, considering
                annuities.

        **Optional trade-link parameters** (for shipping between regions)

        - ``transport_efficiency`` (link, transport_tech, year_op): share of
          the injected flow delivered at the "to" end (boil-off, transfer
          losses). Not applied on links with linepack.
        - ``transport_fuel_area`` (link, transport_tech): the area the fuel
          is drawn at instead of the "from" end, e.g. a global bunker market.
        - ``transport_fuel_consumption`` (link, transport_tech, year_op) and
          ``transport_fuel_resource`` (link, transport_tech): fuel drawn at
          the "from" end per unit injected, and the resource it is taken
          from (e.g. the bunker fuel of a ship). Its emissions belong to
          whatever conversion supplies that resource.
        - ``transport_delay`` (link, transport_tech): hours between injection
          at the "from" end and delivery at the "to" end (loading, voyage,
          unloading): what arrives at hour h left at hour h - delay, cyclic
          over the year like storage levels. Needs a uniform time step and a
          contiguous series of hours (not representative periods); not
          applied on links with linepack. The goods in transit are the
          difference between what has left and what has arrived.

        These additions ensure that transport infrastructure operates within
        technical and economic limits, allowing the model to simulate energy
        transport flows realistically.
    """
    m = model
    p = model_parameters

    # The delay between injection and delivery is given in hours (default 0);
    # the flow is rolled by whole time steps, so it is converted to steps,
    # which needs one step length.
    step = np.unique(p.time_step_duration.values)
    if (p.transport_delay > 0).any() and step.size != 1:
        raise ValueError("transport_delay needs a uniform time_step_duration")
    delay_steps = np.round(p.transport_delay / step[0]).astype("int64")

    # A (link, tech) pair exists where it carries a resource of the model: a
    # study with one tech per resource gives each tech a resource on its own
    # links only, and on the others (no resource) nothing can flow, so none of
    # the flows, capacities and rows of the module is created there.
    link_tech = xr.DataArray(
        np.zeros((p.sizes["link"], p.sizes["transport_tech"])),
        coords=[p.link, p.transport_tech],
    )
    pair_exists = xr.DataArray(
        np.isin(
            p.transport_resource.broadcast_like(link_tech)
            .transpose("link", "transport_tech")
            .values.astype(str),
            p.resource.values.astype(str),
        ),
        coords=[p.link, p.transport_tech],
    )

    # ------------
    # Variables
    # ------------

    # Operation - transport

    operation_transport_power_capacity = m.add_variables(
        name="operation_transport_power_capacity",
        lower=0,
        coords=[p.link, p.transport_tech, p.year_op],
        mask=pair_exists,
    )

    operation_transport_power = m.add_variables(
        name="operation_transport_power",
        lower=0,
        coords=[p.link, p.transport_tech, p.hour, p.year_op],
        mask=pair_exists,
    )

    # Power injected into the link at its transport_area_from end. Equal to
    # operation_transport_power (the flow withdrawn at transport_area_to)
    # unless transport_linepack_capacity is configured, in which case the
    # pipe's own gas/volume buffer can absorb the difference -- see
    # operation_transport_power_in_def and operation_transport_linepack_level_def.
    operation_transport_power_in = m.add_variables(
        name="operation_transport_power_in",
        lower=0,
        coords=[p.link, p.transport_tech, p.hour, p.year_op],
        mask=pair_exists,
    )

    # Linepack buffer level (only meaningful where transport_linepack_capacity
    # is configured; see operation_transport_linepack_level_def).
    operation_transport_linepack_level = m.add_variables(
        name="operation_transport_linepack_level",
        lower=0,
        coords=[p.link, p.transport_tech, p.hour, p.year_op],
        mask=np.isfinite(p.transport_linepack_capacity) & pair_exists,
    )

    # Operation - transport intermediate variables

    operation_transport_net_generation = m.add_variables(
        name="operation_transport_net_generation",
        coords=[p.area, p.hour, p.transport_tech, p.resource, p.year_op],
        mask=np.equal(p.transport_resource, p.resource).any(
            [dim for dim in ["link"] if dim in p.transport_resource.dims]
        ),
    )

    # Planning - transport

    planning_transport_power_capacity = m.add_variables(
        name="planning_transport_power_capacity",
        lower=0,
        coords=[p.link, p.transport_tech, p.year_dec, p.year_inv],
        mask=xr.where(
            cond=p.transport_early_decommissioning,
            x=(p.year_inv < p.year_dec)
            * (p.year_dec <= p.transport_end_of_life)
            * np.logical_or(
                p.year_dec <= p.year_inv.max(),
                p.year_dec == p.transport_end_of_life,
            ),
            y=p.year_dec == p.transport_end_of_life,
        )
        & pair_exists,
    )

    # Costs - transport

    operation_transport_costs = m.add_variables(
        name="operation_transport_costs",
        # no lower bound: hurdle costs may be negative (a subsidy on a link),
        # as the operating costs of conversion and net imports
        coords=[p.area, p.transport_tech, p.year_op],
    )

    planning_transport_costs = m.add_variables(
        name="planning_transport_costs",
        lower=0,
        coords=[p.area, p.transport_tech, p.year_op],
    )

    # ------------------
    # Objective function
    # ------------------

    # Hurdle costs (EUR per MWh carried) are counted in operation_transport_costs below, weighted like the
    # variable costs of the other modules (time step, year normalisation, and discounting through
    # annualised_totex), so that they do not depend on the number of modelled hours or years.
    annualised_totex_def.lhs += (
        operation_transport_costs + planning_transport_costs
    ).sum(["transport_tech"])

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    operation_adequacy_constraint.lhs += (
        operation_transport_net_generation.sum("transport_tech")
    )

    # Fuel burnt by the link (e.g. ship bunker fuel), in proportion to the
    # injected flow, drawn at transport_fuel_area: the "from" end by default
    # (no value, or one that is not an area), or any area, e.g. a global bunker
    # market that the regions supply through their own links.
    if np.isfinite(p.transport_fuel_consumption).any():       # skip the term when no link burns fuel
        fuel_area = xr.where(
            p.transport_fuel_area.isin(p.area.values),
            p.transport_fuel_area,
            p.transport_area_from,
        )
        fuel_drawn = (
            (
                operation_transport_power_in
                * p.transport_fuel_consumption.fillna(0)
                * p.time_step_duration
            )
            .where(p.area == fuel_area)
            .where(p.resource == p.transport_fuel_resource)
            .sum(["link", "transport_tech"])
        )
        operation_adequacy_constraint.lhs -= fuel_drawn

    # Operation - transport

    # m.add_constraints(
    #     operation_transport_power - operation_transport_power_capacity <= 0,
    #     name="operation_transport_power_max_constraint",
    # )


    m.add_constraints(
        operation_transport_power
        - operation_transport_power_capacity
        * xr.where(
            cond=np.isfinite(p.transport_availability),
            x=p.transport_availability,
            y=1,
        )
        <= 0,
        name="operation_transport_power_max_constraint",
        mask=pair_exists,
    )

    # Operation - transport linepack

    # Where linepack isn't configured, the flow withdrawn at the "to" end is
    # the injected flow times the link efficiency (default 1: a lossless
    # link).
    no_linepack = ~np.isfinite(p.transport_linepack_capacity)
    m.add_constraints(
        operation_transport_power_in * p.transport_efficiency
        - operation_transport_power
        == 0,
        name="operation_transport_power_in_def",
        mask=no_linepack & (delay_steps == 0) & pair_exists,
    )

    # With a delay of k time steps, what is withdrawn at hour h was injected
    # at hour h - k (cyclic over the year): one constraint set per delay value.
    for k in np.unique(delay_steps.values[delay_steps.values > 0]):
        m.add_constraints(
            operation_transport_power_in.roll(hour=int(k)) * p.transport_efficiency
            - operation_transport_power
            == 0,
            name=f"operation_transport_power_in_def_delay_{int(k)}",
            mask=no_linepack & (delay_steps == k) & pair_exists,
        )

    # Where configured, the pipe's own gas/volume buffer absorbs the
    # difference between what's injected (power_in) and what's withdrawn
    # (operation_transport_power) -- cyclic over the year, like storage's
    # own level definition.
    m.add_constraints(
        -operation_transport_linepack_level
        + operation_transport_linepack_level.roll(hour=1)
        + (operation_transport_power_in - operation_transport_power)
        * p.time_step_duration
        == 0,
        name="operation_transport_linepack_level_def",
        mask=np.isfinite(p.transport_linepack_capacity) & pair_exists,
    )

    m.add_constraints(
        operation_transport_linepack_level <= p.transport_linepack_capacity,
        name="operation_transport_linepack_level_max_constraint",
        mask=np.isfinite(p.transport_linepack_capacity) & pair_exists,
    )

    m.add_constraints(
        operation_transport_power_in
        - operation_transport_power_capacity
        * xr.where(
            cond=np.isfinite(p.transport_availability),
            x=p.transport_availability,
            y=1,
        )
        <= 0,
        name="operation_transport_power_in_max_constraint",
        mask=np.isfinite(p.transport_linepack_capacity) & pair_exists,
    )

    # Operation - Transport unit commitment

    m.add_constraints(
        -p.transport_ramp_up * operation_transport_power_capacity
        + (
                operation_transport_power
                - operation_transport_power.shift(hour=1)
        )
        / p.time_step_duration
        <= 0,
        name="operation_transport_ramp_up_constraint",
        mask=np.isfinite(p.transport_ramp_up)
        * (p.hour != p.hour[0])
        * pair_exists,
    )

    m.add_constraints(
        -p.transport_ramp_down * operation_transport_power_capacity
        + (
                operation_transport_power.shift(hour=1)
                - operation_transport_power
        )
        / p.time_step_duration
        <= 0,
        name="operation_transport_ramp_down_constraint",
        mask=np.isfinite(p.transport_ramp_down)
        * (p.hour != p.hour[0])
        * pair_exists,
    )

    # Operation - transport intermediate variables

    m.add_constraints(
        -operation_transport_power_capacity
        + planning_transport_power_capacity.where(
            (p.year_inv <= p.year_op) * (p.year_op < p.year_dec)
        ).sum(["year_dec", "year_inv"])
        == 0,
        name="operation_transport_power_capacity_def",
        mask=pair_exists,
    )

    m.add_constraints(
        -operation_transport_net_generation
        + p.time_step_duration
        * (
            operation_transport_power.where(p.area == p.transport_area_to).sum(
                "link"
            )
            # Uses power_in (what's actually injected at the "from" end)
            # rather than operation_transport_power (what's withdrawn at
            # "to") -- identical to it unless linepack is configured, in
            # which case the two legitimately differ while the buffer
            # charges/discharges.
            - operation_transport_power_in.where(
                p.area == p.transport_area_from
            ).sum("link")
        ).where(
            np.equal(p.transport_resource, p.resource).any(
                [
                    dim
                    for dim in ["link", "year_inv"]
                    if dim in p.transport_resource.dims
                ]
            )
        )
        == 0,
        name="operation_transport_net_generation_def",
        mask=np.equal(p.transport_resource, p.resource).any(
            [dim for dim in ["link"] if dim in p.transport_resource.dims]
        ),
    )

    # Operation - other constraints
    m.add_constraints(
        operation_transport_power_capacity <= p.transport_power_capacity_max,
        name="operation_transport_power_capacity_max_constraint",
        mask=np.isfinite(p.transport_power_capacity_max) & pair_exists,
    )

    m.add_constraints(
        operation_transport_power_capacity >= p.transport_power_capacity_min,
        name="operation_transport_power_capacity_min_constraint",
        mask=np.isfinite(p.transport_power_capacity_min) & pair_exists,
    )

    # Planning - transport

    m.add_constraints(
        planning_transport_power_capacity.sum("year_dec")
        >= p.transport_power_capacity_investment_min,
        name="planning_transport_power_capacity_min_constraint",
        mask=np.isfinite(p.transport_power_capacity_investment_min)
        * np.not_equal(
            p.transport_power_capacity_investment_min,
            p.transport_power_capacity_investment_max,
        )
        * pair_exists,
    )

    m.add_constraints(
        planning_transport_power_capacity.sum("year_dec")
        <= p.transport_power_capacity_investment_max,
        name="planning_transport_power_capacity_max_constraint",
        mask=np.isfinite(p.transport_power_capacity_investment_max)
        * np.not_equal(
            p.transport_power_capacity_investment_min,
            p.transport_power_capacity_investment_max,
        )
        * pair_exists,
    )

    m.add_constraints(
        planning_transport_power_capacity.sum("year_dec")
        == p.transport_power_capacity_investment_min,
        name="planning_transport_power_capacity_def",
        mask=np.isfinite(p.transport_power_capacity_investment_max)
        * np.equal(
            p.transport_power_capacity_investment_min,
            p.transport_power_capacity_investment_max,
        )
        * pair_exists,
    )

    # Costs - transport

    operation_year_normalization = (
        p.operation_year_duration / p.time_step_duration.sum("hour")
    )

    m.add_constraints(
        -operation_transport_costs
        # Fixed costs and hurdle costs, split half and half between the two
        # ends of each link
        + 0.5
        * (
            (
                p.transport_fixed_cost * operation_transport_power_capacity
                + p.transport_hurdle_costs.fillna(0)
                * operation_year_normalization
                * (p.time_step_duration * operation_transport_power).sum("hour")
            )
            .where(
                np.logical_or(
                    p.area == p.transport_area_from,
                    p.area == p.transport_area_to,
                )
            )
            .sum("link")
        )
        == 0,
        name="operation_transport_costs_def",
    )

    if annuity_full_life(p, "transport_annuity_cost"):
        # Annuity fixed at investment, paid over the vintage's full life
        # whatever the decommissioning year (sunk cost); each link's
        # annuity is split between its two end areas.
        annuity_terms = 0.5 * (
            (
                planning_transport_power_capacity.sum("year_dec")
                * p.transport_annuity_cost
            )
            .where(
                np.logical_or(
                    p.area == p.transport_area_from,
                    p.area == p.transport_area_to,
                )
            )
            .where(
                full_life_window(p.year_inv, p.transport_life_span, p.year_op)
            )
            .sum(["link", "year_inv"])
        )
    else:
        # The flag has link and year_inv dimensions: choose the annuity per
        # link and vintage, then sum over both (summing first and choosing
        # afterwards gives the expression those dimensions back and writes
        # the cost row once per link and vintage).
        perfect_foresight = (
            (
                p.transport_annuity_cost * planning_transport_power_capacity
            )
            .where(
                np.logical_or(
                    p.area == p.transport_area_from,
                    p.area == p.transport_area_to,
                )
            )
            .where((p.year_inv <= p.year_op) * (p.year_op < p.year_dec))
            .sum("year_dec")
        )
        cheapest = (
            (
                planning_transport_power_capacity.sum("year_dec")
                * p.transport_annuity_cost.min(
                    [
                        dim
                        for dim in ["year_dec"]
                        if dim in p.transport_annuity_cost.dims
                    ]
                )
            )
            .where(
                np.logical_or(
                    p.area == p.transport_area_from,
                    p.area == p.transport_area_to,
                )
            )
            .where(
                (p.year_inv <= p.year_op)
                * (p.year_op < p.transport_end_of_life)
            )
        )
        annuity_terms = 0.5 * perfect_foresight.where(
            cond=p.transport_annuity_perfect_foresight, other=cheapest
        ).sum(["link", "year_inv"])

    m.add_constraints(
        -planning_transport_costs + annuity_terms == 0,
        name="planning_transport_costs_def",
    )

    return m
