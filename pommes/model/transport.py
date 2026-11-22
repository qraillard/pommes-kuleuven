"""Module related to inter-zone resource transport."""

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset

from pommes.model.annuity import annuity_full_life, full_life_window
from pommes.utils import (
    TimeBlocks,
    add_to_balances,
    block_dim,
    block_prefix,
    check_block_transport,
    check_group_coord,
    check_partition,
    condition_coords,
    condition_mask,
    expected_over_conditions,
    time_blocks,
    transport_block_availability,
    transport_pair_exists,
)


def transport_groups(
    p: Dataset, tb: TimeBlocks
) -> list[tuple[int | None, Dataset, None]]:
    """
    The groups ``(L, p_g, None)`` of the transport module.

    ``L`` is None for the hourly technologies, else a block length
    (ascending); a group without technology is left out. ``p_g`` is
    ``p.sel(transport_tech=techs_g)``; a block group holds
    ``transport_block_availability`` (computed on the full ``p``) and no
    ``transport_availability`` (the only transport parameter with an ``hour``
    dimension). Without a block length on a transport technology (conversion
    blocks included) there is one group, ``(None, p, None)``, ``p`` itself.
    Transport has no unit index: the third element is always None.
    """
    lengths = tb.tech_length["transport"].values
    if not (tb.active and np.isfinite(lengths).any()):
        return [(None, p, None)]
    keys = [None, *sorted({int(x) for x in lengths[np.isfinite(lengths)]})]
    parts = check_partition(
        {L: pd.Index(tb.techs("transport", L)) for L in keys},
        p.indexes["transport_tech"],
    )
    out = []
    for L, part in parts.items():
        if L is None:
            p_g = p.sel(transport_tech=list(part))
        else:
            p_g = p.drop_vars("transport_availability", errors="ignore").sel(
                transport_tech=list(part)
            )
            p_g = p_g.assign(
                transport_block_availability=transport_block_availability(
                    p, tb, L, list(part)
                )
            )
        out.append((L, p_g, None))
    return out


def _refuse_unbuilt_rows(p_g: Dataset) -> None:
    """
    Interim guard: refuse an input whose rows the block groups do not build yet.

    A non-zero fuel emission factor with a finite fuel consumption (Task 6,
    which removes the function).
    """

    def finite(name):
        return name in p_g and bool(np.isfinite(p_g[name]).any())

    fuel = finite("transport_fuel_consumption")
    for name in (
        "transport_fuel_emission_factor",
        "transport_fuel_total_emission_factor",
    ):
        if fuel and name in p_g and bool((p_g[name].fillna(0) != 0).any()):
            raise NotImplementedError(
                f"{name}: fuel emissions of a block transport technology are not built yet."
            )


def _group_variables(
    m: Model, p: Dataset, tb: TimeBlocks, L: int | None, pair
):
    """
    Power, injected power, linepack level and net generation of a group.

    On the group's own technologies and time axis (``hour`` for the hourly
    group, ``time_block_{L}h`` for a block group); ``pair`` is the pair mask
    restricted to the group. The linepack level exists for the hourly group
    only (None for a block group, R12).
    """
    prefix = block_prefix("transport", L)
    time = p.hour if L is None else tb.duration[L][block_dim(L)]
    power = m.add_variables(
        name=f"{prefix}power",
        lower=0,
        coords=[
            p.link,
            p.transport_tech,
            time,
            p.year_op,
            *condition_coords(p),
        ],
        mask=condition_mask(p, pair),
    )

    # Power injected into the link at its transport_area_from end. Equal to
    # operation_transport_power (the flow withdrawn at transport_area_to)
    # unless transport_linepack_capacity is configured, in which case the
    # pipe's own gas/volume buffer can absorb the difference -- see
    # operation_transport_power_in_def and operation_transport_linepack_level_def.
    power_in = m.add_variables(
        name=f"{prefix}power_in",
        lower=0,
        coords=[
            p.link,
            p.transport_tech,
            time,
            p.year_op,
            *condition_coords(p),
        ],
        mask=condition_mask(p, pair),
    )

    # Linepack buffer level (only meaningful where transport_linepack_capacity
    # is configured; see operation_transport_linepack_level_def).
    level = None
    if L is None:
        level = m.add_variables(
            name=f"{prefix}linepack_level",
            lower=0,
            coords=[
                p.link,
                p.transport_tech,
                time,
                p.year_op,
                *condition_coords(p),
            ],
            mask=condition_mask(
                p, np.isfinite(p.transport_linepack_capacity) & pair
            ),
        )

    # Operation - transport intermediate variables

    net_generation = m.add_variables(
        name=f"{prefix}net_generation",
        coords=[
            p.area,
            time,
            p.transport_tech,
            p.resource,
            p.year_op,
            *condition_coords(p),
        ],
        mask=condition_mask(
            p,
            np.equal(p.transport_resource, p.resource).any(
                [dim for dim in ["link"] if dim in p.transport_resource.dims]
            ),
        ),
    )
    return power, power_in, level, net_generation


class _TransportRows:
    """
    The operational rows of one transport group.

    ``L`` is None for the hourly group, whose rows are today's; else the rows
    are the hourly rows on the block axis ``T = time_block_{L}h``: the step
    ``D`` is the block duration, the ramp divisor ``D0`` the duration of its
    first step and ``first`` replaces ``hour != hour[0]``. ``capacity`` and
    ``costs`` are the year-level variables restricted to the group, ``pair``
    and ``delay_steps`` restricted too, and ``p_g`` supplies every parameter
    (H7). After each row of a split module the group's coordinate is checked
    (H1/H2). ``variables`` is the tuple of ``_group_variables``.
    """

    def __init__(
        self,
        m,
        p_g,
        tb,
        L,
        variables,
        capacity,
        costs,
        norm,
        pair,
        delay_steps=None,
    ):
        self.m, self.p, self.tb, self.L = m, p_g, tb, L
        self.power, self.power_in, self.level, self.net_generation = variables
        self.capacity, self.costs, self.norm = capacity, costs, norm
        self.pair, self.delay_steps = pair, delay_steps
        self.prefix = block_prefix("transport", L)
        if L is None:
            self.T = "hour"
            self.D = self.D0 = p_g.time_step_duration
            self.first = p_g.hour != p_g.hour[0]
        else:
            self.T = block_dim(L)
            self.D, self.D0 = tb.duration[L], tb.first_step_duration[L]
            self.first = tb.duration[L][self.T] != 0
        # the module is split: some transport tech has a block length
        self.guard = tb.active and bool(
            np.isfinite(tb.tech_length["transport"].values).any()
        )
        self.dim, self.ref = "transport_tech", p_g.indexes["transport_tech"]

    def add(self, constraint, name, mask=None):
        """Add the row ``prefix + name``; check its coordinate in a split module."""
        con = self.m.add_constraints(
            constraint, name=f"{self.prefix}{name}", mask=mask
        )
        if self.guard:
            check_group_coord(con, self.ref, self.dim)
        return con

    def fuel(self, adequacy):
        """Fuel burnt by the link (e.g. ship bunker fuel), drawn from the balances."""
        p = self.p
        # in proportion to the injected flow, drawn at transport_fuel_area: the
        # "from" end by default (no value, or one that is not an area), or any
        # area, e.g. a global bunker market that the regions supply through
        # their own links.
        if np.isfinite(p.transport_fuel_consumption).any():  # skip the term when no link burns fuel
            fuel_area = xr.where(
                p.transport_fuel_area.isin(p.area.values),
                p.transport_fuel_area,
                p.transport_area_from,
            )
            fuel_drawn = (
                (
                    self.power_in
                    * p.transport_fuel_consumption.fillna(0)
                    * self.D
                )
                .where(p.area == fuel_area)
                .where(p.resource == p.transport_fuel_resource)
                .sum(["link", "transport_tech"])
            )
            add_to_balances(adequacy, -fuel_drawn, self.tb, src=self.L)

    def limits(self):
        """Power bounds, delays, linepack and ramps (a block group: power maximum, power_in and ramps)."""
        p, power, power_in = self.p, self.power, self.power_in
        level, capacity, pair = self.level, self.capacity, self.pair
        if self.L is not None:
            # a block group has no linepack and no delay: the hourly mask is
            # the pair mask. The availability is the block minimum
            self.add(
                power - capacity * p.transport_block_availability <= 0,
                "power_max_constraint",
                mask=condition_mask(p, pair),
            )
            self.add(
                power_in * p.transport_efficiency - power == 0,
                "power_in_def",
                mask=condition_mask(p, pair),
            )
            self.ramps()
            return

        self.add(
            power
            - capacity
            * xr.where(
                cond=np.isfinite(p.transport_availability),
                x=p.transport_availability,
                y=1,
            )
            <= 0,
            "power_max_constraint",
            mask=condition_mask(p, pair),
        )

        # Operation - transport linepack

        # Where linepack isn't configured, the flow withdrawn at the "to" end is
        # the injected flow times the link efficiency (default 1: a lossless
        # link).
        no_linepack = ~np.isfinite(p.transport_linepack_capacity)
        self.add(
            power_in * p.transport_efficiency - power == 0,
            "power_in_def",
            mask=condition_mask(
                p, no_linepack & (self.delay_steps == 0) & pair
            ),
        )

        # With a delay of k time steps, what is withdrawn at hour h was injected
        # at hour h - k (cyclic over the year): one constraint set per delay value.
        delay_steps = self.delay_steps
        for k in np.unique(delay_steps.values[delay_steps.values > 0]):
            self.add(
                power_in.roll(hour=int(k)) * p.transport_efficiency - power
                == 0,
                f"power_in_def_delay_{int(k)}",
                mask=condition_mask(p, no_linepack & (delay_steps == k) & pair),
            )

        # Where configured, the pipe's own gas/volume buffer absorbs the
        # difference between what's injected (power_in) and what's withdrawn
        # (operation_transport_power) -- cyclic over the year, like storage's
        # own level definition.
        self.add(
            -level
            + level.roll(hour=1)
            + (power_in - power) * self.D
            == 0,
            "linepack_level_def",
            mask=condition_mask(
                p, np.isfinite(p.transport_linepack_capacity) & pair
            ),
        )

        self.add(
            level <= p.transport_linepack_capacity,
            "linepack_level_max_constraint",
            mask=condition_mask(
                p, np.isfinite(p.transport_linepack_capacity) & pair
            ),
        )

        self.add(
            power_in
            - capacity
            * xr.where(
                cond=np.isfinite(p.transport_availability),
                x=p.transport_availability,
                y=1,
            )
            <= 0,
            "power_in_max_constraint",
            mask=condition_mask(
                p, np.isfinite(p.transport_linepack_capacity) & pair
            ),
        )

        self.ramps()

    def ramps(self):
        """Ramp rows on the hour axis or on the block axis (divisor: the first step of the block)."""
        p, power, capacity, pair = self.p, self.power, self.capacity, self.pair
        T, first = self.T, self.first

        # Operation - Transport unit commitment

        self.add(
            -p.transport_ramp_up * capacity
            + (power - power.shift({T: 1})) / self.D0
            <= 0,
            "ramp_up_constraint",
            mask=condition_mask(
                p, np.isfinite(p.transport_ramp_up) * first * pair
            ),
        )

        self.add(
            -p.transport_ramp_down * capacity
            + (power.shift({T: 1}) - power) / self.D0
            <= 0,
            "ramp_down_constraint",
            mask=condition_mask(
                p, np.isfinite(p.transport_ramp_down) * first * pair
            ),
        )

    def net_generation_def(self):
        """Net generation (energy per step or per block) from the flows at the two ends."""
        p = self.p
        self.add(
            -self.net_generation
            + self.D
            * (
                self.power.where(p.area == p.transport_area_to).sum("link")
                # Uses power_in (what's actually injected at the "from" end)
                # rather than operation_transport_power (what's withdrawn at
                # "to") -- identical to it unless linepack is configured, in
                # which case the two legitimately differ while the buffer
                # charges/discharges.
                - self.power_in.where(p.area == p.transport_area_from).sum(
                    "link"
                )
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
            "net_generation_def",
            mask=condition_mask(
                p,
                np.equal(p.transport_resource, p.resource).any(
                    [dim for dim in ["link"] if dim in p.transport_resource.dims]
                ),
            ),
        )

    def costs_def(self):
        """Operational costs: fixed cost on the capacity, hurdle cost on the energy."""
        p = self.p
        self.add(
            -self.costs
            # Fixed costs and hurdle costs, split half and half between the two
            # ends of each link
            + 0.5
            * (
                (
                    p.transport_fixed_cost * self.capacity
                    + p.transport_hurdle_costs.fillna(0)
                    * self.norm
                    * (self.D * self.power).sum(self.T)
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
            "costs_def",
            mask=condition_mask(p),
        )

def add_transport(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_adequacy_constraint: Constraint,
    tb: TimeBlocks | None = None,
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
        tb (pommes.utils.TimeBlocks, optional):
            Block lengths and maps of the dataset, computed once by
            ``build_model`` (``time_blocks(model_parameters)`` when None).
            Without block inputs it is inactive and the statements are
            today's.

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
          from (e.g. the bunker fuel of a ship).
        - ``transport_fuel_emission_factor`` and
          ``transport_fuel_total_emission_factor`` (link, transport_tech,
          year_op): tCO2 per MWh of fuel burnt (default 0, NaN reads as 0),
          the first for the emissions that the carbon tax sees, the second
          for those of the carbon goals. The emissions of the fuel burnt
          (factor x power injected x consumption x time step) are counted by
          the carbon module in the arrival area of the link
          (``transport_area_to``), whatever ``transport_fuel_area`` is, at the
          step the fuel is burnt (the injection step, also with a delay). If
          the supplier of the fuel already carries its emissions (an emission
          factor on the conversion that produces it, or on its import), leave
          these factors at 0, or they are counted twice. When
          ``transport_fuel_resource`` is not a resource of the model, nothing
          is drawn from a balance, and the two factors still apply to the
          consumption: a study can count the emissions of a fuel it does not
          model.
        - ``transport_delay`` (link, transport_tech): hours between injection
          at the "from" end and delivery at the "to" end (loading, voyage,
          unloading): what arrives at hour h left at hour h - delay, cyclic
          over the year like storage levels. Needs a uniform time step and a
          contiguous series of hours (not representative periods); not
          applied on links with linepack. The goods in transit are the
          difference between what has left and what has arrived. An empty
          value (NaN) reads as 0.

        These additions ensure that transport infrastructure operates within
        technical and economic limits, allowing the model to simulate energy
        transport flows realistically.
    """
    tb = time_blocks(model_parameters) if tb is None else tb
    m = model
    p = model_parameters
    # R11, R12 on the full dataset, before the uniform-step error below
    check_block_transport(p, tb)

    # The delay between injection and delivery is given in hours (default 0);
    # the flow is rolled by whole time steps, so it is converted to steps,
    # which needs one step length.
    step = np.unique(p.time_step_duration.values)
    # An empty delay is no delay: a NaN would match no delay value below and leave the two ends of the link untied.
    delay = p.transport_delay.fillna(0)
    if (delay > 0).any() and step.size != 1:
        raise ValueError("transport_delay needs a uniform time_step_duration")
    delay_steps = np.round(delay / step[0]).astype("int64")

    # A (link, tech) pair exists where it carries a resource of the model: a
    # study with one tech per resource gives each tech a resource on its own
    # links only, and on the others (no resource) nothing can flow, so none of
    # the flows, capacities and rows of the module is created there.
    pair_exists = transport_pair_exists(p)

    # Today's statements on p itself unless some transport tech has a block
    # length: then the hourly techs and each block length form a group.
    groups = transport_groups(p, tb)
    blocked = groups[0][1] is not p
    hourly = [g for g in groups if g[0] is None]
    blocks = [g for g in groups if g[0] is not None]

    operation_year_normalization = (
        p.operation_year_duration / p.time_step_duration.sum("hour")
    )

    def restrict(x, p_g):
        """``x`` on the technologies of the group (H1), only when the module is split."""
        if blocked and "transport_tech" in x.dims:
            return x.sel(transport_tech=p_g.indexes["transport_tech"])
        return x

    def rows_of(L, p_g, variables):
        """The row object of a group, its partners restricted to its technologies (H1)."""
        return _TransportRows(
            m,
            p_g,
            tb,
            L,
            variables,
            restrict(operation_transport_power_capacity, p_g),
            restrict(operation_transport_costs, p_g),
            operation_year_normalization,
            restrict(pair_exists, p_g),
            restrict(delay_steps, p_g),
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

    # The hourly group's flows, power injected, linepack level and net
    # generation, in today's order
    if hourly:
        _, p_h, _ = hourly[0]
        hourly_variables = _group_variables(
            m, p_h, tb, None, restrict(pair_exists, p_h)
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
        coords=[p.area, p.transport_tech, p.year_op, *condition_coords(p)],
        mask=condition_mask(p),
    )

    planning_transport_costs = m.add_variables(
        name="planning_transport_costs",
        lower=0,
        coords=[p.area, p.transport_tech, p.year_op],
    )

    if hourly:
        hourly_rows = rows_of(None, p_h, hourly_variables)

    # ------------------
    # Objective function
    # ------------------

    # Hurdle costs (EUR per MWh carried) are counted in operation_transport_costs below, weighted like the
    # variable costs of the other modules (time step, year normalisation, and discounting through
    # annualised_totex), so that they do not depend on the number of modelled hours or years.
    annualised_totex_def.lhs += expected_over_conditions(
        operation_transport_costs.sum("transport_tech"), p
    ) + planning_transport_costs.sum("transport_tech")

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    if hourly:
        add_to_balances(
            operation_adequacy_constraint,
            hourly_variables[3].sum("transport_tech"),
            tb,
        )

        # Fuel burnt by the link
        hourly_rows.fuel(operation_adequacy_constraint)

        # Operation - transport (power bounds, delays, linepack, ramps)
        hourly_rows.limits()

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

    if hourly:
        hourly_rows.net_generation_def()

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

    if hourly:
        hourly_rows.costs_def()

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

    # Block groups, after the module's last statement: their labels follow
    # every hourly label, and each row is the hourly row on the block axis.

    for L, p_g, _ in blocks:
        _refuse_unbuilt_rows(p_g)
        variables = _group_variables(m, p_g, tb, L, restrict(pair_exists, p_g))
        add_to_balances(
            operation_adequacy_constraint,
            variables[3].sum("transport_tech"),
            tb,
            src=L,
        )
        rows = rows_of(L, p_g, variables)
        rows.fuel(operation_adequacy_constraint)
        rows.limits()
        rows.net_generation_def()
        rows.costs_def()

    return m
