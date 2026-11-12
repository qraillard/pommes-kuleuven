"""Module to write in model flexdem related constraints."""

import numpy as np
from linopy import Constraint, Model
from xarray import Dataset

from pommes.utils import (
    OPERATING_CONDITION_WEIGHT,
    TimeBlocks,
    add_to_balances,
    block_dim,
    block_prefix,
    check_axis,
    check_block_flexdem,
    check_group,
    condition_coords,
    condition_mask,
    select_resources,
    time_blocks,
)

# Parameters read by the statements below (the dataset a coarse group is selected on)
_READ = (
    "time_step_duration",
    "flexdem_demand",
    "flexdem_conservation_hrs",
    "flexdem_maxload_ratio",
    "flexdem_minload_ratio",
    "flexdem_ramp_up",
    "flexdem_ramp_down",
    "flexdem_window_offset_hrs",
    "flexdem_shifted_energy_initial",
    OPERATING_CONDITION_WEIGHT,
)


def add_flexdem(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_adequacy_constraint: Constraint,
    tb: TimeBlocks | None = None,
) -> Model:
    """
    Add the flexible demand to the model, per balance resolution.

    The resources with an hourly balance get today's variable and rows
    (``operation_flexdem_*``); the resources of a coarse balance of length
    ``L`` get the same rows on the blocks of the balance
    (``operation_flexdem_block_{L}h_*``, section 6.4). Without a coarse
    balance the group is ``model_parameters`` itself.
    """
    tb = time_blocks(model_parameters) if tb is None else tb
    p = model_parameters
    check_block_flexdem(p, tb)  # R9, on the full dataset, before any sel
    if tb.balance_lengths:
        groups = {
            L: select_resources(p, _READ, tb.resources(L))
            for L in [None, *tb.balance_lengths]
        }
    else:
        groups = {None: p}
    for L, p_g in groups.items():
        _add_group(model, p_g, tb, L, operation_adequacy_constraint)
    return model


def _add_group(
    m: Model,
    p: Dataset,
    tb: TimeBlocks,
    L: int | None,
    operation_adequacy_constraint: Constraint,
) -> None:
    """
    Flexible demand of the resources of one balance (``L`` None: hourly).

    ``T`` is ``hour`` or ``time_block_{L}h``; the hourly statements are
    today's. On blocks, ``FD`` is the demand energy of each block and the ramp
    rows are written on mean powers ``F / D``, the shift applied after the
    division (each block over its own duration).
    """
    prefix = block_prefix("flexdem", L)
    T = "hour" if L is None else block_dim(L)

    flexdem_demand = p.flexdem_demand
    for dim in ["area","resource","year_op","hour"]:
        if dim not in flexdem_demand.dims:
            flexdem_demand = flexdem_demand.expand_dims(dim={dim: p[dim]})

    conservation_hrs = p.flexdem_conservation_hrs
    for dim in ["area","year_op","resource"]:
        if dim not in conservation_hrs.dims:
            conservation_hrs = conservation_hrs.expand_dims(dim={dim: p[dim]})

    active = (flexdem_demand != 0).any("hour")
    if L is None:
        time, demand, ratio_demand = p.hour, flexdem_demand, p.flexdem_demand
    else:  # demand energy of each block
        group = tb.block_of_hour[L]
        check_group(group, None)
        check_axis(flexdem_demand, "hour", group.indexes["hour"])
        demand = flexdem_demand.groupby(group).sum()
        time, ratio_demand = tb.duration[L][T], demand

    # ------------
    # Variables
    # ------------

    operation_flexdem_demand = m.add_variables(
        name=f"{prefix}demand",
        lower=0,
        coords=[p.area, p.resource, time, p.year_op, *condition_coords(p)],
        mask=condition_mask(p, active),
    )

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    add_to_balances(
        operation_adequacy_constraint, -operation_flexdem_demand, tb, src=L
    )

    # Ratio constraints

    m.add_constraints(operation_flexdem_demand - ratio_demand*p.flexdem_maxload_ratio <=0,
                      name=f"{prefix}demand_max",
                      mask=condition_mask(p, np.isfinite(p.flexdem_maxload_ratio)))

    m.add_constraints(operation_flexdem_demand - ratio_demand * p.flexdem_minload_ratio >= 0,
                      name=f"{prefix}demand_min",
                      mask=condition_mask(p, p.flexdem_minload_ratio > 0))

    # Conservation constraints. Like demand, flexdem_demand and operation_flexdem_demand are
    # energies per time step, or per block (the adequacy balance adds them to net generations,
    # which are time_step_duration x power), so they are summed as they are.
    shifted_energy = operation_flexdem_demand - demand
    # Optional rolling-horizon inputs (absent: windows start at the first step, nothing carried in).
    offset = p.flexdem_window_offset_hrs if "flexdem_window_offset_hrs" in p else 0.0
    carry = p.flexdem_shifted_energy_initial if "flexdem_shifted_energy_initial" in p else None
    if carry is not None and not bool((carry != 0).any()):
        carry = None  # nothing carried in: build exactly the rows of a stand-alone horizon

    # Whole horizon, counting the energy carried in: with windows it is their sum, so in a rolling
    # horizon it does not force conservation over every block (without a window it still does).
    horizon_energy = shifted_energy.sum(T)
    m.add_constraints((horizon_energy if carry is None else horizon_energy + carry) == 0,
                      name=f"{prefix}demand_conservation",
                      mask=condition_mask(p, active),
    )

    # Windows: the same within each window of flexdem_conservation_hrs hours. A time step belongs
    # to window floor(start time / window length), its start time being flexdem_window_offset_hrs
    # plus the cumulative duration of the steps before it, so steps longer than 1 h are handled and
    # the windows stay on the calendar when the horizon is a block of a longer period. The windows
    # cut by the horizon start and end close at them; the first one gets the shifted energy already
    # committed before the horizon (flexdem_shifted_energy_initial). One constraint call per
    # distinct window length. On blocks the start time is the block's (R9: a window is a union of
    # blocks), with the tolerance of the block anchoring.
    if L is None:
        duration = p.time_step_duration
        start_time = offset + duration.cumsum("hour") - duration
    else:
        start_time = offset + tb.start[L] + 1e-9
    for window_hrs in np.unique(conservation_hrs.values):
        if not window_hrs > 0:
            continue
        window = (start_time // window_hrs).astype("int64").rename("window")
        if L is not None:
            check_group(window, T)
            check_axis(shifted_energy, T, tb.start[L].indexes[T])
        lhs = shifted_energy.groupby(window).sum()
        if carry is not None:
            lhs = lhs + carry * (lhs.coords["window"] == window.isel({T: 0}, drop=True))
        m.add_constraints(
            lhs == 0,
            name=f"{prefix}demand_conservation_{int(window_hrs)}h",
            mask=condition_mask(p, active & (conservation_hrs == window_hrs)),
        )



    # Ramp constraints
    if L is None:
        ramp = operation_flexdem_demand
        first = p.hour != p.hour[0]
        up = (ramp - ramp.shift(hour=1)) / p.time_step_duration
        down = (ramp.shift(hour=1) - ramp) / p.time_step_duration
    else:  # mean powers: the shift comes after the division
        mean = operation_flexdem_demand / tb.duration[L]
        first = time != time[0]
        up = mean - mean.shift({T: 1})
        down = mean.shift({T: 1}) - mean

    m.add_constraints(
        up - p.flexdem_ramp_up * demand
        <= 0,
        name=f"{prefix}ramp_up_constraint",
        mask=condition_mask(p, np.isfinite(p.flexdem_ramp_up) * first),
    )

    m.add_constraints(
        down
        -p.flexdem_ramp_down * demand
        <= 0,
        name=f"{prefix}ramp_down_constraint",
        mask=condition_mask(p, np.isfinite(p.flexdem_ramp_down) * first),
    )



    # TODO add eventually some costs
