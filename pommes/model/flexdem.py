"""Module to write in model flexdem related constraints."""

import numpy as np
import xarray as xr
from linopy import Constraint, Model
from xarray import Dataset


def add_flexdem(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
    operation_adequacy_constraint: Constraint,
) -> Model:
    m = model
    p = model_parameters

    flexdem_demand = p.flexdem_demand
    for dim in ["area","resource","year_op","hour"]:
        if dim not in flexdem_demand.dims:
            flexdem_demand = flexdem_demand.expand_dims(dim={dim: p[dim]})

    conservation_hrs = p.flexdem_conservation_hrs
    for dim in ["area","year_op","resource"]:
        if dim not in conservation_hrs.dims:
            conservation_hrs = conservation_hrs.expand_dims(dim={dim: p[dim]})


    # ------------
    # Variables
    # ------------

    operation_flexdem_demand = m.add_variables(
        name="operation_flexdem_demand",
        lower=0,
        coords=[p.area, p.resource, p.hour, p.year_op],
        mask=(flexdem_demand != 0).any("hour"),
    )

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    operation_adequacy_constraint.lhs += -operation_flexdem_demand

    # Ratio constraints

    m.add_constraints(operation_flexdem_demand -p.flexdem_demand*p.flexdem_maxload_ratio <=0,
                      name="operation_flexdem_demand_max",
                      mask=np.isfinite(p.flexdem_maxload_ratio))

    m.add_constraints(operation_flexdem_demand - p.flexdem_demand * p.flexdem_minload_ratio >= 0,
                      name="operation_flexdem_demand_min",
                      mask=p.flexdem_minload_ratio > 0)

    # Conservation constraints. Like demand, flexdem_demand and operation_flexdem_demand are
    # energies per time step (the adequacy balance adds them to net generations, which are
    # time_step_duration x power), so they are summed as they are.
    duration = p.time_step_duration
    shifted_energy = operation_flexdem_demand - flexdem_demand
    active = (flexdem_demand != 0).any("hour")
    # Optional rolling-horizon inputs (absent: windows start at the first step, nothing carried in).
    offset = p.flexdem_window_offset_hrs if "flexdem_window_offset_hrs" in p else 0.0
    carry = p.flexdem_shifted_energy_initial if "flexdem_shifted_energy_initial" in p else None
    if carry is not None and not bool((carry != 0).any()):
        carry = None  # nothing carried in: build exactly the rows of a stand-alone horizon

    # Whole horizon, counting the energy carried in: with windows it is their sum, so in a rolling
    # horizon it does not force conservation over every block (without a window it still does).
    horizon_energy = shifted_energy.sum("hour")
    m.add_constraints((horizon_energy if carry is None else horizon_energy + carry) == 0,
                      name="operation_flexdem_demand_conservation",
                      mask=active,
    )

    # Windows: the same within each window of flexdem_conservation_hrs hours. A time step belongs
    # to window floor(start time / window length), its start time being flexdem_window_offset_hrs
    # plus the cumulative duration of the steps before it, so steps longer than 1 h are handled and
    # the windows stay on the calendar when the horizon is a block of a longer period. The windows
    # cut by the horizon start and end close at them; the first one gets the shifted energy already
    # committed before the horizon (flexdem_shifted_energy_initial). One constraint call per
    # distinct window length.
    start_time = offset + duration.cumsum("hour") - duration
    for window_hrs in np.unique(conservation_hrs.values):
        if not window_hrs > 0:
            continue
        window = (start_time // window_hrs).astype("int64").rename("window")
        lhs = shifted_energy.groupby(window).sum()
        if carry is not None:
            lhs = lhs + carry * (lhs.coords["window"] == window.isel(hour=0, drop=True))
        m.add_constraints(
            lhs == 0,
            name=f"operation_flexdem_demand_conservation_{int(window_hrs)}h",
            mask=active & (conservation_hrs == window_hrs),
        )



    # Ramp constraints
    m.add_constraints(
        (operation_flexdem_demand
        - operation_flexdem_demand.shift(hour=1)) / p.time_step_duration
        - p.flexdem_ramp_up * flexdem_demand
        <= 0,
        name="operation_flexdem_ramp_up_constraint",
        mask=np.isfinite(p.flexdem_ramp_up) * (p.hour != p.hour[0]),
    )

    m.add_constraints(
        (operation_flexdem_demand.shift(hour=1)
        - operation_flexdem_demand) / p.time_step_duration
        -p.flexdem_ramp_down * flexdem_demand
        <= 0,
        name="operation_flexdem_ramp_down_constraint",
        mask=np.isfinite(p.flexdem_ramp_down) * (p.hour != p.hour[0]),
    )



    # TODO add eventually some costs

    return m