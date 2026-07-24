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

    # Conservation constraint
    m.add_constraints(operation_flexdem_demand.sum("hour") - flexdem_demand.sum("hour") == 0,
                      name="operation_flexdem_demand_conservation"
    )



    for res in p.resource.values:
        for area in p.area.values:
            for year in p.year_op.values:
                cons_hr = conservation_hrs.sel(year_op=year, resource=res, area=area).values
                if cons_hr > 0:
                    hour_range=[]
                    for hr in list(range(p.hour.values.min(), p.hour.values.max()-cons_hr+1,cons_hr)):
                        if hr+cons_hr<=p.hour.values.max():
                            hour_range.append(list(range(hr,hr+cons_hr)))
                    for hr_range in hour_range:
                        m.add_constraints(
                        operation_flexdem_demand.sel(hour=hr_range,resource=res, area=area, year_op=year).sum(
                            "hour")- flexdem_demand.sel(hour=hr_range,resource=res,area=area,year_op=year).sum("hour")
                            == 0, name=f"operation_flexdem_demand_conservation_{hr_range[0]}_{res}_{area}_{year}")



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