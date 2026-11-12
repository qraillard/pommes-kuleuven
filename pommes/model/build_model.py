"""Module with the core function to build the linopy Model."""

import numpy as np
from linopy import Model
from xarray import Dataset

from pommes.model import (
    add_carbon,
    add_combined,
    add_conversion,
    add_net_import,
    add_retrofit_combined,
    add_retrofit_conversion,
    add_retrofit_storage,
    add_retrofit_transport,
    add_storage,
    add_transport,
    add_turpe,
    add_process,
    add_flexdem
)
from pommes.utils import (
    OPERATING_CONDITION,
    OPERATING_CONDITION_WEIGHT,
    block_dim,
    check_block_dims,
    condition_coords,
    condition_mask,
    expected_over_conditions,
    time_blocks,
)

# Parameters read by the shedding, spillage and adequacy statements below
_READ = (
    "time_step_duration",
    "demand",
    "load_shedding_max_capacity",
    "spillage_max_capacity",
    "load_shedding_cost",
    "spillage_cost",
    OPERATING_CONDITION_WEIGHT,
)


def _select_resources(p, resources):
    """``p`` restricted to ``resources``, on the names read here only."""
    q = p[[n for n in _READ if n in p]]
    dims = ("area", "hour", "resource", "year_op", OPERATING_CONDITION)
    q = q.assign_coords({d: p[d] for d in dims if d in p.coords})
    return q.sel(resource=resources)


def build_model(model_parameters: Dataset) -> Model:
    """
    Build a Linopy Model based on the provided model parameters.

    Args:
        model_parameters (xarray.Dataset):
            The dataset containing parameters for the energy system model.
            It includes details such as areas, resources, and temporal indices.

    Returns:
        linopy.Model:
            A Linopy Model object representing the energy system.

    Note:
        The xarray Dataset `model_parameters` must have the following core
        indexes:

        - **Area (`area`)**: str
          Represents the different geographical areas covered by the model.
        - **Hour (`hour`)**: int64
          Time index representing different hours for which the model is
          evaluated.
        - **Resource (`resource`)**: str
          Different types of resources, such as renewable or non-renewable
          energy.
        - **Year of Decision (`year_dec`)**: int64
          The year in which the decision is made.
        - **Year of Investment (`year_inv`)**: int64
          The year when an investment takes place.
        - **Year of Operation (`year_op`)**: int64
          The year in which the operation occurs.

        Depending on the present assets, the following additional indexes may
        be required for specific modules:

        - **Conversion Module**:
            - **Conversion Technology (`conversion_tech`)**: str
              Specifies the type of conversion technology being used, such as
              a gas turbine or wind turbine.

        - **Combined Module**:
            - **Combined Technology (`combined_tech`)**: str
              Specifies the combined technology being used.
            - **Mode (`mode`)**: str
              Represents different operating modes of the combined technology.

        - **Storage Module**:
            - **Storage Technology (`storage_tech`)**: str
              Represents the type of storage technology, such as batteries or
              pumped hydro storage.

        - **Transport Module**:
            - **Link (`link`)**: str
              Represents the connections between areas.
            - **Transport Technology (`transport_tech`)**: str
              Specifies the type of transport technology, such as power lines
              or gas pipelines.

        - **Turpe Module**:
            - **Hour Type (`hour_type`)**: str
              Represents different types of hours, such as peak or off-peak.

        Data variables are described in the `pommes/dataset_description.yaml`
        file.

        The `check_inputs` function from the `pommes.io.build_input_dataset`
        module ensures data consistency.
    """
    p = model_parameters
    tb = time_blocks(p)
    if "carbon" in p.keys() and p.carbon and any(
        np.isfinite(a.values).any() for a in tb.tech_length.values()
    ):
        # interim (Task 9): the carbon rows read the hourly power only
        raise ValueError(
            "carbon with a block conversion or combined technology is not "
            "supported yet (the carbon support of Task 9 of the multi-"
            "resolution plan is missing): set carbon to False or drop the "
            "block lengths."
        )

    operation_year_normalization = (
        p.operation_year_duration / p.time_step_duration.sum("hour")
    )

    # Hourly resources only when some balance is coarse (C5: else p itself)
    coarse = tb.active and bool(tb.balance_lengths)
    p_h = _select_resources(p, tb.resources(None)) if coarse else p
    p_L = {L: _select_resources(p, tb.resources(L)) for L in tb.balance_lengths}

    m = Model()

    # ------------
    # Variables
    # ------------

    # Operation - load_shedding & spillage (powers [MW]: energy per step = time_step_duration x power;
    # their caps load_shedding_max_capacity / spillage_max_capacity are in MW)

    def shedding_variables(q, coord, name):
        """Shedding and spillage powers of ``q`` on ``coord``, named ``name``."""
        out = {}
        for kind in ("load_shedding", "spillage"):
            cap = q[f"{kind}_max_capacity"]
            out[kind] = (
                m.add_variables(
                    name=f"operation_{kind}{name}_power",
                    lower=0,
                    mask=condition_mask(
                        q, np.logical_or(np.isnan(cap), cap > 0)
                    ),
                    coords=[q.area, coord, q.resource, q.year_op, *condition_coords(q)],
                )
            )
        return out

    hourly_power = shedding_variables(p_h, p_h.hour, "")
    operation_load_shedding_power = hourly_power["load_shedding"]
    operation_spillage_power = hourly_power["spillage"]
    block_power = {
        L: shedding_variables(p_L[L], tb.duration[L][block_dim(L)], f"_block_{L}h")
        for L in tb.balance_lengths
    }

    # Costs - load_shedding & spillage

    operation_load_shedding_costs = m.add_variables(
        name="operation_load_shedding_costs",
        mask=condition_mask(p),
        coords=[p.area, p.year_op, *condition_coords(p)],
    )

    operation_spillage_costs = m.add_variables(
        name="operation_spillage_costs",
        mask=condition_mask(p),
        coords=[p.area, p.year_op, *condition_coords(p)],
    )

    # Annualised costs

    annualised_totex = m.add_variables(
        name="annualised_totex", coords=[p.area, p.year_op]
    )

    # ------------------
    # Objective function
    # ------------------

    m.add_objective(
        (p.discount_factor * annualised_totex.sum("area")).sum("year_op")
    )

    # --------------
    # Constraints
    # --------------

    # Adequacy constraint

    operation_adequacy_constraint = m.add_constraints(
        p_h.time_step_duration
        * (operation_load_shedding_power - operation_spillage_power)
        - p_h.demand
        == 0,
        mask=condition_mask(p_h),
        name="operation_adequacy_constraint",
    )

    # Coarse balances, right after the hourly one (add_to_balances needs them)
    for L, power in block_power.items():
        shed, spill = power["load_shedding"], power["spillage"]
        q, D = p_L[L], tb.duration[L]
        demand = (
            q.demand.broadcast_like(p.hour)
            .fillna(0)  # R8: a NaN demand reads as 0, as in the hourly row
            .groupby(tb.block_of_hour[L])
            .sum(skipna=False)
        )
        m.add_constraints(
            D * (shed - spill) - demand == 0,
            mask=condition_mask(q),
            name=f"operation_adequacy_block_{L}h_constraint",
        )

    # Operation - load_shedding & spillage

    for kind, hourly in (
        ("load_shedding", operation_load_shedding_power),
        ("spillage", operation_spillage_power),
    ):
        for q, var, name in [
            (p_h, hourly, ""),
            *((p_L[L], block_power[L][kind], f"_block_{L}h") for L in block_power),
        ]:
            cap = q[f"{kind}_max_capacity"]
            m.add_constraints(
                var <= cap,
                mask=condition_mask(
                    q, np.isfinite(cap) * (cap > 0)
                ),
                name=f"operation_{kind}{name}_max_constraint",
            )

    # Costs - load_shedding & spillage

    for kind, costs, hourly in (
        ("load_shedding", operation_load_shedding_costs, operation_load_shedding_power),
        ("spillage", operation_spillage_costs, operation_spillage_power),
    ):
        expr = -costs + (
            p_h.time_step_duration * hourly * p_h[f"{kind}_cost"]
        ).sum(["hour", "resource"]) * operation_year_normalization
        for L in block_power:  # parts meet on [area, year_op, *cond]
            part = (
                tb.duration[L] * block_power[L][kind] * p_L[L][f"{kind}_cost"]
            ).sum([block_dim(L), "resource"])
            expr = expr.add(part * operation_year_normalization, join="left")
        m.add_constraints(
            expr == 0, mask=condition_mask(p), name=f"operation_{kind}_costs_def"
        )

    # Annualised costs

    annualised_totex_def = m.add_constraints(
        -annualised_totex
        + expected_over_conditions(operation_load_shedding_costs, p)
        + expected_over_conditions(operation_spillage_costs, p)
        == 0,
        name="annualised_totex_def",
    )

    # -----------------------------------------
    # Other sets of variables and constraints
    # -----------------------------------------

    if "conversion" in p.keys() and p.conversion:
        m = add_conversion(
            m,
            p,
            annualised_totex_def,
            operation_year_normalization,
            operation_adequacy_constraint,
            tb=tb,
        )

    if "retrofit_conversion" in p.keys() and p.retrofit_conversion:
        m = add_retrofit_conversion(m, p, annualised_totex_def)

    if "storage" in p.keys() and p.storage:
        m = add_storage(
            m, p, annualised_totex_def, operation_adequacy_constraint,
            tb=tb,
        )

    if "retrofit_storage" in p.keys() and p.retrofit_storage:
        m = add_retrofit_storage(m, p, annualised_totex_def)

    if "transport" in p.keys() and p.transport:
        m = add_transport(
            m, p, annualised_totex_def, operation_adequacy_constraint,
            tb=tb,
        )

    if "retrofit_transport" in p.keys() and p.retrofit_transport:
        m = add_retrofit_transport(m, p, annualised_totex_def)

    if "combined" in p.keys() and p.combined:
        m = add_combined(
            m,
            p,
            annualised_totex_def,
            operation_year_normalization,
            operation_adequacy_constraint,
            tb=tb,
        )

    if "retrofit_combined" in p.keys() and p.retrofit_combined:
        m = add_retrofit_combined(m, p, annualised_totex_def)

    if "process" in p.keys() and p.process:
        m = add_process(
            m,
            p,
            annualised_totex_def,
            operation_year_normalization,
            operation_adequacy_constraint,
            tb=tb,
        )

    if "flexdem" in p.keys() and p.flexdem:
        m = add_flexdem(
            m, p, annualised_totex_def, operation_adequacy_constraint,
            tb=tb,
        )

    if "net_import" in p.keys() and p.net_import:
        m = add_net_import(
            m,
            p,
            annualised_totex_def,
            operation_year_normalization,
            operation_adequacy_constraint,
            tb=tb,
        )

    if "carbon" in p.keys() and p.carbon:
        m = add_carbon(
            m, p, annualised_totex_def, operation_year_normalization,
            tb=tb,
        )

    if "turpe" in p.keys() and p.turpe:
        m = add_turpe(m, p, annualised_totex_def)

    if tb.active:
        check_block_dims(m)

    return m
