"""Full life annuities in every module that charges annuities.

Conversion, combined, process, storage, transport and the four retrofit blocks each pay, under
``annuity_full_life``, the annuity of a vintage over its full life whatever the decommissioning
year. The existing fixtures (annuity tables by year_dec, flag 0: cheapest annuity paid until the
end of life) describe exactly that cost on a grid where the life span lands on year_dec, so a
dataset converted to year_dec-free annuities must give the same costs and objective as the
fixture itself, and the annuity must still be charged in years where early decommissioning has
removed the capacity.
"""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model

# Fixtures of the module tests, reused here (pytest resolves them by name).
from pommes.tests.test_combined import (  # noqa: F401
    parameters_combined,
    parameters_combined_multi_horizon,
)
from pommes.tests.test_conversion import (  # noqa: F401
    parameters_conversion_multi_horizon,
)
from pommes.tests.test_process import (  # noqa: F401
    parameters_process,
    parameters_process_multi_horizon,
)
from pommes.tests.test_retrofit_conversion import (
    p_retrofit_value,  # noqa: F401
)
from pommes.tests.test_retrofit_other_modules import (  # noqa: F401
    p_retrofit_combined_value,
    p_retrofit_storage_value,
    p_retrofit_transport_value,
)
from pommes.tests.test_storage import (  # noqa: F401
    parameters_storage,
    parameters_storage_multi_horizon,
)
from pommes.tests.test_transport import (  # noqa: F401
    parameters_transport,
    parameters_transport_multi_horizon,
)


def full_life_copy(p):
    """Same dataset with year_dec-free annuities (cheapest over year_dec) and annuity_full_life."""
    q = p.copy(deep=True)
    for name in list(q.data_vars):
        if "annuity_cost" in name and "year_dec" in q[name].dims:
            q[name] = q[name].min("year_dec")
    q["annuity_full_life"] = True
    return q


def solve(p):
    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok"
    return model


def planning_costs(model):
    return {
        name: model.solution[name]
        for name in model.variables
        if name.startswith("planning_") and name.endswith("_costs")
    }


def assert_same_costs(legacy, full):
    assert full.objective.value == pytest.approx(
        legacy.objective.value, rel=1e-9, abs=1e-9
    )
    costs_legacy, costs_full = planning_costs(legacy), planning_costs(full)
    assert costs_legacy.keys() == costs_full.keys() and costs_legacy
    for name in costs_legacy:
        np.testing.assert_allclose(
            costs_full[name].values,
            costs_legacy[name].values,
            rtol=1e-9,
            atol=1e-9,
            err_msg=name,
        )


# module: (fixture, early decommissioning variable, planning cost variable, operating capacity)
MODULES = {
    "conversion": (
        "parameters_conversion_multi_horizon",
        "conversion",
        "operation_conversion_power_capacity",
    ),
    "combined": (
        "parameters_combined_multi_horizon",
        "combined",
        "operation_combined_power_capacity",
    ),
    "process": (
        "parameters_process_multi_horizon",
        "process",
        "operation_process_power_capacity",
    ),
    "storage": (
        "parameters_storage_multi_horizon",
        "storage",
        "operation_storage_power_capacity",
    ),
    "transport": (
        "parameters_transport_multi_horizon",
        "transport",
        "operation_transport_power_capacity",
    ),
}


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("module", list(MODULES))
def test_full_life_gives_the_sunk_cost_of_the_previous_convention_without_perfect_foresight(
    module, early, request
):
    fixture, prefix, _ = MODULES[module]
    p = request.getfixturevalue(fixture).copy(deep=True)
    p[f"{prefix}_early_decommissioning"] = early
    assert not bool(p.annuity_full_life)
    legacy = solve(p)
    full = solve(full_life_copy(p))
    assert_same_costs(legacy, full)


@pytest.mark.parametrize("module", list(MODULES))
def test_full_life_annuity_is_still_charged_after_early_decommissioning(
    module, request
):
    fixture, prefix, capacity = MODULES[module]
    p = request.getfixturevalue(fixture).copy(deep=True)
    p[f"{prefix}_early_decommissioning"] = True
    full = solve(full_life_copy(p))
    costs = planning_costs(full)[f"planning_{prefix}_costs"].values.sum(
        axis=tuple(range(0, 1))
    )
    operating = full.solution[capacity]
    # the capacity is decommissioned in some operation year ...
    assert (
        operating.sum([d for d in operating.dims if d != "year_op"]) < 1e-9
    ).any()
    # ... and the annuity is charged in every operation year of the life (never zero here)
    cost_by_year = planning_costs(full)[f"planning_{prefix}_costs"]
    cost_by_year = cost_by_year.sum(
        [d for d in cost_by_year.dims if d != "year_op"]
    )
    assert (cost_by_year.values > 0).all(), costs


@pytest.mark.parametrize(
    "fixture",
    [
        "p_retrofit_value",
        "p_retrofit_combined_value",
        "p_retrofit_storage_value",
        "p_retrofit_transport_value",
    ],
)
def test_full_life_retrofit_costs_match_the_sunk_cost_of_the_previous_convention(
    fixture, request
):
    p = request.getfixturevalue(fixture).copy(deep=True)
    if fixture == "p_retrofit_value":
        # as test_retrofit_cost_enters_the_objective: demand for hydrogen, direct smr_ccs out of reach
        p["demand"].loc[dict(resource="hydrogen", year_op=2030)] = 10.0
        p["conversion_annuity_cost"] = xr.where(
            p.conversion_tech == "smr_ccs", 5000.0, p.conversion_annuity_cost
        )
    # price the retrofit (the other fixtures pin the transfer by equal bounds, as the cost tests do)
    for name in [
        n
        for n in p.data_vars
        if n.startswith("retrofit_") and "annuity_cost" in n
    ]:
        p[name] = xr.full_like(p[name], 1.0)
    assert not bool(p.annuity_full_life)
    legacy = solve(p)
    full = solve(full_life_copy(p))
    transfers = [
        v
        for v in full.variables
        if v.startswith("planning_retrofit_") and v.endswith("_capacity")
    ]
    assert (
        transfers and sum(float(full.solution[v].sum()) for v in transfers) > 0
    )
    retrofit_costs = [
        v
        for v in full.variables
        if v.startswith("planning_retrofit_") and v.endswith("_costs")
    ]
    assert sum(float(full.solution[v].sum()) for v in retrofit_costs) > 0
    assert_same_costs(legacy, full)
