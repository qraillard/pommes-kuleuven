"""T0: without block inputs the model has no trace of the time blocks."""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.utils import time_blocks
from pommes.tests.dataset_nnodes import build_nnode_parameters
from pommes.tests.test_flexdem import _with_defaults, _with_flexdem
from pommes.tests.time_blocks_helper import (
    assert_t0,
    hand_toy,
    lp_text,
    with_blocks,
)


def _process_dataset(parameters_dispatch_invest, process):
    """As ``test_process.py::parameters_process`` builds it."""
    p = xr.merge([parameters_dispatch_invest, process])
    p = p.sel(
        area=["area_1"],
        conversion_tech=["wind_onshore"],
        resource=["electricity", "heat", "methane"],
    ).copy(deep=True)
    p["load_shedding_cost"].loc[dict(resource="methane")] = 5
    p["demand"].loc[dict(resource="electricity")] = np.array([0, 0, 0, 0])
    p["conversion_power_capacity_investment_max"] = 10
    p["conversion_power_capacity_investment_min"] = 10
    p["conversion_annuity_cost"] = 0
    p["conversion_fixed_cost"] = 0
    return p


def _transport_fuel_dataset(parameters):
    """``parameters`` with one transport tech burning methane (the ``-fuel`` site)."""
    p = parameters.copy(deep=True)
    techs = p.transport_tech.values
    p["transport_fuel_consumption"] = xr.DataArray(
        [np.nan, 0.1] + [np.nan] * (len(techs) - 2),
        dims="transport_tech",
        coords={"transport_tech": techs},
    )
    p["transport_fuel_resource"] = xr.DataArray("methane")
    p["transport_fuel_area"] = xr.DataArray("area_1")
    return p


FIXTURES = [
    "parameters",
    "parameters_dispatch_invest",
    "parameters_retrofit_conversion",
    "parameters_retrofit_combined",
    "parameters_retrofit_storage",
    "parameters_retrofit_transport",
]


@pytest.mark.parametrize(
    "case", [*FIXTURES, "process", "flexdem", "transport_fuel"]
)
def test_t0_lp_identical(case, request):
    if case in FIXTURES:
        p = request.getfixturevalue(case)
    elif case == "process":
        p = _process_dataset(
            request.getfixturevalue("parameters_dispatch_invest"),
            request.getfixturevalue("process"),
        )
    elif case == "flexdem":
        p = _with_defaults(
            _with_flexdem(build_nnode_parameters(n_areas=2, n_hours=48))
        )
    else:
        p = _transport_fuel_dataset(request.getfixturevalue("parameters"))
    assert_t0(p)


@pytest.mark.parametrize("module", ["conversion", "combined"])
def test_t0_length_on_a_module_that_is_off(parameters, module):
    off = parameters.assign({module: np.array(False)})
    lengths = {
        t: 24 for t in off[f"{module}_tech"].values
    }
    with_length = with_blocks(off, **{module: lengths})
    assert time_blocks(with_length).active is False
    m = build_model(with_length)
    assert all("_block_" not in n for n in [*m.variables, *m.constraints])
    assert lp_text(m) == lp_text(build_model(off))


def test_t0_balances():
    assert_t0(hand_toy([1.0] * 4, [0, 2, 0, 2]))
