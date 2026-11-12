"""T0: without block inputs the model has no trace of the time blocks."""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.utils import time_blocks
from pommes.tests.dataset_nnodes import build_nnode_parameters
from pommes.tests.test_flexdem import _with_defaults, _with_flexdem
from pommes.tests.time_blocks_helper import (
    BLOCK_PARAMETERS,
    assert_t0,
    coarse_step_dataset,
    combined_toy,
    conversion_t2_dataset,
    conversion_toy,
    hand_toy,
    lp_text,
    process_hourly,
    with_blocks,
    with_values,
)


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
        p = process_hourly(
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


def test_t0_combined_with_block_tech_and_nan_lengths():
    """C11: a second combined tech, finite ramps and a constant-operation tech, every length NaN."""
    p = combined_toy([1.0] * 10)
    assert p.combined_tech.size == 3 and float(p.combined_ramp_up.min()) == 0.0
    assert_t0(p)


def test_t0_conversion_with_every_row():
    """C11: availability, must-run 1 and 0.5, ramps, constant operation, yearly, factor, daily and cumulative caps; every length NaN."""
    p = conversion_t2_dataset().drop_vars(BLOCK_PARAMETERS)
    p = with_values(p, "conversion_max_daily_production", {"kiln_b": 1e6})
    p = with_values(p, "conversion_max_cumulative_production", {"kiln_c": 1e9})
    p = p.assign(operation_year_weight=("year_op", [10.0]))
    assert_t0(p)


def test_t0_daily_cap_on_20_minute_steps():
    """R12: the hourly daily row is 0.6.1+kul's, a tolerance-free ``start // day``, even on non-dyadic steps.

    ``assert_t0`` compares the branch with itself, so the day map is also checked against the plain floor:
    step 72 of 144 steps of 1/3 h starts an ulp below hour 24 and, with a 1e-9 tolerance, would move to day 1.
    """
    p = conversion_toy([1 / 3] * 144)
    p = with_values(p, "conversion_max_daily_production", {"kiln_b": 1e6})
    assert_t0(p)
    row = build_model(p).constraints["operation_conversion_daily_production_max_constraint"]
    step = p.time_step_duration
    start = step.cumsum("hour") - step
    expected = (start // p.operation_day_duration).astype("int64")
    assert expected.values[72] == 0  # the ulp case stays in day 0
    terms = (row.lhs.vars != -1).sum("_term").sel(day=0)
    assert terms.max().item() == int((expected == 0).sum())


@pytest.mark.filterwarnings("ignore:Time blocks give no gain")
def test_t0_flexdem_group_is_the_input(monkeypatch):
    """C5: without a coarse resource flexdem runs on ``p`` itself: no ``sel``, every resource of ``p``."""
    import pommes.model.flexdem as flexdem

    def refuse(*args, **kwargs):
        raise AssertionError("flexdem selected resources without a coarse balance")

    monkeypatch.setattr(flexdem, "select_resources", refuse)
    p = _with_defaults(_with_flexdem(build_nnode_parameters(n_areas=2, n_hours=48)))
    # block technologies that leave every balance hourly (the heater and the well stay hourly): tb is active, no balance is coarse
    hourly = with_blocks(coarse_step_dataset(4, 24).drop_vars(BLOCK_PARAMETERS), combined={"boiler_block": 4})
    assert time_blocks(hourly).active and not time_blocks(hourly).balance_lengths
    for dataset in (p, hourly):
        m = build_model(dataset)
        var = m.variables["operation_flexdem_demand"]
        assert list(var.labels.indexes["resource"]) == list(dataset.indexes["resource"])
        assert not any("flexdem_block" in n for n in [*m.variables, *m.constraints])


def test_t0_flexdem_with_nan_lengths():
    """C11: the flexdem dataset of ``test_flexdem.py`` with the four parameters present and all NaN."""
    p = with_blocks(_with_defaults(_with_flexdem(build_nnode_parameters(n_areas=2, n_hours=48))))
    assert all(name in p for name in BLOCK_PARAMETERS)
    assert all(bool(p[name].isnull().all()) for name in BLOCK_PARAMETERS if name != "conversion_time_block_rule")
    assert set(p.conversion_time_block_rule.values.tolist()) == {"min"}
    assert_t0(p)
    m = build_model(p)
    assert "operation_flexdem_demand_conservation_24h" in m.constraints
    assert all("_block_" not in n for n in [*m.variables, *m.constraints])


def test_t0_carbon_with_nan_lengths(parameters):
    """C11: carbon on with an emission factor on a conversion and a combined tech, the four parameters present and all NaN."""
    assert bool(parameters.carbon) and float(parameters.conversion_emission_factor.max()) > 0
    assert float(parameters.combined_emission_factor.max()) > 0
    p = with_blocks(parameters)
    assert all(name in p for name in BLOCK_PARAMETERS)
    assert all(bool(p[name].isnull().all()) for name in BLOCK_PARAMETERS if name != "conversion_time_block_rule")
    assert set(p.conversion_time_block_rule.values.tolist()) == {"min"}
    assert_t0(p)
    m = build_model(p)
    assert "operation_carbon_emissions_def" in m.constraints
    assert all("_block_" not in n for n in [*m.variables, *m.constraints])
