"""A study does not have to list ``year_dec``: ``build_input_parameters`` derives it from the life spans.

``year_dec`` is the grid of years at which a vintage can leave. An end of life that is not on the grid used to be
rounded down to the previous grid year (``compute_end_of_life``), so a study had to choose a grid that holds every
``year_inv + life_span``. Without ``year_dec`` in the config coords, ``derive_year_dec`` builds the grid: the ends of
life, ``year_inv``, ``year_op`` and ``year_ret``. A ``year_dec`` that is listed behaves exactly as before.

What is checked, with the ``test_case`` example (all life spans 20 years, early decommissioning off):

- the grid on a small dataset written by hand, for each source of years and each life span that is skipped;
- a derived grid and a listed grid that holds the ends of life give the same optimum (both annuity modes);
- an end of life off the listed grid is rounded down there and exact on the derived grid, where it gives a capacity
  window at least as long, so an optimum that is at most as high;
- a parameter indexed by ``year_dec`` cannot be read when ``year_dec`` is not listed;
- a retrofit study (the ``parameters_retrofit_conversion`` fixture) builds on the derived grid, which holds ``year_ret``;
- the dataset of a derived study passes ``check_inputs``.
"""
import copy

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pommes import test_case_path
from pommes.io.build_input_dataset import (
    build_input_parameters,
    derive_year_dec,
    load_inputs_as_dict_of_df,
    read_config_file,
)
from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs


def _config(annuity_mode: str, drop_year_dec: bool) -> dict:
    config = read_config_file(file_path=f"{test_case_path}/config.yaml")
    config["pre_process"]["annuity_mode"] = annuity_mode
    if drop_year_dec:
        del config["coords"]["year_dec"]
    return config


def _optimum(p: xr.Dataset) -> float:
    model = build_model(check_inputs(p))
    model.solve(solver_name="highs")
    return float(model.objective.value)


def _set_life_spans(config: dict, dict_df: dict, life_span: float) -> None:
    for name, param in config["input"]["parameters"].items():
        if name.endswith("_life_span"):
            dict_df[param["file"]][param["column"]] = life_span


def test_the_grid_of_a_small_dataset():
    """Ends of life (rounded down), study years, and no year for a life span that is NaN or not positive."""
    ds = xr.Dataset(
        coords={
            "year_inv": np.array([2020, 2030]),
            "year_op": np.array([2033]),
            "year_ret": np.array([2025]),
        }
    )
    ds["conversion_life_span"] = ("conversion_tech", [25.0, np.nan, 0.0, 12.5, -7.0])
    ds["retrofit_conversion_life_span"] = ("retrofit_tech", [10.0])
    grid = derive_year_dec(ds)
    # year_inv 2020, 2030; year_op 2033; year_ret 2025; the ends: 2020/2030 + 25 = 2045/2055, + 12.5 = 2032.5/2042.5
    # -> 2032/2042 (rounded down); retrofit from year_ret: 2025 + 10 = 2035. Each source holds a year of its own, and the
    # life spans that are 0 or -7 (2020/2030 and 2013/2023 if they were kept) and NaN give none.
    assert grid.tolist() == [2020, 2025, 2030, 2032, 2033, 2035, 2042, 2045, 2055]
    assert grid.dtype == np.int64
    assert set(ds.year_ret.values) <= set(grid)  # a retrofit needs year_ret among the decommissioning years


def test_a_retrofit_life_span_needs_year_ret():
    ds = xr.Dataset(coords={"year_inv": np.array([2020])})
    ds["retrofit_conversion_life_span"] = ("retrofit_tech", [10.0])
    with pytest.raises(ValueError, match="year_ret"):
        derive_year_dec(ds)


@pytest.mark.parametrize("annuity_mode", ["full_life", "until_decommissioning"])
def test_a_derived_grid_gives_the_optimum_of_a_listed_grid(annuity_mode):
    listed = build_input_parameters(_config(annuity_mode, drop_year_dec=False))
    derived = build_input_parameters(_config(annuity_mode, drop_year_dec=True))
    # The listed grid (2020 ... 2070) holds the ends of life 2030 ... 2070; the derived one adds the first year_inv, 2010.
    assert set(listed.year_dec.values) < set(derived.year_dec.values)
    assert derived.year_dec.dtype == np.int64
    assert derived.year_dec.values.tolist() == sorted(derived.year_dec.values.tolist())
    for name in listed.data_vars:
        if name.endswith("_end_of_life"):
            xr.testing.assert_equal(listed[name], derived[name])
    objective_listed, objective_derived = _optimum(listed), _optimum(derived)
    assert objective_derived == pytest.approx(objective_listed, rel=1e-7)


def test_an_end_of_life_off_the_listed_grid_is_exact_on_the_derived_grid():
    life_span = 15.0  # 2010 + 15 = 2025, 2020 + 15 = 2035 ...: none is on the listed grid 2020, 2030, ...
    results = {}
    for drop_year_dec in (False, True):
        config = _config("full_life", drop_year_dec)
        dict_df = load_inputs_as_dict_of_df(config)
        _set_life_spans(config, dict_df, life_span)
        p = build_input_parameters(config, dict_df)
        results[drop_year_dec] = p
    listed, derived = results[False], results[True]
    exact = derived.year_inv + life_span
    assert (derived.conversion_end_of_life == exact).all()  # the end of life is year_inv + life_span
    assert (listed.conversion_end_of_life <= exact).all()  # rounded down on the listed grid
    assert (listed.conversion_end_of_life < exact).any()
    # A longer capacity window for the same annuity (full_life): the optimum cannot be higher.
    assert _optimum(derived) <= _optimum(listed) * (1 + 1e-9)


def test_a_parameter_indexed_by_year_dec_needs_year_dec():
    config = _config("full_life", drop_year_dec=True)
    dict_df = load_inputs_as_dict_of_df(config)
    index = pd.MultiIndex.from_product([["boiler"], [2020, 2030]], names=["conversion_tech", "year_dec"])
    dict_df["by_year_dec.csv"] = pd.DataFrame({"value": [1.0, 2.0]}, index=index)
    config["input"]["parameters"]["conversion_by_year_dec"] = {
        "file": "by_year_dec.csv",
        "column": "value",
        "type": "float64",
        "fill": 0.0,
        "index_input": ["conversion_tech", "year_dec"],
    }
    with pytest.raises(ValueError, match="year_dec"):
        build_input_parameters(config, dict_df)


def test_a_retrofit_study_builds_on_the_derived_grid(parameters_retrofit_conversion):
    """The retrofit module needs year_ret among year_inv and year_dec (``_check_year_ret_grid``); the derived grid has it.

    The fixture lists year_dec 2030, 2040, 2050 while its ends of life reach 2060 (the vintage of 2040 lives 20 years):
    that vintage has no capacity variable on the listed grid, and has one on the derived grid, which holds 2060.
    """
    listed = parameters_retrofit_conversion.sel(
        conversion_tech=["smr", "smr_ccs"],
        area=["area_1"],
        hour=[0],
        resource=["electricity", "hydrogen", "methane"],
        year_dec=[2030, 2040, 2050],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030],
        year_ret=[2030, 2040],
    ).copy(deep=True)
    grid = derive_year_dec(listed)
    assert {2030, 2040}.issubset(grid)  # year_ret
    assert {2040, 2050, 2060}.issubset(grid)  # the ends of life of the fixture
    derived = listed.reindex(year_dec=grid)

    def capacity_variables(p):
        model = build_model(p)  # raises if year_ret is not a subset of year_inv and year_dec
        return int((model.variables["planning_conversion_power_capacity"].labels.values >= 0).sum())

    assert capacity_variables(derived) > capacity_variables(listed)
