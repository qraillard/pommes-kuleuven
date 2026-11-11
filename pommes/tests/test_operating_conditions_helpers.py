"""Tests of the operating-condition helpers of pommes.utils."""

import numpy as np
import pytest
import xarray as xr
from linopy import Model

from pommes.tests.conditions_helper import with_conditions
from pommes.utils import (
    annual_quantity,
    condition_coords,
    condition_dual_scale,
    condition_mask,
    expected_over_conditions,
    has_operating_conditions,
    operating_condition_weight,
)

YEARS = [2020, 2030]


def base():
    return xr.Dataset(coords={"year_op": YEARS})


def test_helpers_are_neutral_without_conditions():
    p = base()
    x = xr.DataArray([1.0, 2.0], dims="year_op")
    mask = xr.DataArray([True, False], dims="year_op")
    assert not has_operating_conditions(p)
    assert operating_condition_weight(p) is None
    assert condition_coords(p) == []
    assert condition_mask(p) is None
    assert condition_mask(p, mask) is mask
    assert expected_over_conditions(x, p) is x
    assert annual_quantity(x, p, "carbon_goal_expected") is x
    assert float(condition_dual_scale(p)) == 1


def test_weight_without_year_op_is_broadcast():
    p = with_conditions(base(), [0.25, 0.75])
    w = operating_condition_weight(p)
    assert w.dims == ("year_op", "operating_condition")
    assert w.shape == (2, 2)
    assert (w.sel(operating_condition="b") == 0.75).all()


def test_single_condition_without_weight_gets_one():
    p = base().assign_coords(operating_condition=["a"])
    assert (operating_condition_weight(p) == 1).all()
    p["operating_condition_weight"] = xr.DataArray(np.nan)
    assert (operating_condition_weight(p) == 1).all()
    p = base().assign_coords(operating_condition=["a", "b"])
    p["operating_condition_weight"] = xr.DataArray(np.nan)
    with pytest.raises(ValueError):
        operating_condition_weight(p)


def test_string_labels_keep_their_order():
    p = with_conditions(base(), [0.2, 0.3, 0.5], labels=["z", "b", "m"])
    w = operating_condition_weight(p)
    assert list(w.operating_condition.values) == ["z", "b", "m"]
    assert list(condition_coords(p)[0].values) == ["z", "b", "m"]
    assert w.isel(year_op=0).values.tolist() == [0.2, 0.3, 0.5]


def test_expectation_reads_nan_as_zero():
    p = with_conditions(base(), [0.5, np.nan, 0.5])
    x = xr.DataArray(
        [2.0, 4.0, 8.0], coords={"operating_condition": ["a", "b", "c"]}
    )
    assert (expected_over_conditions(x, p) == 5).all()
    mask = condition_mask(p)
    assert mask.sel(operating_condition="b").sum() == 0
    assert mask.sel(operating_condition="a").all()


def test_expectation_of_a_linopy_expression():
    p = with_conditions(base(), [0.25, 0.75])
    m = Model()
    v = m.add_variables(coords=[p.year_op, p.operating_condition], name="v")
    e = expected_over_conditions(v, p)
    assert "operating_condition" not in e.dims
    assert sorted(np.unique(e.coeffs.values)) == [0.25, 0.75]
    assert expected_over_conditions(v * 2, p).coeffs.max() == 1.5


def test_annual_quantity_follows_the_flag():
    p = with_conditions(base(), [0.5, 0.5])
    x = xr.DataArray(
        [2.0, 4.0], coords={"operating_condition": ["a", "b"]}
    )
    p["on"] = True
    p["off"] = False
    assert (annual_quantity(x, p, "on") == 3).all()
    assert annual_quantity(x, p, "off") is x
    assert annual_quantity(x, p, "absent") is x
