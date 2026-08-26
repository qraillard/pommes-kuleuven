import numpy as np
import xarray as xr

from pommes.utils import sort_coords_and_vars


def test_already_sorted():
    """Test a dataset where coordinates are already in ascending order
    and variables are alphabetically sorted.
    """
    coords = {"x": ("x", [0, 1, 2]), "y": ("y", [10, 11])}
    data_vars = {
        "aaa": (("x", "y"), np.array([[1, 2], [3, 4], [5, 6]])),
        "bbb": (("x", "y"), np.array([[10, 20], [30, 40], [50, 60]])),
    }
    ds_in = xr.Dataset(data_vars=data_vars, coords=coords)

    ds_out = sort_coords_and_vars(ds_in)

    assert ds_out.coords["x"].values.tolist() == [0, 1, 2]
    assert ds_out.coords["y"].values.tolist() == [10, 11]

    assert list(ds_out.data_vars.keys()) == ["aaa", "bbb"]

    xr.testing.assert_identical(ds_in, ds_out)


def test_unordered_coords():
    """Test that numeric coordinates get sorted correctly if they are out of order."""
    coords = {"time": ("time", [3, 1, 2])}
    data_vars = {
        "var1": (("time",), [30, 10, 20]),
    }
    ds_in = xr.Dataset(data_vars=data_vars, coords=coords)

    ds_out = sort_coords_and_vars(ds_in)

    assert ds_out.coords["time"].values.tolist() == [1, 2, 3]
    np.testing.assert_array_equal(ds_out["var1"].values, [10, 20, 30])


def test_unordered_vars():
    """Test that data variables get sorted alphabetically."""
    coords = {"x": ("x", [0, 1])}
    data_vars = {
        "z_var": (("x",), [100, 200]),
        "a_var": (("x",), [1, 2]),
        "m_var": (("x",), [10, 20]),
    }
    ds_in = xr.Dataset(data_vars=data_vars, coords=coords)

    ds_out = sort_coords_and_vars(ds_in)

    expected_vars = ["a_var", "m_var", "z_var"]
    assert list(ds_out.data_vars) == expected_vars

    assert ds_out.coords["x"].values.tolist() == [0, 1]

    np.testing.assert_array_equal(
        ds_out["a_var"].values, ds_in["a_var"].values
    )
    np.testing.assert_array_equal(
        ds_out["m_var"].values, ds_in["m_var"].values
    )
    np.testing.assert_array_equal(
        ds_out["z_var"].values, ds_in["z_var"].values
    )


def test_multiple_dimensions():
    """Test dataset with more than one dimension needing sorting."""
    coords = {
        "time": ("time", [2, 1]),
        "level": ("level", [500, 300, 400]),
    }
    data_vars = {
        "varA": (("time", "level"), [[1, 2, 3], [4, 5, 6]]),
        "varB": (("time", "level"), [[10, 20, 30], [40, 50, 60]]),
    }
    ds_in = xr.Dataset(data_vars=data_vars, coords=coords)

    ds_out = sort_coords_and_vars(ds_in)

    assert ds_out.coords["time"].values.tolist() == [1, 2]
    assert ds_out.coords["level"].values.tolist() == [300, 400, 500]

    assert ds_out["varA"].shape == (2, 3)
    assert ds_out["varB"].shape == (2, 3)

    sorted_back = ds_out.sel(time=[2, 1], level=[500, 300, 400])
    xr.testing.assert_equal(sorted_back, ds_in)


def test_empty_dataset():
    """Test edge case with an empty Dataset (no variables, no coords)."""
    ds_in = xr.Dataset()
    ds_out = sort_coords_and_vars(ds_in)

    assert len(ds_out.coords) == 0
    assert len(ds_out.data_vars) == 0
    xr.testing.assert_identical(ds_in, ds_out)


def test_single_variable_single_coord():
    """Test minimal dataset with one coord and one data var."""
    coords = {"dim": ("dim", [2, 3, 1])}
    data_vars = {"my_var": (("dim",), [20, 30, 10])}
    ds_in = xr.Dataset(data_vars=data_vars, coords=coords)

    ds_out = sort_coords_and_vars(ds_in)

    assert ds_out.coords["dim"].values.tolist() == [1, 2, 3]
    np.testing.assert_array_equal(ds_out["my_var"].values, [10, 20, 30])

    assert list(ds_out.data_vars) == ["my_var"]
