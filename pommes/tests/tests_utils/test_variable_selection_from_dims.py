"""Module to test select_dataset_variables_from_dims in utils."""

import numpy as np
import pytest
import xarray as xr

from pommes.utils import select_dataset_variables_from_dims

rng = np.random.default_rng(seed=42)


@pytest.fixture
def sample_dataset():
    """A sample xarray Dataset for testing."""
    time = np.arange(5)
    lat = np.linspace(-90, 90, 3)
    lon = np.linspace(0, 360, 4)

    data_var1 = (("time", "lat", "lon"), rng.uniform(size=(5, 3, 4)))
    data_var2 = (("time", "lat"), rng.uniform(size=(5, 3)))
    data_var3 = (("lat", "lon"), rng.uniform(size=(3, 4)))
    data_var4 = ("x", rng.uniform(size=10))

    ds = xr.Dataset(
        {
            "var1": data_var1,
            "var2": data_var2,
            "var3": data_var3,
            "var4": data_var4,
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
            "x": np.arange(10),
        },
    )
    return ds


def test_select_single_dim(sample_dataset):
    ds_subset = select_dataset_variables_from_dims(sample_dataset, ["time"])
    assert "var1" in ds_subset.data_vars
    assert "var2" in ds_subset.data_vars
    assert "var3" not in ds_subset.data_vars
    assert "var4" not in ds_subset.data_vars


def test_select_single_dim_str(sample_dataset):
    ds_subset = select_dataset_variables_from_dims(sample_dataset, "time")
    assert "var1" in ds_subset.data_vars
    assert "var2" in ds_subset.data_vars
    assert "var3" not in ds_subset.data_vars
    assert "var4" not in ds_subset.data_vars


def test_select_multiple_dims(sample_dataset):
    ds_subset = select_dataset_variables_from_dims(
        sample_dataset, ["time", "lat"]
    )
    assert "var1" in ds_subset.data_vars
    assert "var2" in ds_subset.data_vars
    assert "var3" not in ds_subset.data_vars
    assert "var4" not in ds_subset.data_vars


def test_select_no_common_dims(sample_dataset):
    ds_subset = select_dataset_variables_from_dims(
        sample_dataset, ["foo", "bar"]
    )
    assert len(ds_subset.data_vars) == 0


def test_select_no_common_dims_str(sample_dataset):
    ds_subset = select_dataset_variables_from_dims(
        sample_dataset, ["foo", "bar"]
    )
    assert len(ds_subset.data_vars) == 0


def test_select_empty_dims(sample_dataset):
    ds_subset = select_dataset_variables_from_dims(sample_dataset, [])
    assert len(ds_subset.data_vars) == 4
    assert set(ds_subset.data_vars) == {"var1", "var2", "var3", "var4"}


def test_select_set_dims(sample_dataset):
    ds_subset = select_dataset_variables_from_dims(
        sample_dataset, {"lat", "lon"}
    )
    assert "var1" in ds_subset.data_vars
    assert "var3" in ds_subset.data_vars
    assert "var2" not in ds_subset.data_vars
    assert "var4" not in ds_subset.data_vars
