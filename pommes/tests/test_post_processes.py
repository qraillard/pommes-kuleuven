import numpy as np
import pytest
import xarray as xr

from pommes.post_process import reindex_from_tech_to_asset

rng = np.random.default_rng(seed=42)


@pytest.fixture
def sample_da():
    """Return a sample DataArray with a <category>_tech dimension."""
    data = rng.uniform(size=3)
    return xr.DataArray(
        data,
        dims=["foo_tech"],
        coords={
            "foo_tech": ["tech1", "tech2", "tech3"],
        },
        name="test_dataarray",
    )


@pytest.fixture
def sample_ds():
    """Return a sample Dataset with a <category>_tech dimension."""
    data = rng.uniform(size=(3, 2))
    return xr.Dataset(
        {
            "var1": (("foo_tech", "y"), data),
            "var2": (("foo_tech",), data[:, 0]),
        },
        coords={"foo_tech": ["techA", "techB", "techC"], "y": [0, 1]},
    )


def test_dataarray_rename(sample_da):
    category = "foo"
    result = reindex_from_tech_to_asset(sample_da, category)

    # Check dimension name
    assert "asset" in result.dims
    assert f"{category}_tech" not in result.dims

    # Check coordinate
    assert "asset" in result.coords
    assert "category" in result.coords

    # Verify the values in 'category' coordinate
    np.testing.assert_array_equal(
        result.coords["category"].values,
        np.array([category, category, category], dtype=str),
    )


def test_dataset_rename(sample_ds):
    category = "foo"
    result = reindex_from_tech_to_asset(sample_ds, category)

    assert "asset" in result.dims
    assert f"{category}_tech" not in result.dims

    for var in result.data_vars:
        assert "asset" in result[var].dims

    assert "asset" in result.coords
    assert "category" in result.coords

    expected_category_values = np.array(
        [category, category, category], dtype=str
    )
    np.testing.assert_array_equal(
        result.coords["category"].values, expected_category_values
    )


def test_missing_dimension():
    data = xr.DataArray(
        rng.uniform(size=3), dims=["some_other_dim"], name="test_data"
    )
    category = "foo"
    with pytest.raises(ValueError):
        _ = reindex_from_tech_to_asset(data, category)


def test_multiple_dimensions_same_name():
    data = xr.DataArray(
        rng.uniform(size=(3, 4)), dims=["foo_tech", "foo_tech"]
    )
    category = "foo"
    with pytest.raises(ValueError):
        _ = reindex_from_tech_to_asset(data, category)
