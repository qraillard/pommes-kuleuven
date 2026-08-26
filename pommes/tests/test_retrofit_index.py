"""Unit tests for pommes.model.retrofit._index."""

import numpy as np
import xarray as xr
from linopy import Model

from pommes.model.retrofit._index import gather_pairs_to_tech


def test_gather_pairs_to_tech_places_each_pair_on_its_target():
    tech = xr.DataArray(
        ["alpha", "beta", "gamma"],
        dims="demo_tech",
        coords={"demo_tech": ["alpha", "beta", "gamma"]},
    )
    pair_tech_names = xr.DataArray(
        ["gamma", "alpha"],
        dims="demo_pair",
        coords={"demo_pair": ["p0", "p1"]},
    )

    m = Model()
    v = m.add_variables(
        name="v",
        lower=0,
        coords=[pair_tech_names.demo_pair],
    )

    gathered = gather_pairs_to_tech(
        1 * v, tech, pair_tech_names, "demo_pair"
    )

    assert "demo_tech" in gathered.dims
    assert "demo_pair" not in gathered.dims
    # p1 targets alpha, p0 targets gamma, beta receives nothing.
    coeffs = gathered.coeffs.sel(demo_tech="beta").values
    assert np.nansum(np.abs(coeffs)) == 0
    assert gathered.nterm >= 1
