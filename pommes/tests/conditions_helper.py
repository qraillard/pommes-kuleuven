"""Test helper building datasets with operating conditions."""

import string

import pandas as pd
import xarray as xr
from xarray import Dataset

from pommes.utils import OPERATING_CONDITION, OPERATING_CONDITION_WEIGHT


def with_conditions(
    p: Dataset, weights, labels=None, **per_condition
) -> Dataset:
    """
    Add the operating_condition coordinate and weights to a dataset.

    weights: a sequence (dims operating_condition) or a DataArray (as given).
    labels: default "a", "b", ...; kept in the given order.
    per_condition: parameter name -> one value (DataArray or scalar) per
        condition; the parameter is replaced by their concatenation.
    """
    n = len(weights) if not isinstance(weights, xr.DataArray) else None
    if labels is None:
        n = n or weights.sizes[OPERATING_CONDITION]
        labels = list(string.ascii_lowercase[:n])
    index = pd.Index(labels, name=OPERATING_CONDITION)
    p = p.assign_coords({OPERATING_CONDITION: index})
    if not isinstance(weights, xr.DataArray):
        weights = xr.DataArray(
            list(weights), coords={OPERATING_CONDITION: index}
        )
    p = p.assign({OPERATING_CONDITION_WEIGHT: weights})
    for name, values in per_condition.items():
        parts = [
            v if isinstance(v, xr.DataArray) else xr.DataArray(v)
            for v in values
        ]
        p = p.assign({name: xr.concat(parts, dim=index)})
    return p
