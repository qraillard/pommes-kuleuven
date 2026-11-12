"""Shared helpers of the time-blocks tests (identical in both engines)."""

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from pommes.model.build_model import build_model

try:  # sparse engine: solved variables sit on a unit index
    from pommes.model.sparsity import densify_any
except ImportError:  # dense engine

    def densify_any(obj):
        return obj


_PARAMETERS = (
    ("conversion_time_block_hrs", "conversion_tech"),
    ("combined_time_block_hrs", "combined_tech"),
    ("resource_time_block_hrs", "resource"),
)


def _per(p, dim, values):
    da = xr.DataArray(
        np.full(p[dim].size, np.nan), dims=dim, coords={dim: p[dim].values}
    )
    for label, value in (values or {}).items():
        da.loc[{dim: label}] = value
    return da


def with_blocks(
    p: xr.Dataset,
    conversion: dict[str, float] | None = None,
    combined: dict[str, float] | None = None,
    resource: dict[str, float] | None = None,
    rule: dict[str, str] | None = None,
) -> xr.Dataset:
    """
    ``p`` with the four block parameters set (NaN / ``"min"`` elsewhere).

    A parameter whose dimension is not in ``p`` is left out.
    """
    p = p.copy()
    given = (conversion, combined, resource)
    for (name, dim), values in zip(_PARAMETERS, given):
        if dim in p.coords:
            p[name] = _per(p, dim, values)
    if "conversion_tech" in p.coords:
        labels = p["conversion_tech"].values
        r = np.array(["min"] * len(labels), dtype=object)
        for label, value in (rule or {}).items():
            r[list(labels).index(label)] = value
        p["conversion_time_block_rule"] = xr.DataArray(
            r.astype(str), dims="conversion_tech", coords={"conversion_tech": labels}
        )
    return p


def lp_text(m) -> str:
    """The LP file of ``m``, read back as text."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.lp"
        m.to_file(path, io_api="lp")
        return path.read_text()


def assert_t0(p: xr.Dataset) -> None:
    """
    Without block inputs the model has no trace of them (T0 plumbing pin).

    Builds ``p`` without the four parameters and with them all NaN: same LP
    text, no ``_block_`` name, no ``time_block_`` dimension, hourly adequacy
    over the full resource coordinate. Both models take the inactive path:
    this guards the NaN parameters and the ``tb`` plumbing, not drift against
    0.6.1+kul (``compare_lp.py`` does).
    """
    names = [n for n, _ in _PARAMETERS] + ["conversion_time_block_rule"]
    bare = build_model(p.drop_vars(names, errors="ignore"))
    nan = build_model(with_blocks(p))
    assert lp_text(bare) == lp_text(nan)
    for name in [*nan.variables, *nan.constraints]:
        assert "_block_" not in name, name
    for name in nan.variables:
        assert not any(d.startswith("time_block_") for d in nan.variables[name].dims)
    for name in nan.constraints:
        assert not any(d.startswith("time_block_") for d in nan.constraints[name].dims)
    adequacy = nan.constraints["operation_adequacy_constraint"]
    assert adequacy.labels.indexes["resource"].equals(p.indexes["resource"])


def solved(m, name: str) -> xr.DataArray:
    """
    Solution of variable ``name`` on ``[area, tech, ...]`` (engine-agnostic).

    Sparse: densified; the technologies of another group (all NaN) dropped.
    """
    da = densify_any(m.solution[name])
    techs = [d for d in da.dims if d.endswith("_tech") and d != "link"]
    for tech in techs:
        da = da.dropna(tech, how="all")
    lead = [d for d in ("area", *techs) if d in da.dims]
    return da.transpose(*lead, ...)
