"""Module with consistency check with reference configuration."""

import logging

import numpy as np
import xarray as xr
from xarray import Dataset

from pommes.model.annuity import infer_annuity_full_life
from pommes.model.data_validation import ref_inputs

logger = logging.getLogger(__name__)


def _complete_year_dec_with_ends_of_life(ds: Dataset) -> Dataset:
    """
    Make every end of life a decommissioning year, or warn where that is unsafe.

    Without early decommissioning, a vintage has a capacity variable only at
    ``year_dec == *_end_of_life``. An end of life that is not in ``year_dec``
    leaves the vintage without any variable: it cannot be built, and an
    investment minimum on it is dropped without a message. A dataset built by
    ``build_input_parameters`` never has one (``compute_end_of_life`` rounds
    down onto ``year_dec``, and a study without ``year_dec`` gets a grid that
    holds the ends of life), but a dataset written by hand can. Entries whose
    life span is NaN or not positive hold no capacity and are not counted.

    With full life annuities (``annuity_full_life``), the annuity of a vintage
    does not depend on ``year_dec``, so the missing ends of life are added to
    ``year_dec`` (one warning lists them). With the previous convention the
    annuity tables are indexed by ``year_dec`` and nothing can say what the
    new rows cost, which the model would read as no cost: the dataset is left
    as it is and one warning per ``*_end_of_life`` variable says so.

    Args:
        ds (xarray.Dataset): Dataset with its ``year_dec`` coordinate and, if
            it has one, ``annuity_full_life``.

    Returns:
        xarray.Dataset: The dataset, with the ends of life added to ``year_dec``
            where that is safe.
    """
    if "year_dec" not in ds.coords:
        return ds
    full_life = bool(ds["annuity_full_life"]) if "annuity_full_life" in ds else False
    missing: set[int] = set()
    found: list[tuple[str, int, int, dict, object]] = []
    for name in ds.data_vars:
        if not str(name).endswith("_end_of_life"):
            continue
        end = ds[name]
        life_span = str(name).replace("end_of_life", "life_span")
        held = end.notnull() if life_span not in ds else ds[life_span] > 0
        off = held & ~end.isin(ds.year_dec)
        count = int(off.sum())
        if not count:
            continue
        first = tuple(np.argwhere(off.values)[0])
        example = {
            dim: (off[dim].values[i].item() if dim in off.coords else i)
            for dim, i in zip(off.dims, first)
        }
        broadcast = end.broadcast_like(off).transpose(*off.dims)
        missing.update(int(year) for year in np.unique(broadcast.where(off).values[off.values]))
        found.append(
            (str(name), count, int(held.broadcast_like(off).sum()), example, broadcast.values[first].item())
        )
    if not found:
        return ds

    if full_life:
        years = sorted(missing)
        logger.warning(
            f"year_dec completed with {years}: the end of life of "
            + ", ".join(f"{name} ({count} entries)" for name, count, *_ in found)
            + " was not a decommissioning year, so such a vintage had no "
            "capacity variable (it could not be built, and an investment "
            "minimum on it was ignored). Add these years to year_dec, or "
            "leave year_dec out of the study's coords (it is then derived "
            "from the life spans), to silence this warning."
        )
        return ds.reindex(year_dec=np.union1d(ds.year_dec.values, years).astype("int64"))

    why = (
        "year_dec is not completed: the annuity tables of the previous "
        "convention depend on year_dec and the new rows would cost nothing"
    )
    for name, count, total, example, value in found:
        logger.warning(
            f"{name}: {count} of {total} entries have an end of life that is "
            f"not in year_dec (for example {example}: {value}). Without early "
            f"decommissioning such a vintage has no capacity variable, so it "
            f"cannot be built and an investment minimum on it is ignored; "
            f"{why}. Put the end of life in year_dec, or leave year_dec out "
            f"of the study's coords (it is then derived from the life spans)."
        )
    return ds


def check_inputs(ds: Dataset) -> Dataset:
    """
    Validate and update the input dataset.

    The function verifies that the coordinates and data types of the
    dataset's variables match the expected values from the reference file.
    If any required data variables are missing from the dataset for a
    particular module, the function adds them with default values from the
    reference configuration.

    Args:
        ds (xarray.Dataset): Input dataset to validate. Each variable is
            checked against the reference configuration defined in
            "pommes/dataset_description.yaml".

    Raises:
        ValueError: If mismatches occur between the dataset's coordinates
            and types and the reference configuration.

    Warnings:
        Missing data variables are added with default values based on
        the reference configuration.

    Returns:
        xarray.Dataset: The validated and updated dataset, with missing
            variables added if necessary.
    """
    if "annuity_full_life" not in ds.data_vars:
        # Dataset built without compute_annuity_cost: tables with a year_dec
        # dimension are the previous convention, others full life.
        full_life = infer_annuity_full_life(ds)
        logging.warning(
            f"annuity_full_life missing: inferred {full_life} from the "
            f"annuity tables (year_dec dimension means until_decommissioning)."
        )
        ds = ds.assign(annuity_full_life=xr.DataArray(full_life))

    for variable in ds.data_vars:
        da = ds[variable]
        # print(variable,ref_inputs[variable])
        if not set(da.coords).issubset(ref_inputs[variable]["index_set"]):
            raise ValueError(
                f"For {variable}: {list(da.coords)} not in "
                f"{ref_inputs[variable]['index_set']}"
            )

        if not isinstance(
            da.dtype, type(np.dtype(ref_inputs[variable]["type"]))
        ):
            try:
                ds[variable] = ds[variable].astype(
                    ref_inputs[variable]["type"]
                )
                logging.warning(
                    f"{variable}: Type {da.dtype} converted to "
                    f"{ref_inputs[variable]['type']}"
                )
            except ValueError as e:
                raise ValueError(
                    f"{variable}: Type {da.dtype} should be "
                    f"{ref_inputs[variable]['type']}"
                ) from e

    modules = [
        "carbon",
        "combined",
        "conversion",
        "transport",
        "net_import",
        "storage",
        "turpe",
        "retrofit_conversion",
        "retrofit_combined",
        "retrofit_storage",
        "retrofit_transport",
        "process",
        "flexdem"
    ]
    variables = list(ref_inputs.keys())

    for module in modules:
        if module not in ds.data_vars or not ds[module]:
            variables = [var for var in variables if module not in var]

    for variable in ds.data_vars:
        if variable in variables:
            variables.remove(variable)

    if variables:
        logging.warning(
            "Missing variables in input dataset. Adding defaults:\n"
            + "\n".join(
                f"{var}: {ref_inputs[var]['default']} "
                f"(type={ref_inputs[var]['type']})"
                for var in variables
            )
        )
        for variable in variables:
            ds = ds.assign(
                {
                    variable: np.array(
                        ref_inputs[variable]["default"],
                        dtype=ref_inputs[variable]["type"],
                    )
                }
            )

    # Maximum of each life span variable on its own dimensions (NaN ignored).
    # Stacking them with to_dataarray would build the union of all their
    # dimensions (about 105 GiB on a large study).
    life_span_vars = [var for var in ds.data_vars if "life_span" in var]
    maxima = [
        float(ds[var].max(skipna=True))
        for var in life_span_vars
        if ds[var].size  # a variable with a zero-length dimension has no value
    ]
    maxima = [value for value in maxima if not np.isnan(value)]
    max_life_span = max(maxima) if maxima else 0

    if max_life_span + ds.year_inv.max() > ds.year_dec.max():
        raise ValueError(
            f"Max life span: {max_life_span}, "
            f"Max year_inv: {int(ds.year_inv.max())}, "
            f"Max year_dec: {int(ds.year_dec.max())}"
        )

    return _complete_year_dec_with_ends_of_life(ds)
