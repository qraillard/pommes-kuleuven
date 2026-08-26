"""Module with consistency check with reference configuration."""

import logging

import numpy as np
from xarray import Dataset

from pommes.model.data_validation import ref_inputs


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

    life_span_vars = [var for var in ds.data_vars if "life_span" in var]

    if life_span_vars:
        max_life_span = (
            ds[life_span_vars]
            .to_dataarray(name="tech")
            .to_dataframe()
            .max()
            .iloc[0]
        )
    else:
        max_life_span = 0

    if max_life_span + ds.year_inv.max() > ds.year_dec.max():
        raise ValueError(
            f"Max life span: {max_life_span}, "
            f"Max year_inv: {int(ds.year_inv.max())}, "
            f"Max year_dec: {int(ds.year_dec.max())}"
        )

    return ds
