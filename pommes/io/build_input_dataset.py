"""Module for the building of the xarray input Dataset."""

import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import logging

from pommes.utils import crf, discount_factor

# Set up a logger for this module
logger = logging.getLogger(__name__)


def read_config_file(
    filename: str = "config",
    file_path: str | None = None,
    study: str | None = None,
) -> dict[str, Any]:
    """
    Reads the configuration file for a study.

    Args:
        filename (str): Name of the config file without the ".yaml" extension.
        file_path (str | None): Path to the configuration file. If None, a
            default path based on `study` is used.
        study (str | None): Name of the study.
            Must be provided if `file_path` is None.

    Returns:
        dict[str, Any]: Dictionary containing the configuration of the study.
    """
    if file_path is None:
        file_path = f"study/{study}/{filename}.yaml"
    with open(file_path) as file:
        data = yaml.safe_load(file)

    if data["input"].get("path") is None:
        data["input"]["path"] = Path(file_path).parent / "data"
    else:
        if (data["input"]["path"] != f"study/{study}/data") and (
            study is not None
        ):
            warnings.warn(
                f"Config seems to be inconsistent with study name: \n "
                f"Study name '{study}' \n "
                f"Config path: '{data['input']['path']}'",
            )

    return data


def build_coordinates(config_: dict[str, Any]) -> xr.Coordinates:
    """
    Build coordinates based on the configuration.

    Args:
        config_ (dict[str, Any]): Configuration dictionary.

    Returns:
        xarray.Coordinates: Xarray coordinates constructed from the
        configuration.
    """
    return xr.Coordinates(
        {
            coord_label: np.array(x["values"], dtype=x["type"])
            for coord_label, x in config_["coords"].items()
        }
    )


def load_inputs_as_dict_of_df(
    config_: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    Loads input data from all CSV files given in data variable descriptions.

    Args:
        config_ (dict[str, Any]): Configuration dictionary.

    Returns:
        dict[str, pandas.DataFrame]: Dictionary of DataFrames containing the
            loaded data.
    """
    parameters = config_["input"]["parameters"]
    path = config_["input"]["path"]

    file_index = {}

    for variable in parameters.keys():
        file = parameters[variable]["file"]
        index_input = parameters[variable]["index_input"]
        if file in file_index.keys():
            if file_index[file] != tuple(sorted(index_input)):
                raise ValueError(
                    f"Same file has not identical index in data variable "
                    f"descriptions!\n"
                    f'Check index of variable "{variable}".'
                )
        else:
            file_index[file] = tuple(sorted(index_input))

    dict_df = {}

    for (file, index_input) in file_index.items():
        # print(file, index_input)
        logger.info("Start reading csv file: " + file + " ...")
        dict_df[file] =  pd.read_csv(
                            f"{path}/{file}",
                            index_col=index_input,
                            dtype={
                                coord: np.dtype(config_["coords"][coord]["type"])
                                for coord in index_input
                            },
                            sep=";",
                        )
        logger.info("Completed reading csv file: " + file)

    return dict_df


def compute_end_of_life(ds: xr.Dataset) -> xr.Dataset:
    """
    Add annuity costs to all corresponding investment costs in input dataset.

    Args:
        ds (xarray.Dataset): Dataset built from the input parameters before
        pre-processing.

    Returns:
        xarray.Dataset: Updated dataset with new data variables
        'x_end_of_life' for all 'x_life_span'.
    """
    for data_var in ds.data_vars:
        if "life_span" in data_var:
            # Retrofit assets are financed from the year the retrofit
            # happens; everything else from its investment year.
            if "retrofit" in str(data_var):
                if "year_ret" not in ds.coords:
                    raise ValueError(
                        f"{data_var} needs a 'year_ret' coordinate. "
                        f"Add year_ret to the study's coords."
                    )
                year_start = ds.year_ret
            else:
                year_start = ds.year_inv
            ds[str(data_var).replace("life_span", "end_of_life")] = (
                ds.year_dec.broadcast_like(year_start + ds[data_var])
                .where(ds.year_dec <= (year_start + ds[data_var]))
                .max("year_dec")
                .astype("int64")
            )
    return ds


def build_annuity_rate(
    years_dec_: xr.DataArray,
    years_inv_: xr.DataArray,
    max_finance_rate: float = 0.15, #previously 0.1
    finance_rate_step: float = 0.005,
) -> xr.DataArray:
    """
    Build a DataArray containing capital recovery factor for financial data.

    Args:
        years_dec_ (xarray.DataArray): Decommissioning year coordinates.
        years_inv_ (xarray.DataArray): Investment year coordinates.
        max_finance_rate (float): Maximum financing rate. Default is 0.1.
        finance_rate_step (float): Step size for financing rate.
            Default is 0.005.

    Returns:
        xarray.DataArray: DataArray containing repayment rates indexed with
            financing rates, decommissioning year, and investment year.
    """
    temp = np.arange(0, max_finance_rate, finance_rate_step, dtype="float64")
    finance_rate = xr.DataArray(temp, coords=dict(finance_rate=temp))

    df = (
        xr.DataArray(np.nan, coords=[years_dec_, years_inv_, finance_rate])
        .to_dataframe(name="crf")
        .reset_index()
    )

    df["crf"] = df.apply(
        lambda row: crf(row.finance_rate, row.year_dec - row.year_inv), axis=1
    )
    return (
        df.set_index(["year_dec", "year_inv", "finance_rate"]).to_xarray().crf
    )


def compute_annuity_cost(
    ds: xr.Dataset, config_: dict[str, Any]
) -> xr.Dataset:
    """
    Add annuity costs to all corresponding investment costs in input dataset.

    Args:
        ds (xarray.Dataset): Dataset built from the input parameters before
            pre-processing.
        config_ (dict[str, Any]): Configuration dictionary.

    Returns:
        xarray.Dataset: Updated dataset with new data variables
            'x_annuity_cost' for all 'x_invest_cost'.
    """
    annuity_rate = build_annuity_rate(ds.year_inv, ds.year_dec)
    annuity_to_compute = config_["pre_process"]["annuity_computation"]

    for category, dict_annuity in annuity_to_compute.items():
        if ds[category].any():
            for name, param in dict_annuity.items():
                year_start_name = param.get("year_start", "year_inv")
                if year_start_name not in ds.coords:
                    raise ValueError(
                        f"{name} needs a {year_start_name!r} coordinate "
                        f"(its configured 'year_start'). Add "
                        f"{year_start_name} to the study's coords."
                    )
                year_start = ds[year_start_name]
                if year_start_name == "year_inv":
                    rate = annuity_rate
                else:
                    # build_annuity_rate keys its crf lookup on columns
                    # literally named "year_dec" and "year_inv"; borrow
                    # that shape for the retrofit's own start year, then
                    # relabel the axis back to year_start_name so it
                    # aligns with the rest of the dataset.
                    rate = build_annuity_rate(
                        ds.year_dec,
                        year_start.rename({year_start_name: "year_inv"}),
                    ).rename({"year_inv": year_start_name})
                ds = ds.assign(
                    {
                        name: ds[param["invest_cost"]]
                        * xr.where(
                            cond=(
                                (year_start < ds.year_dec)
                                * (
                                    ds.year_dec
                                    <= (year_start + ds[param["life_span"]])
                                )
                                * np.isfinite(ds[param["finance_rate"]])
                            ),
                            x=rate.sel(
                                finance_rate=ds[param["finance_rate"]].where(
                                    ds[param["finance_rate"]].isin(
                                        rate.finance_rate
                                    ),
                                    0,
                                )
                            ).drop_vars("finance_rate"),
                            y=np.nan,
                        )
                    }
                )
    return ds


def compute_discount(ds: xr.Dataset, config_: dict[str, Any]) -> xr.Dataset:
    """
    Build a DataArray containing the discount factor and add it to the input.

    Args:
        ds (xarray.Dataset): Dataset built from the input parameters before
            pre-processing.
        config_ (dict[str, Any]): Configuration dictionary.

    Returns:
        xarray.Dataset: Updated dataset with a new data variable 'discount
            factor'.
    """
    discount_rate = ds[
        config_["pre_process"]["discount_factor"]["discount_rate"]
    ]
    year_ref = ds[config_["pre_process"]["discount_factor"]["year_ref"]]

    discount_rate = xr.broadcast(xr.DataArray(discount_rate), ds.year_op)[0]

    ds = ds.assign(
        discount_factor=xr.DataArray(
            np.array(
                np.vectorize(discount_factor, excluded=["year_ref"])(
                    r=discount_rate,
                    year=ds.year_op.astype(np.float64),
                    year_ref=year_ref,
                ),
                dtype=np.float64,
            ),
            coords=[ds.year_op],
        )
    )

    return ds


def build_input_parameters(config_: dict[str, Any], dict_df: Optional[dict[str, pd.DataFrame]]=None) -> xr.Dataset:
    """
    Constructs and validates the input parameters based on the configuration.

    Args:
        config_ (dict[str, Any]): Configuration dictionary containing
            information about input parameters, modules, and coordinate types.

    Raises:
        ValueError: If there is an inconsistency between the index input and
            index set for any parameter.

    Returns:
        xarray.Dataset: Dataset built from the input parameters, with modules,
            coordinates, and parameters set according to the configuration.
            Missing data is filled with default values where necessary,
            and annuity cost and discount values are computed.

    Notes:
        - This function first builds the dataset coordinates and assigns
            modules based on the `config_` dictionary.
        - Parameters are validated against their expected index sets. If there
            is any inconsistency, a `ValueError` is raised.
        - The parameter data is loaded from the source files and columns, cast
            to the required data type, and broadcast to match the dataset's
            coordinates.
        - Any missing data in the parameter arrays is filled with the specified
            fill values.
        - After assigning all parameters, additional computations for annuity
            costs and discounting are performed using `compute_annuity_cost`
            and `compute_discount`.
    """
    parameters = config_["input"]["parameters"]

    if dict_df is None:
        dict_df = load_inputs_as_dict_of_df(config_)

    ds = xr.Dataset(coords=build_coordinates(config_))

    for module, boolean in config_["add_modules"].items():
        ds = ds.assign({module: boolean})

    for key, param in parameters.items():
        # print(key,param)
        series = dict_df[param["file"]][param["column"]].astype(param["type"])

        if series.index.names[0] is None:
            da = xr.DataArray(np.array(series[0]))
        else:
            da = series.to_xarray().astype(param["type"])

            da = da.broadcast_like(ds[list(da.dims)])
            da = da.assign_coords(
                {
                    coord: da[coord].astype(config_["coords"][coord]["type"])
                    for coord in da.dims
                }
            )

        if np.any(da.isnull()):
            da = da.fillna(np.array(param["fill"], dtype=param["type"]))
        ds = ds.assign({key: da})

    ds = compute_end_of_life(ds)
    ds = compute_annuity_cost(ds, config_)
    ds = compute_discount(ds, config_)

    return ds
