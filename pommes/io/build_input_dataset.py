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


def derive_year_dec(ds: xr.Dataset) -> np.ndarray:
    """
    Decommissioning years of a study that does not list ``year_dec``.

    The grid is the sorted union of

    - the end of every asset, ``year_start + life_span`` for each
      ``*_life_span`` variable (``year_start`` is ``year_ret`` for a retrofit
      life span and ``year_inv`` otherwise, as in ``compute_end_of_life``). A
      life span that is NaN or not positive holds no capacity and is skipped;
      a fractional end of life is rounded down to a whole year;
    - ``year_inv``, ``year_op`` and ``year_ret`` when the dataset has them: an
      asset can be decommissioned early at a decision year, a retrofit needs
      ``year_ret`` among the decommissioning years, and an operation year is
      a boundary of the window in which a vintage serves.

    Every end of life is then a decommissioning year, so ``compute_end_of_life``
    gives the exact end (``year_inv + life_span``) instead of rounding it down to
    a grid chosen by the user, and ``check_inputs`` finds a grid that is long
    enough. The grid stays short: at most the number of distinct
    ``year_start + life_span`` plus the study years.

    Args:
        ds (xarray.Dataset): Dataset with its coordinates and its life spans,
            before ``compute_end_of_life``.

    Returns:
        numpy.ndarray: Sorted unique int64 years.
    """
    years = [
        np.asarray(ds[coord].values, dtype="float64").ravel()
        for coord in ("year_inv", "year_op", "year_ret")
        if coord in ds.coords
    ]
    for data_var in ds.data_vars:
        if "life_span" not in data_var:
            continue
        if "retrofit" in str(data_var):
            if "year_ret" not in ds.coords:
                raise ValueError(
                    f"{data_var} needs a 'year_ret' coordinate. "
                    f"Add year_ret to the study's coords."
                )
            year_start = ds.year_ret
        else:
            year_start = ds.year_inv
        life_span = ds[data_var]
        end = (year_start + life_span).where(life_span > 0)
        years.append(np.asarray(end.values, dtype="float64").ravel())
    all_years = np.concatenate(years) if years else np.array([])
    return np.unique(np.floor(all_years[np.isfinite(all_years)])).astype("int64")


def build_annuity_rate(
    years_dec_: xr.DataArray,
    years_inv_: xr.DataArray,
    max_finance_rate: float = 0.15,
    finance_rate_step: float = 0.005,
    finance_rates: Optional[Any] = None,
) -> xr.DataArray:
    """
    Build a DataArray containing capital recovery factor for financial data.

    The factor is tabulated for each pair (decommissioning year, start year):
    ``crf(finance_rate, year_dec - year_inv)``, NaN where ``year_dec <=
    year_inv``. The number of annuities follows the decommissioning year.

    Args:
        years_dec_ (xarray.DataArray): Decommissioning year coordinates.
        years_inv_ (xarray.DataArray): Investment (or start) year coordinates.
        max_finance_rate (float): Legacy grid only, used when `finance_rates`
            is None: the grid is ``np.arange(0, max_finance_rate,
            finance_rate_step)``. Default is 0.15.
        finance_rate_step (float): Legacy grid only, step of the grid.
            Default is 0.005.
        finance_rates (array-like | None): Exact finance rates (fractions per
            year) to tabulate. When given, the grid arguments are ignored.
            Rates must be finite and greater than -1.

    Returns:
        xarray.DataArray: DataArray containing repayment rates indexed with
            decommissioning year, investment year and financing rate.

    Raises:
        ValueError: If a finance rate is not finite or not greater than -1, or
            if a capital recovery factor is not finite where it is defined.
    """
    if finance_rates is None:
        temp = np.arange(0, max_finance_rate, finance_rate_step, dtype="float64")
    else:
        temp = np.unique(np.asarray(finance_rates, dtype="float64").ravel())
        if temp.size == 0 or not np.isfinite(temp).all() or (temp <= -1).any():
            raise ValueError(
                "finance rates must be a non-empty set of finite values "
                f"greater than -1, got {temp}"
            )
    finance_rate = xr.DataArray(temp, coords=dict(finance_rate=temp))

    df = (
        xr.DataArray(np.nan, coords=[years_dec_, years_inv_, finance_rate])
        .to_dataframe(name="crf")
        .reset_index()
    )

    df["crf"] = df.apply(
        lambda row: crf(row.finance_rate, row.year_dec - row.year_inv), axis=1
    )
    table = (
        df.set_index(["year_dec", "year_inv", "finance_rate"]).to_xarray().crf
    )

    defined = (table.year_dec - table.year_inv) > 0
    undefined = defined & ~np.isfinite(table)
    if bool(undefined.any()):
        bad = undefined.any(["year_dec", "year_inv"])
        raise ValueError(
            "capital recovery factor is not finite for finance rates "
            f"{table.finance_rate.values[bad.values]}"
        )
    return table


def _finance_rates_used(ds: xr.Dataset, params: list[dict]) -> np.ndarray:
    """Sorted unique finite values of the finance rate variables of `params`."""
    values = [
        np.asarray(ds[param["finance_rate"]].values, dtype="float64").ravel()
        for param in params
    ]
    rates = np.unique(np.concatenate(values)) if values else np.array([])
    return rates[np.isfinite(rates)]


ANNUITY_MODES = ("full_life", "until_decommissioning")


def _legacy_annuity_cost(
    ds: xr.Dataset, active: dict[str, dict]
) -> xr.Dataset:
    """
    Annuity tables by decommissioning year (``annuity_mode: until_decommissioning``).

    The annuity of an asset that starts at `year_start` and is decommissioned
    in `year_dec` is ``invest_cost * crf(finance_rate, year_dec -
    year_start)`` for ``year_start < year_dec <= year_start + life_span``, NaN
    otherwise. The capital recovery factor is computed for the exact finance
    rates present in the dataset (no rate grid). Assets with a non-finite
    finance rate get NaN.
    """
    # One table per start year, tabulated for the rates actually used
    # (a placeholder rate if none is finite: every annuity is then NaN).
    rates = _finance_rates_used(ds, list(active.values()))
    if rates.size == 0:
        rates = np.array([0.0])
    tables: dict[str, xr.DataArray] = {}

    for name, param in active.items():
        year_start_name = param.get("year_start", "year_inv")
        year_start = ds[year_start_name]
        if year_start_name not in tables:
            if year_start_name == "year_inv":
                tables[year_start_name] = build_annuity_rate(
                    ds.year_dec, ds.year_inv, finance_rates=rates
                )
            else:
                # build_annuity_rate keys its crf lookup on columns
                # literally named "year_dec" and "year_inv"; borrow
                # that shape for the retrofit's own start year, then
                # relabel the axis back to year_start_name so it
                # aligns with the rest of the dataset.
                tables[year_start_name] = build_annuity_rate(
                    ds.year_dec,
                    year_start.rename({year_start_name: "year_inv"}),
                    finance_rates=rates,
                ).rename({"year_inv": year_start_name})
        rate = tables[year_start_name]

        finance_rate = ds[param["finance_rate"]]
        is_finite = np.isfinite(finance_rate)
        # Exact lookup key; non-finite rates only need some valid key, their
        # annuity is masked to NaN by the condition below.
        key = finance_rate.astype("float64").where(is_finite, rates[0])
        try:
            factor = rate.sel(finance_rate=key).drop_vars("finance_rate")
        except KeyError as error:
            raise ValueError(
                f"{name}: finance rate missing from the annuity table "
                f"(rates tabulated: {rates})"
            ) from error
        ds = ds.assign(
            {
                name: ds[param["invest_cost"]]
                * xr.where(
                    cond=(
                        (year_start < ds.year_dec)
                        * (ds.year_dec <= (year_start + ds[param["life_span"]]))
                        * is_finite
                    ),
                    x=factor,
                    y=np.nan,
                )
            }
        )
    return ds


def _full_life_annuity_cost(
    ds: xr.Dataset, active: dict[str, dict]
) -> xr.Dataset:
    """
    Annuities fixed at investment (``annuity_mode: full_life``).

    The annuity of a vintage is ``invest_cost * crf(finance_rate,
    life_span)``, independent of the decommissioning year (no ``year_dec``
    dimension); the model pays it in every operation year of the vintage's
    full life. NaN where the finance rate is not finite or the life span is
    not positive.
    """
    for name, param in active.items():
        finance_rate = ds[param["finance_rate"]].astype("float64")
        is_finite = np.isfinite(finance_rate)
        usable = finance_rate.where(is_finite)
        if bool((usable <= -1).any()):
            raise ValueError(
                f"{name}: finance rates must be greater than -1, got "
                f"{np.unique(usable.values[usable.values <= -1])}"
            )
        life_span = ds[param["life_span"]].astype("float64")
        factor = xr.apply_ufunc(crf, usable, life_span)
        undefined = is_finite & (life_span > 0) & ~np.isfinite(factor)
        if bool(undefined.any()):
            raise ValueError(
                f"{name}: capital recovery factor is not finite for finance "
                f"rates {np.unique(usable.where(undefined).values[undefined.values])}"
            )
        ds = ds.assign({name: ds[param["invest_cost"]] * factor})
    return ds


def _warn_off_grid_life_spans(ds: xr.Dataset, active: dict[str, dict]) -> None:
    """
    Warn where ``year_start + life_span`` is not a decommissioning year.

    Capacity ends at the largest ``year_dec`` not after ``year_start +
    life_span`` (``compute_end_of_life``) while the ``full_life`` annuity runs
    over the whole life span: operation years between the two pay an annuity
    for capacity that is gone. One warning per annuity variable, with the
    number of entries and one example.

    Args:
        ds (xarray.Dataset): Dataset built from the input parameters.
        active (dict[str, dict]): Annuity parameters of the active modules.
    """
    year_dec = np.asarray(ds.year_dec.values)
    for name, param in active.items():
        cost = ds[param["invest_cost"]]
        life = ds[param["life_span"]]
        end = ds[param.get("year_start", "year_inv")] + life
        built = np.isfinite(cost) & (cost != 0) & (life > 0)
        off = built & ~end.isin(year_dec)
        count = int(off.sum())
        if not count:
            continue
        first = tuple(np.argwhere(off.values)[0])
        example = {
            dim: (off[dim].values[i].item() if dim in off.coords else i)
            for dim, i in zip(off.dims, first)
        }
        end_value = float(end.broadcast_like(off).transpose(*off.dims).values[first])
        rounded = year_dec[year_dec <= end_value]
        logger.warning(
            f"{name}: for {count} of {int(built.broadcast_like(off).sum())} "
            f"entries with an invest "
            f"cost, year_start + life_span is not in year_dec (for example "
            f"{example}: {end_value:g}, capacity ends at "
            f"{rounded.max() if rounded.size else 'no year_dec'}). With "
            f"annuity_mode full_life the capacity ends at the rounded-down "
            f"year_dec while the annuity runs the full life span; leaving "
            f"year_dec out of the study's coords (it is then derived from "
            f"the life spans) or a finer year_dec grid (for example "
            f"yearly) removes the gap."
        )


def _check_annuity_inputs(ds: xr.Dataset, active: dict[str, dict]) -> None:
    """
    Raise if an asset that can hold capacity has an invest cost but no rate.

    A finite, nonzero invest cost with a positive life span and a finance rate
    that is not finite would give a NaN annuity that the model drops without
    notice (a capex of zero). A vintage whose life span is NaN or not positive
    holds no capacity, so its invest cost is never paid: it only gets a
    warning (studies use a life span of 0 for vintages that cannot be built).

    Args:
        ds (xarray.Dataset): Dataset built from the input parameters.
        active (dict[str, dict]): Annuity parameters of the active modules.

    Raises:
        ValueError: Listing, per annuity variable, the number of entries.
    """
    problems = []
    for name, param in active.items():
        cost = ds[param["invest_cost"]]
        has_cost = np.isfinite(cost) & (cost != 0)
        has_life = ds[param["life_span"]] > 0
        no_rate = has_cost & has_life & ~np.isfinite(ds[param["finance_rate"]])
        count = int(no_rate.sum())
        if count:
            problems.append(
                f"{name}: {count} entries (finance rate not finite: {count})"
            )
        no_life = int((has_cost & ~has_life).sum())
        if no_life:
            logger.warning(
                f"{name}: {no_life} entries have an invest cost but a life "
                f"span that is NaN or not positive; these vintages hold no "
                f"capacity, so the invest cost is ignored."
            )
    if problems:
        raise ValueError(
            "Annuity cannot be computed where the invest cost is finite and "
            "nonzero and the life span is positive but the finance rate is "
            "missing (the capex would silently be zero). Give a finance rate, "
            "or set the invest cost to 0 (or NaN) for assets that are not "
            "built:\n" + "\n".join(problems)
        )


def compute_annuity_cost(
    ds: xr.Dataset, config_: dict[str, Any]
) -> xr.Dataset:
    """
    Add annuity costs to all corresponding investment costs in input dataset.

    The convention is chosen by ``pre_process: annuity_mode`` in the study
    configuration:

    - ``full_life`` (default): the annuity of a vintage is fixed at
      investment, ``invest_cost * crf(finance_rate, life_span)``, without
      ``year_dec`` dimension. The model pays it in every operation year of
      the vintage's full life (``year_start <= year_op < year_start +
      life_span``) even if the capacity is decommissioned earlier (sunk
      cost): early decommissioning only saves fixed and variable costs.
    - ``until_decommissioning`` (previous behaviour): annuity tables by
      decommissioning year, ``invest_cost * crf(finance_rate, year_dec -
      year_start)`` for ``year_start < year_dec <= year_start + life_span``;
      the per-vintage ``annuity_perfect_foresight`` flag then chooses between
      paying it until the decommissioning year and the cheapest annuity until
      the end of life.

    In both cases the capital recovery factor uses the exact finance rates of
    the dataset (no rate grid). Assets without invest cost (0 or NaN) may lack
    a finance rate or life span and get a NaN annuity; an asset with a finite,
    nonzero invest cost, a positive life span and no finite rate is an error,
    not a free asset (without a positive life span it holds no capacity: a
    warning). The scalar ``annuity_full_life`` is added to the
    dataset for the model.

    Args:
        ds (xarray.Dataset): Dataset built from the input parameters before
            pre-processing.
        config_ (dict[str, Any]): Configuration dictionary.

    Returns:
        xarray.Dataset: Updated dataset with new data variables
            'x_annuity_cost' for all 'x_invest_cost', and 'annuity_full_life'.

    Raises:
        ValueError: If ``annuity_mode`` is unknown, a finance rate cannot be
            used (not greater than -1), an asset with an invest cost lacks a
            finite finance rate or a positive life span, or a configured
            'year_start' is not a coordinate.
    """
    mode = config_["pre_process"].get("annuity_mode", "full_life")
    if mode not in ANNUITY_MODES:
        raise ValueError(
            f"pre_process: annuity_mode must be one of {ANNUITY_MODES}, "
            f"got {mode!r}"
        )
    annuity_to_compute = config_["pre_process"]["annuity_computation"]
    active = {
        name: param
        for category, dict_annuity in annuity_to_compute.items()
        if ds[category].any()
        for name, param in dict_annuity.items()
    }
    for name, param in active.items():
        year_start_name = param.get("year_start", "year_inv")
        if year_start_name not in ds.coords:
            raise ValueError(
                f"{name} needs a {year_start_name!r} coordinate "
                f"(its configured 'year_start'). Add "
                f"{year_start_name} to the study's coords."
            )

    _check_annuity_inputs(ds, active)
    logger.info("Annuities computed with annuity_mode: %s", mode)

    if mode == "full_life":
        _warn_off_grid_life_spans(ds, active)
        ds = _full_life_annuity_cost(ds, active)
    else:
        ds = _legacy_annuity_cost(ds, active)
    return ds.assign(annuity_full_life=xr.DataArray(mode == "full_life"))


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
    # A study that does not list year_dec gets it from the life spans below.
    derive_year_dec_grid = "year_dec" not in config_["coords"]

    for module, boolean in config_["add_modules"].items():
        ds = ds.assign({module: boolean})

    for key, param in parameters.items():
        # print(key,param)
        series = dict_df[param["file"]][param["column"]].astype(param["type"])

        if derive_year_dec_grid and "year_dec" in series.index.names:
            raise ValueError(
                f"{key} is indexed by year_dec, but year_dec is not in the "
                f"study's coords: list year_dec in the config, or index {key} "
                f"by something else."
            )

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

    if derive_year_dec_grid:
        ds = ds.assign_coords(
            year_dec=xr.DataArray(derive_year_dec(ds), dims="year_dec")
        )

    ds = compute_end_of_life(ds)
    ds = compute_annuity_cost(ds, config_)
    ds = compute_discount(ds, config_)

    return ds
