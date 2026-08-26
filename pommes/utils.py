"""Module implementing utility functions."""

import datetime
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Constraint, Model
from pandas import DataFrame
from xarray import DataArray, Dataset


def sort_coords_and_vars(ds: Dataset) -> Dataset:
    """
    Sort the coordinates of a xarray Dataset.

    Sort the coordinates of a xarray Dataset by their coordinate values (for
    all dimension coords), then sort the Dataset variables and coordinate
    names alphabetically.

    Args:
        ds (xarray.Dataset):
            The input Dataset to be sorted.

    Returns:
        xarray.Dataset:
            A new Dataset with:
             1) Each dimension coordinate sorted in ascending order by value.
             2) Data variable names sorted alphabetically.
             3) Coordinate names sorted alphabetically.
    """
    for dim in ds.dims:
        ds = ds.sortby(dim)

    sorted_data_vars = np.sort(ds.data_vars).tolist()
    sorted_coord_names = np.sort(ds.coords).tolist()

    new_ds = xr.Dataset(
        data_vars={var: ds[var] for var in sorted_data_vars},
        coords={coord: ds[coord] for coord in sorted_coord_names},
    )
    return new_ds


def array_to_datetime(
    hours: DataArray, year: int = 2000, datetime_format: str = "%d/%mH:%M"
) -> DataArray:
    """
    Convert an array of hours into datetime objects.

    Args:
        hours (DataArray): Array of hours to convert.
        year (int, optional): Reference year. Defaults to 2000.
        datetime_format (str, optional): Format for datetime objects.
            Defaults to "%d/%mH:%M".

    Returns:
        DataArray: Array of formatted datetime strings.
    """
    h0 = datetime.datetime(year=year, month=1, day=1)
    return np.vectorize(
        lambda hour: (h0 + datetime.timedelta(hours=int(hour))).strftime(
            datetime_format
        ),
    )(hours)


def model_solve(model_: Model, solver: str) -> Model:
    """
    Solve a Linopy model using the specified solver.

    Args:
        model_ (Model): The Linopy model to solve.
        solver (str): Name of the solver to use (e.g., "gurobi", "highs", "xpress").

    Returns:
        Model: The solved Linopy model.
    """
    if solver == "gurobi":
        model_.solve(
            solver_name=solver, method=2, crossover=0
        )  # crossover=0, numericfocus=3)
    elif solver == "highs":
        model_.solve(
            solver_name=solver,
            presolve="on",
            solver="ipm",
            parallel="on",
            run_crossover="on",
            ipm_optimality_tolerance=1e-8,
        )
    elif solver == "xpress":
        model_.solve(solver_name=solver, DEFAULTALG=4, CROSSOVER=2)
    else:
        model_.solve(solver_name=solver)
    return model_


def crf(r: float, m: int) -> float:
    """
    Calculate the capital recovery factor.

    Args:
        r (float): Finance rate per period.
        m (int): Number of periods.

    Returns:
        float: The capital recovery factor or NaN if `m` <= 0.
    """
    if m <= 0:
        return np.nan
    if r == 0:
        return 1 / m
    return r / (1 - (1 + r) ** (-np.array(m, dtype=np.float64)))


def discount_factor(r: float, year: int, year_ref: int) -> float:
    """
    Compute the discount factor between a year and a reference year.

    Args:
        r (float): Discount rate.
        year (int): Year of cash flow.
        year_ref (int): Reference year.

    Returns:
        float: Discount factor.
    """
    return (1 + r) ** (-np.array(year - year_ref, dtype=np.float64))


def square_array_by_diagonals(
    shape: int, diags: dict, fill: float = 0, dtype: str | None = None
) -> np.ndarray:
    """
    Build a square array with specified diagonal values.

    Args:
        shape (int): Size of the square array.
        diags (dict): Dictionary mapping diagonal indices to their values.
        fill (float, optional): Fill value for non-diagonal elements. Defaults to 0.
        dtype (type, optional): Data type of the array. Defaults to None.

    Returns:
        np.ndarray: Square array with specified diagonal values.

    Examples:
        >>> dict_diag = {0: 5, 1: [1, 2], -2:[1, 2, 3]}
        >>> square_array_by_diagonals(shape=5, diags=dict_diag, fill=np.nan)
        array([[ 5.,  1., nan, nan, nan],
               [nan,  5.,  2., nan, nan],
               [ 1., nan,  5.,  1., nan],
               [nan,  2., nan,  5.,  2.],
               [nan, nan,  3., nan,  5.]])
    """
    res = np.zeros((shape, shape), dtype=dtype)
    res.fill(fill)
    for k, value in diags.items():
        if k >= 0:
            i = k
        else:
            i = (-k) * shape
        res[: shape - k].flat[i :: shape + 1] = value
    return res


def combine_csv_to_excel(
    repo_path: str, output_excel_path: str, sep: str = ";"
) -> None:
    """
    Combine multiple CSV files from a directory into a single Excel file.

    Args:
        repo_path (str): Path to the directory containing the CSV files.
        output_excel_path (str): Path to save the combined Excel file.
        sep (str, optional): Delimiter for CSV files. Defaults to ";".

    Returns:
        None: Saves the combined Excel file at the specified path.

    Examples:
        >>> repo = "/path/to/your/repository"
        >>> excel_path = "/path/to/output/output.xlsx"
        >>> combine_csv_to_excel(repo, excel_path)
    """
    csv_files = [
        file for file in os.listdir(repo_path) if file.endswith(".csv")
    ]

    with pd.ExcelWriter(output_excel_path) as writer:
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(repo_path, csv_file), sep=sep)
            df.to_excel(
                writer, sheet_name=os.path.splitext(csv_file)[0], index=False
            )
    return None


def split_excel_to_csv(
    input_excel_path: str, output_folder: str, sep: str = ";"
) -> None:
    """
    Split an Excel file with multiple sheets into separate CSV files.

    Args:
        input_excel_path (str): Path to the input Excel file.
        output_folder (str): Path to save the generated CSV files.
        sep (str, optional): Delimiter for CSV files. Defaults to ";".

    Returns:
        None: Saves separate CSV files in the specified folder.

    Examples:
        >>> excel_path = "/path/to/input/input.xlsx"
        >>> folder = "/path/to/output/csv_files/"
        >>> split_excel_to_csv(excel_path, folder)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    excel_file = pd.ExcelFile(input_excel_path)

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        output_csv_path = os.path.join(output_folder, f"{sheet_name}.csv")
        df.to_csv(output_csv_path, index=False, sep=sep)
    return None


def get_main_resource(model_parameters: Dataset) -> DataArray:
    """
    Extract the main resource associated with each technology.

    Args:
        model_parameters (Dataset): Model parameters dataset containing resource
            and technology information.

    Returns:
        DataArray: A DataArray containing the main resource for each technology.
    """
    p = model_parameters

    storage = xr.DataArray([], coords=dict(tech=[]))
    transport = xr.DataArray([], coords=dict(tech=[]))

    conversion = p.resource.isel(
        resource=np.equal(p.conversion_factor, 1).argmax(dim="resource")
    )
    conversion = conversion.drop_vars("resource").rename(
        conversion_tech="tech"
    )

    if "storage" in p.keys() and p.storage:
        storage = p.storage_main_resource.rename(storage_tech="tech")
    if "transport" in p.keys() and p.transport:
        is_link = p.transport_is_link.any(
            [dim for dim in p.transport_is_link.dims if "area" not in dim]
        )
        mask = is_link.sel(
            area_from=p.area_from.isel(area_from=0),
            area_to=p.area_to.sel(
                area_to=is_link.isel(area_from=0)
                .where(is_link)
                .sum("area_from")
                > 0
            ).isel(area_to=0),
        )
        transport = (
            p.isel(year_inv=0, drop=True)
            .transport_resource.sel(
                area_from=mask.area_from, area_to=mask.area_to, drop=True
            )
            .rename(transport_tech="tech")
        )

    da = xr.concat([conversion, storage, transport], dim="tech")

    return da


def squeeze_dataset(
    ds: Dataset,
    exclude_dims: list[str] | None = None,
    exclude_data_vars: list[str] | None = None,
    copy: bool = True,
) -> Dataset:
    """
    Squeeze dimensions in the dataset where all values are identical.

    Args:
        ds (Dataset): Input dataset to be squeezed.
        exclude_dims (list[str] | None, optional): Dimensions to exclude from squeezing.
            Defaults to None.
        exclude_data_vars (list[str] | None, optional): Data variables to exclude from squeezing.
            Defaults to None.
        copy (bool, optional): Whether to create a copy of the dataset. Defaults to True.

    Returns:
        Dataset: The dataset with squeezed dimensions.

    Note:
        - The function logs a warning for each squeezed dimension.
        - If `exclude_dims` or `exclude_data_vars` are not provided, the function
            will attempt to squeeze all dimensions and data variables.
    """
    if copy:
        ds = ds.copy(deep=True)
    for var_name in ds.data_vars:
        data_var = ds[var_name]
        if exclude_data_vars is None or data_var not in exclude_data_vars:
            for dim in data_var.dims:
                if exclude_dims is None or dim not in exclude_dims:
                    # Check if all values along the specified dimension are
                    # equal
                    if np.equal(data_var, data_var.isel({dim: 0})).all():
                        logging.warning(
                            msg=f"squeezing dim {dim} in data_var "
                            f"{data_var.name}"
                        )
                        ds[var_name] = data_var.isel({dim: 0})
    return ds


def get_infeasibility_constraint_name(
    constraint_label: Constraint, model: Model, model_parameters: Dataset
) -> str:
    """
    Retrieve the name of an infeasibility constraint.

    Args:
        constraint_label (Constraint): The label or ID of the constraint.
        model (Model): The Linopy model containing the constraints.
        model_parameters (Dataset): The dataset containing model parameters.

    Returns:
        str: The name of the infeasibility constraint.
    """
    if isinstance(constraint_label, str):
        constraint_label = int(constraint_label[1:])
    p = model_parameters
    m = model
    constraint_name = m.constraints.get_name_by_label(constraint_label)
    constraint = m.constraints[constraint_name]
    index = {}
    for dim in list(constraint.coord_dims):
        index[dim] = p[dim].where(
            np.equal(constraint.labels, constraint_label).any(
                dim=[d for d in list(constraint.coord_dims) if d != dim]
            ),
            drop=True,
        )
        index[dim] = index[dim].to_numpy().astype(p[dim].dtype)[0]
    print(
        f"\n{constraint_name}\n"
        + "\n".join([f"{key}: {value}" for key, value in index.items()])
    )
    return constraint.sel(index)


def write_to_excel(
    dataframes: dict[str, DataFrame],
    excel_file: str | None,
) -> None:
    """
    Write multiple DataFrames to an Excel file.

    Each DataFrame is written to a separate sheet in the specified Excel file.

    Args:
        dataframes (dict): A dictionary of pandas DataFrames to be written.
        excel_file (Path): The path to the output Excel file.

    Returns:
        None
    """
    with pd.ExcelWriter(excel_file) as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def select_dataset_variables_from_dims(
    dataset: Dataset,
    dims: str | list[str] | set[str],
    deep_copy: bool = True,
) -> Dataset:
    """
    Select variables from a xarray Dataset that include all given dimensions.

    If dims is empty, return all variables.

    Args:
        dataset (xarray.Dataset):
            The dataset from which variables will be selected.
        dims (str | list[str] | set[str]):
            The dimensions that each selected variable must include. If a list
            is provided, it is automatically converted to a set.
        deep_copy (bool, optional):
            Whether to create a deep copy of the data array or dataset.
            Defaults to True.

    Returns:
        xarray.Dataset:
            A new dataset containing only the variables that have all the
            specified dimensions.
    """
    if isinstance(dims, list):
        dims = set(dims)
    if isinstance(dims, str):
        dims = {dims}

    vars_with_dims = [
        var_name
        for var_name in dataset.data_vars
        if dims.issubset(dataset[var_name].dims)
    ]
    res = dataset[vars_with_dims]
    if deep_copy:
        res = res.copy(deep=True)
    return res
