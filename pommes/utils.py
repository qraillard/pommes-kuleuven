"""Module implementing utility functions."""

import datetime
import logging
import os
import warnings
from dataclasses import dataclass

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
        r (float | array-like): Finance rate per period.
        m (int | array-like): Number of periods. Arrays are broadcast
            together and the result is an array.

    Returns:
        float: The capital recovery factor (1 / m if `r` is 0) or NaN if
            `m` <= 0, `r` <= -1 or `r` is NaN. Computed as
            ``r / -expm1(-m * log1p(r))``, accurate to rounding for every
            `r` > -1 (the closed form ``r / (1 - (1 + r) ** -m)`` loses
            digits for rates near 0 and overflows to inf below 1e-16).
    """
    if np.ndim(r) == 0 and np.ndim(m) == 0:
        if not m > 0 or not r > -1:
            return np.nan
        if r == 0:
            return 1 / m
        return r / -np.expm1(-m * np.log1p(r))
    r, m = np.broadcast_arrays(
        np.asarray(r, dtype=np.float64), np.asarray(m, dtype=np.float64)
    )
    out = np.full(r.shape, np.nan)
    valid = (m > 0) & (r > -1)
    zero = valid & (r == 0)
    other = valid & (r != 0)
    out[zero] = 1 / m[zero]
    out[other] = r[other] / -np.expm1(-m[other] * np.log1p(r[other]))
    return out


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


# --- Operating conditions (weighted weather years) ---

OPERATING_CONDITION = "operating_condition"
OPERATING_CONDITION_WEIGHT = "operating_condition_weight"
NO_CONDITION_COORD_MESSAGE = (
    f"{OPERATING_CONDITION} is a dimension without a coordinate: the "
    f"conditions and their weights would be ignored. Call "
    f"assign_coords({OPERATING_CONDITION}=[...]) first."
)


def has_operating_conditions(p: Dataset) -> bool:
    """
    Whether the dataset carries the operating_condition coordinate.

    Raises:
        ValueError: If operating_condition is a dimension without a
            coordinate (the conditions and their weights would be ignored).
    """
    if OPERATING_CONDITION in p.dims and OPERATING_CONDITION not in p.coords:
        raise ValueError(NO_CONDITION_COORD_MESSAGE)
    return OPERATING_CONDITION in p.coords


def operating_condition_weight(p: Dataset) -> DataArray | None:
    """
    Weight of each operating condition, dims (year_op, operating_condition).

    None without the coordinate. A weight that does not carry the dimension
    and is nan means "no weight given": 1 for a single condition, ValueError
    for several. nan stays nan (pair absent). Labels keep their order.
    """
    if not has_operating_conditions(p):
        return None
    conditions = p[OPERATING_CONDITION]
    w = p.get(OPERATING_CONDITION_WEIGHT)
    if w is None or (OPERATING_CONDITION not in w.dims and w.isnull().all()):
        if conditions.size != 1:
            raise ValueError(
                f"{OPERATING_CONDITION_WEIGHT} is required with several "
                f"{OPERATING_CONDITION} values."
            )
        w = xr.DataArray(1.0)
    w = w.astype(float)
    if "year_op" in p.coords and "year_op" not in w.dims:
        w = w.expand_dims(year_op=p["year_op"].values)
    w, _ = xr.broadcast(w, conditions)
    dims = [d for d in ("year_op", OPERATING_CONDITION) if d in w.dims]
    return w.transpose(*dims, ...)


def condition_coords(p: Dataset) -> list[DataArray]:
    """Coordinates to append after year_op for state-of-the-world variables."""
    return [p[OPERATING_CONDITION]] if has_operating_conditions(p) else []


def condition_mask(p: Dataset, mask: DataArray | None = None):
    """`mask & isfinite(weight)`; `mask` unchanged without the coordinate."""
    if not has_operating_conditions(p):
        return mask
    present = np.isfinite(operating_condition_weight(p))
    return present if mask is None else mask & present


def expected_over_conditions(expr, p: Dataset):
    """
    Weighted sum over operating conditions, nan weights read as 0.

    Works on linopy variables/expressions and DataArrays; returns `expr`
    itself when the dataset has no coordinate or `expr` lacks the dimension.

    On a result DataArray (a solution, a summary) nan is skipped by the sum:
    an absent pair, or a variable masked in one condition, counts as 0, and an
    entry that is nan in every condition gives 0, not nan.
    """
    if not has_operating_conditions(p) or OPERATING_CONDITION not in expr.dims:
        return expr
    w = operating_condition_weight(p).fillna(0)
    return (expr * w).sum(OPERATING_CONDITION)


def annual_quantity(expr, p: Dataset, expected_flag: str):
    """Expectation over conditions when `p[expected_flag]` is true, else expr."""
    if expected_flag in p and bool(p[expected_flag]):
        return expected_over_conditions(expr, p)
    return expr


def annual_mask(p: Dataset, expected_flag: str, mask):
    """Mask of a flagged annual constraint: `mask` on the expectation, else per condition."""
    if expected_flag in p and bool(p[expected_flag]):
        return mask
    return condition_mask(p, mask)


def condition_dual_scale(p: Dataset) -> DataArray:
    """
    Weight per (year_op, condition); 1 without the coordinate.

    Divide the raw dual of an operational constraint by it to get its price
    (the dual of a row is the weight of its condition times the price). Only
    for constraints that carry `operating_condition`: the one-row-per-year
    constraints of the `_expected` flags already give the price of the year. A
    condition of weight 0 has no price: the division gives nan or inf there,
    whatever the dual. A condition absent in a year (weight nan) has no row.
    """
    w = operating_condition_weight(p)
    return xr.DataArray(1.0) if w is None else w


# --- Time blocks (multi-resolution) ---


def block_dim(L: int) -> str:
    """Name of the block dimension of ``L`` hours, e.g. ``time_block_168h``."""
    return f"time_block_{int(L)}h"


def _per_label(p: Dataset, name: str, dim: str):
    """Labels of ``dim`` and values of ``p[name]`` along it (None: absent)."""
    labels = p[dim].values if dim in p.coords else np.array([])
    da = p.get(name)
    if da is None:
        return labels, None
    if dim in da.dims:
        da = da.reindex({dim: labels})
    else:
        da = da.expand_dims({dim: labels})
    return labels, da.transpose(dim).values


def _length_along(p: Dataset, name: str, dim: str, module: str | None):
    """
    Block length per label of ``dim``, NaN when absent (R1 on present values).

    ``module`` is the flag that switches the lengths off (None: always read).
    """
    labels, values = _per_label(p, name, dim)
    values = (
        np.full(len(labels), np.nan)
        if values is None
        else values.astype(float)
    )
    bad = ~np.isnan(values) & ~(
        np.isfinite(values) & (values > 0) & (values == np.floor(values))
    )
    if bad.any():
        k = np.flatnonzero(bad)[0]
        raise ValueError(
            f"{name}[{labels[k]}] = {values[k]}: a block length is a positive "
            f"whole number of hours (or NaN for hourly)."
        )
    if module is not None and not (module in p and bool(p[module])):
        values = np.full(len(labels), np.nan)
    return xr.DataArray(values, dims=dim, coords={dim: labels})


def conversion_block_rule(p: Dataset) -> DataArray:
    """
    Rule turning hourly availability into a block value, per conversion tech.

    ``"min"`` or ``"mean"``; ``"nan"``, ``""``, float NaN and an absent
    parameter read as ``"min"``. Any other value raises ``ValueError`` (R8).
    """
    labels, values = _per_label(
        p, "conversion_time_block_rule", "conversion_tech"
    )
    if values is None:
        values = np.full(len(labels), "min", dtype=object)
    rule = [
        "min"
        if (isinstance(x, float) and np.isnan(x)) or x in ("", "nan")
        else x
        for x in values.astype(object)
    ]
    for tech, x in zip(labels, rule):
        if x not in ("min", "mean"):
            raise ValueError(
                f"conversion_time_block_rule[{tech}] = {x!r}: "
                "expected 'min' or 'mean'."
            )
    return xr.DataArray(
        np.array(rule, dtype=str),
        dims="conversion_tech",
        coords={"conversion_tech": labels},
    )


@dataclass(frozen=True)
class TimeBlocks:
    """Block lengths and maps of one dataset, see ``time_blocks``."""

    tech_length: dict[str, DataArray]
    resource_length: DataArray
    block_of_hour: dict[int, DataArray]
    duration: dict[int, DataArray]
    first_hour: dict[int, DataArray]
    first_step_duration: dict[int, DataArray]
    start: dict[int, DataArray]
    step: DataArray | None = None  # time_step_duration (hour), for block_mean

    @property
    def active(self) -> bool:
        """Whether any length in use is finite (else the model is hourly)."""
        arrays = [*self.tech_length.values(), self.resource_length]
        return any(bool(np.isfinite(a.values).any()) for a in arrays)

    def resources(self, L: int | None) -> list[str]:
        """Resources whose balance has length ``L`` (None: hourly)."""
        r = self.resource_length
        hit = r.isnull().values if L is None else (r == L).values
        return [str(x) for x in r["resource"].values[hit]]

    @property
    def balance_lengths(self) -> list[int]:
        """Distinct finite balance lengths, ascending."""
        v = self.resource_length.values
        return sorted({int(x) for x in v[np.isfinite(v)]})

    def techs(self, module: str, L: int | None) -> list[str]:
        """Technologies of ``module`` with length ``L`` (None: hourly)."""
        a = self.tech_length[module]
        hit = a.isnull().values if L is None else (a == L).values
        return [str(x) for x in a[f"{module}_tech"].values[hit]]


def _on(p: Dataset, flag: str) -> bool:
    return flag in p and bool(p[flag])


def _hits(hit: DataArray, dim: str, labels, other_dim=None):
    """(label of ``dim``, label of ``other_dim``) bool matrix, True where ``hit`` holds for some other index."""
    reduce = [d for d in hit.dims if d not in (dim, other_dim)]
    if reduce:
        hit = hit.any(reduce)
    for d, lab in ((dim, labels[0]), (other_dim, labels[1])):
        if d is None:
            continue
        hit = (
            hit.reindex({d: lab}, fill_value=False)
            if d in hit.dims
            else hit.expand_dims({d: lab})
        )
    return hit.transpose(*[d for d in (dim, other_dim) if d]).values


def _contributors(p: Dataset, tech_length: dict) -> dict:
    """
    Counted contributors per resource, in the order of section 4.3.

    Returns ``{resource: [(kind, name, length)]}``, ``length`` NaN for hourly.
    Factors are read on the full grid (a unit pinned to zero still counts).
    """
    res = p["resource"].values if "resource" in p.coords else np.array([])
    out = {str(r): [] for r in res}

    def add(kind, dim, hit, lengths=None):
        names = p[dim].values
        for i, j in zip(*np.nonzero(hit)):
            L = np.nan if lengths is None else lengths[i]
            out[str(res[j])].append((kind, str(names[i]), float(L)))

    def factor(*names):
        hit = None
        for n in names:
            if n in p:
                x = p[n]
                h = np.isfinite(x) & (x != 0)
                hit = h if hit is None else hit | h
        return hit

    for module, kind in (
        ("conversion", "conversion tech"),
        ("combined", "combined tech"),
    ):
        if _on(p, module) and f"{module}_factor" in p:
            dim = f"{module}_tech"
            hit = _hits(
                factor(f"{module}_factor"),
                dim,
                (p[dim].values, res),
                "resource",
            )
            add(kind, dim, hit, tech_length[module].values)
    if _on(p, "storage"):
        hit = factor(
            "storage_factor_in", "storage_factor_keep", "storage_factor_out"
        )
        if hit is not None:
            add(
                "storage tech",
                "storage_tech",
                _hits(
                    hit,
                    "storage_tech",
                    (p.storage_tech.values, res),
                    "resource",
                ),
            )
    if _on(p, "transport") and "transport_resource" in p:
        techs = p.transport_tech.values
        hit = np.zeros((len(techs), len(res)), dtype=bool)
        for j, r in enumerate(res):
            by_res = p.transport_resource == r
            if "transport_fuel_resource" in p:
                fuel = p.transport_fuel_resource == r
                if "transport_fuel_consumption" in p:
                    fuel = fuel & np.isfinite(p.transport_fuel_consumption)
                else:
                    fuel = fuel & False
                by_res = by_res | fuel
            hit[:, j] = _hits(by_res, "transport_tech", (techs, None))
        add("transport tech", "transport_tech", hit)
    if _on(p, "process") and "process_factor" in p:
        add(
            "process tech",
            "process_tech",
            _hits(
                factor("process_factor"),
                "process_tech",
                (p.process_tech.values, res),
                "resource",
            ),
        )
    return out


def time_blocks(p: Dataset) -> TimeBlocks:
    """
    Block lengths and hour-to-block maps of the dataset ``p`` (pure function).

    Blocks are anchored at the first step: ``start = cumsum(dt) - dt`` and
    ``block = floor((start + 1e-9) / L)``; the last block may be shorter.
    Raises ``ValueError`` for R1 (length not a positive whole number), R2
    (a step crossing a block boundary), R3 (a non-positive or non-finite step),
    R4 (a technology block length that does not divide a forced balance
    length, or derived lengths that do not nest), R5 (a coarse electricity
    balance with TURPE), R8 (unknown availability rule) and R10 (every
    balance coarse). Warns (``UserWarning``) why a balance stays hourly when
    some length is in use. The result carries ``step``, the
    ``time_step_duration`` that ``block_mean`` weights by. Call it on the
    full input dataset, before ``to_sparse_units``.
    """
    conversion_block_rule(p)  # R8
    tech_length = {
        module: _length_along(
            p, f"{module}_time_block_hrs", f"{module}_tech", module
        )
        for module in ("conversion", "combined")
    }
    resource_length = _length_along(
        p, "resource_time_block_hrs", "resource", None
    )
    resource_length, why_hourly = _derive_balance_lengths(
        p, tech_length, resource_length
    )
    lengths = sorted(
        {
            int(x)
            for a in (*tech_length.values(), resource_length)
            for x in a.values[np.isfinite(a.values)]
        }
    )
    keys = (
        "block_of_hour",
        "duration",
        "first_hour",
        "first_step_duration",
        "start",
    )
    maps = {key: {} for key in keys}
    if lengths:
        if "time_step_duration" not in p:
            raise ValueError(
                "time_step_duration is required when a block length is used."
            )
        dt = p["time_step_duration"]
        hours = dt["hour"].values
        d = dt.values.astype(float)
        if not (np.isfinite(d) & (d > 0)).all():
            k = np.flatnonzero(~(np.isfinite(d) & (d > 0)))[0]
            raise ValueError(
                f"time_step_duration[hour={hours[k]}] = {d[k]}: block "
                f"lengths need positive finite time steps."
            )
        t0 = np.cumsum(d) - d
        for L in lengths:
            # R5: the R2 tolerance also anchors steps that land an ulp low
            b = np.floor((t0 + 1e-9) / L).astype("int64")
            cross = t0 + d > (b + 1) * L + 1e-9
            if cross.any():
                k = np.flatnonzero(cross)[0]
                raise ValueError(
                    f"time_step_duration: the step at hour {hours[k]} (start "
                    f"{t0[k]}, duration {d[k]}) crosses a block boundary of "
                    f"length {L} h."
                )
            dim = block_dim(L)
            nb = int(b.max()) + 1
            coord = np.arange(nb, dtype="int64")
            first = np.searchsorted(b, coord)  # b is non-decreasing
            as_block = lambda v: xr.DataArray(  # noqa: E731
                v, dims=dim, coords={dim: coord}
            )
            maps["block_of_hour"][L] = xr.DataArray(
                b, dims="hour", coords={"hour": hours}, name=dim
            )
            maps["duration"][L] = as_block(
                np.bincount(b, weights=d, minlength=nb)
            )
            maps["first_hour"][L] = as_block(hours[first])
            maps["first_step_duration"][L] = as_block(d[first])
            maps["start"][L] = as_block(t0[first])
    tb = TimeBlocks(
        tech_length,
        resource_length,
        **maps,
        step=p["time_step_duration"] if lengths else None,
    )
    if tb.active and why_hourly:
        warnings.warn(
            "Time blocks give no gain on these balances: "
            + "; ".join(why_hourly)
            + ". Force a length with resource_time_block_hrs.",
            UserWarning,
            stacklevel=2,
        )
    return tb


def _derive_balance_lengths(p: Dataset, tech_length: dict, forced: DataArray):
    """
    Balance length per resource (section 4.3) with the nesting refusals.

    Returns the length array (forced lengths override the derived ones) and
    the why-hourly messages. Raises R4, R5 and R10.
    """
    if not any(
        np.isfinite(a.values).any() for a in (*tech_length.values(), forced)
    ):
        return forced, []  # T0: nothing to derive, nothing to read
    res = [str(r) for r in forced["resource"].values]
    found = _contributors(p, tech_length)
    values = forced.values.copy()
    why = []
    for i, r in enumerate(res):
        hourly = [c for c in found[r] if np.isnan(c[2])]
        block = [c for c in found[r] if not np.isnan(c[2])]
        if np.isfinite(values[i]):
            Lr = int(values[i])
            for kind, name, L in block:
                if int(L) % Lr and Lr % int(L):
                    raise ValueError(
                        f"{r}: {kind} {name} ({int(L)} h) and the forced "
                        f"balance length ({Lr} h) do not nest (neither "
                        f"divides the other)."
                    )
        elif block and hourly:
            why.append(f"{r}: hourly because of {hourly[0][0]} {hourly[0][1]}")
        elif block:
            Lr = min(int(c[2]) for c in block)
            for kind, name, L in block:
                if int(L) % Lr:
                    raise ValueError(
                        f"{r}: {kind} {name} ({int(L)} h) is not a multiple "
                        f"of the smallest block length of the balance "
                        f"({Lr} h)."
                    )
            values[i] = Lr
    out = xr.DataArray(values, dims="resource", coords={"resource": res})
    if _on(p, "turpe") and "electricity" in res:
        if np.isfinite(out.sel(resource="electricity").item()):
            raise ValueError(
                "electricity: a coarse balance (forced or derived) cannot be "
                "priced by turpe, which needs the hourly profile of imports."
            )
    if len(res) and np.isfinite(values).all():
        raise ValueError(
            "Every resource balance is coarse: keep at least one hourly "
            "resource (an idle one is enough)."
        )
    return out, why


def check_block_daily_cap(p: Dataset, tb: TimeBlocks) -> None:
    """R6: a block length must nest with the day when a daily cap is set."""
    if not tb.active or "conversion_max_daily_production" not in p:
        return
    d = (
        int(p["operation_day_duration"])
        if "operation_day_duration" in p
        else 24
    )
    cap = np.isfinite(p["conversion_max_daily_production"])
    cap = cap.any([x for x in cap.dims if x != "conversion_tech"])
    for tech in (
        tb.tech_length["conversion"]
        .dropna("conversion_tech")["conversion_tech"]
        .values
    ):
        L = int(tb.tech_length["conversion"].sel(conversion_tech=tech))
        capped = (
            bool(cap.sel(conversion_tech=tech))
            if "conversion_tech" in cap.dims
            else bool(cap)
        )
        if capped and L % d and d % L:
            raise ValueError(
                f"{tech}: block length {L} h neither divides nor is a "
                f"multiple of operation_day_duration ({d} h), with a "
                f"conversion_max_daily_production."
            )


def check_block_flexdem(p: Dataset, tb: TimeBlocks) -> None:
    """R9: flexdem windows and offset must be unions of the balance blocks."""
    if not (tb.active and _on(p, "flexdem") and "flexdem_demand" in p):
        return
    fd = p["flexdem_demand"] != 0
    hrs = (
        p["flexdem_conservation_hrs"]
        if "flexdem_conservation_hrs" in p
        else None
    )
    offset = (
        float(p["flexdem_window_offset_hrs"])
        if "flexdem_window_offset_hrs" in p
        else 0.0
    )
    offset = 0.0 if np.isnan(offset) else offset  # NaN reads as 0 (flexdem.py)
    for r in tb.resource_length["resource"].values:
        L = tb.resource_length.sel(resource=r).item()
        if not np.isfinite(L):
            continue
        L = int(L)
        sel = lambda x: x.sel(resource=r) if "resource" in x.dims else x  # noqa: E731
        if not bool(sel(fd).any()):
            continue
        if hrs is not None:
            for w in np.unique(sel(hrs).values):
                if w > 0 and np.isfinite(w) and w % L:
                    raise ValueError(
                        f"{r}: flexdem_conservation_hrs = {w} is not a "
                        f"multiple of the balance block length {L} h."
                    )
        if offset % L:
            raise ValueError(
                f"{r}: flexdem_window_offset_hrs = {offset} is not a "
                f"multiple of the balance block length {L} h."
            )


def block_mean(x: DataArray, tb: TimeBlocks, L: int) -> DataArray:
    """Duration-weighted mean of ``x`` over the blocks of ``L`` hours (section 4.2)."""
    if "hour" not in x.dims:
        return x
    total = (tb.step * x).groupby(tb.block_of_hour[L]).sum(skipna=False)
    return total / tb.duration[L]


def _check_group(group: DataArray, T_L: str | None) -> None:
    """H3: a groupby array is 1-D over ``hour`` or the block dimension."""
    assert group.ndim == 1 and group.dims in {("hour",), (T_L,)}, (
        group.dims
    )


def _check_axis(expr, dim: str, ref: pd.Index) -> None:
    """M1: linopy's groupby and ``isel`` act by position, so the axis must be the map's."""
    assert expr.indexes[dim].equals(ref), (
        f"{dim} of the expression is not the time map's axis"
    )


def change_resolution(expr, tb: TimeBlocks, src: int | None, dst: int | None):
    """
    Energy-per-step expression from resolution ``src`` to ``dst`` (section 7.1).

    ``None`` is hourly, ``L`` the blocks of ``L`` hours. Same resolution:
    ``expr`` itself. Hourly to blocks and finer to coarser blocks sum; blocks
    to hourly and coarser to finer blocks broadcast the duration share.
    """
    if src == dst:
        out = expr
    elif src is None:
        group = tb.block_of_hour[dst]
        _check_group(group, None)
        _check_axis(expr, "hour", group.indexes["hour"])
        out = expr.groupby(group).sum()
    elif dst is not None and dst > src:  # coarser row: sum its blocks
        group = (
            tb.block_of_hour[dst]
            .sel(hour=tb.first_hour[src])
            .drop_vars("hour")
        )
        _check_group(group, block_dim(src))
        _check_axis(expr, block_dim(src), tb.duration[src].indexes[block_dim(src)])
        out = expr.groupby(group).sum()
    else:  # hourly or finer row: broadcast the share of the source block
        t_src = block_dim(src)
        if dst is None:
            idx, row_duration = tb.block_of_hour[src], tb.step
        else:
            idx = (
                tb.block_of_hour[src]
                .sel(hour=tb.first_hour[dst])
                .drop_vars("hour")
            )
            row_duration = tb.duration[dst]
        _check_group(idx, "hour" if dst is None else block_dim(dst))
        _check_axis(expr, t_src, tb.duration[src].indexes[t_src])
        w = (
            row_duration
            / tb.duration[src].isel({t_src: idx}).drop_vars(t_src)
        )
        out = expr.isel({t_src: idx}).drop_vars(t_src) * w
    levels = {
        n
        for i in out.indexes.values()
        if isinstance(i, pd.MultiIndex)
        for n in i.names
    }
    assert set(out.coords) <= set(out.dims) | levels, set(out.coords)
    return out


def add_to_balances(adequacy, expr, tb: TimeBlocks, src: int | None = None) -> None:
    """
    Add ``expr`` to the adequacy constraint of every resource it holds.

    Each resource goes to the balance of its length (hourly: ``adequacy``),
    converted from ``src`` by ``change_resolution`` and joined by label.
    Without blocks it is today's ``adequacy.lhs += expr``.
    """
    if not tb.active:
        adequacy.lhs += expr
        return
    routed = []
    for dst in [None, *tb.balance_lengths]:
        res = [r for r in tb.resources(dst) if r in expr.indexes["resource"]]
        if not res:
            continue
        routed += res
        sub = expr.sel(resource=res)
        if src != dst:
            # I1: a slice with no live term and no constant adds nothing but dead terms
            # (a bare variable has labels and no constant)
            labels = sub.vars if hasattr(sub, "vars") else sub.labels
            live = (labels != -1).any([d for d in labels.dims if d != "resource"])
            if hasattr(sub, "const"):
                live = live | (sub.const != 0).any(
                    [d for d in sub.const.dims if d != "resource"]
                )
            if not live.any():
                continue
            sub = sub.sel(resource=live.indexes["resource"][live.values])
        part = change_resolution(sub, tb, src, dst)
        con = (
            adequacy
            if dst is None
            else adequacy.model.constraints[
                f"operation_adequacy_block_{dst}h_constraint"
            ]
        )
        for d in set(part.dims) & set(con.lhs.dims) - {"_term"}:
            # a left join would drop extra labels silently
            assert part.indexes[d].isin(con.lhs.indexes[d]).all(), (
                con.name,
                d,
            )
        con.lhs = con.lhs.add(part, join="left")  # by label, never position
    assert sorted(routed) == sorted(
        expr.indexes["resource"]
    ), "resource not routed exactly once"


def check_block_dims(m) -> None:
    """
    H5: no term carries a dimension its row lacks, no object mixes resolutions.

    Raises ``RuntimeError`` naming the object when a constraint's ``vars``
    dims are not within its labels' dims plus ``_term``, or when a variable
    or a constraint (labels or vars) carries ``hour`` together with a
    ``time_block_*`` dimension, or two different ``time_block_*`` dimensions.
    Reads dimension names only.
    """
    for kind, items in (("variable", m.variables), ("constraint", m.constraints)):
        for name in items:
            obj = items[name]
            dims = [set(obj.labels.dims)]
            if kind == "constraint":
                extra = set(obj.vars.dims) - set(obj.labels.dims) - {"_term"}
                if extra:
                    raise RuntimeError(
                        f"{kind} {name}: terms carry {sorted(extra)}, "
                        f"which its rows lack (H5)."
                    )
                dims.append(set(obj.vars.dims))
            for d in dims:
                blocks = {x for x in d if x.startswith("time_block_")}
                if blocks and ("hour" in d or len(blocks) > 1):
                    raise RuntimeError(
                        f"{kind} {name}: dimensions {sorted(d)} mix "
                        f"resolutions (H5)."
                    )


def check_group_coord(obj, ref: pd.Index, dim: str) -> None:
    """
    H1/H2 guard: ``obj`` (a variable or a constraint) carries the group's own
    coordinate along ``dim``, order included.

    Raises ``RuntimeError`` otherwise: a partner restricted in another order
    or a forgotten restriction would be paired by position, or grow the group.
    """
    if not obj.indexes[dim].equals(ref):
        raise RuntimeError(
            f"{getattr(obj, 'name', obj)}: {dim} is not the group's "
            f"coordinate (H1/H2): {list(obj.indexes[dim])} != {list(ref)}."
        )


def check_partition(
    parts: dict[int | None, pd.Index], full: pd.Index
) -> dict[int | None, pd.Index]:
    """
    The groups partition the full index (C11): disjoint, union equal to it.

    Raises ``AssertionError`` otherwise (H10). Returns the non-empty parts,
    in order, so that an empty group is never built.
    """
    live = {key: part for key, part in parts.items() if len(part)}
    members = [x for part in live.values() for x in part]
    assert len(members) == len(set(members)), "groups overlap"
    assert set(members) == set(full), "groups do not cover the full index"
    return live
