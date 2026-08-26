"""
Module for extracting and processing results from optimization models.

This module provides functions to calculate and return capacities, net generation,
and costs of various assets based on the optimization solution and model parameters.

Functions:
    get_capacities(solution, model_parameters):
        Calculate and return the capacities of various assets.

    get_net_generation(solution, model_parameters):
        Compute the net generation for different categories (e.g., demand, conversion).

    get_costs(solution, model_parameters):
        Calculate and return the costs (operation and planning) for various assets.
"""

from typing import TypeVar

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

XarrayObject = TypeVar("XarrayObject", xr.DataArray, xr.Dataset)


def reindex_from_tech_to_asset(
    x: XarrayObject,
    category: str,
) -> XarrayObject:
    """
    Reindex the given DataArray or Dataset to asset.

    By renaming the "{category}_tech" dimension to "asset".

    This function renames the dimension in the input object from
    "<category>_tech" to "asset", and assigns a new coordinate named
    "category" that has the same dimension as "asset".

    Args:
        x (xarray.DataArray | xarray.Dataset):
            The input xarray object whose dimension will be renamed.
        category (str):
            The category name used to build the dimension name
            "<category>_tech". For instance, if `category` is "foo", the
            dimension to be renamed is "foo_tech".

    Returns:
        xarray.DataArray | xarray.Dataset:
            A new xarray object with the renamed dimension ("asset") and an
            added "category" coordinate. Delete the {category} from all
            variables.

    Example:
        >>> data = xr.DataArray(
        ...     np.random.rand(3),
        ...     dims=["foo_tech"],
        ...     coords={"foo_tech": ["tech1", "tech2", "tech3"]},
        ...     name="some_data"
        ... )
        >>> print(data)
        <xarray.DataArray 'some_data' (foo_tech: 3)>
        array([0.236, 0.887, 0.112])
        Coordinates:
          * foo_tech  (foo_tech) <U5 'tech1' 'tech2' 'tech3'

        >>> result = reindex_from_tech_to_asset(data, category="foo")
        >>> print(result)
        <xarray.DataArray 'some_data' (asset: 3)>
        array([0.236, 0.887, 0.112])
        Coordinates:
          * asset     (asset) <U5 'tech1' 'tech2' 'tech3'
            category  (asset) <U3 'foo' 'foo' 'foo'
    """
    x = x.rename({f"{category}_tech": "asset"})

    def delete_category_in_name(name: str, cat: str) -> str:
        name_list = name.split("_")
        name_list.remove(cat)
        new_name = "_".join(name_list)
        return new_name

    if isinstance(x, xr.DataArray) and category in x.name:
        x = x.rename(delete_category_in_name(str(x.name), category))

    if isinstance(x, xr.Dataset):
        x = x.rename(
            {
                variable: delete_category_in_name(str(variable), category)
                for variable in x.data_vars
                if category in variable
            }
        )

    x = x.assign_coords({"category": xr.DataArray(category, coords=[x.asset])})
    return x


def get_capacities(
    solution: Dataset,
    model_parameters: Dataset,
) -> DataArray:
    """
    Calculate and return the capacities of assets based on the solution.

    Args:
        solution (xarray.Dataset): The solution dataset from an optimization
        model,
            containing operation capacities for combined, conversion, storage, and
            transport technologies.
        model_parameters (xarray.Dataset): The dataset containing model
        parameters.

    Returns:
        xr.DataArray: A concatenated xarray DataArray representing the capacity
            of various assets (combined, conversion, storage, transport)  after
            summing across investment years.
            Dimensions include:
            - asset: Stack of category, capacity type, and technology (tech) for
                each asset type.
            - area
            - year_op
    """
    p = model_parameters

    da_combined, da_conversion, da_storage, da_transport = (
        xr.DataArray(),
    ) * 4

    if "combined" in p.keys() and p.combined:
        da = solution["operation_combined_power_capacity"]
        da = da.copy(deep=True)
        da = da.rename(None)
        da = da.expand_dims(
            category=np.array(["combined"], dtype=str),
            capacity_type=np.array(["power"], dtype=str),
        )
        da = da.rename(combined_tech="tech")
        da = da.stack(asset=["category", "capacity_type", "tech"])
        da_combined = da

    if "conversion" in p.keys() and p.conversion:
        da = solution["operation_conversion_power_capacity"]
        da = da.copy(deep=True)
        da = da.rename(None)
        da = da.expand_dims(
            category=np.array(["conversion"], dtype=str),
            capacity_type=np.array(["power"], dtype=str),
        )
        da = da.rename(conversion_tech="tech")
        da = da.stack(asset=["category", "capacity_type", "tech"])
        da_conversion = da

    if "storage" in p.keys() and p.storage:
        ds = solution[
            [
                "operation_storage_power_capacity",
                "operation_storage_energy_capacity",
            ]
        ]
        ds = ds.copy(deep=True)
        ds = ds.expand_dims(category=np.array(["storage"], dtype=str))
        ds = ds.rename(
            storage_tech="tech",
            operation_storage_power_capacity="power",
            operation_storage_energy_capacity="energy",
        )
        da = ds.to_dataarray(dim="capacity_type", name=None)
        da = da.stack(asset=["category", "capacity_type", "tech"])
        da_storage = da

    if "transport" in p.keys() and p.transport:
        da = solution["operation_transport_power_capacity"]
        da = da.copy(deep=True)
        da = da.rename(None)
        da = da.expand_dims(capacity_type=np.array(["power"], dtype=str))
        da = da.assign_coords(
            area_from=(["link", "transport_tech"], p.transport_area_from.data),
            area_to=(["link", "transport_tech"], p.transport_area_to.data),
        )
        da_imports = da.groupby(["area_from", "transport_tech"]).sum()
        da_imports = da_imports.expand_dims(
            category=np.array(["transport_import"], dtype=str)
        )
        da_imports = da_imports.rename(area_from="area")
        da_exports = da.groupby(["area_to", "transport_tech"]).sum()
        da_exports = da_exports.expand_dims(
            category=np.array(["transport_export"], dtype=str)
        )
        da_exports = da_exports.rename(area_to="area")

        da = xr.combine_by_coords([da_imports, da_exports])

        da = da.rename(transport_tech="tech")
        da = da.stack(asset=["category", "capacity_type", "tech"])
        da_transport = da

    da = xr.concat(
        [
            d
            for d in [da_combined, da_conversion, da_storage, da_transport]
            if not d.isnull().all()
        ],
        dim="asset",
    )

    return da


def get_net_generation(
    solution: Dataset, model_parameters: Dataset
) -> xr.DataArray:
    """
    Compute the net generation for different assets based on the solution.

    Args:
        solution (xarray.Dataset): The solution dataset from an optimization
        model.
        model_parameters (xarray.Dataset): The dataset containing model
        parameters.

    Returns:
        xr.DataArray: A concatenated DataArray representing the net generation for
            various categories
            (e.g., demand, load shedding, spillage, conversion, combined).
            Dimensions include:
            - category: Different types of generation or consumption
                (e.g., demand, spillage).
            - other relevant dimensions depending on the data.
    """
    p = model_parameters
    s = solution

    demand = (-1 * p.demand).expand_dims(category=["demand"])
    load_shedding = s.operation_load_shedding.expand_dims(
        category=["load_shedding"]
    )
    spillage = (-1 * s.operation_spillage).expand_dims(category=["spillage"])
    conversion = s.operation_conversion_net_generation.rename(
        dict(conversion_tech="category")
    )
    combined = xr.DataArray([], coords=dict(category=[]))
    storage = xr.DataArray([], coords=dict(category=[]))
    net_import = xr.DataArray([], coords=dict(category=[]))
    transport = xr.DataArray([], coords=dict(category=[]))

    if "combined" in p.keys() and p.combined.any():
        combined = s.operation_combined_net_generation.rename(
            dict(combined_tech="category")
        )
    if "storage" in p.keys() and p.storage.any():
        storage = s.operation_storage_net_generation.rename(
            dict(storage_tech="category")
        )
    if "net_import" in p.keys() and p.net_import.any():
        net_import = s.operation_net_import_net_generation.expand_dims(
            category=["net_import"]
        )
    if "transport" in p.keys() and p.transport.any():
        transport = s.operation_transport_net_generation.rename(
            dict(transport_tech="category")
        )

    da = xr.concat(
        [
            demand,
            load_shedding,
            spillage,
            combined,
            conversion,
            storage,
            net_import,
            transport,
        ],
        dim="category",
        coords="all",
        join="outer",
        compat="broadcast_equals",
    )
    return da


def get_costs(solution: Dataset, model_parameters: Dataset) -> xr.DataArray:
    """
    Calculate and return the costs (operation and planning).

    Args:
        solution (xarray.Dataset): The solution dataset from an optimization
        model.
        model_parameters (xarray.Dataset): The dataset containing model
        parameters.

    Returns:
        xr.DataArray: A concatenated DataArray representing the costs of various
            assets (combined, conversion,
            storage, transport, net import).
            Dimensions include:
            - asset: Stack of category, cost type, and technology (tech) for each
                asset type.
            - area
            - year_op
    """
    p = model_parameters

    da_combined, da_conversion, da_net_import, da_storage, da_transport = (
        xr.DataArray(),
    ) * 5

    if "combined" in p.keys() and p.conversion:
        ds = solution[["operation_combined_costs", "planning_combined_costs"]]
        ds = ds.copy()
        ds = ds.expand_dims(category=np.array(["combined"], dtype=str))
        ds = ds.rename(
            combined_tech="tech",
            operation_combined_costs="operation",
            planning_combined_costs="planning",
        )
        da = ds.to_dataarray(dim="cost_type", name=None)
        da = da.stack(asset=["category", "cost_type", "tech"])
        da_combined = da

    if "conversion" in p.keys() and p.conversion:
        ds = solution[
            ["operation_conversion_costs", "planning_conversion_costs"]
        ]
        ds = ds.copy()
        ds = ds.expand_dims(category=np.array(["conversion"], dtype=str))
        ds = ds.rename(
            conversion_tech="tech",
            operation_conversion_costs="operation",
            planning_conversion_costs="planning",
        )
        da = ds.to_dataarray(dim="cost_type", name=None)
        da = da.stack(asset=["category", "cost_type", "tech"])
        da_conversion = da

    if "net_import" in p.keys() and p.net_import:
        da = solution["operation_net_import_costs"]
        da = da.copy()
        da = da.expand_dims(
            category=np.array(["net_import"], dtype=str),
            cost_type=np.array(["operation"], dtype=str),
        )
        da = da.rename(resource="tech")
        da = da.rename(None)
        da = da.stack(asset=["category", "cost_type", "tech"])
        da_net_import = da

    if "storage" in p.keys() and p.storage:
        ds = solution[["operation_storage_costs", "planning_storage_costs"]]
        ds = ds.copy()
        ds = ds.expand_dims(category=np.array(["storage"], dtype=str))
        ds = ds.rename(
            storage_tech="tech",
            operation_storage_costs="operation",
            planning_storage_costs="planning",
        )
        da = ds.to_dataarray(dim="cost_type", name=None)
        da = da.stack(asset=["category", "cost_type", "tech"])
        da_storage = da

    if "transport" in p.keys() and p.transport:
        ds = solution[
            ["operation_transport_costs", "planning_transport_costs"]
        ]
        ds = ds.copy()
        ds = ds.expand_dims(category=np.array(["transport"], dtype=str))
        ds = ds.rename(
            transport_tech="tech",
            operation_transport_costs="operation",
            planning_transport_costs="planning",
        )
        da = ds.to_dataarray(dim="cost_type", name=None)
        da = da.stack(asset=["category", "cost_type", "tech"])
        da_transport = da

    da = xr.concat(
        [
            d
            for d in [
                da_combined,
                da_conversion,
                da_net_import,
                da_storage,
                da_transport,
            ]
            if not d.isnull().all()
        ],
        dim="asset",
    )

    return da
