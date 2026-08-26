"""Module with function to post_process results and produce summaries."""

from typing import Literal

import pandas as pd
from pandas import DataFrame
from xarray import DataArray, Dataset


def simplify_from_postfix(
    dataset: DataFrame | Dataset, postfixes: list[str], name: str
) -> Dataset:
    """
    Simplify dataset dimensions based on given postfixes.

    For each postfix in `postfixes`, the function checks the dimensions of a dataset.
    If a dimension ends with the postfix, it renames it to just the postfix.
    If no matching dimension exists, it adds a dimension with the postfix and fills
    it with the given `name`.

    Args:
        dataset (Union[pandas.DataFrame, xarray.Dataset]): Input dataset to
            process.
        postfixes (list of str): List of postfixes to check.
        name (str): The value to use for new dimensions if created.

    Returns:
        xarray.Dataset: Processed dataset with simplified dimensions.

    Raises:
        ValueError: If more than one dimension matches a given postfix.
    """
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.squeeze().to_xarray()
    for postfix in postfixes:
        tech_dims = [dim for dim in dataset.dims if str(dim).endswith(postfix)]
        if len(tech_dims) == 0:
            dataset = dataset.expand_dims({postfix: [name]})
        elif len(tech_dims) == 1:
            dataset = dataset.rename({tech_dims[0]: postfix})
        else:
            raise ValueError(
                f"Dataset has more than one dimension ending with '{postfix}'."
                f"Unable to proceed."
            )

    return dataset


def reindex_by_area(
    dataset: pd.DataFrame | Dataset,
    transport_area_from: DataArray,
    transport_area_to: DataArray,
) -> pd.DataFrame:
    """
    Reindex a dataset by areas based on transport mappings.

    Transforms a dataset initially indexed by "link" into a dataset indexed by "area".
    Index associations are described in `transport_area_from` and `transport_area_to`.

    Args:
        dataset (Union[pandas.DataFrame, xarray.Dataset]): Input dataset to
            reindex.
        transport_area_from (xarray.DataArray): Mapping from "link" to
            "area_from".
        transport_area_to (xarray.DataArray): Mapping from "link" to "area_to".

    Returns:
        pandas.DataFrame: Reindexed dataset.

    Examples:
        >>> exchange = solution.operation_transport_power_capacity.\
        to_dataframe().groupby(['link', 'transport_tech', 'year_op']).sum()
        >>> reindex_by_area(exchange, parameters['transport_area_from'],\
        parameters['transport_area_to'])
    """
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.squeeze().to_xarray()
    input_name = dataset.name
    indexes = list(dataset.coords)
    indexes_without_link = [x for x in indexes if x != "link"]
    if "transport_tech" in transport_area_from.coords:
        reindexed_dataset = dataset.assign_coords(
            area_from=(["link", "transport_tech"], transport_area_from.data),
            area_to=(["link", "transport_tech"], transport_area_to.data),
        )
    else:
        reindexed_dataset = dataset.assign_coords(
            area_from=(["link"], transport_area_from.data),
            area_to=(["link"], transport_area_to.data),
        )

    reindexed_dataset_s = (
        reindexed_dataset.to_dataframe()
        .reset_index()
        .set_index(["area_from", "area_to"] + indexes_without_link)[
            [input_name]
        ]
    )

    return reindexed_dataset_s


def aggregate_link_data(
    dataset: DataFrame,
    stacked_into_transport_tech: bool = False,
    output_name: str | None = None,
) -> DataFrame:
    """
    Aggregate dataset by areas, computing incoming and outgoing flows.

    Sums incoming and outgoing power (capacity or energy flow per hour).
    Results can be optionally stacked into a single column with an indicator for
    direction.

    Args:
        dataset (pandas.DataFrame): Input dataset indexed by "area_from" and
            "area_to".
        stacked_into_transport_tech (bool, optional): Whether to stack the results
            into one column. Defaults to False.
        output_name (str, optional): Custom output column name. Defaults to None.

    Returns:
        pandas.DataFrame: Aggregated dataset.

    Examples:
        >>> exchange = reindex_by_area(dataset, transport_area_from, transport_area_to)
        >>> aggregate_link_data(exchange)
        >>> aggregate_link_data(exchange, stacked_into_transport_tech=True)
    """
    input_name = dataset.columns[0]
    if output_name is None:
        output_name = input_name
    incoming_power = dataset.groupby(
        [
            col
            for col in dataset.reset_index().columns
            if (col != "area_to" and col != input_name)
        ]
    ).sum()
    incoming_power = incoming_power.rename(
        columns={input_name: output_name + "_total_incoming"}
    ).rename_axis(index={"area_from": "area"})
    outgoing_power = dataset.groupby(
        [
            col
            for col in dataset.reset_index().columns
            if (col != "area_from" and col != input_name)
        ]
    ).sum()
    outgoing_power = outgoing_power.rename(
        columns={input_name: output_name + "_total_outgoing"}
    ).rename_axis(index={"area_to": "area"})
    aggregated_reindexed_dataset = pd.merge(
        outgoing_power, incoming_power, left_index=True, right_index=True
    )

    if stacked_into_transport_tech:
        aggregated_reindexed_dataset = (
            aggregated_reindexed_dataset.stack().reset_index()
        )
        indexes = list(aggregated_reindexed_dataset.columns[:-2])
        aggregated_reindexed_dataset.columns = indexes + [
            "direction",
            output_name,
        ]
        aggregated_reindexed_dataset["transport_tech"] = (
            aggregated_reindexed_dataset["transport_tech"]
            + "_"
            + aggregated_reindexed_dataset["direction"].map(get_last_word)
        )
        aggregated_reindexed_dataset = aggregated_reindexed_dataset.drop(
            columns=["direction"]
        ).set_index(indexes)

    return aggregated_reindexed_dataset


def get_previous_word(input_str: str, word: str) -> str:
    """
    Removes a specified suffix from the input string, if present.

    Args:
        input_str (str): The input string to process.
        word (str): The suffix to remove from the input string.

    Returns:
        str: The modified string without the specified suffix.

    Examples:
        >>> get_previous_word("transport_tech", "tech")
        'transport'
    """
    if input_str.endswith("_" + word):
        input_str = input_str[: -(len(word) + 1)]
    return input_str


def calculate_total_power_capacity(
    solution: Dataset,
    parameters: Dataset,
    by_year_op: bool = True,
    by_area: bool = True,
) -> DataFrame:
    """
    Calculate total power capacity, aggregated by year or area.

    Computes total power capacities from the given solution and parameters,
    with options to group by `year_op` or `area`.

    Args:
        solution (xarray.Dataset): The solution dataset containing power
            capacity data.
        parameters (xarray.Dataset): Model parameters to be used in
        calculations.
        by_year_op (bool, optional): Whether to aggregate by operational year.
            Defaults to True.
        by_area (bool, optional): Whether to aggregate by area. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing total power capacities.

    Examples:
        >>> calculate_total_power_capacity(solution, parameters)
        >>> calculate_total_power_capacity(solution, parameters, by_area=False)
        >>> calculate_total_power_capacity(solution, parameters, by_year_op=False)
    """
    return calculate_total_of_a_type(
        "power_capacity",
        solution,
        parameters,
        by_year_op=by_year_op,
        by_area=by_area,
    )


def calculate_total_costs(
    solution: Dataset,
    parameters: Dataset,
    by_year_op: bool = True,
    by_area: bool = True,
) -> DataFrame:
    """
    Compute total costs, optionally aggregated by year or area.

    Splits costs into operation and planning, with options for further aggregation.

    Args:
        solution (xarray.Dataset): The solution dataset containing cost data.
        parameters (xarray.Dataset): Model parameters to be used in
        calculations.
        by_year_op (bool, optional): Whether to aggregate by operational year.
            Defaults to True.
        by_area (bool, optional): Whether to aggregate by area. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing total costs, indexed by
        operation or planning.

    Examples:
        >>> calculate_total_costs(solution, parameters)
        >>> calculate_total_costs(solution, parameters, by_area=False)
    """
    total_costs = {
        "operation": calculate_total_of_a_type(
            "costs",
            solution,
            parameters,
            operation_or_planning="operation",
            by_year_op=by_year_op,
            by_area=by_area,
        ),
        "planning": calculate_total_of_a_type(
            "costs",
            solution,
            parameters,
            operation_or_planning="planning",
            by_year_op=by_year_op,
            by_area=by_area,
        ),
    }

    total_costs["planning"] = total_costs["planning"].reorder_levels(
        order=total_costs["operation"].index.names
    )

    return pd.concat(
        total_costs.values(),
        keys=total_costs.keys(),
        names=["operation_or_planning"],
    )


def calculate_total_net_generation(
    solution: Dataset,
    parameters: Dataset,
    by_year_op: bool = True,
    by_area: bool = True,
) -> DataFrame:
    """
    Calculate total net generation, optionally aggregated by year or area.

    Args:
        solution (xarray.Dataset): The solution dataset containing net
            generation data.
        parameters (xarray.Dataset): Model parameters to be used in
        calculations.
        by_year_op (bool, optional): Whether to aggregate by operational year. Defaults to True.
        by_area (bool, optional): Whether to aggregate by area. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing total net generation.

    Examples:
        >>> calculate_total_net_generation(solution, parameters)
        >>> calculate_total_net_generation(solution, parameters, by_area=False)
    """
    return calculate_total_of_a_type(
        "net_generation",
        solution,
        parameters,
        by_year_op=by_year_op,
        by_area=by_area,
    )


def calculate_total_emissions(
    solution: Dataset,
    parameters: Dataset,
    by_year_op: bool = True,
    by_area: bool = True,
) -> DataFrame:
    """
    Compute total emissions, optionally aggregated by year or area.

    Args:
        solution (xarray.Dataset): The solution dataset containing emission
        data.
        parameters (Dataset): Model parameters to be used in calculations.
        by_year_op (bool, optional): Whether to aggregate by operational year.
            Defaults to True.
        by_area (bool, optional): Whether to aggregate by area. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing total emissions.

    Examples:
        >>> calculate_total_emissions(solution, parameters)
        >>> calculate_total_emissions(solution, parameters, by_area=False)
    """
    return calculate_total_of_a_type(
        "emissions",
        solution,
        parameters,
        by_year_op=by_year_op,
        by_area=by_area,
    )


def get_last_word(input_str: str, sep: str = "_") -> str:
    """
    Extract the last word from a string based on a given separator.

    Args:
        input_str (str): The input string to process.
        sep (str): The separator to split the string. Defaults to "_".

    Returns:
        str: The last word in the string, or the entire string if no separator is found.

    Examples:
        >>> get_last_word("transport_tech")
        'tech'
        >>> get_last_word("area_from")
        'from'
    """
    words = input_str.split(sep=sep)
    return words[-1] if words else None


def get_sum_dims(
    variable: DataArray, by_year_op: bool = True, by_area: bool = True
) -> list[str]:
    """
    Determine the dimensions to sum over for a variable.

    Constructs a list of dimensions to aggregate based on the provided flags.

    Args:
        variable (xarray.DataArray): The variable to analyze.
        by_year_op (bool, optional): Whether to exclude "year_op" from the sum
            dimensions. Defaults to True.
        by_area (bool, optional): Whether to exclude "area" from the sum dimensions.
            Defaults to True.

    Returns:
        list of str: A list of dimensions to sum over.

    Examples:
        >>> get_sum_dims(variable, by_year_op=False, by_area=True)
        ['year_inv', 'hour', 'area']
    """
    sum_dims = []
    if "year_inv" in variable.dims:
        sum_dims.append("year_inv")
    if "hour" in variable.dims:
        sum_dims.append("hour")
    if not by_year_op and "year_op" in variable.dims:
        sum_dims.append("year_op")
    if not by_area and "area" in variable.dims:
        sum_dims.append("area")
    return sum_dims


def calculate_total_of_a_type(
    type: str,
    solution: Dataset,
    parameters: Dataset,
    operation_or_planning: Literal["operation", "planning"] = "operation",
    by_year_op: bool = True,
    by_area: bool = True,
) -> DataFrame | None:
    r"""
    Calculate totals of a specific type, optionally aggregated by year or area.

    This function calculates total power capacity or total net_generation or
    total costs depending on the value of type and over variables prefixed with
    operation_or_planning. Re-indexation of link index is performed, and
    variables names are simplified to fit in a simple panda table by default
    the sum is computed by year_op and by area but two parameters allow
    to modify that.

    Note:
        - `type` can be 'power_capacity'
        - `operation_or_planning` should be "operation" or "planning"
        - not possible : type=="net_generation" and
            `operation_or_planning` == "planning"

        It replaces
        - get_capacities(solution, model_parameters)
            with calculate_total_of_a_type("power_capacity", solution,
            model_parameters)
        - get_net_generation(solution, model_parameters)
            with calculate_total_of_a_type("net_generation", solution
            model_parameters)
        - get_costs(solution, model_parameters)
            with calculate_total_of_a_type("costs", solution,
            model_parameters,operation_or_planning="operation")
            of operation_or_planning="planning" for planning costs

    Args:
        type (str): The type of variable to sum (e.g., "power_capacity", "costs",
            "emissions").
        solution (xarray.Dataset): The solution dataset containing the
            variables.
        parameters (xarray.Dataset): Model parameters to use for reindexing
        and mappings.
        operation_or_planning (str, optional): Indicates operation or planning.
            Defaults to "operation".
        by_year_op (bool, optional): Whether to aggregate by operational year.
            Defaults to True.
        by_area (bool, optional): Whether to aggregate by area. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing aggregated totals.

    Raises:
        KeyError: If a required variable or dimension is missing.

    Examples:
        >>> calculate_total_of_a_type("power_capacity", solution, parameters)
        >>> calculate_total_of_a_type("costs", solution, parameters,
        \operation_or_planning="planning")
    """
    try:
        variables_of_type_type = [
            var
            for var in solution.data_vars
            if str(var).startswith(operation_or_planning + "_")
            and str(var).endswith("_" + type)
        ]
        calculated_total_dict = {}
        for var in variables_of_type_type:
            variable_name = get_previous_word(
                str(var).replace(operation_or_planning + "_", ""), type
            )
            sum_dims = get_sum_dims(solution[var], by_year_op, by_area)

            calculated_total = solution[var].sum(dim=sum_dims)
            if type == "costs" and variable_name == "net_import":
                calculated_total = calculated_total.rename(
                    {"resource": "tech"}
                )

            if "link" in solution[var].dims:
                calculated_total = reindex_by_area(
                    calculated_total,
                    parameters["transport_area_from"],
                    parameters["transport_area_to"],
                )
                calculated_total = aggregate_link_data(
                    calculated_total,
                    stacked_into_transport_tech=True,
                    output_name=type,
                )

            calculated_total = simplify_from_postfix(
                calculated_total, postfixes=["tech"], name=variable_name
            )
            calculated_total = calculated_total.to_dataframe()
            calculated_total_dict[variable_name] = calculated_total

        combined_df = pd.concat(
            calculated_total_dict.values(),
            keys=calculated_total_dict.keys(),
            names=["type"],
            sort=False,
            join="outer",
            ignore_index=False,
        )
        combined_df.index.set_names(
            ["type"] + list(calculated_total_dict.values())[0].index.names,
            inplace=True,
        )
        combined_df = combined_df.sort_index()

        return combined_df

    except KeyError as e:
        print(e)


def generate_summary_dataframes_from_results(
    solution: Dataset,
    parameters: Dataset,
    by_year_op: bool = True,
    by_area: bool = True,
) -> dict[str, DataFrame]:
    """
    Generate summary DataFrames based on the solution and parameters.

    Creates multiple DataFrames representing different aspects of the solution
    results,
    including costs, emissions, power capacities, and more.

    Args:
        solution (xarray.Dataset): The solution dataset containing study
            results.
        parameters (xarray.Dataset): Model parameters associated with the
        solution.
        by_year_op (bool, optional): Whether to aggregate by operational year.
            Defaults to True.
        by_area (bool, optional): Whether to aggregate by area. Defaults to True.

    Returns:
        dict: A dictionary of pandas DataFrames, each keyed by a descriptive
        name.

    Examples:
        >>> generate_summary_dataframes_from_results(solution, parameters)
    """
    dataframes = {
        "Operation costs - EUR": solution.annualised_totex,
        "Production capacity - MW": calculate_total_power_capacity(
            solution, parameters, by_year_op=by_year_op, by_area=by_area
        ),
        "Total costs - EUR": calculate_total_costs(
            solution, parameters, by_year_op=by_year_op, by_area=by_area
        ),
        "CO2 emissions - tCO2eq": calculate_total_emissions(
            solution, parameters, by_year_op=by_year_op, by_area=by_area
        ),
        "Production - MWh": calculate_total_net_generation(
            solution, parameters, by_year_op=by_year_op, by_area=by_area
        ),
    }
    return dataframes
