import itertools

import numpy as np
import pytest

from pommes.io.calculate_summaries import (
    calculate_total_of_a_type,
    generate_summary_dataframes_from_results,
)
from pommes.model.build_model import build_model

try:  # the sparse solution stores some variables over "<tech>_unit" dims
    from pommes.model.sparsity import densify_any
except ImportError:  # dense pommes-milp: nothing to unstack

    def densify_any(data):
        return data


TYPES = ["power_capacity", "net_generation", "costs", "emissions"]


@pytest.fixture(scope="module")
def solved(parameters):
    model = build_model(parameters)
    model.solve(solver_name="highs")
    return model.solution


def test_test_case(parameters, solved):
    combinations = itertools.product(
        TYPES, ["operation", "planning"], [True, False], [True, False]
    )
    for type_value, operation_or_planning, by_year_op, by_area in combinations:
        # net generation and emissions have no planning variables
        if type_value in ("net_generation", "emissions") and (
            operation_or_planning == "planning"
        ):
            continue
        table = calculate_total_of_a_type(
            type=type_value,
            solution=solved,
            parameters=parameters,
            operation_or_planning=operation_or_planning,
            by_year_op=by_year_op,
            by_area=by_area,
        )
        assert not table.empty, (
            type_value,
            operation_or_planning,
            by_year_op,
            by_area,
        )

    summaries = generate_summary_dataframes_from_results(solved, parameters)
    assert all(table is not None for table in summaries.values())


def _row_labels_are_names(table, parameters, tech_of_type):
    """Every area, tech and year_op label of every row is what it names."""
    areas = set(parameters.area.values.tolist())
    years = set(parameters.year_op.values.tolist())
    for row in table.index:
        labels = dict(zip(table.index.names, row))
        assert labels["area"] in areas, labels
        assert labels["year_op"] in years, labels
        assert labels["tech"] in tech_of_type(labels["type"]), labels


def _tech_of_type(parameters, type_value, resource_for_net_import):
    techs = {
        "combined": parameters.combined_tech.values,
        "conversion": parameters.conversion_tech.values,
        "storage": parameters.storage_tech.values,
        "transport": parameters.transport_tech.values,
    }
    if type_value == "net_import" and resource_for_net_import:
        return set(parameters.resource.values.tolist())
    names = techs.get(type_value, [type_value])  # no tech dimension: its name
    return set(names.tolist() if hasattr(names, "tolist") else names)


def _solution_sum(solved, variable, **labels):
    data = densify_any(solved[variable])
    other = [d for d in data.dims if d not in labels and d != "link"]
    return float(data.sum(other).sel(**labels))


def test_row_labels_are_names(parameters, solved):
    tables = {
        t: calculate_total_of_a_type(t, solved, parameters)
        for t in ["costs", "net_generation", "power_capacity"]
    }

    def power_capacity_techs(type_value):
        names = _tech_of_type(parameters, type_value, False)
        if type_value == "transport":  # incoming / outgoing, per link end
            return {
                f"{n}_{d}" for n in names for d in ("incoming", "outgoing")
            }
        return names

    _row_labels_are_names(
        tables["costs"],
        parameters,
        lambda type_value: _tech_of_type(parameters, type_value, True),
    )
    _row_labels_are_names(
        tables["net_generation"],
        parameters,
        lambda type_value: _tech_of_type(parameters, type_value, False),
    )
    _row_labels_are_names(
        tables["power_capacity"], parameters, power_capacity_techs
    )

    # One non-zero conversion row per table equals the solution summed the
    # same way (over every dimension but area, tech, year_op and resource).
    # The operation costs of the conversion technologies are 0 in this case:
    # take its planning costs.
    tables["planning costs"] = calculate_total_of_a_type(
        "costs", solved, parameters, "planning"
    )
    for key, column in [
        ("planning costs", "planning_conversion_costs"),
        ("net_generation", "operation_conversion_net_generation"),
        ("power_capacity", "operation_conversion_power_capacity"),
    ]:
        conversion = (
            tables[key].xs("conversion", level="type")[column].dropna()
        )
        labels = conversion.abs().idxmax()
        row = dict(zip(conversion.index.names, labels))
        expected = _solution_sum(
            solved,
            column,
            area=row["area"],
            conversion_tech=row["tech"],
            year_op=row["year_op"],
            **({"resource": row["resource"]} if "resource" in row else {}),
        )
        assert expected != 0, key
        assert conversion.loc[labels] == pytest.approx(expected)


@pytest.mark.parametrize("operation_or_planning", ["operation", "planning"])
def test_power_capacity_without_area(
    parameters, solved, operation_or_planning
):
    # The transport capacities are 0 in the solution: give them values.
    variable = f"{operation_or_planning}_transport_power_capacity"
    solution = solved.copy()
    solution[variable] = solved[variable] + np.arange(
        1, solved[variable].size + 1
    ).reshape(solved[variable].shape)

    by_area = calculate_total_of_a_type(
        "power_capacity", solution, parameters, operation_or_planning
    )
    no_area = calculate_total_of_a_type(
        "power_capacity",
        solution,
        parameters,
        operation_or_planning,
        by_area=False,
    )
    assert "area" not in no_area.index.names
    transport = no_area.xs("transport", level="type", drop_level=False)
    summed = (
        by_area.xs("transport", level="type", drop_level=False)
        .groupby(list(no_area.index.names))
        .sum()
    )
    assert transport["power_capacity"].abs().sum() > 0
    assert transport["power_capacity"].to_numpy() == pytest.approx(
        summed.reindex(transport.index)["power_capacity"].to_numpy()
    )


def test_missing_parameter_is_raised(parameters, solved):
    with pytest.raises(KeyError, match="Cannot compute the total of"):
        calculate_total_of_a_type(
            "power_capacity",
            solved,
            parameters.drop_vars("transport_area_from"),
        )
