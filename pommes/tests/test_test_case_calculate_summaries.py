import itertools

from pommes.io.calculate_summaries import (
    calculate_total_of_a_type,
    generate_summary_dataframes_from_results,
)
from pommes.model.build_model import build_model


def test_test_case(parameters):
    model = build_model(parameters)
    model.solve(solver_name="highs")
    p = model.parameters
    s = model.solution

    types = ["power_capacity", "net_generation", "costs"]
    operation_or_planning_options = ["operation", "planning"]
    by_year_op_options = [True, False]
    by_area_options = [True, False]

    combinations = itertools.product(
        types,
        operation_or_planning_options,
        by_year_op_options,
        by_area_options,
    )

    for type_value, operation_or_planning, by_year_op, by_area in combinations:
        if not (
            type_value == "net_generation"
            and operation_or_planning == "planning"
        ):
            print(
                f"Testing with type={type_value}, "
                f"operation_or_planning={operation_or_planning}, "
                f"by_year_op={by_year_op}, by_area={by_area}"
            )
            calculate_total_of_a_type(
                type=type_value,
                solution=s,
                parameters=p,
                operation_or_planning=operation_or_planning,
                by_year_op=by_year_op,
                by_area=by_area,
            )
    print("Testing generate_summary_dataframes_from_results")
    generate_summary_dataframes_from_results(s, p)

    assert True


def test_no_level_reordering_without_operating_condition(parameters, monkeypatch):
    """Without the coordinate the row labels are those of 0.5.0+kul."""
    import pandas as pd

    model = build_model(parameters)
    model.solve(solver_name="highs")

    def fail(self, *args, **kwargs):
        raise AssertionError("reorder_levels must not run without the coordinate")

    monkeypatch.setattr(pd.DataFrame, "reorder_levels", fail)
    for type_value in ["costs", "net_generation"]:
        calculate_total_of_a_type(
            type=type_value,
            solution=model.solution,
            parameters=model.parameters,
            operation_or_planning="operation",
        )
