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
