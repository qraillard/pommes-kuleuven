from pommes.io.calculate_summaries import aggregate_link_data, reindex_by_area
from pommes.model.build_model import build_model


def test_test_case(parameters):
    model = build_model(parameters)
    model.solve(solver_name="highs")

    s = model.solution

    exchange = (
        s.operation_transport_power_capacity.to_dataframe()
        .groupby(["link", "transport_tech", "year_op"])
        .sum()
    )
    exchange_ri = reindex_by_area(
        dataset=exchange,
        transport_area_from=parameters["transport_area_from"],
        transport_area_to=parameters["transport_area_to"],
    )
    aggregate_link_data(exchange_ri)
    aggregate_link_data(exchange_ri, stacked_into_transport_tech=True)
