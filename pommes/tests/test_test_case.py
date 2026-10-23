from pommes.model.build_model import build_model


def test_test_case(parameters):
    model = build_model(parameters)
    model.solve(solver_name="highs")
    assert True
