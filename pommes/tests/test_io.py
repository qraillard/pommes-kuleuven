import shutil
from pathlib import Path

from pommes.io.save_solution import save_solution
from pommes.model.build_model import build_model

test_folder = Path(__file__).parent


def test_save_solution(parameters):
    model = build_model(parameters)
    model.solve(solver_name="highs")

    output_folder = test_folder / "test_results"

    save_solution(
        model=model,
        output_folder=output_folder.as_posix(),
        model_parameters=parameters,
    )

    assert output_folder.exists()
    assert output_folder.is_dir()

    folder_list = ["constraints", "inputs", "variables"]

    for folder in folder_list:
        assert (output_folder / folder).exists()
        assert (output_folder / folder).is_dir()

    file_list = [
        "dual.nc",
        "input.nc",
        "model.nc",
        "objective.csv",
        "solution.nc",
    ]

    for file_name in file_list:
        assert (output_folder / file_name).exists()
        assert (output_folder / file_name).is_file()

    shutil.rmtree(output_folder)
