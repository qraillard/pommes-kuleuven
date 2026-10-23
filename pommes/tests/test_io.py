import shutil
from pathlib import Path

from pommes.io.save_solution import save_solution
from pommes.model.build_model import build_model

test_folder = Path(__file__).parent


def test_save_solution(parameters, tmp_path):
    model = build_model(parameters)
    model.solve(solver_name="highs")

    # A temporary folder: the tracked pommes/tests/test_results is neither overwritten nor
    # deleted, and a synced folder (OneDrive) cannot lock the rmtree below.
    output_folder = tmp_path / "test_results"

    save_solution(
        model=model,
        output_folder=output_folder.as_posix(),
        model_parameters=parameters,
    )

    assert output_folder.exists()
    assert output_folder.is_dir()

    # "inputs" (a per-parameter CSV dump) and "model.nc" (gated behind
    # save_solution's save_model flag, off by default and not passed
    # here) are not part of save_solution's actual output; input.nc,
    # dual.nc and solution.nc already cover persisting the study.
    # (as in pommes-kul; these assertions used to pass only on stale
    # files left in the tracked test_results folder)
    folder_list = ["constraints", "variables"]

    for folder in folder_list:
        assert (output_folder / folder).exists()
        assert (output_folder / folder).is_dir()

    file_list = [
        "dual.nc",
        "input.nc",
        "solution.nc",
    ]

    for file_name in file_list:
        assert (output_folder / file_name).exists()
        assert (output_folder / file_name).is_file()

    shutil.rmtree(output_folder)
