"""Module setting the example path."""

from pathlib import Path

from pommes.io.build_input_dataset import (
    build_input_parameters,
    read_config_file,
)
from pommes.io.calculate_summaries import (
    generate_summary_dataframes_from_results,
)
from pommes.io.save_solution import save_solution
from pommes.model.build_model import build_model
from pommes.model.data_validation.dataset_check import check_inputs
from pommes.post_process import get_capacities, get_net_generation

pommes_path = Path(__file__).parent
test_case_path = pommes_path / "examples" / "test_case"

__all__ = [
    "build_model",
    "build_input_parameters",
    "check_inputs",
    "read_config_file",
    "save_solution",
    "generate_summary_dataframes_from_results",
    "get_net_generation",
    "get_capacities",
]
