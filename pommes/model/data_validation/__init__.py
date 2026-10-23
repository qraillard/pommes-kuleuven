"""Package for building the linopy model in POMMES."""

from pathlib import Path

import yaml

data_validation_path = Path(__file__).parent

with open(data_validation_path / "dataset_description.yaml") as file:
    ref_inputs = yaml.safe_load(file)
