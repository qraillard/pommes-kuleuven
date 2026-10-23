"""Module to convert the data validation yaml to rst."""

import os
from pathlib import Path

import yaml


def yaml_to_rst(input_file: str, output_file: str) -> None:
    """Convert the yaml to restructures text."""
    file_name = os.path.splitext(os.path.basename(input_file))[0]

    with open(input_file) as yaml_file:
        data = yaml.safe_load(yaml_file)

    with open(output_file, "w") as rst_file:
        # Write the file reference header
        rst_file.write(f".. _{file_name}:\n\n")

        # Write the title
        rst_file.write("Dataset Description\n")
        rst_file.write("===================\n\n")

        # Write the list-table structure
        rst_file.write(".. list-table::\n")
        rst_file.write("    :header-rows: 1\n\n")

        # Write the table header
        headers = ["Key"] + list(next(iter(data.values())).keys())
        rst_file.write("    * - " + "\n      - ".join(headers) + "\n\n")

        last_key = list(data.keys())[-1]
        # Write each row of the table
        for key, attributes in data.items():
            row = [key] + [str(attributes.get(col, "")) for col in headers[1:]]
            rst_file.write("    * - " + "\n      - ".join(row) + "\n")
            if key != last_key:
                rst_file.write("\n")


if __name__ == "__main__":
    doc_path = Path(__file__).resolve().parent
    project_path = doc_path.parent
    input_file = (
        project_path
        / "pommes"
        / "model"
        / "data_validation"
        / "dataset_description.yaml"
    )
    output_file = (
        doc_path / "source" / "methodology" / "dataset_description.rst"
    )
    yaml_to_rst(input_file, output_file)
