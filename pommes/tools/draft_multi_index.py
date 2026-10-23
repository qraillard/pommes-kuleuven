"""Draft module."""

from pommes.io.build_input_dataset import (
    build_input_parameters,
    read_config_file,
)
from pommes.model.data_validation.dataset_check import check_inputs

if __name__ == "__main__":
    study = "test_case"
    scenario = "ref"
    suffix = "_02161113"

    output_folder = f"study/{study}/output/{scenario}{suffix}"
    solver = "highs"  # ["gurobi", "xpress", "highs", "mosek"]

    config = read_config_file(study=study)

    model_parameters = build_input_parameters(config)
    model_parameters = check_inputs(model_parameters)

    ds = model_parameters[
        [
            variable
            for variable in model_parameters.data_vars
            if "conversion" in variable
        ]
    ]
    ds = ds.expand_dims(
        {
            dim: model_parameters[dim]
            for dim in model_parameters.dims
            if dim in ["area", "conversion_tech", "year_inv"]
            and dim not in ds.dims
        }
    )

    ds = ds.stack(asset=["area", "conversion_tech", "year_inv", "year_dec"])
    ds = ds.dropna(dim="asset", how="all", subset=["conversion_annuity_cost"])

    ds.assign(areas=ds.area, conversion_techs=ds.conversion_tech).groupby(
        ["areas", "conversion_techs"]
    ).sum().rename(conversion_techs="conversion_tech", areas="area")
