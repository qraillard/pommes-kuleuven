"""Retrofit between conversion technologies (e.g. smr -> smr_ccs)."""

from linopy import Constraint, Model
from xarray import Dataset

from pommes.model.retrofit._core import add_retrofit_block
from pommes.model.retrofit._spec import RetrofitSpec

SPEC = RetrofitSpec(
    module="conversion", site_dim="area", capacity_kinds=("power",)
)


def add_retrofit_conversion(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
) -> Model:
    """Add conversion-to-conversion retrofit. Requires add_conversion."""
    return add_retrofit_block(
        model, model_parameters, annualised_totex_def, SPEC
    )
