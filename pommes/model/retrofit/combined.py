"""Retrofit between combined technologies (e.g. bf_bof -> bf_bof_ccs).

The ``mode`` dimension plays no part here: it exists only at dispatch
level, while retrofit acts on planned capacity.
"""

from linopy import Constraint, Model
from xarray import Dataset

from pommes.model.retrofit._core import add_retrofit_block
from pommes.model.retrofit._spec import RetrofitSpec

SPEC = RetrofitSpec(
    module="combined", site_dim="area", capacity_kinds=("power",)
)


def add_retrofit_combined(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
) -> Model:
    """Add combined-to-combined retrofit. Requires add_combined."""
    return add_retrofit_block(
        model, model_parameters, annualised_totex_def, SPEC
    )
