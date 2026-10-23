"""Retrofit between storage technologies.

Storage is the one module with two capacities, and they do not convert
at the same rate: turning a methane cavern into a hydrogen one barely
changes its MW injection rating while collapsing its MWh, because
hydrogen's volumetric energy density is roughly a third of methane's.
Each kind therefore gets its own transfer variable and its own ratio.

Setting ``storage_energy_power_ratio`` as well as two independent
retrofit ratios over-determines the retrofitted asset and can make the
model infeasible.
"""

from linopy import Constraint, Model
from xarray import Dataset

from pommes.model.retrofit._core import add_retrofit_block
from pommes.model.retrofit._spec import RetrofitSpec

SPEC = RetrofitSpec(
    module="storage",
    site_dim="area",
    capacity_kinds=("power", "energy"),
)


def add_retrofit_storage(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
) -> Model:
    """Add storage-to-storage retrofit. Requires add_storage."""
    return add_retrofit_block(
        model, model_parameters, annualised_totex_def, SPEC
    )
