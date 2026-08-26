"""Retrofit between transport technologies (e.g. methane -> hydrogen pipe).

Transport plans on ``link`` rather than ``area``, so the transfer
variable is link-indexed while costs must still reach ``area`` to join
the objective. The split mirrors ``planning_transport_costs_def``: each
link's cost is shared evenly between the two areas it connects.
"""

import numpy as np
from linopy import Constraint, LinearExpression, Model
from xarray import Dataset

from pommes.model.retrofit._core import add_retrofit_block
from pommes.model.retrofit._spec import RetrofitSpec


class TransportRetrofitSpec(RetrofitSpec):
    """RetrofitSpec that folds link costs onto endpoint areas."""

    def fold_costs(
        self, expr: LinearExpression, p: Dataset
    ) -> LinearExpression:
        """Split each link's cost evenly across its two endpoints."""
        return 0.5 * expr.where(
            np.logical_or(
                p.area == p.transport_area_from,
                p.area == p.transport_area_to,
            )
        ).sum("link")


SPEC = TransportRetrofitSpec(
    module="transport", site_dim="link", capacity_kinds=("power",)
)


def add_retrofit_transport(
    model: Model,
    model_parameters: Dataset,
    annualised_totex_def: Constraint,
) -> Model:
    """Add transport-to-transport retrofit. Requires add_transport."""
    return add_retrofit_block(
        model, model_parameters, annualised_totex_def, SPEC
    )
