"""Description of a host module, for the generic retrofit core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linopy import LinearExpression
    from xarray import Dataset


@dataclass(frozen=True)
class RetrofitSpec:
    """
    Everything the retrofit core needs to know about a host module.

    Only three fields are structural; every name derives from them,
    because all four host modules follow the same conventions.

    Args:
        module: host module name, e.g. ``"conversion"``.
        site_dim: ``"area"`` for every module except ``transport``,
            which plans on ``"link"``.
        capacity_kinds: ``("power",)``, or ``("power", "energy")`` for
            storage, whose power and energy ratings convert at
            genuinely different ratios.
    """

    module: str
    site_dim: str
    capacity_kinds: tuple[str, ...]

    @property
    def tech_dim(self) -> str:
        return f"{self.module}_tech"

    @property
    def pair_dim(self) -> str:
        return f"retrofit_{self.module}_pair"

    @property
    def flag(self) -> str:
        return f"retrofit_{self.module}"

    def host_planning_var(self, kind: str) -> str:
        """Host modules always carry the kind in capacity names."""
        return f"planning_{self.module}_{kind}_capacity"

    def host_capacity_def(self, kind: str) -> str:
        return f"operation_{self.module}_{kind}_capacity_def"

    def param(self, name: str, kind: str | None = None) -> str:
        """
        Retrofit parameter name.

        The kind is appended only when the module has more than one,
        matching the host convention (``conversion_invest_cost`` versus
        ``storage_invest_cost_power``). Parameters that are not
        per-kind pass ``kind=None``.
        """
        suffix = ""
        if kind is not None and len(self.capacity_kinds) > 1:
            suffix = f"_{kind}"
        return f"retrofit_{self.module}_{name}{suffix}"

    def fold_costs(
        self, expr: LinearExpression, p: Dataset
    ) -> LinearExpression:
        """
        Bring the cost expression onto ``area``.

        Identity for area-sited modules. A module that plans on a
        different site dimension overrides this to split each site's
        cost across the areas it touches, mirroring how that module
        already folds its own planning costs onto ``area``.
        """
        return expr

    def bound(self, kind: str, which: str) -> str:
        """
        Capacity investment bound name.

        Unlike costs, bounds always infix the kind, again matching the
        host (``conversion_power_capacity_investment_min`` carries
        ``power`` even though conversion has only one kind).
        """
        return (
            f"retrofit_{self.module}_{kind}"
            f"_capacity_investment_{which}"
        )
