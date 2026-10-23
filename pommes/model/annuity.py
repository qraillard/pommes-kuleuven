"""Switch between the two annuity conventions of the planning costs.

``annuity_full_life`` (scalar boolean of the input dataset, set from the
``pre_process: annuity_mode`` option of the study configuration):

- True (default, ``annuity_mode: full_life``): the annuity of a vintage is
  fixed at investment, ``invest_cost x crf(finance_rate, life_span)``, and
  paid in every operation year of the vintage's full life
  (``year_start <= year_op < year_start + life_span``), even if the capacity
  is decommissioned earlier (sunk cost). Early decommissioning then only saves
  fixed O&M and variable costs. The annuity has no ``year_dec`` dimension.
- False (``annuity_mode: until_decommissioning``, previous behaviour): the
  annuity table depends on ``year_dec`` and each vintage's
  ``annuity_perfect_foresight`` flag chooses between paying it until the
  decommissioning year (1) and the cheapest annuity until ``end_of_life`` (0).

A dataset without the variable (built by hand, without ``compute_annuity_cost``)
is read from its annuity tables, with a warning: a ``year_dec`` dimension means
``until_decommissioning``, otherwise ``full_life``.
"""

import logging

import xarray as xr
from xarray import Dataset


def infer_annuity_full_life(ds: Dataset) -> bool:
    """
    Convention of a dataset that does not say which one it uses.

    Annuity tables with a ``year_dec`` dimension can only be the previous
    convention; without one the dataset is read as full life.

    Args:
        ds (xarray.Dataset): Dataset with ``*_annuity_cost*`` variables.

    Returns:
        bool: False if an annuity cost depends on ``year_dec``, else True.
    """
    return not any(
        "year_dec" in ds[name].dims
        for name in ds.data_vars
        if "annuity_cost" in str(name)
    )


def annuity_full_life(p: Dataset, *annuity_names: str) -> bool:
    """
    Tell whether planning costs use full life (sunk) annuities.

    Args:
        p (xarray.Dataset): Model parameters.
        *annuity_names (str): Names of the annuity cost parameters about to be
            used; with full life annuities none may depend on ``year_dec``.

    Returns:
        bool: The value of ``annuity_full_life``; if the dataset has none (built
            without ``compute_annuity_cost`` or ``check_inputs``), the
            convention inferred from its annuity tables with a warning.

    Raises:
        ValueError: If ``annuity_full_life`` is true while an annuity depends
            on ``year_dec`` (dataset built for the previous behaviour: rebuild
            it or set ``annuity_full_life`` to False).
    """
    if "annuity_full_life" in p:
        full_life = bool(p["annuity_full_life"])
    else:
        full_life = infer_annuity_full_life(p)
        logging.warning(
            "No 'annuity_full_life' variable in the model parameters: "
            f"inferred {full_life} from the annuity tables (year_dec "
            "dimension means until_decommissioning)."
        )
    if full_life:
        for name in annuity_names:
            if "year_dec" in p[name].dims:
                raise ValueError(
                    f"annuity_full_life is True but {name} depends on "
                    f"year_dec. Rebuild the dataset with compute_annuity_cost "
                    f"(annuity_mode full_life) or set annuity_full_life to "
                    f"False for the previous behaviour."
                )
    return full_life


def full_life_window(
    year_start: xr.DataArray, life_span: xr.DataArray, year_op: xr.DataArray
) -> xr.DataArray:
    """Operation years of a vintage's full life: start <= year_op < start + life."""
    return (year_start <= year_op) * (year_op < year_start + life_span)
