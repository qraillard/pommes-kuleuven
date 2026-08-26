"""Invariants protecting existing studies from the retrofit module."""

import ast
import pathlib

import numpy as np
import pytest
import xarray as xr

from pommes.io.build_input_dataset import (
    compute_annuity_cost,
    compute_end_of_life,
)
from pommes.model.build_model import build_model
from pommes.model.data_validation import ref_inputs
from pommes.model.data_validation.dataset_check import check_inputs
from pommes.model.retrofit._spec import RetrofitSpec

# Retrofit deliberately does NOT cover `process`. That module is solved
# as a MILP -- its capacity is tied to an integer unit count through
# ``operation_process_power_capacity == nb_units * process_unit_size`` --
# and integer unit commitment is not suited to the capacity expansion
# retrofit exists to model. Concretely, a retrofit into a unit-sized
# technology is quantised, so a transfer whose derated capacity is not a
# whole multiple of the unit size is infeasible rather than rounded.
MODULES = ("conversion", "combined", "storage", "transport")


def test_every_retrofit_param_names_its_module():
    """
    check_inputs filters unused variables by substring. A parameter
    called `retrofit_foo` rather than `retrofit_conversion_foo` would
    escape filtering and be injected as a default into studies that
    never enabled retrofit.
    """
    offenders = [
        name
        for name in ref_inputs
        if "retrofit" in name
        and not any(f"retrofit_{m}" in name for m in MODULES)
    ]
    assert offenders == []


def test_old_retrofit_parameters_are_gone():
    for stale in (
        "retrofit_factor",
        "retrofit_year",
        "retrofit_tech_from",
        "retrofit_tech_to",
        "retrofit_invest_cost",
    ):
        assert stale not in ref_inputs


@pytest.mark.parametrize("module", MODULES)
def test_each_module_declares_its_core_parameters(module):
    spec = RetrofitSpec(
        module,
        "link" if module == "transport" else "area",
        ("power", "energy") if module == "storage" else ("power",),
    )
    assert spec.flag in ref_inputs
    assert spec.param("tech_from") in ref_inputs
    assert spec.param("tech_to") in ref_inputs
    assert spec.param("eligible_share") in ref_inputs
    for kind in spec.capacity_kinds:
        assert spec.param("ret_ratio", kind) in ref_inputs
        assert spec.bound(kind, "min") in ref_inputs
        assert spec.bound(kind, "max") in ref_inputs


def test_eligible_share_is_indexed_on_the_source_technology():
    """
    Not on the pair. The cap constraint sums over every pair sharing a
    source, so a per-pair share would be ill-defined the moment two
    pairs share one source technology.
    """
    entry = ref_inputs["retrofit_conversion_eligible_share"]
    assert "conversion_tech" in entry["index_set"]
    assert "retrofit_conversion_pair" not in entry["index_set"]


def test_pair_parameters_are_indexed_on_the_pair_dimension():
    entry = ref_inputs["retrofit_conversion_ret_ratio"]
    assert "retrofit_conversion_pair" in entry["index_set"]
    assert "year_ret" in entry["index_set"]


def _ds_with_ret():
    # year_ret=2040 is deliberately absent from year_inv=[2020, 2030]:
    # with retrofit life span 10, anchoring on year_ret gives
    # 2040 + 10 = 2050, while anchoring on year_inv (the pre-fix bug)
    # gives max(2020, 2030) + 10 = 2040. The two paths disagree, so a
    # regression that silently falls back to year_inv is caught. (A
    # year_ret that coincides with one of the year_inv values, as in
    # an earlier draft of this fixture, made both paths land on the
    # same number by coincidence and the test could not fail.)
    return xr.Dataset(
        {
            "conversion_life_span": xr.DataArray(20, coords={}),
            "retrofit_conversion_life_span": xr.DataArray(10, coords={}),
            "retrofit_conversion": xr.DataArray(True, coords={}),
            "retrofit_conversion_invest_cost": xr.DataArray(100.0),
            "retrofit_conversion_finance_rate": xr.DataArray(0.0),
        },
        coords={
            "year_inv": [2020, 2030],
            "year_ret": [2040],
            "year_dec": [2030, 2040, 2050],
        },
    )


def test_end_of_life_uses_year_ret_for_retrofit_variables():
    ds = compute_end_of_life(_ds_with_ret())
    # Retrofit: 2040 + 10 = 2050. Anchored on year_ret, not year_inv
    # (the year_inv path would give 2030 + 10 = 2040 instead).
    assert int(ds.retrofit_conversion_end_of_life.max()) == 2050
    # Non-retrofit is unchanged: 2030 + 20 = 2050.
    assert int(ds.conversion_end_of_life.max()) == 2050


def test_end_of_life_untouched_when_no_retrofit_present():
    ds = xr.Dataset(
        {"conversion_life_span": xr.DataArray(20, coords={})},
        coords={"year_inv": [2020], "year_dec": [2030, 2040]},
    )
    out = compute_end_of_life(ds)
    assert int(out.conversion_end_of_life.max()) == 2040


def _ds_annuity_discriminating():
    # year_inv=2020, year_ret=2030, year_dec=2050, life_span=40,
    # finance_rate=0.0 (so crf(0, m) = 1/m exactly, no grid-snap
    # rounding to worry about):
    #   - anchored on year_ret: window is 2030 < year_dec <= 2070,
    #     duration = 2050 - 2030 = 20, annuity = 100 * 1/20 = 5.0
    #   - anchored on year_inv: window is 2020 < year_dec <= 2060,
    #     duration = 2050 - 2020 = 30, annuity = 100 * 1/30 = 3.333...
    # Both windows admit year_dec=2050 but at different durations, so
    # the resulting value discriminates the two anchors outright (a
    # life span that lines up exactly with the year_dec grid spacing,
    # as in an earlier draft, gives the same duration -- and thus the
    # same annuity value -- under either anchor, and could not fail).
    return xr.Dataset(
        {
            "retrofit_conversion_life_span": xr.DataArray(40, coords={}),
            "retrofit_conversion": xr.DataArray(True, coords={}),
            "retrofit_conversion_invest_cost": xr.DataArray(100.0),
            "retrofit_conversion_finance_rate": xr.DataArray(0.0),
        },
        coords={
            "year_inv": [2020],
            "year_ret": [2030],
            "year_dec": [2050],
        },
    )


def _annuity_param(**extra):
    return {
        "pre_process": {
            "annuity_computation": {
                "retrofit_conversion": {
                    "retrofit_conversion_annuity_cost": {
                        "invest_cost": "retrofit_conversion_invest_cost",
                        "finance_rate": (
                            "retrofit_conversion_finance_rate"
                        ),
                        "life_span": "retrofit_conversion_life_span",
                        **extra,
                    }
                }
            }
        }
    }


def test_annuity_uses_year_start_when_provided():
    ds = _ds_with_ret()
    config = _annuity_param(year_start="year_ret")
    out = compute_annuity_cost(ds, config)
    assert "retrofit_conversion_annuity_cost" in out
    assert np.isfinite(
        out.retrofit_conversion_annuity_cost
    ).any()

    # The two checks above would pass even if year_start were ignored
    # outright (see _ds_with_ret's own comment on why); pin down an
    # actual VALUE that only the year_ret-anchored repayment window
    # can produce, using a fixture purpose-built to discriminate.
    out_discriminating = compute_annuity_cost(
        _ds_annuity_discriminating(), config
    )
    value = float(
        np.nanmax(out_discriminating.retrofit_conversion_annuity_cost.values)
    )
    assert value == pytest.approx(100 / 20)


def test_annuity_defaults_to_year_inv_when_year_start_omitted():
    """
    Every existing study omits year_start entirely -- it is a new key
    this feature introduces. This is the single most important path in
    the file for backward compatibility: it must resolve to the exact
    year_inv-anchored behaviour every pre-retrofit study already
    depends on.
    """
    config = _annuity_param()  # no year_start key at all
    out = compute_annuity_cost(_ds_annuity_discriminating(), config)
    value = float(np.nanmax(out.retrofit_conversion_annuity_cost.values))
    # year_inv-anchored: duration = 2050 - 2020 = 30, 100 * 1/30.
    assert value == pytest.approx(100 / 30)


def test_annuity_year_start_missing_from_coords_is_rejected():
    """
    ``compute_end_of_life`` raises a clear ``ValueError`` when a
    retrofit param needs ``year_ret`` and the study's coords don't
    have it; ``compute_annuity_cost``'s own ``year_start`` lookup had
    no equivalent guard, so the same misconfiguration -- a study whose
    ``annuity_computation`` config names a coordinate the dataset
    never declares -- surfaced as a bare, unexplained ``KeyError``
    from the raw ``ds[year_start_name]`` lookup instead.
    """
    ds = _ds_annuity_discriminating().drop_vars("year_ret")
    config = _annuity_param(year_start="year_ret")

    with pytest.raises(ValueError, match="year_ret"):
        compute_annuity_cost(ds, config)


def test_check_inputs_never_injects_retrofit_defaults_for_a_plain_study():
    """
    The mechanism behind the headline promise -- a user upgrades this
    package and existing studies keep working with no config change --
    is dataset_check.check_inputs' per-module substring filter, not
    the wording of any particular study's config file. Build a study
    dataset resembling a real one: ordinary modules turned on
    (conversion, storage), retrofit never mentioned anywhere. Nothing
    that check_inputs adds as a default may have "retrofit" in its
    name, because every such default is gated behind its own
    retrofit_<module> flag, and none of those flags is set here.
    """
    ds = xr.Dataset(
        {
            "conversion": xr.DataArray(True, coords={}),
            "storage": xr.DataArray(True, coords={}),
        },
        coords={
            "year_inv": [2020],
            "year_dec": [2030, 2040],
        },
    )
    out = check_inputs(ds)
    retrofit_vars = [str(v) for v in out.data_vars if "retrofit" in str(v)]
    assert retrofit_vars == []


# Resolved from this file's location, not the working directory pytest
# is invoked from: pommes/tests/test_retrofit_generic.py -> parents[3]
# is the directory that holds this checkout and its sibling studies.
_MODELLING_DIR = pathlib.Path(__file__).resolve().parents[3]

STUDY_CONFIGS = [
    _MODELLING_DIR / "POMMES-BE-IND-EU" / "config_REACTORS.yaml",
    _MODELLING_DIR / "POMMES-BE-IND-Adequacy" / "config.yaml",
]


@pytest.mark.parametrize(
    "config_path", STUDY_CONFIGS, ids=lambda p: p.parent.name
)
def test_real_study_configs_gain_no_retrofit_variables(config_path):
    """
    Secondary, words-only check: these two real study configs happen
    not to mention retrofit either. Skips cleanly if the sibling study
    checkouts are absent. The mechanism itself is proved by
    test_check_inputs_never_injects_retrofit_defaults_for_a_plain_study
    above; this test alone would not catch the filter being deleted.
    """
    if not config_path.exists():
        pytest.skip(f"{config_path} not available in this checkout")

    from pommes.io.build_input_dataset import read_config_file

    config = read_config_file(file_path=str(config_path))
    declared = set(config["input"]["parameters"].keys())
    assert not any("retrofit" in name for name in declared)


def test_no_technology_name_appears_in_the_retrofit_package():
    """
    The old retrofit.py hardcoded conversion_tech="smr_ccs". Nothing in
    the new package may name a technology: every pair comes from data.
    """
    import pommes.model.retrofit as pkg

    banned = (
        "smr", "ccgt", "ocgt", "electrolysis", "wind", "solar",
        "battery", "bf_bof", "pipe", "tank", "nuclear",
    )
    root = pathlib.Path(pkg.__file__).parent
    offenders = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Docstrings are documentation, not logic. An illustrative
        # "e.g. smr -> smr_ccs" in a module docstring is wanted, so
        # collect docstrings and exclude them from the literal scan.
        docs = set()
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Module,
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    docs.add(doc)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str) or node.value in docs:
                continue
            low = node.value.lower()
            for name in banned:
                if name in low:
                    offenders.append((path.name, name, node.lineno))
    assert offenders == []


def test_retrofit_works_with_arbitrary_technology_names(
    parameters_retrofit_conversion,
):
    """
    Rename the technologies to foo/bar. The model must build with the
    same variable dimensions and the same constraint set.

    The year grids below are NOT the ones this test was originally
    drafted with. ``year_ret`` must be a subset of both ``year_inv``
    and ``year_dec`` or ``_check_year_ret_grid`` raises -- see its
    docstring for why chaining and the source-cap constraint each
    depend on that. The original draft paired ``year_ret=[2030, 2040]``
    with ``year_inv=[2020, 2030]``, leaving 2040 without a ``year_inv``
    counterpart, so it would have errored rather than tested anything.
    It predates the ``year_inv`` guard. ``year_dec`` below includes
    2020 for the same reason, now that the guard also covers
    ``year_dec``: ``year_ret=[2020, 2030]`` would otherwise leave 2020
    without a ``year_dec`` counterpart.
    """
    p = parameters_retrofit_conversion.sel(
        conversion_tech=["smr", "smr_ccs"],
        area=["area_1"],
        hour=[0],
        resource=["electricity", "hydrogen", "methane"],
        year_dec=[2020, 2030, 2040, 2050],
        year_inv=[2020, 2030, 2040],
        year_op=[2020, 2030],
        year_ret=[2020, 2030],
    ).copy(deep=True)

    renamed = p.assign_coords(conversion_tech=["foo", "bar"])
    pair = renamed.retrofit_conversion_pair
    renamed["retrofit_conversion_tech_from"] = xr.DataArray(
        np.array(["foo"], dtype=str),
        dims="retrofit_conversion_pair",
        coords={"retrofit_conversion_pair": pair},
    )
    renamed["retrofit_conversion_tech_to"] = xr.DataArray(
        np.array(["bar"], dtype=str),
        dims="retrofit_conversion_pair",
        coords={"retrofit_conversion_pair": pair},
    )

    original = build_model(p)
    generic = build_model(renamed)

    assert set(original.variables) == set(generic.variables)
    assert set(original.constraints) == set(generic.constraints)
    assert (
        original.variables[
            "planning_retrofit_conversion_power_capacity"
        ].size
        == generic.variables[
            "planning_retrofit_conversion_power_capacity"
        ].size
    )
