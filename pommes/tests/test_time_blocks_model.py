"""Whole-model exactness of the time blocks (multi-resolution, sections 9, 15.1, 15.2, 16.2, 16.3 and 16.5).

The fixtures' whole model (storage, transport, turpe, net_import, carbon, conversion, combined, an idle resource) at
T1, T2 and T3(b); the uniform coarse step (T3(c)) and net_import at scale against today's code run with steps of 4 h;
retrofit between an hourly and a block technology against the flat twin; operating conditions on block models.
Each model is built (and solved) once per module and shared by the tests that read it.
"""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import assert_condition_dims_consistent, with_conditions
from pommes.tests.time_blocks_helper import (
    BLOCK_PARAMETERS,
    assert_same,
    assert_twin_equal,
    binds,
    coarse_step_reference,
    conditions_block_dataset,
    flat_twin,
    net_import_expected_conditions_dataset,
    net_import_twin_dataset,
    retrofit_combined_block_dataset,
    retrofit_conversion_block_dataset,
    retrofit_source_block_dataset,
    solve_model,
    solved,
    T2_ELECTROLYSIS_FACTOR,
    t3c_dataset,
    t3c_ratio_dataset,
    to_hours,
    variables_in,
    whole_t1_dataset,
    whole_t2_dataset,
    whole_t3b_dataset,
    with_blocks,
    with_idle,
)
from pommes.utils import block_dim, condition_dual_scale, time_blocks

# a balance that no technology of a whole model can make coarse says so once per build: the tests that build such a
# model (the models are cached, so every test that reads one of them) expect it
no_gain = pytest.mark.filterwarnings("ignore:Time blocks give no gain")

EMISSIONS = "operation_carbon_emissions"


def _clean(p):
    """``p`` without block inputs: today's hourly model."""
    return p.drop_vars(BLOCK_PARAMETERS, errors="ignore")


def _lengths(p) -> dict:
    """The conversion and combined block lengths of ``p`` as ``flat_twin`` takes them."""
    out = {}
    for module in ("conversion", "combined"):
        if f"{module}_tech" not in p.coords:
            continue
        length = p[f"{module}_time_block_hrs"].dropna(f"{module}_tech")
        if length.size:
            out[module] = {str(t): int(v) for t, v in zip(length[f"{module}_tech"].values, length.values)}
    return out


def _conditioned(p):
    return with_conditions(p, [0.25, 0.75])


# the retrofit pair of each module: (builder, source, target)
RETROFIT = {
    "conversion": (retrofit_conversion_block_dataset, "old", "new"),
    "combined": (retrofit_combined_block_dataset, "electric_boiler", "boiler_block"),
}

# every block model of this file, by name: a builder each (built once, solved once when a test reads a value)
BUILDERS = {
    "t1": whole_t1_dataset,
    "t1_hourly": lambda: _clean(whole_t1_dataset()),
    "t2": whole_t2_dataset,
    "t3b": whole_t3b_dataset,
    "t3b_hourly": lambda: _clean(whole_t3b_dataset()),
    "t3c": t3c_dataset,
    "t3c_ratio": t3c_ratio_dataset,
    "net_import_maxima": net_import_twin_dataset,
    "net_import_expected": net_import_expected_conditions_dataset,
    "retrofit_conversion_target": RETROFIT["conversion"][0],
    "retrofit_conversion_source": lambda: retrofit_source_block_dataset("conversion"),
    "retrofit_combined_target": RETROFIT["combined"][0],
    "retrofit_combined_source": lambda: retrofit_source_block_dataset("combined"),
    "conditions": conditions_block_dataset,
    "conditions_even": lambda: with_conditions(conditions_block_dataset(), [0.5, 0.5]),
    "conditions_two": lambda: _conditioned(conditions_block_dataset()),
}
_BUILT, _SOLVED = {}, {}


def built(key):
    """The model of ``BUILDERS[key]``, built once (not solved)."""
    if key not in _BUILT:
        _BUILT[key] = build_model(BUILDERS[key]())
    return _BUILT[key]


def solved_model(key):
    """The model of ``BUILDERS[key]``, built and solved once."""
    if key not in _SOLVED:
        _SOLVED[key] = solve_model(built(key))
    return _SOLVED[key]


def reference_model(key, L):
    """Today's model of ``BUILDERS[key]`` run with steps of ``L`` hours, solved once."""
    name = f"{key}_reference"
    if name not in _SOLVED:
        _SOLVED[name] = solve_model(build_model(coarse_step_reference(BUILDERS[key](), L)))
    return _SOLVED[name]


def test_the_idle_resource_changes_nothing(parameters):
    """The idle resource added to the fixture's whole model holds nothing: the hourly optimum is the same."""
    p = parameters.sel(year_dec=[2040], year_inv=[2020], year_op=[2020])
    assert solve_model(build_model(with_idle(p))).objective.value == pytest.approx(
        solve_model(build_model(p)).objective.value, rel=1e-9
    )


# --- T1: block length = step ---


@no_gain
def test_t1_whole_model():
    """Every conversion and combined length is the step: every value equals the hourly model, hour by hour."""
    p = whole_t1_dataset()
    m, hourly = solved_model("t1"), solved_model("t1_hourly")
    tb = time_blocks(p)
    assert tb.balance_lengths == [1] and tb.resources(1) == ["heat"]  # heat has block contributors only
    assert "operation_adequacy_block_1h_constraint" in m.constraints
    assert m.objective.value == pytest.approx(hourly.objective.value, rel=1e-9)
    pairs = [
        ("operation_conversion_block_1h_power", "operation_conversion_power"),
        ("operation_combined_block_1h_power", "operation_combined_power"),
    ]
    for name, hourly_name in pairs:
        block, want = to_hours(solved(m, name), tb), solved(hourly, hourly_name)
        assert_same(block, want.sel({d: block.indexes[d] for d in block.dims if d.endswith("_tech")}), name)
        assert float(abs(block).sum()) > 1.0  # something runs
    for kind in ("import", "export"):  # net_import follows the coarse heat balance
        block = to_hours(solved(m, f"operation_net_import_block_1h_{kind}"), tb)
        want = solved(hourly, f"operation_net_import_{kind}").sel(resource=block.indexes["resource"])
        assert_same(block, want, kind)
    assert float(solved(m, "operation_net_import_block_1h_import").sum()) > 1.0
    assert float(solved(m, EMISSIONS).sum()) > 0.1  # carbon is on and heat imports emit
    assert_same(solved(m, EMISSIONS), solved(hourly, EMISSIONS), EMISSIONS)


# --- T2: the flat twin, every balance at the step ---


@no_gain
def test_t2_whole_model_flat_twin():
    """Block conversion and combined technologies on hourly balances equal the flat twin, emissions hour by hour."""
    p = whole_t2_dataset()
    lengths = _lengths(p)
    assert set(lengths["conversion"].values()) == {2} and lengths["combined"] == {"electric_boiler": 2}
    assert "wind_onshore" not in lengths["conversion"]  # must-run on an availability that varies: a relaxation
    tb = time_blocks(p)
    assert tb.balance_lengths == [1]  # the heat balance forced to the step; every other balance is hourly
    m, twin = solved_model("t2"), solve_model(flat_twin(p, **lengths))
    assert_twin_equal(m, twin, tb, ["operation_conversion_block_2h_power", "operation_combined_block_2h_power"])
    assert_same(solved(m, EMISSIONS), solved(twin, EMISSIONS), EMISSIONS)
    assert float(solved(m, EMISSIONS).sum()) > 0.1
    # the block technologies emit, apart from the heat import: the electrolyser's share alone is above 0.1 t
    electrolysis = solved(m, "operation_conversion_block_2h_power").sel(conversion_tech="electrolysis")
    assert float(electrolysis.sum()) * T2_ELECTROLYSIS_FACTOR * 2 > 0.1
    assert float(abs(solved(m, "operation_combined_block_2h_power")).sum()) > 1.0  # the block boiler runs
    assert float(abs(solved(m, "operation_conversion_block_2h_power")).sum()) > 1.0  # and a block conversion tech


# --- T3(b): a forced coarse balance is a relaxation ---


def test_t3b_forced_coarse_is_a_relaxation():
    """Methane and hydrogen forced to 2 h: no dearer than hourly; each coarse row holds the demand of its block."""
    p = whole_t3b_dataset()
    assert not bool(p.get("flexdem", False))  # no flexible demand, hence no ramp: the coarse model can only relax
    tb = time_blocks(p)
    assert tb.resources(2) == ["hydrogen", "methane"]
    assert p.sizes["hour"] % 2 == 1 and list(tb.duration[2].values) == [2.0, 2.0, 2.0, 2.0, 1.0]  # a short last block
    for resource in tb.resources(2):  # net_import prices and factors are constant inside each block
        for name in ("net_import_import_price", "net_import_export_price", "net_import_emission_factor"):
            x = p[name].sel(resource=resource)
            if "hour" in x.dims:
                assert float(x.groupby(tb.block_of_hour[2]).std().max()) == 0.0, (resource, name)
    m, hourly = solved_model("t3b"), solved_model("t3b_hourly")
    # strictly cheaper on this dataset (probed: 25 964 567 against 25 964 755)
    assert m.objective.value < hourly.objective.value - 1e-6
    rhs = m.constraints["operation_adequacy_block_2h_constraint"].rhs
    block_sum = p.demand.fillna(0).groupby(tb.block_of_hour[2]).sum().sel(resource=tb.resources(2))
    assert list(rhs.indexes["resource"]) == tb.resources(2)
    np.testing.assert_allclose(rhs.transpose(*block_sum.dims).values, block_sum.values, atol=1e-9)
    last = p.demand.isel(hour=-1).sel(resource=tb.resources(2)).transpose(*rhs.isel(time_block_2h=-1).dims)
    np.testing.assert_allclose(rhs.isel(time_block_2h=-1).values, last.values, atol=1e-9)  # one step: its own demand
    assert float(block_sum.max() - block_sum.min()) > 0.1  # the rows differ from block to block


# --- T3(c): the uniform coarse step ---

L = 4


def _block_vs_step(m, reference, name):
    """A block variable of ``m`` equals the reference's per-step variable (the block index is the reference's hour)."""
    got = solved(m, name).rename({block_dim(L): "hour"})
    want = solved(reference, name.replace(f"_block_{L}h_", "_"))
    if "resource" in got.dims:  # the block model holds the coarse resources only
        want = want.sel(resource=got.indexes["resource"])
    assert_same(got, want, name)
    return got


def _assert_equals_reference(key, with_conversion):
    """
    The block model of ``key`` equals today's model with steps of ``L`` h on the quantities that have one optimum
    (probed with random-weighted min and max at the optimal objective): objective, net_import imports, shedding and
    spillage block by block; the total export, the combined and flexdem energies (and the conversion energies when
    ``with_conversion``) summed over the blocks, since equal-cost shifts between blocks leave their distribution open;
    carbon on yearly totals.
    """
    p = BUILDERS[key]()
    m, reference = solved_model(key), reference_model(key, L)
    assert m.objective.value == pytest.approx(reference.objective.value, rel=1e-7)
    assert float(abs(_block_vs_step(m, reference, f"operation_net_import_block_{L}h_import")).sum()) > 0.1
    for name in (f"operation_load_shedding_block_{L}h_power", f"operation_spillage_block_{L}h_power"):
        _block_vs_step(m, reference, name)
    export = f"operation_net_import_block_{L}h_export"  # unique only as a total over areas and blocks in the ratio cell
    np.testing.assert_allclose(
        float(solved(m, export).sum()), float(solved(reference, export.replace(f"_block_{L}h_", "_")).sum()), rtol=1e-6
    )
    names = [f"operation_combined_block_{L}h_power", f"operation_flexdem_block_{L}h_demand"]
    for name in names + ([f"operation_conversion_block_{L}h_power"] if with_conversion else []):
        got = solved(m, name).sum(block_dim(L))
        want = solved(reference, name.replace(f"_block_{L}h_", "_"))
        if "resource" in got.dims:
            want = want.sel(resource=got.indexes["resource"])
        assert_same(got, want.sum("hour"), name)
        assert float(abs(got).sum()) > 0.1, name
    if bool(p.carbon):  # compared on yearly totals only: the block model's emission rows stay hourly
        for name in (EMISSIONS, "operation_total_carbon_emissions"):
            got, want = solved(m, name).sum("hour"), solved(reference, name).sum("hour")
            np.testing.assert_allclose(got.values, want.transpose(*got.dims).values, atol=1e-6, err_msg=name)


@pytest.mark.parametrize(
    "key, rows",
    [
        ("t3c", [f"operation_flexdem_block_{L}h_{r}" for r in ("demand_conservation_8h", "ramp_up_constraint", "ramp_down_constraint")]),
        ("t3c_ratio", [f"operation_flexdem_block_{L}h_{r}" for r in ("demand_max", "demand_min")]),
    ],
    ids=["window_and_ramps", "ratios"],
)
def test_t3c_uniform_coarse_step(key, rows):
    """T3(c): ratios, window, ramps and both yearly maxima equal today's model with steps of 4 h; each family binds."""
    _assert_equals_reference(key, with_conversion=key == "t3c")
    m = solved_model(key)
    for name in rows:
        assert binds(m, name, equality="conservation" in name), name
    for kind in ("import", "export"):  # the yearly maxima of net_import
        assert binds(m, f"operation_net_import_block_{L}h_{kind}_yearly_max_constraint"), kind
    if bool(BUILDERS[key]().carbon):
        assert float(solved(m, "operation_carbon_costs").sum()) > 1e-3  # the tax is paid
        free = solve_model(build_model(BUILDERS[key]().assign(carbon_tax=np.array(0.0))))
        assert float(solved(m, EMISSIONS).sum()) < float(solved(free, EMISSIONS).sum()) - 1e-6  # and it binds


# --- net_import at scale (section 16.2) ---


@pytest.mark.parametrize("key", ["net_import_maxima", "net_import_expected", "t3c"])
def test_net_import_twin_at_scale(key):
    """
    Constant-price blocks against steps of 4 h: binding import and export maxima; both on the expectation under two
    conditions; a binding carbon tax. Objective, the imports block by block and the total exports equal (the export
    distribution has several optima).
    """
    m, reference = solved_model(key), reference_model(key, L)
    assert m.objective.value == pytest.approx(reference.objective.value, rel=1e-7)
    for kind, resource in (("import", "heat"), ("export", "gas")):
        got = solved(m, f"operation_net_import_block_{L}h_{kind}").sel(resource=resource).rename({block_dim(L): "hour"})
        want = solved(reference, f"operation_net_import_{kind}").sel(resource=resource)
        # one optimum block by block; the export is unique only as a total (shifts between blocks and areas)
        if kind == "import":
            assert_same(got, want, kind)
        else:
            np.testing.assert_allclose(float(got.sum()), float(want.sum()), rtol=1e-6, err_msg=kind)
        assert float(got.max()) > 0.1  # the rest of the world is used
        row = f"operation_net_import_block_{L}h_{kind}_yearly_max_constraint"
        assert binds(m, row), row  # the maximum binds
        if key == "net_import_expected":
            assert "operating_condition" not in m.constraints[row].labels.dims  # one row per year, on the expectation
    if key == "t3c":  # the tax binds
        free = BUILDERS[key]().assign(carbon_tax=np.array(0.0))
        free_emissions = solve_model(build_model(free))
        # the tax binds
        assert float(solved(m, EMISSIONS).sum()) < float(solved(free_emissions, EMISSIONS).sum()) - 1e-6
        assert float(solved(m, "operation_carbon_costs").sum()) > 1e-3
        got, want = solved(m, EMISSIONS).sum("hour"), solved(reference, EMISSIONS).sum("hour")
        np.testing.assert_allclose(got.values, want.transpose(*got.dims).values, atol=1e-6)


# --- retrofit (section 16.3) ---

RETROFIT_CASES = [
    ("conversion", "target"),
    ("conversion", "source"),
    ("combined", "target"),
    ("combined", "source"),
]


def _retrofit_twin(module, side):
    key = f"retrofit_{module}_{side}"
    p = BUILDERS[key]()
    m = solved_model(key)
    twin = solve_model(flat_twin(p, **_lengths(p)))
    # objective, every planning variable (transfers included), capacities. The conversion capacity of the combined
    # datasets is not unique per vintage (probed: the 2020, 2030 and 2040 vintages substitute at the same objective;
    # the capacity of each operation year is unique and still compared)
    skip = ("planning_conversion_power_capacity",) if module == "combined" else ()
    assert_twin_equal(m, twin, time_blocks(p), [], skip=skip)
    transfer = solved(m, f"planning_retrofit_{module}_power_capacity")
    assert float(transfer.sum()) > 1e-3  # a transfer really happens
    return transfer


@no_gain
@pytest.mark.parametrize("module", ["conversion", "combined"])
def test_retrofit_hourly_source_block_target(module):
    """The source is hourly and the target on blocks: objective, transfers and capacities equal the flat twin."""
    p = BUILDERS[f"retrofit_{module}_target"]()
    assert _lengths(p)[module] == {RETROFIT[module][2]: 2}  # only the target is a block technology
    _retrofit_twin(module, "target")


@no_gain
@pytest.mark.parametrize("module", ["conversion", "combined"])
def test_retrofit_block_source_hourly_target(module):
    """The source is on blocks and the target hourly: objective, transfers and capacities equal the flat twin."""
    p = BUILDERS[f"retrofit_{module}_source"]()
    assert _lengths(p)[module] == {RETROFIT[module][1]: 2}
    _retrofit_twin(module, "source")


@no_gain
@pytest.mark.parametrize("module, side", RETROFIT_CASES)
def test_retrofit_rows_read_no_operation_variable(module, side):
    """No retrofit row has a term of a variable with an ``hour`` or ``time_block_*`` dimension (read on the labels)."""
    m = built(f"retrofit_{module}_{side}")
    rows = [c for c in m.constraints if c.startswith(("retrofit_", "planning_retrofit_"))]
    assert f"retrofit_{module}_power_cap_constraint" in rows and f"planning_retrofit_{module}_costs_def" in rows
    read = set()
    for name in rows:
        for variable in variables_in(m, m.constraints[name]):
            read.add(variable)
            dims = m.variables[variable].dims
            assert not any(d == "hour" or d.startswith("time_block_") for d in dims), (name, variable, dims)
    assert f"planning_retrofit_{module}_power_capacity" in read and f"planning_{module}_power_capacity" in read
    # the retrofit does write terms into the host's year-level capacity row: that row is not a retrofit row
    host = m.constraints[f"operation_{module}_power_capacity_def"]
    assert f"planning_retrofit_{module}_power_capacity" in variables_in(m, host)


@no_gain
@pytest.mark.parametrize("module, side", RETROFIT_CASES)
def test_retrofit_unit_index_unchanged_by_blocks(module, side):
    """The index of ``planning_<module>_power_capacity`` (the unit index a sparse retrofit reads) ignores the blocks."""
    key = f"retrofit_{module}_{side}"
    with_blocks_model = built(key)
    without = build_model(_clean(BUILDERS[key]()))
    name = f"planning_{module}_power_capacity"
    a, b = with_blocks_model.variables[name], without.variables[name]
    assert a.dims == b.dims
    for dim in a.dims:
        assert a.indexes[dim].equals(b.indexes[dim]), dim
    np.testing.assert_array_equal((a.labels != -1).values, (b.labels != -1).values)  # the same live cells
    assert (a.labels != -1).any()


# --- operating conditions (section 16.5) ---


@no_gain
def test_block_models_have_consistent_condition_dims():
    """No row of a block model has a term on a dimension the row lacks (every model here, conditions included)."""
    assert any("operating_condition" in built(k).variables[n].dims for k in ("net_import_expected", "conditions_two") for n in built(k).variables)
    for key in BUILDERS:
        assert_condition_dims_consistent(built(key))


def test_two_identical_conditions_are_neutral_with_blocks():
    """Two conditions of weight 1/2 and the same inputs give the optimum without conditions, block variables too."""
    m0, m2 = solved_model("conditions"), solved_model("conditions_even")
    assert m2.objective.value == pytest.approx(m0.objective.value, rel=1e-9)
    for name in ("operation_conversion_block_2h_power", "operation_load_shedding_block_2h_power", "operation_spillage_block_2h_power"):
        want = solved(m0, name)
        for condition in m2.variables[name].indexes["operating_condition"]:
            got = solved(m2, name).sel(operating_condition=condition, drop=True)
            assert_same(got, want, f"{name} {condition}")
    assert float(abs(solved(m0, "operation_load_shedding_block_2h_power")).sum()) > 1.0  # shedding really occurs


@pytest.mark.parametrize("name", ["operation_adequacy_block_2h_constraint", "operation_adequacy_constraint"])
def test_coarse_duals_scaled_by_condition_dual_scale(name):
    """The raw dual of a row under conditions, divided by ``condition_dual_scale``, is the dual without them."""
    p = _conditioned(conditions_block_dataset())
    ref = solved_model("conditions").constraints[name].dual
    raw = solved_model("conditions_two").constraints[name].dual
    assert float(abs(ref).max()) > 1.0  # priced rows: a unique dual, not zeros (hourly: ``power``, shed at 5)
    if name.startswith("operation_adequacy_block"):
        assert float(ref.max() - ref.min()) > 0.1  # the prices differ from block to block
    price = raw / condition_dual_scale(p)
    for k in p["operating_condition"].values:
        np.testing.assert_allclose(
            price.sel(operating_condition=k, drop=True).transpose(*ref.dims).values, ref.values, rtol=1e-6, atol=1e-9
        )
