"""Outputs of block models: expand_to_hours, summaries, net generation, plots, save and read (multi-resolution, section 11)."""

import importlib.util
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

import pommes
from pommes.io.calculate_summaries import calculate_total_net_generation
from pommes.io.read_solution import read_dual, read_solution
from pommes.io.save_solution import save_solution
from pommes.model.build_model import build_model
from pommes.post_process import get_net_generation, module_hourly
from pommes.tests.time_blocks_helper import (
    BLOCK_PARAMETERS,
    KILNS,
    combined_all_block_dataset,
    combined_derived_coarse_dataset,
    combined_t2_dataset,
    conversion_all_block_dataset,
    conversion_t2_dataset,
    conversion_two_lengths_dataset,
    densify_any,
    flat_twin,
    hand_toy_b,
    net_import_two_lengths,
    solve_model,
    transport_t2_availability_dataset,
    transport_hand_toy,
    transport_t2_two_lengths_dataset,
    _transport_t2,
    with_blocks,
)
from pommes.utils import expand_to_hours, time_blocks
from pommes.visualisation.energy_cost_plot import (
    adequacy_price,
    shedding_spillage_energy,
)

OUTER_JOIN = "Coordinates across variables not equal"
ALL_CONVERSION = {t: 4 for t in ["well", "backup", *KILNS]}
ALL_COMBINED = {t: 2 for t in ["electric_boiler", "boiler_block", "steady_kiln"]}
# dataset builder -> lengths of the flat twin (the technologies of the module that sit on blocks)
CASES = {
    "conversion_all_block": (conversion_all_block_dataset, dict(conversion=ALL_CONVERSION)),
    "combined_all_block": (combined_all_block_dataset, dict(combined=ALL_COMBINED)),
    "conversion_t2": (conversion_t2_dataset, dict(conversion={t: 4 for t in KILNS})),
    "combined_t2": (combined_t2_dataset, dict(combined={"boiler_block": 2})),
}


def read(m) -> xr.Dataset:
    """
    The whole solution of ``m`` as the readers get it: outer-join warning filtered, units unstacked.

    The Dataset counterpart of ``solved`` (which gives one variable, groups dropped).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=OUTER_JOIN)
        return m.solution.map(densify_any)


def read_dual_of(m, names) -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=OUTER_JOIN)
        return xr.Dataset({n: m.constraints[n].dual for n in names})


def solve_pair(p, **lengths):
    """A solved block model and its solved flat twin, with their solutions."""
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, **lengths))
    return m, twin, read(m), read(twin)


def hourly_params(p: xr.Dataset) -> xr.Dataset:
    return p.drop_vars(BLOCK_PARAMETERS, errors="ignore")


def assert_same(a: xr.DataArray, b: xr.DataArray, name: str = "") -> None:
    """Equal on the union of the labels, a label missing on one side counting as 0."""
    a, b = xr.align(a, b, join="outer", fill_value=0)
    np.testing.assert_allclose(a.fillna(0).values, b.transpose(*a.dims).fillna(0).values, atol=1e-6, err_msg=name)


# ---------------------------------------------------------------------------
# expand_to_hours
# ---------------------------------------------------------------------------


def _blocks_of_four():
    """Steps (1, 1, 2, 1, 1, 2, 1, 1) h in blocks of 4 h: durations 4, 4, 2."""
    p = conversion_t2_dataset()
    tb = time_blocks(p, warn=False)
    values = xr.DataArray(
        [[8.0, 6.0, 5.0], [1.0, 2.0, 3.0]],
        dims=("area", "time_block_4h"),
        coords={"area": ["a", "b"], "time_block_4h": [0, 1, 2]},
    )
    return tb, values


def test_expand_to_hours_energy_keeps_the_block_sum():
    tb, values = _blocks_of_four()
    out = expand_to_hours(values, tb, "energy")
    assert out.dims == ("area", "hour")
    # a step takes its share of the block: hour 2 lasts 2 h of the 4 h block
    np.testing.assert_allclose(out.sel(area="a").values, [2, 2, 4, 1.5, 1.5, 3, 2.5, 2.5])
    back = out.groupby(tb.block_of_hour[4]).sum()
    np.testing.assert_allclose(back.values, values.values)


def test_expand_to_hours_intensive_repeats():
    tb, values = _blocks_of_four()
    out = expand_to_hours(values, tb, "intensive")
    np.testing.assert_allclose(out.sel(area="a").values, [8, 8, 8, 6, 6, 6, 5, 5])
    np.testing.assert_array_equal(out["hour"].values, tb.block_of_hour[4]["hour"].values)


def test_expand_to_hours_refuses_an_unknown_kind_and_a_stale_axis():
    tb, values = _blocks_of_four()
    with pytest.raises(ValueError, match="kind"):
        expand_to_hours(values, tb, "power")
    with pytest.raises(ValueError, match="time_block_4h"):
        expand_to_hours(values.isel(time_block_4h=[0, 1]), tb, "energy")


def test_time_blocks_warn_false_skips_only_the_why_hourly_warning():
    p = conversion_t2_dataset()
    with pytest.warns(UserWarning, match="no gain"):
        time_blocks(p)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        time_blocks(p, warn=False)
        with pytest.raises(ValueError):  # a refusal is not a warning
            time_blocks(with_blocks(p, conversion={"kiln_a": 2.5}), warn=False)


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def test_summaries_combine_two_lengths():
    p = conversion_two_lengths_dataset()
    m, twin, s, s_twin = solve_pair(p, conversion={"kiln_a": 4, "kiln_b": 4, "kiln_c": 2, "kiln_d": 2})
    assert "operation_conversion_block_2h_net_generation" in s and "operation_conversion_block_4h_net_generation" in s
    table = calculate_total_net_generation(s, p)
    ref = calculate_total_net_generation(s_twin, hourly_params(p))
    assert table.index.is_unique
    assert list(table.columns) == list(ref.columns)
    table, ref = table.sort_index(), ref.sort_index()
    assert table.index.equals(ref.index)
    np.testing.assert_allclose(table.values, ref.values, atol=1e-6)
    # something is produced by each length, so a table overwritten by the other one would show
    for kiln in ("kiln_a", "kiln_c"):
        assert table.xs(kiln, level="tech").abs().values.sum() > 1e-6


def test_summaries_combine_net_import_hourly_and_blocks_along_resource():
    p = net_import_two_lengths()
    m = solve_model(build_model(p))
    s = read(m)
    table = calculate_total_net_generation(s, p)
    assert table.index.is_unique
    net = table.xs("net_import", level="type")["operation_net_import_net_generation"]
    by_resource = net.groupby(level="resource").sum()
    assert set(by_resource.index) == {"idle", "gas", "heat"}
    hand = {
        "idle": s["operation_net_import_net_generation"].sel(resource="idle"),
        "gas": s["operation_net_import_block_2h_net_generation"].sel(resource="gas"),
        "heat": s["operation_net_import_block_4h_net_generation"].sel(resource="heat"),
    }
    for resource, da in hand.items():
        assert by_resource[resource] == pytest.approx(float(da.sum()), abs=1e-6)
    assert abs(by_resource["gas"]) > 1e-6 and abs(by_resource["heat"]) > 1e-6  # the coarse groups are read


# ---------------------------------------------------------------------------
# net generation, plots
# ---------------------------------------------------------------------------


def test_get_net_generation_with_an_all_block_module():
    p = combined_all_block_dataset()
    m, twin, s, s_twin = solve_pair(p, combined=ALL_COMBINED)
    assert "operation_combined_net_generation" not in s  # no hourly combined variable
    net = get_net_generation(s, p)
    ref = get_net_generation(s_twin, hourly_params(p))
    assert set(net.category.values) == set(ref.category.values)
    assert abs(net.sel(category=list(ALL_COMBINED)).sum()) > 1e-6
    assert_same(net, ref)


def test_get_net_generation_includes_coarse_shedding_and_spill():
    p = combined_derived_coarse_dataset()
    m, twin, s, s_twin = solve_pair(p, combined=ALL_COMBINED)
    tb = time_blocks(p, warn=False)
    assert tb.resources(2) == ["heat"]
    net = get_net_generation(s, p)
    ref = get_net_generation(s_twin, hourly_params(p))
    shed, spill = (net.sel(category=c).sel(resource="heat") for c in ("load_shedding", "spillage"))
    assert shed.sum() > 0.1 and spill.sum() < -0.1  # both exist in the block model, and are read
    # the block variables are powers: the energy of a block is D_L x power
    want = expand_to_hours(tb.duration[2] * s["operation_load_shedding_block_2h_power"], tb, "energy").sel(resource="heat")
    assert_same(shed, want.reindex_like(shed, fill_value=0))
    assert_same(net, ref)


@pytest.mark.parametrize("case", CASES)
def test_readers_run_on_a_block_model_and_equal_the_twin(case):
    """The data behind the plots (net generation, conversion or combined power) equals the twin's, without a warning."""
    make, lengths = CASES[case]
    p = make()
    m, twin, s, s_twin = solve_pair(p, **lengths)
    module = "combined" if "combined" in lengths else "conversion"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the readers never re-emit the why-hourly warning
        net = get_net_generation(s, p)
        power = module_hourly(s, module, "power", p)
    # a module with only block technologies has no hourly variable
    assert ("operation_conversion_power" in s) == (case != "conversion_all_block")
    assert ("operation_combined_power" in s) == (case == "combined_t2")
    assert abs(power).sum() > 1e-6
    assert_same(net, get_net_generation(s_twin, hourly_params(p)))
    assert_same(power, module_hourly(s_twin, module, "power", hourly_params(p)))


@pytest.mark.parametrize("case", CASES)
def test_plots_run_on_a_block_model(case, tmp_path):
    pytest.importorskip("plotly")  # an optional dependency of the plots
    from pommes.common_plots import plot_power

    make, lengths = CASES[case]
    p = make()
    m, twin, s, s_twin = solve_pair(p, **lengths)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_power(s, str(tmp_path), parameters=p)
    fig_twin = plot_power(s_twin, str(tmp_path / "twin"), parameters=hourly_params(p))
    assert (tmp_path / "power.html").is_file()

    def traces(f):
        return {
            t.name: sorted(zip(np.asarray(t.x, float), np.nan_to_num(np.asarray(t.y, float), nan=-1.0)))
            for t in f.data
        }

    got, want = traces(fig), traces(fig_twin)
    assert got.keys() == want.keys()
    for tech in want:
        np.testing.assert_allclose(got[tech], want[tech], atol=1e-6, err_msg=tech)
    assert sum(abs(y) for t in got.values() for _, y in t) > 1e-6


def test_plot_power_asks_for_the_parameters_of_a_block_solution(tmp_path):
    pytest.importorskip("plotly")
    from pommes.common_plots import plot_power

    p = conversion_all_block_dataset()
    s = read(solve_model(build_model(p)))
    with pytest.raises(ValueError, match="parameters"):
        plot_power(s, str(tmp_path))


def test_module_hourly_returns_the_hourly_variable_untouched_without_blocks():
    p = hourly_params(conversion_all_block_dataset())
    s = read(solve_model(build_model(p)))
    xr.testing.assert_identical(module_hourly(s, "conversion", "power", p), s["operation_conversion_power"])
    assert module_hourly(s, "combined", "power", p) is None  # off: skipped


def test_energy_cost_plot_reads_coarse_shedding_spillage_and_duals():
    """C8: the readers of ``plot_energy_cost`` (its own run needs a model with every module, which fails for other reasons)."""
    p = combined_derived_coarse_dataset()
    m, twin, s, s_twin = solve_pair(p, combined=ALL_COMBINED)
    tb = time_blocks(p, warn=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the readers of the plot never re-emit the why-hourly warning
        shed, spill = shedding_spillage_energy(s, p)
    for got, name in ((shed, "load_shedding"), (spill, "spillage")):
        block = s[f"operation_{name}_block_2h_power"]
        want = expand_to_hours(tb.duration[2] * block, tb, "energy")  # coarse: D_L x power, spread over the steps
        assert got.sel(resource="heat").sum() > 0.1  # a binding case: the energy exists
        assert_same(got.sel(resource=["heat"]), want)
    shed_twin, spill_twin = shedding_spillage_energy(s_twin, hourly_params(p))
    assert_same(shed, shed_twin)
    assert_same(spill, spill_twin)
    # the coarse dual of heat is read with kind="intensive": one value per block, repeated over its steps
    names = ["operation_adequacy_constraint", "operation_adequacy_block_2h_constraint"]
    dual = read_dual_of(m, names)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        price = adequacy_price(dual, p)
    assert abs(price.sel(resource="heat")).max() > 1e-6
    heat = expand_to_hours(dual[names[1]], tb, "intensive")
    assert_same(price.sel(resource=["heat"]), heat)
    others = [r for r in price.resource.values if r != "heat"]
    assert_same(price.sel(resource=others), dual[names[0]].sel(resource=others))


def test_coarse_dual_one_value_per_block():
    p = combined_derived_coarse_dataset()
    m = solve_model(build_model(p))
    tb = time_blocks(p, warn=False)
    dual = read_dual_of(m, ["operation_adequacy_block_2h_constraint"])["operation_adequacy_block_2h_constraint"]
    assert dual.dims == ("area", "time_block_2h", "resource", "year_op")
    assert dual.sizes["time_block_2h"] == tb.duration[2].size
    assert list(dual.resource.values) == ["heat"]
    assert abs(dual).max() > 1e-6
    # the dual of an energy balance is in EUR/MWh whatever the length: expanded, it is constant inside each block
    hours = expand_to_hours(dual, tb, "intensive")
    for b in range(tb.duration[2].size):
        v = hours.isel(hour=np.flatnonzero(tb.block_of_hour[2].values == b)).values
        assert (v == v[..., :1]).all()


# ---------------------------------------------------------------------------
# save and read
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:In a future version of xarray the default value for join")  # read_solution
@pytest.mark.filterwarnings("ignore:The inplace keyword in DataFrame.set_index")  # read_solution
def test_save_and_read_a_block_solution(tmp_path):
    p = conversion_all_block_dataset()
    m = solve_model(build_model(p))
    save_solution(m, str(tmp_path), model_parameters=p)
    sol = xr.open_dataset(tmp_path / "solution.nc")
    dual = xr.open_dataset(tmp_path / "dual.nc")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=OUTER_JOIN)
        orig = m.solution
    name = "operation_conversion_block_4h_power"
    assert name in sol and "operation_adequacy_block_4h_constraint" in dual
    np.testing.assert_allclose(sol[name].transpose(*orig[name].dims).values, orig[name].values, atol=1e-9)
    coarse = m.constraints["operation_adequacy_block_4h_constraint"].dual
    saved = dual["operation_adequacy_block_4h_constraint"].sel(resource=coarse.indexes["resource"])
    np.testing.assert_allclose(saved.transpose(*coarse.dims).values, coarse.values, atol=1e-9)
    # CSV
    csv = read_solution(str(tmp_path / "variables"))
    for variable in (name, "operation_conversion_block_4h_net_generation"):
        assert float(csv[variable].sum()) == pytest.approx(float(np.nansum(orig[variable].values)), abs=1e-5)
    csv_dual = read_dual(str(tmp_path / "constraints"))
    assert float(csv_dual["operation_adequacy_block_4h_constraint"].sum()) == pytest.approx(float(coarse.sum()), abs=1e-5)
    # the saved input still gives the same blocks (the rule is a string array)
    back = time_blocks(xr.open_dataset(tmp_path / "input.nc"), warn=False)
    ref = time_blocks(p, warn=False)
    np.testing.assert_array_equal(back.tech_length["conversion"].values, ref.tech_length["conversion"].values)
    assert back.balance_lengths == ref.balance_lengths == [4]


def test_outer_join_warning_is_filtered(tmp_path):
    p = hand_toy_b()
    m = solve_model(build_model(p))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_solution(m, str(tmp_path), model_parameters=p)
    assert (tmp_path / "solution.nc").is_file()
    # the filter is local: the warning is still raised by a plain read of the solution
    with pytest.warns(UserWarning, match=OUTER_JOIN):
        m.solution


# ---------------------------------------------------------------------------
# transport technologies on blocks (0.8.0)
# ---------------------------------------------------------------------------


def _all_block_transport() -> xr.Dataset:
    """The ship ``S`` alone, on 24 h blocks: the model has no hourly transport variable."""
    return with_blocks(_transport_t2([1.0] * 48, {"S": 3.0}, techs=("S",)), transport={"S": 24})


def test_get_net_generation_with_an_all_block_transport():
    p = _all_block_transport()
    m, twin, s, s_twin = solve_pair(p, transport={"S": 24})
    assert "operation_transport_net_generation" not in s and "operation_transport_block_24h_net_generation" in s
    net = get_net_generation(s, p)
    ref = get_net_generation(s_twin, hourly_params(p))
    assert set(net.category.values) == set(ref.category.values)
    assert abs(net.sel(category="S")).sum() > 1e-6
    assert_same(net, ref)


def test_get_net_generation_with_a_mix():
    p = transport_t2_availability_dataset()
    m, twin, s, s_twin = solve_pair(p, transport={"S": 24})
    assert "operation_transport_net_generation" in s and "operation_transport_block_24h_net_generation" in s
    net = get_net_generation(s, p)
    ref = get_net_generation(s_twin, hourly_params(p))
    assert set(net.category.values) >= {"H", "S"}
    for tech in ("H", "S"):  # both groups are read
        assert abs(net.sel(category=tech)).sum() > 1e-6
    assert_same(net, ref)


def test_module_hourly_transport():
    p = transport_t2_availability_dataset()
    m, twin, s, s_twin = solve_pair(p, transport={"S": 24})
    for stem in ("power", "power_in", "net_generation"):
        got = module_hourly(s, "transport", stem, p)
        assert got.dims[0] != "time_block_24h" and "hour" in got.dims
        assert abs(got.sel(transport_tech="S")).sum() > 1e-6
        assert_same(got, module_hourly(s_twin, "transport", stem, hourly_params(p)), stem)


def test_expand_to_hours_transport_energy_sums_back():
    p = transport_t2_availability_dataset()
    tb = time_blocks(p, warn=False)
    s = read(solve_model(build_model(p)))
    block = s["operation_transport_block_24h_net_generation"]
    hours = expand_to_hours(block, tb, "energy")
    assert hours.sizes["hour"] == 48 and abs(hours).sum() > 1e-6
    b = tb.block_of_hour[24].values
    for k in range(block.sizes["time_block_24h"]):
        assert_same(hours.isel(hour=np.flatnonzero(b == k)).sum("hour"), block.isel(time_block_24h=k, drop=True))


def test_summaries_transport_two_lengths():
    p = transport_t2_two_lengths_dataset()
    m, twin, s, s_twin = solve_pair(p, transport={"S": 12, "T": 24})
    assert {"operation_transport_block_12h_net_generation", "operation_transport_block_24h_net_generation"} <= set(s)
    table = calculate_total_net_generation(s, p)
    ref = calculate_total_net_generation(s_twin, hourly_params(p))
    assert table.index.is_unique
    table, ref = table.sort_index(), ref.sort_index()
    assert table.index.equals(ref.index)
    np.testing.assert_allclose(table.values, ref.values, atol=1e-6)
    for tech in ("S", "T"):  # each length carries something, so a table overwritten by the other would show
        assert np.nansum(np.abs(table.xs(tech, level="tech").values)) > 1e-6


@pytest.mark.filterwarnings("ignore:In a future version of xarray the default value for join")  # read_solution
@pytest.mark.filterwarnings("ignore:The inplace keyword in DataFrame.set_index")  # read_solution
def test_save_and_read_a_block_transport_solution(tmp_path):
    p = transport_hand_toy()  # the cargo balance is 24 h as well: coarse duals
    m = solve_model(build_model(p))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the outer-join warning on transport_tech is filtered by save_solution
        save_solution(m, str(tmp_path), model_parameters=p)
    sol = xr.open_dataset(tmp_path / "solution.nc")
    dual = xr.open_dataset(tmp_path / "dual.nc")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=OUTER_JOIN)
        orig = m.solution
    name = "operation_transport_block_24h_power"
    assert name in sol and "operation_adequacy_block_24h_constraint" in dual
    np.testing.assert_allclose(sol[name].transpose(*orig[name].dims).values, orig[name].values, atol=1e-9)
    coarse = m.constraints["operation_adequacy_block_24h_constraint"].dual
    saved = dual["operation_adequacy_block_24h_constraint"].sel(resource=coarse.indexes["resource"])
    np.testing.assert_allclose(saved.transpose(*coarse.dims).values, coarse.values, atol=1e-9)
    csv = read_solution(str(tmp_path / "variables"))
    for variable in (name, "operation_transport_block_24h_net_generation"):
        assert float(np.nansum(csv[variable].values)) == pytest.approx(float(np.nansum(orig[variable].values)), abs=1e-5)
    assert abs(float(np.nansum(csv[name].values))) > 1e-6
    csv_dual = read_dual(str(tmp_path / "constraints"))
    assert float(csv_dual["operation_adequacy_block_24h_constraint"].sum()) == pytest.approx(float(coarse.sum()), abs=1e-5)


def test_power_max_block_dual_is_per_block():
    """One 3 h block, availability 0.5 in one step: the dual of its power_max is that of the twin's binding hour, not three times it."""
    p = with_blocks(_transport_t2([1.0] * 3, {"S": 3.0}, techs=("S",), availability={"S": [1.0, 0.5, 1.0]}), transport={"S": 3})
    m, twin = solve_model(build_model(p)), solve_model(flat_twin(p, transport={"S": 3}))
    names = ("operation_transport_block_3h_power_max_constraint", "operation_transport_power_max_constraint")
    block, hourly = (read_dual_of(x, [n])[n].sel(transport_tech="S", year_op=2030) for x, n in zip((m, twin), names))
    block = block.squeeze("time_block_3h", drop=True)
    link = abs(hourly).sum("hour").idxmax("link").item()  # the link that carries the flow (the other one is degenerate)
    per_hour = hourly.sel(link=link).values  # [0, d, 0]: only the 0.5 hour binds in the twin
    assert abs(per_hour[1]) > 1e-6
    assert per_hour[0] == pytest.approx(0, abs=1e-6) and per_hour[2] == pytest.approx(0, abs=1e-6)
    assert float(block.sel(link=link)) == pytest.approx(per_hour[1], rel=1e-6)  # not repeated over the 3 hours


def test_toy_examples_read_the_hourly_variables():
    """``toy_trade.py`` and ``toy_policies.py`` set no length and read the hourly transport variables."""
    folder = Path(pommes.__file__).parent / "examples" / "toy_trade"
    for name in ("toy_trade.py", "toy_policies.py"):
        text = (folder / name).read_text()
        assert "transport_time_block_hrs" not in text and "operation_transport_power" in text
    spec = importlib.util.spec_from_file_location("toy_trade_outputs", folder / "toy_trade.py")
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    ds = toy.build_toy_dataset()
    assert "transport_time_block_hrs" not in ds or ds["transport_time_block_hrs"].isnull().all()
    m = build_model(ds)
    assert "operation_transport_power" in m.variables
    assert not [v for v in m.variables if v.startswith("operation_transport_block_")]
