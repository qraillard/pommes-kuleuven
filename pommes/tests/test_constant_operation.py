"""Constant operation where the ramp is 0.

A conversion unit whose ramp up and ramp down are both 0 holds ``P_t = P_(t-1)``. pommes writes this as ONE equality per
unit and step (``operation_conversion_constant_operation_constraint``) instead of the two ramp inequalities. A combined
unit does the same on the sum over modes (``operation_combined_constant_operation_constraint``) when the aggregate ramps
are 0, and on every mode (``operation_combined_mode_constant_operation_constraint``) when the mode ramps are 0 too, which
replaces the aggregate equality and the pairwise mode rows (a constant sum and constant pairwise differences fix every
mode). A single zero (ramp up 0, ramp down NaN or positive) and every other value keep the inequalities.

The objective values and the row counts "before" written in the tests are the ones of the code BEFORE this change
(pommes-milp 776b540), run once on the same data (toys ``_conversion_toy`` and ``_combined_toy``). Each toy has 10 steps,
so a constraint on one unit has 9 rows (the first step has none) and one on 2 modes has 18.
"""

import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model

CONV_EQ = "operation_conversion_constant_operation_constraint"
CONV_UP = "operation_conversion_ramp_up_constraint"
CONV_DOWN = "operation_conversion_ramp_down_constraint"
COMB_EQ = "operation_combined_constant_operation_constraint"
COMB_MODE_EQ = "operation_combined_mode_constant_operation_constraint"
COMB_UP = "operation_combined_ramp_up_constraint"
COMB_DOWN = "operation_combined_ramp_down_constraint"
PAIR_UP = "operation_combined_mode_ramp_up_constraint_electric_fossil"
PAIR_DOWN = "operation_combined_mode_ramp_down_constraint_electric_fossil"

NAN = np.nan
STEPS = 9  # rows of one constraint on one unit (10 steps, the first has no previous step)

# electricity demand of the conversion toy [MWh per 1 h step]; the wind availability of the conftest is
# [0.5, 0.7, 0.1, 0.0, 0.2, 0.4, 0.8, 0.9, 0.6, 0.3]
CONVERSION_DEMAND = np.array([4.0, 6.0, 8.0, 3.0, 9.0, 7.0, 5.0, 3.0, 6.0, 10.0])
# heat demand [MWh per step] and wind availability of the combined toy: the electric mode is cheaper than the fossil
# one (the methane of the fossil mode is shed at 5 per MWh), and the wind (10 MW) never falls under 4.5 MW
HEAT_DEMAND = np.array([2.0, 3.0, 6.0, 7.0, 5.0, 4.0, 3.0, 2.0, 6.0, 7.0])
WIND_AVAILABILITY = np.array([0.5, 0.7, 0.45, 0.6, 0.5, 0.8, 0.9, 0.45, 0.6, 0.55])


# ---------------------------------------------------------------------------------------------------------------------
# Toys
# ---------------------------------------------------------------------------------------------------------------------


def _conversion_toy(base, ramp_up, ramp_down, areas=("area_1",)):
    """ocgt and wind_onshore on ``areas``, 10 steps, varying electricity demand.

    ``ramp_up`` and ``ramp_down``: one value per technology ``[ocgt, wind_onshore]``, or a list of those per area.
    """
    p = base.sel(
        area=list(areas),
        conversion_tech=["ocgt", "wind_onshore"],
        resource=["electricity"],
        year_dec=[2040],
        year_inv=[2020],
        year_op=[2020],
    ).copy(deep=True)
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    p["demand"] = xr.DataArray(
        CONVERSION_DEMAND[:, None, None],
        coords={"hour": p.hour, "resource": p.resource, "year_op": p.year_op},
        dims=["hour", "resource", "year_op"],
    )
    for name, values in (("conversion_ramp_up", ramp_up), ("conversion_ramp_down", ramp_down)):
        values = np.asarray(values, dtype="float64")
        if values.ndim == 1:
            p[name] = xr.DataArray(values, coords={"conversion_tech": p.conversion_tech}, dims=["conversion_tech"])
        else:
            p[name] = xr.DataArray(
                values,
                coords={"area": p.area, "conversion_tech": p.conversion_tech},
                dims=["area", "conversion_tech"],
            )
    return p


def _combined_toy(base, combined, agg_up, agg_down, mode_up, mode_down, areas=("area_1",)):
    """The electric boiler (electric and fossil mode) next to a 10 MW wind farm, varying heat demand.

    ``agg_*`` and ``mode_*``: the values of ``combined_ramp_*`` and ``combined_mode_ramp_*`` (one value for all
    modes), a number or one number per area.
    """
    p = (
        xr.merge([base, combined])
        .sel(
            area=list(areas),
            conversion_tech=["wind_onshore"],
            resource=["electricity", "heat", "methane"],
            year_dec=[2040],
            year_inv=[2020],
            year_op=[2020],
        )
        .copy(deep=True)
    )
    p.operation_year_duration[:] = p.time_step_duration.sum().values
    p["load_shedding_cost"].loc[dict(resource="methane")] = 5
    p["demand"] = xr.DataArray(
        np.zeros((len(p.hour), len(p.resource), 1)),
        coords={"hour": p.hour, "resource": p.resource, "year_op": p.year_op},
        dims=["hour", "resource", "year_op"],
    )
    p["demand"].loc[dict(resource="heat")] = HEAT_DEMAND[:, None]
    p["conversion_power_capacity_investment_max"] = 10
    p["conversion_power_capacity_investment_min"] = 10
    p["conversion_annuity_cost"] = 0
    p["conversion_fixed_cost"] = 0
    p["conversion_availability"].loc[dict(conversion_tech="wind_onshore")] = WIND_AVAILABILITY
    for name, values in (
        ("combined_ramp_up", agg_up),
        ("combined_ramp_down", agg_down),
        ("combined_mode_ramp_up", mode_up),
        ("combined_mode_ramp_down", mode_down),
    ):
        values = np.asarray(values, dtype="float64")
        if values.ndim == 0:
            p[name] = xr.DataArray(float(values))
        else:
            p[name] = xr.DataArray(values, coords={"area": p.area}, dims=["area"])
    return p

def _three_modes(p):
    """The boiler of ``_combined_toy`` with a third, hybrid mode: 1 MWh of heat from 0.5 MWh of electricity and 0.75 MWh
    of methane."""
    p = p.reindex(mode=["electric", "fossil", "hybrid"]).copy(deep=True)
    p["combined_factor"].loc[dict(mode="hybrid")] = np.array([-0.5, 1.0, -0.75])
    p["combined_emission_factor"].loc[dict(mode="hybrid")] = 5.0
    return p


def _pairwise_rows(model):
    """Rows of all the pairwise mode ramp constraints (``operation_combined_mode_ramp_*_<mode>_<mode>``)."""
    return sum(_rows(model, name) for name in model.constraints if name.startswith("operation_combined_mode_ramp_"))


# ---------------------------------------------------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------------------------------------------------


def _solve(p):
    model = build_model(p)
    model.solve(solver_name="highs")
    assert model.status == "ok" and model.termination_condition == "optimal"
    return model, model.solution


def _rows(model, name):
    """Number of rows of a constraint (0 if the constraint does not exist)."""
    if name not in model.constraints:
        return 0
    return int((model.constraints[name].labels != -1).sum())


def _rows_per_unit(model, name, tech_dim):
    """Rows of a constraint per ``(area, tech)`` (summed over the other dims)."""
    labels = model.constraints[name].labels
    return (labels != -1).sum([d for d in labels.dims if d not in ("area", tech_dim)])


def _assert_difference_rows(model, name, summed_terms=1):
    """Every row of the constraint is ``x_t - x_(t-1) == 0``, with ``x`` a sum of ``summed_terms`` variables: coefficients
    +1 and -1, sign =, rhs 0."""
    flat = model.constraints[name].flat
    assert len(flat) > 0
    assert (flat.groupby("labels").size() == 2 * summed_terms).all()
    assert sorted(flat.coeffs.unique()) == [-1.0, 1.0]
    assert (flat.groupby("labels").coeffs.sum() == 0).all()
    assert set(flat.sign.unique()) == {"="}
    assert (flat.rhs == 0).all()


def _ocgt_power(s, area="area_1"):
    """Power of ocgt in ``area`` over the hours."""
    return s.operation_conversion_power.sel(conversion_tech="ocgt", area=area, year_op=2020)


def _combined_power(s, area="area_1"):
    """Power of the boiler in ``area`` as an array ``(mode, hour)``."""
    power = s.operation_combined_power.sel(area=area, combined_tech="electric_boiler", year_op=2020)
    return power.transpose("mode", "hour")


def _is_constant(x):
    return float(x.max() - x.min()) < 1e-9


# ---------------------------------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------------------------------


def test_conversion_ramps_0_make_one_equality(parameters_dispatch_invest):
    """Check 1: ocgt with ramp up 0 and ramp down 0 (wind_onshore: no ramp)."""
    p = _conversion_toy(parameters_dispatch_invest, [0.0, NAN], [0.0, NAN])
    model, s = _solve(p)

    assert _rows(model, CONV_EQ) == STEPS
    per_unit = _rows_per_unit(model, CONV_EQ, "conversion_tech")
    assert int(per_unit.sel(conversion_tech="ocgt").sum()) == STEPS
    assert int(per_unit.sel(conversion_tech="wind_onshore").sum()) == 0
    _assert_difference_rows(model, CONV_EQ)

    # the ramp rows of that unit are gone (wind_onshore has none either)
    assert _rows(model, CONV_UP) == 0
    assert _rows(model, CONV_DOWN) == 0

    assert _is_constant(_ocgt_power(s))
    # master: 85 rows (9 + 9 ramp rows replaced by 9), objective 830.0
    assert model.ncons == 85 - STEPS
    assert model.nvars == 61
    assert model.objective.value == pytest.approx(830.0, rel=1e-9)


def test_conversion_ramp_up_0_and_ramp_down_nan_keeps_the_ramp_up_row(parameters_dispatch_invest):
    """Check 2: a single 0 means "never increase": no equality, the ramp-up row stays."""
    p = _conversion_toy(parameters_dispatch_invest, [0.0, NAN], [NAN, NAN])
    model, s = _solve(p)

    assert CONV_EQ not in model.constraints
    assert _rows(model, CONV_UP) == STEPS
    assert _rows(model, CONV_DOWN) == 0
    power = _ocgt_power(s).values
    assert (np.diff(power) <= 1e-9).all()  # never increases
    assert not _is_constant(_ocgt_power(s))
    # master: same model
    assert model.ncons == 76
    assert model.objective.value == pytest.approx(628.0, rel=1e-9)


def test_conversion_ramp_up_0_and_ramp_down_positive_keeps_both_ramp_rows(parameters_dispatch_invest):
    p = _conversion_toy(parameters_dispatch_invest, [0.0, NAN], [0.3, NAN])
    model, _ = _solve(p)

    assert CONV_EQ not in model.constraints
    assert _rows(model, CONV_UP) == STEPS
    assert _rows(model, CONV_DOWN) == STEPS
    assert model.ncons == 85
    assert model.objective.value == pytest.approx(637.6, rel=1e-9)


def test_conversion_equality_only_on_the_units_with_both_ramps_0(parameters_dispatch_invest):
    """Two areas: ocgt has ramps 0 in area_1 and 0.4 in area_2, so the equality is on one unit and the rows of the
    other unit stay."""
    p = _conversion_toy(
        parameters_dispatch_invest,
        [[0.0, NAN], [0.4, NAN]],
        [[0.0, NAN], [0.4, NAN]],
        areas=("area_1", "area_2"),
    )
    model, s = _solve(p)

    eq = _rows_per_unit(model, CONV_EQ, "conversion_tech")
    up = _rows_per_unit(model, CONV_UP, "conversion_tech")
    down = _rows_per_unit(model, CONV_DOWN, "conversion_tech")
    for area, expected_eq, expected_ramp in (("area_1", STEPS, 0), ("area_2", 0, STEPS)):
        assert int(eq.sel(area=area, conversion_tech="ocgt")) == expected_eq
        assert int(up.sel(area=area, conversion_tech="ocgt")) == expected_ramp
        assert int(down.sel(area=area, conversion_tech="ocgt")) == expected_ramp
    assert int(eq.sel(conversion_tech="wind_onshore").sum() + up.sel(conversion_tech="wind_onshore").sum()) == 0
    assert _is_constant(_ocgt_power(s, "area_1"))
    # master: 170 rows (ramp rows 18 + 18), objective 1416.0
    assert model.ncons == 170 - STEPS
    assert model.nvars == 122
    assert model.objective.value == pytest.approx(1416.0, rel=1e-9)


# ---------------------------------------------------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------------------------------------------------


def test_combined_ramps_and_mode_ramps_0_make_one_equality_per_mode(parameters_dispatch_invest, combined):
    """Check 3: aggregate and mode ramps all 0 -> every mode constant, no aggregate or pairwise row."""
    p = _combined_toy(parameters_dispatch_invest, combined, 0.0, 0.0, 0.0, 0.0)
    model, s = _solve(p)

    assert _rows(model, COMB_MODE_EQ) == 2 * STEPS  # 2 modes
    _assert_difference_rows(model, COMB_MODE_EQ)
    for name in (COMB_EQ, COMB_UP, COMB_DOWN, PAIR_UP, PAIR_DOWN):
        assert _rows(model, name) == 0, name

    power = _combined_power(s)
    assert _is_constant(power.sel(mode="electric"))
    assert _is_constant(power.sel(mode="fossil"))
    assert float(power.sel(mode="electric").mean()) == pytest.approx(4.5)  # the wind never gives less than 4.5 MW
    assert float(power.sel(mode="fossil").mean()) == pytest.approx(2.5)  # 7 MW at the peak of the heat demand
    # master: 148 rows (9 + 9 aggregate, 9 + 9 pairwise), objective 1027.5
    assert model.ncons == 148 - 4 * STEPS + 2 * STEPS
    assert model.nvars == 121
    assert model.objective.value == pytest.approx(1027.5, rel=1e-9)


def test_combined_aggregate_ramps_0_only_fixes_the_sum_not_the_modes(parameters_dispatch_invest, combined):
    """Check 4: aggregate ramps 0, no mode ramp -> the sum is constant, the modes shift."""
    p = _combined_toy(parameters_dispatch_invest, combined, 0.0, 0.0, NAN, NAN)
    model, s = _solve(p)

    assert _rows(model, COMB_EQ) == STEPS
    _assert_difference_rows(model, COMB_EQ, summed_terms=2)  # the sum over the 2 modes
    for name in (COMB_MODE_EQ, COMB_UP, COMB_DOWN, PAIR_UP, PAIR_DOWN):
        assert _rows(model, name) == 0, name

    power = _combined_power(s)
    assert _is_constant(power.sum("mode"))
    assert float(power.sum("mode").mean()) == pytest.approx(7.0)
    assert not _is_constant(power.sel(mode="electric"))  # it follows the wind
    assert not _is_constant(power.sel(mode="fossil"))
    # master: 130 rows (9 + 9 aggregate ramp rows), objective 933.75
    assert model.ncons == 130 - STEPS
    assert model.objective.value == pytest.approx(933.75, rel=1e-9)


def test_combined_mode_ramp_up_0_only_keeps_the_pairwise_row_and_the_aggregate_equality(
    parameters_dispatch_invest, combined
):
    """Aggregate ramps 0, mode ramp up 0 and mode ramp down NaN: the modes are not fixed, so the sum equality and the
    pairwise ramp-up rows stay."""
    p = _combined_toy(parameters_dispatch_invest, combined, 0.0, 0.0, 0.0, NAN)
    model, s = _solve(p)

    assert _rows(model, COMB_EQ) == STEPS
    assert _rows(model, PAIR_UP) == STEPS
    for name in (COMB_MODE_EQ, COMB_UP, COMB_DOWN, PAIR_DOWN):
        assert _rows(model, name) == 0, name
    assert _is_constant(_combined_power(s).sum("mode"))
    # master: 139 rows, objective 1020.0
    assert model.ncons == 139 - STEPS
    assert model.objective.value == pytest.approx(1020.0, rel=1e-9)


def test_combined_aggregate_ramp_up_0_and_ramp_down_nan_keeps_the_ramp_up_row(parameters_dispatch_invest, combined):
    p = _combined_toy(parameters_dispatch_invest, combined, 0.0, NAN, NAN, NAN)
    model, _ = _solve(p)

    assert COMB_EQ not in model.constraints and COMB_MODE_EQ not in model.constraints
    assert _rows(model, COMB_UP) == STEPS
    assert _rows(model, COMB_DOWN) == 0
    assert model.ncons == 121
    assert model.objective.value == pytest.approx(933.75, rel=1e-9)


def test_combined_mode_ramps_0_with_nonzero_aggregate_ramps_keep_every_row(parameters_dispatch_invest, combined):
    """Mode ramps 0 but aggregate ramps 0.5: the modes may still shift together, nothing is replaced."""
    p = _combined_toy(parameters_dispatch_invest, combined, 0.5, 0.5, 0.0, 0.0)
    model, _ = _solve(p)

    assert COMB_EQ not in model.constraints and COMB_MODE_EQ not in model.constraints
    for name in (COMB_UP, COMB_DOWN, PAIR_UP, PAIR_DOWN):
        assert _rows(model, name) == STEPS, name
    assert model.ncons == 148
    assert model.objective.value == pytest.approx(880.75, rel=1e-9)


def test_combined_equalities_only_on_the_units_with_the_zero_ramps(parameters_dispatch_invest, combined):
    """Two areas: area_1 has aggregate and mode ramps 0 (equality per mode), area_2 only the aggregate ramps 0 (sum
    equality). Same data in both areas, so the objective is the sum of the two single-area objectives (1027.5 and
    933.75)."""
    p = _combined_toy(
        parameters_dispatch_invest,
        combined,
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, NAN],
        [0.0, NAN],
        areas=("area_1", "area_2"),
    )
    model, s = _solve(p)

    mode_eq = _rows_per_unit(model, COMB_MODE_EQ, "combined_tech")
    eq = _rows_per_unit(model, COMB_EQ, "combined_tech")
    assert int(mode_eq.sel(area="area_1").sum()) == 2 * STEPS and int(mode_eq.sel(area="area_2").sum()) == 0
    assert int(eq.sel(area="area_1").sum()) == 0 and int(eq.sel(area="area_2").sum()) == STEPS
    for name in (COMB_UP, COMB_DOWN, PAIR_UP, PAIR_DOWN):
        assert _rows(model, name) == 0, name

    assert _is_constant(_combined_power(s, "area_1").sel(mode="electric"))
    assert _is_constant(_combined_power(s, "area_1").sel(mode="fossil"))
    assert _is_constant(_combined_power(s, "area_2").sum("mode"))
    assert not _is_constant(_combined_power(s, "area_2").sel(mode="electric"))
    # master: (9 + 9 + 9 + 9) + (9 + 9) rows, objective 1027.5 + 933.75
    assert model.objective.value == pytest.approx(1027.5 + 933.75, rel=1e-9)


def test_combined_three_modes_all_ramps_0(parameters_dispatch_invest, combined):
    """Three modes: 3 pairs of modes give 6 x 9 pairwise rows on master, replaced by one equality per mode."""
    p = _three_modes(_combined_toy(parameters_dispatch_invest, combined, 0.0, 0.0, 0.0, 0.0))
    model, s = _solve(p)

    assert _rows(model, COMB_MODE_EQ) == 3 * STEPS
    _assert_difference_rows(model, COMB_MODE_EQ)
    assert _pairwise_rows(model) == 0
    for name in (COMB_EQ, COMB_UP, COMB_DOWN):
        assert _rows(model, name) == 0, name
    power = _combined_power(s)
    assert list(power.mode.values) == ["electric", "fossil", "hybrid"]
    for mode in power.mode.values:
        assert _is_constant(power.sel(mode=mode)), mode
    # master: 184 rows (2 x 9 aggregate and 6 x 9 pairwise ramp rows), objective 1027.5
    assert model.ncons == 184 - 8 * STEPS + 3 * STEPS
    assert model.objective.value == pytest.approx(1027.5, rel=1e-9)


# ---------------------------------------------------------------------------------------------------------------------
# Nonzero ramps: nothing changes
# ---------------------------------------------------------------------------------------------------------------------


def test_conversion_nonzero_ramps_are_unchanged(parameters_dispatch_invest):
    """Check 5, conversion: both technologies with ramp up 0.5 and ramp down 0.4."""
    p = _conversion_toy(parameters_dispatch_invest, [0.5, 0.5], [0.4, 0.4])
    model, _ = _solve(p)

    assert CONV_EQ not in model.constraints
    assert _rows(model, CONV_UP) == 2 * STEPS
    assert _rows(model, CONV_DOWN) == 2 * STEPS
    # master: the same model
    assert model.nvars == 61 and model.ncons == 103
    assert model.objective.value == pytest.approx(796.0, rel=1e-9)


def test_combined_nonzero_ramps_are_unchanged(parameters_dispatch_invest, combined):
    """Check 5, combined: aggregate ramps 0.5 and 0.4, mode ramps 0.3 and 0.2."""
    p = _combined_toy(parameters_dispatch_invest, combined, 0.5, 0.4, 0.3, 0.2)
    model, _ = _solve(p)

    assert COMB_EQ not in model.constraints and COMB_MODE_EQ not in model.constraints
    for name in (COMB_UP, COMB_DOWN, PAIR_UP, PAIR_DOWN):
        assert _rows(model, name) == STEPS, name
    # master: the same model
    assert model.nvars == 121 and model.ncons == 148
    assert model.objective.value == pytest.approx(848.55, rel=1e-9)


def test_no_ramp_gives_no_constant_operation_constraint(parameters_dispatch_invest, combined):
    """The default (ramps NaN): the model has none of the new constraints."""
    p = _conversion_toy(parameters_dispatch_invest, [NAN, NAN], [NAN, NAN])
    model, _ = _solve(p)
    assert CONV_EQ not in model.constraints
    assert model.nvars == 61 and model.ncons == 67  # master
    assert model.objective.value == pytest.approx(518.5, rel=1e-9)

    p = _combined_toy(parameters_dispatch_invest, combined, NAN, NAN, NAN, NAN)
    model, _ = _solve(p)
    assert COMB_EQ not in model.constraints and COMB_MODE_EQ not in model.constraints
    assert model.nvars == 121 and model.ncons == 112  # master
    assert model.objective.value == pytest.approx(845.0, rel=1e-9)
