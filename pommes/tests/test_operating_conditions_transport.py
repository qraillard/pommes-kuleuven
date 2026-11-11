"""Operating conditions in the transport module.

Two areas joined by two links (link_1 from area_1 to area_2, link_2 back), one electricity line,
wind in both areas. The demand and the wind move between the areas from one condition to the other.
"""
import numpy as np
import pytest
import xarray as xr

from pommes.model.build_model import build_model
from pommes.tests.conditions_helper import (
    assert_condition_dims_consistent,
    with_conditions,
)

HOURS = [0, 1, 2]
DEMAND = 10.0
OTHER_DIMS = ("hour", "operating_condition", "year_op")


@pytest.fixture()
def p_base(parameters_dispatch_invest, transport):
    p = xr.merge([parameters_dispatch_invest, transport])
    p = p.sel(
        conversion_tech=["wind_onshore"],
        transport_tech=["power_line"],
        hour=HOURS,
        resource=["electricity"],
        year_dec=[2040, 2050],
        year_inv=[2030],
        year_op=[2030],
    ).copy(deep=True)
    # finite and slack ramps build the ramp rows without changing the solution
    p["transport_ramp_up"] = 1e3
    p["transport_ramp_down"] = 1e3
    p["load_shedding_cost"] = xr.DataArray(1000.0)
    p["load_shedding_max_capacity"] = xr.DataArray(np.nan)
    p["carbon"] = False
    return p


def _by_area(p, values):
    """A (area, hour) array: `values` is a pair (area_1, area_2) of lists over the hours."""
    return xr.DataArray(
        np.array(values, dtype=float),
        dims=("area", "hour"),
        coords=dict(area=p.area, hour=p.hour),
    )


def _demand(p, values):
    return _by_area(p, values).expand_dims(
        resource=p.resource, year_op=p.year_op
    )


def _availability(p, values):
    return _by_area(p, values).expand_dims(conversion_tech=p.conversion_tech)


def _solve(p):
    m = build_model(p)
    if "operating_condition" in p.coords:
        assert_condition_dims_consistent(m)
    status, _ = m.solve(solver_name="highs")
    assert status == "ok"
    return m


def _flow(m, name, link, condition):
    da = m.solution[name].sel(link=link, operating_condition=condition)
    return da.sum([d for d in da.dims if d not in OTHER_DIMS]).squeeze()


def _reversed_conditions(p, weights=(0.5, 0.5)):
    # a: area_2 consumes and only area_1 produces; b: the other way round
    on, off = [DEMAND] * 3, [0.0] * 3
    wind_on, wind_off = [1.0] * 3, [0.0] * 3
    return with_conditions(
        p,
        list(weights),
        demand=[
            _demand(p, [off, on]),
            _demand(p, [on, off]),
        ],
        conversion_availability=[
            _availability(p, [wind_on, wind_off]),
            _availability(p, [wind_off, wind_on]),
        ],
    )


def test_flows_differ_by_condition_on_a_shared_line(p_base):
    m = _solve(_reversed_conditions(p_base))
    power = "operation_transport_power"
    # a: area_1 sends to area_2 over link_1; b: area_2 sends to area_1 over link_2
    assert float(_flow(m, power, "link_1", "a").sum()) == pytest.approx(
        DEMAND * len(HOURS)
    )
    assert float(_flow(m, power, "link_2", "a").sum()) == pytest.approx(0.0)
    assert float(_flow(m, power, "link_1", "b").sum()) == pytest.approx(0.0)
    assert float(_flow(m, power, "link_2", "b").sum()) == pytest.approx(
        DEMAND * len(HOURS)
    )
    # one capacity per link serves both conditions
    capacity = m.solution["planning_transport_power_capacity"]
    assert "operating_condition" not in capacity.dims
    assert float(capacity.sel(link="link_1").sum()) == pytest.approx(DEMAND)
    assert float(capacity.sel(link="link_2").sum()) == pytest.approx(DEMAND)
    shed = m.solution["operation_load_shedding_power"]
    assert float(shed.sum()) == pytest.approx(0.0)


def test_delay_stays_inside_each_condition(p_base):
    p = p_base.copy(deep=True)
    p["transport_delay"] = 1.0
    # only area_2 consumes, in every hour; a: area_1 produces in the last hour only; b: it never
    # produces
    off = [0.0] * 3
    p = with_conditions(
        p,
        [0.5, 0.5],
        demand=[_demand(p, [off, [DEMAND] * 3])] * 2,
        conversion_availability=[
            _availability(p, [[0.0, 0.0, 1.0], off]),
            _availability(p, [off, off]),
        ],
    )
    m = _solve(p)
    power_in = "operation_transport_power_in"
    power = "operation_transport_power"
    sent = _flow(m, power_in, "link_1", "a")
    got = _flow(m, power, "link_1", "a")
    assert float(sent.sel(hour=2)) > 0
    # a's last-hour injection arrives at a's first hour, through the cyclic delay
    assert float(got.sel(hour=0)) == pytest.approx(float(sent.sel(hour=2)))
    # b injects nothing and receives nothing: a's injection arrives nowhere in b
    assert float(_flow(m, power_in, "link_1", "b").sum()) == pytest.approx(0.0)
    assert float(_flow(m, power, "link_1", "b").sum()) == pytest.approx(0.0)
    shed = m.solution["operation_load_shedding_power"].sel(area="area_2")
    shed_b = shed.sel(operating_condition="b").sum()
    assert float(shed_b) == pytest.approx(DEMAND * len(HOURS))


WITH_CONDITION = [
    "operation_transport_power",
    "operation_transport_power_in",
    "operation_transport_linepack_level",
    "operation_transport_net_generation",
    "operation_transport_costs",
    "operation_transport_power_max_constraint",
    "operation_transport_power_in_def",
    "operation_transport_linepack_level_def",
    "operation_transport_linepack_level_max_constraint",
    "operation_transport_power_in_max_constraint",
    "operation_transport_ramp_up_constraint",
    "operation_transport_ramp_down_constraint",
    "operation_transport_net_generation_def",
    "operation_transport_costs_def",
]


def _with_linepack(p, delay=False):
    # a finite linepack on link_1 builds the linepack rows and the power-in bound
    p = p.copy(deep=True)
    p["transport_linepack_capacity"] = xr.DataArray(
        [[5.0]],
        dims=("link", "transport_tech"),
        coords=dict(link=["link_1"], transport_tech=p.transport_tech),
    ).reindex(link=p.link)
    if not delay:
        return p
    # a one-hour delay on link_2 (no linepack there) builds the delayed power-in rows
    # (and leaves no link for the plain power-in definition, hence the two datasets)
    p["transport_delay"] = xr.DataArray(
        [[0.0], [1.0]],
        dims=("link", "transport_tech"),
        coords=dict(link=["link_1", "link_2"], transport_tech=p.transport_tech),
    ).reindex(link=p.link, fill_value=0.0)
    return p


def test_an_absent_pair_has_no_transport_row_and_weight_zero_keeps_them(p_base):
    delayed = "operation_transport_power_in_def_delay_1"
    cases = [
        (w, absent, delay)
        for w, absent in (([1.0, np.nan], True), ([1.0, 0.0], False))
        for delay in (False, True)
    ]
    for weights, absent, delay in cases:
        required = (
            set(WITH_CONDITION) - {"operation_transport_power_in_def"} | {delayed}
            if delay
            else set(WITH_CONDITION)
        )
        m = build_model(_reversed_conditions(_with_linepack(p_base, delay), weights))
        assert_condition_dims_consistent(m)
        checked = []
        for container in (m.variables, m.constraints):
            for name in container:
                if "transport" not in name:
                    continue
                labels = container[name].labels
                if "operating_condition" not in labels.dims:
                    continue
                checked.append(name)
                if absent:
                    assert (labels.sel(operating_condition="b") == -1).all(), name
                    if name in required:
                        assert (labels.sel(operating_condition="a") != -1).any(), name
                elif name in required:
                    assert (labels.sel(operating_condition="b") != -1).any(), name
        assert required <= set(checked)


def test_weight_zero_condition_adds_nothing_to_the_objective_but_keeps_its_flows(p_base):
    full = _reversed_conditions(p_base, (1.0, 0.0))
    m = _solve(full)
    # condition a alone: same data, a single condition of weight 1
    a = with_conditions(
        p_base,
        [1.0],
        demand=[_demand(p_base, [[0.0] * 3, [DEMAND] * 3])],
        conversion_availability=[_availability(p_base, [[1.0] * 3, [0.0] * 3])],
    )
    assert m.objective.value == pytest.approx(_solve(a).objective.value, rel=1e-9)
    # b keeps its rows; link_2 has no capacity (sized for a only), so b's demand is shed
    labels = m.constraints["operation_transport_power_max_constraint"].labels
    assert (labels.sel(operating_condition="b") != -1).any()
    shed_b = m.solution["operation_load_shedding_power"].sel(operating_condition="b")
    assert float(shed_b.sum()) > 0
def test_identical_conditions_leave_the_objective_unchanged(p_base):
    p = p_base.copy(deep=True)
    p["demand"] = _demand(p, [[0.0] * 3, [DEMAND] * 3])
    p["conversion_availability"] = _availability(p, [[1.0] * 3, [0.0] * 3])
    # the fuel the line burns is drawn from the adequacy rows, which carry the dimension
    p["transport_fuel_consumption"] = 0.1
    p["transport_fuel_resource"] = "electricity"
    reference = _solve(p)
    same = with_conditions(
        p,
        [0.3, 0.7],
        demand=[p.demand] * 2,
        conversion_availability=[p.conversion_availability] * 2,
    )
    m = _solve(same)
    assert float(m.solution["planning_transport_power_capacity"].sum()) > 0
    assert float(m.solution["operation_transport_costs"].sum()) > 0
    assert m.objective.value == pytest.approx(reference.objective.value, rel=1e-9)
