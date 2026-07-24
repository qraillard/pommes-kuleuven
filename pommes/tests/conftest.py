"""Module describing the configuration tests."""

import numpy as np
import pytest
import xarray as xr
from xarray import Dataset

from pommes.utils import sort_coords_and_vars, square_array_by_diagonals

# --- Test model ---


@pytest.fixture(scope="module")
def study():
    return "test_case"


# Index
@pytest.fixture(scope="module")
def coords() -> Dataset:
    return xr.Dataset(
        coords=dict(
            area=np.array(["area_1", "area_2"], dtype=str),
            combined_tech=np.array(["electric_boiler"], dtype=str),
            conversion_tech=np.array(
                ["electrolysis", "ocgt", "smr", "smr_ccs", "wind_onshore"],
                dtype=str,
            ),
            # conversion_tech_from=np.array(["smr"], dtype=str),
            # conversion_tech_to=np.array(["smr_ccs"], dtype=str),
            transport_tech=np.array(
                ["big_methane_pipe", "methane_pipe", "power_line"], dtype=str
            ),
            hour=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="int64"),
            hour_type=np.array(["HCE", "HPE", "HCH", "HPH", "P"], dtype=str),
            link=np.array(["link_1", "link_2"], dtype=str),
            mode=np.array(["electric", "fossil"], dtype=str),
            process_tech=np.array(["chp"], dtype=str),
            resource=np.array(
                ["electricity", "heat", "hydrogen", "methane"], dtype=str
            ),
            storage_tech=np.array(
                ["battery", "tank_hydrogen", "tank_methane"], dtype=str
            ),
            year_dec=np.array(
                [2020, 2030, 2040, 2050, 2060, 2070], dtype="int64"
            ),
            year_inv=np.array([2010, 2020, 2030, 2040, 2050], dtype="int64"),
            year_op=np.array([2020, 2030, 2040, 2050], dtype="int64"),
            # year_rep=np.array([2020, 2030, 2040, 2050], dtype="int64"),
        )
    )


# Parameters


@pytest.fixture(scope="module")
def time_slice(coords):
    ds = xr.Dataset(
        data_vars=dict(
            time_step_duration=(
                ["hour"],
                np.array(
                    [1 for i in coords.hour],
                    dtype="float64",
                ),
            ),
            operation_year_duration=(
                ["year_op"],
                np.array(
                    [8760 for i in coords.year_op],
                    dtype="float64",
                ),
            ),
            operation_day_duration=np.array(24, dtype="int64"),
        ),
        coords=dict(hour=coords.hour, year_op=coords.year_op),
    )
    return ds


@pytest.fixture(scope="module")
def demand(coords):
    ds = xr.Dataset(
        data_vars=dict(
            demand=(
                ["year_op", "resource"],
                np.array(
                    [
                        [10.0, 2, 3, 0.0],
                        [10.0, 2, 2, 0.0],
                        [10.0, 2, 4, 0.0],
                        [10.0, 2, 0, 0.0],
                    ],
                    dtype="float64",
                ),
            ),
        ),
        coords=dict(resource=coords.resource, year_op=coords.year_op),
    )
    return ds


@pytest.fixture(scope="module")
def load_shedding(coords):
    ds = xr.Dataset(
        data_vars=dict(
            load_shedding_max_capacity=(
                ["resource"],
                np.array([0, 0, 0, np.nan], dtype="float64"),
            ),
            load_shedding_cost=(
                ["resource"],
                np.array([0, 0, 0, 1000], dtype="float64"),
            ),
        ),
        coords=dict(resource=coords.resource),
    )
    return ds


@pytest.fixture(scope="module")
def spillage(coords):
    ds = xr.Dataset(
        data_vars=dict(
            spillage_max_capacity=np.array(np.nan, dtype="float64"),
            spillage_cost=np.array(0, dtype="float64"),
        ),
        coords=dict(resource=coords.resource),
    )
    return ds


@pytest.fixture(scope="module")
def combined(coords):
    ds = xr.Dataset(
        data_vars=dict(
            combined=np.array(True, dtype="bool"),
            combined_early_decommissioning=np.array(False, dtype="bool"),
            combined_annuity_perfect_foresight=np.array(False, dtype="bool"),
            combined_variable_cost=np.array(1, dtype="int64"),
            combined_fixed_cost=np.array(10, dtype="int64"),
            combined_invest_cost=np.array(2000, dtype="int64"),
            combined_life_span=np.array(20, dtype="int64"),
            combined_emission_factor=(
                ["combined_tech", "mode"],
                np.array([[0, 10]], dtype="float64"),
            ),
            combined_must_run=np.array(np.nan, dtype="float64"),
            combined_factor=(
                ["combined_tech", "mode", "resource"],
                np.array(
                    [[[-1, 1, 0, 0], [0, 1, 0, -1.5]]],
                    dtype="float64",
                ),
            ),
            combined_ramp_up=np.array(np.nan, dtype="float64"),
            combined_ramp_down=np.array(np.nan, dtype="float64"),
            combined_max_load=np.array(1.0, dtype="float64"),
            combined_min_load=np.array(0.0, dtype="float64"),
            combined_power_capacity_max=np.array(np.nan, dtype="float64"),
            combined_power_capacity_min=np.array(np.nan, dtype="float64"),
            combined_finance_rate=np.array(0, dtype="float64"),
            combined_power_capacity_investment_max=np.array(
                np.nan, dtype="float64"
            ),
            combined_power_capacity_investment_min=np.array(
                np.nan, dtype="float64"
            ),
        ),
        coords=dict(
            combined_tech=coords.combined_tech,
            resource=coords.resource,
            hour=coords.hour,
            mode=coords.mode,
        ),
    )
    ds = ds.assign(
        combined_annuity_cost=ds.combined_invest_cost
        * xr.DataArray(
            square_array_by_diagonals(
                6, {0: 1 / 20, 1: 1 / 10}, fill=np.nan, dtype="float64"
            )[:, 1:],
            coords=[coords.year_dec, coords.year_inv],
        ),
        combined_end_of_life=ds.combined_life_span + coords.year_inv,
    )
    return ds


@pytest.fixture(scope="module")
def process(coords):
    ds = xr.Dataset(
        data_vars=dict(
            process=np.array(True, dtype="bool"),
            process_early_decommissioning=np.array(False, dtype="bool"),
            process_annuity_perfect_foresight=np.array(False, dtype="bool"),
            process_variable_cost=np.array(1, dtype="float64"),
            process_fixed_cost=np.array(10, dtype="float64"),
            process_invest_cost=np.array(2000, dtype="float64"),
            process_life_span=np.array(20, dtype="float64"),
            process_emission_factor=(
                ["process_tech", "mode"],
                np.array([[0, 10]], dtype="float64"),
            ),
            process_factor=(
                ["process_tech", "mode", "resource"],
                np.array(
                    [[[-1, 1, 0, 0], [0, 1, 0, -1.5]]],
                    dtype="float64",
                ),
            ),
            process_max_load=np.array(1.0, dtype="float64"),
            process_min_load=np.array(0.0, dtype="float64"),
            process_unit_size=np.array(0, dtype="float64"),
            process_startup_cost=np.array(0, dtype="float64"),
            process_shutdown_cost=np.array(0, dtype="float64"),
            process_minimum_startup_time=np.array(1, dtype="float64"),
            process_minimum_shutdown_time=np.array(1, dtype="float64"),
            process_no_shutdown=np.array(False, dtype="bool"),
            process_ramp_up=np.array(np.nan, dtype="float64"),
            process_ramp_down=np.array(np.nan, dtype="float64"),
            process_power_capacity_max=np.array(1000, dtype="float64"),
            process_power_capacity_min=np.array(0, dtype="float64"),
            process_power_capacity_investment_max=np.array(
                np.nan, dtype="float64"
            ),
            process_power_capacity_investment_min=np.array(
                np.nan, dtype="float64"
            ),
        ),
        coords=dict(
            process_tech=coords.process_tech,
            resource=coords.resource,
            mode=coords.mode,
        ),
    )
    ds = ds.assign(
        process_annuity_cost=ds.process_invest_cost
        * xr.DataArray(
            square_array_by_diagonals(
                6, {0: 1 / 20, 1: 1 / 10}, fill=np.nan, dtype="float64"
            )[:, 1:],
            coords=[coords.year_dec, coords.year_inv],
        ),
        process_end_of_life=ds.process_life_span + coords.year_inv,
    )
    return ds


@pytest.fixture(scope="module")
def conversion(coords):
    ds = xr.Dataset(
        data_vars=dict(
            conversion=np.array(True, dtype="bool"),
            conversion_early_decommissioning=np.array(False, dtype="bool"),
            conversion_annuity_perfect_foresight=np.array(False, dtype="bool"),
            conversion_variable_cost=(
                ["conversion_tech"],
                np.array([0, 8, 1, 1, 0], dtype="float64"),
            ),
            conversion_fixed_cost=(
                ["conversion_tech"],
                np.array([0, 10, 30, 35, 0], dtype="float64"),
            ),
            conversion_invest_cost=(
                ["conversion_tech"],
                np.array([2000, 400, 1000, 1200, 200], dtype="float64"),
            ),
            conversion_life_span=np.array(20, dtype="int64"),
            conversion_emission_factor=(
                ["conversion_tech"],
                np.array([0, 0.5, 1, 0.2, 0], dtype="float64"),
            ),
            conversion_must_run=(
                ["conversion_tech"],
                np.array([0, 0, 0, 0, 1], dtype="float64"),
            ),
            conversion_factor=(
                ["conversion_tech", "resource"],
                np.array(
                    [
                        [-1.5, 0, 1.0, 0.0],
                        [1.0, 0, 0.0, -1.5],
                        [0.0, 0, 1.0, -1.5],
                        [-0.1, 0, 1.0, -1.5],
                        [1.0, 0, 0.0, 0.0],
                    ],
                    dtype="float64",
                ),
            ),
            conversion_availability=(
                ["conversion_tech", "hour"],
                np.array(
                    [[np.nan] * len(coords.hour)]
                    * (len(coords.conversion_tech) - 1)
                    + [[0.5, 0.7, 0.1, 0.0, 0.2, 0.4, 0.8, 0.9, 0.6, 0.3]],
                    dtype="float64",
                ),
            ),
            conversion_ramp_up=np.array(np.nan, dtype="float64"),
            conversion_ramp_down=np.array(np.nan, dtype="float64"),
            conversion_finance_rate=np.array(0, dtype="float64"),
            conversion_power_capacity_investment_max=np.array(
                1000, dtype="float64"
            ),
            conversion_power_capacity_investment_min=np.array(
                0, dtype="float64"
            ),
            conversion_max_yearly_production=np.array(np.nan, dtype="float64"),
            conversion_max_daily_production=np.array(np.nan, dtype="float64"),
            conversion_power_capacity_max=np.array(1000, dtype="float64"),
            conversion_power_capacity_min=np.array(0, dtype="float64"),
        ),
        coords=dict(
            conversion_tech=coords.conversion_tech,
            resource=coords.resource,
            hour=coords.hour,
        ),
    )
    ds = ds.assign(
        conversion_annuity_cost=ds.conversion_invest_cost
        * xr.DataArray(
            square_array_by_diagonals(
                6, {0: 1 / 20, 1: 1 / 10}, fill=np.nan, dtype="float64"
            )[:, 1:],
            coords=[coords.year_dec, coords.year_inv],
        ),
        conversion_end_of_life=ds.conversion_life_span + coords.year_inv,
    )
    return ds


@pytest.fixture(scope="module")
def storage(coords):
    ds = xr.Dataset(
        data_vars=dict(
            storage=np.array(True, dtype="bool"),
            storage_early_decommissioning=np.array(False, dtype="bool"),
            storage_annuity_perfect_foresight=np.array(False, dtype="bool"),
            storage_energy_capacity_investment_max=np.array(
                1000, dtype="float64"
            ),
            storage_energy_capacity_investment_min=np.array(
                np.nan, dtype="float64"
            ),
            storage_power_capacity_investment_max=(
                ["storage_tech"],
                np.array([200, 100, 100], dtype="float64"),
            ),
            storage_power_capacity_investment_min=np.array(
                np.nan, dtype="float64"
            ),
            storage_power_capacity_max=np.array(np.nan, dtype="float64"),
            storage_power_capacity_min=np.array(np.nan, dtype="float64"),
            storage_energy_capacity_max=np.array(np.nan, dtype="float64"),
            storage_energy_capacity_min=np.array(np.nan, dtype="float64"),
            storage_energy_power_ratio=np.array(np.nan, dtype="float64"),
            storage_fixed_cost_energy=np.array(0, dtype="float64"),
            storage_fixed_cost_power=np.array(0, dtype="float64"),
            storage_invest_cost_energy=(
                ["storage_tech"],
                np.array([1000, 2000, 400], dtype="float64"),
            ),
            storage_invest_cost_power=(
                ["storage_tech"],
                np.array([2000, 2000, 400], dtype="float64"),
            ),
            storage_life_span=np.array(20, dtype="float64"),
            storage_main_resource=(
                ["storage_tech"],
                np.array(["electricity", "hydrogen", "methane"], dtype=str),
            ),
            storage_dissipation=(
                ["storage_tech"],
                np.array([0.2, 0.0, 0.0], dtype="float64"),
            ),
            storage_finance_rate=(
                ["storage_tech"],
                np.array([0, 0, 0], dtype="float64"),
            ),
            storage_charge_ramp=np.array(np.nan, dtype="float64"),
            storage_discharge_ramp=np.array(np.nan, dtype="float64"),
            storage_factor_in=(
                ["storage_tech", "resource"],
                np.array(
                    [[-2, 0, 0, 0], [-1, 0, -1, 0], [-1, 0, 0, -1]],
                    dtype="float64",
                ),
            ),
            storage_factor_keep=(
                ["storage_tech", "resource"],
                np.array(
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype="float64"
                ),
            ),
            storage_factor_out=(
                ["storage_tech", "resource"],
                np.array(
                    [[0.5, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    dtype="float64",
                ),
            ),
        ),
        coords=dict(
            storage_tech=coords.storage_tech, resource=coords.resource
        ),
    )

    ds = ds.assign(
        storage_annuity_cost_power=ds.storage_invest_cost_power
        * xr.DataArray(
            square_array_by_diagonals(
                6, {0: 1 / 20, 1: 1 / 10}, fill=np.nan, dtype="float64"
            )[:, 1:],
            coords=[coords.year_dec, coords.year_inv],
        ),
        storage_annuity_cost_energy=ds.storage_invest_cost_energy
        * xr.DataArray(
            square_array_by_diagonals(
                6, {0: 1 / 20, 1: 1 / 10}, fill=np.nan, dtype="float64"
            )[:, 1:],
            coords=[coords.year_dec, coords.year_inv],
        ),
        storage_end_of_life=ds.storage_life_span + coords.year_inv,
    )
    return ds


@pytest.fixture(scope="module")
def net_import(coords):
    ds = xr.Dataset(
        data_vars=dict(
            net_import=np.array(True, dtype="bool"),
            net_import_emission_factor=(
                ["resource"],
                np.array([0.05, 0, 0.01, 0.200], dtype="float64"),
            ),
            net_import_total_emission_factor=(
                ["resource"],
                np.array([0.05, 0, 0.01, 0.200], dtype="float64"),
            ),
            net_import_import_price=(
                ["resource"],
                np.array([0, 0, 0, 20], dtype="float64"),
            ),
            net_import_export_price=(
                ["resource"],
                np.array([0, 0, 0, 20], dtype="float64"),
            ),
            net_import_max_yearly_energy_export=(
                ["resource"],
                np.array([0, 0, 0, 0], dtype="float64"),
            ),
            net_import_max_yearly_energy_import=(
                ["resource"],
                np.array([0, 0, 0, 1000], dtype="float64"),
            ),
        ),
        coords=dict(resource=coords.resource),
    )
    return ds


@pytest.fixture(scope="module")
def retrofit(coords):
    ds = xr.Dataset(
        data_vars=dict(
            retrofit_factor=(
                ["conversion_tech_from", "conversion_tech_to"],
                np.array([[0]], dtype="float64"),
            )
        ),
        coords=dict(
            conversion_tech_from=coords.conversion_tech_from,
            conversion_tech_to=coords.conversion_tech_to,
        ),
    )
    ds.assign(retrofit=True)
    return ds


@pytest.fixture(scope="module")
def carbon(coords):
    ds = xr.Dataset(
        data_vars=dict(
            carbon=np.array(True, dtype="bool"),
            carbon_tax=np.array(2, dtype="float64"),
            carbon_goal=np.array(14, dtype="float64"),
            carbon_goal_area=np.array(np.nan, dtype="float64"),
            carbon_goal_group_areas=np.array(False, dtype="bool"),
            carbon_goal_group_max=np.array(np.nan, dtype="float64"),
        ),
        coords=dict(),
    )
    return ds


@pytest.fixture(scope="module")
def turpe(coords):
    ds = xr.Dataset(
        data_vars=dict(
            turpe=np.array(True, dtype="bool"),
            turpe_fixed_cost=(
                ["hour_type"],
                np.array([5, 8, 10, 12, 15], dtype="float64"),
            ),
            turpe_variable_cost=(
                ["hour_type"],
                np.array([0.01, 0.05, 0.1, 0.2, 0.5], dtype="float64"),
            ),
            turpe_calendar=(
                ["hour"],
                np.array(
                    [
                        "HCH",
                        "HPH",
                        "P",
                        "HPH",
                        "HCH",
                        "HCE",
                        "HCE",
                        "HPE",
                        "HPE",
                        "HCE",
                    ],
                    dtype=str,
                ),
            ),
        ),
        coords=dict(hour=coords.hour, hour_type=coords.hour_type),
    )
    return ds


@pytest.fixture(scope="module")
def eco(coords):
    ds = xr.Dataset(
        data_vars=dict(
            discount_rate=(
                ["year_op"],
                np.array([0, 0, 0, 0], dtype=np.float64),
            ),
            discount_factor=(
                ["year_op"],
                np.array([1, 1, 1, 1], dtype=np.float64),
            ),
            year_ref=([], np.array(2020, dtype="int64")),
            planning_step=([], np.array(10, dtype="int64")),
        ),
        coords=dict(year_op=coords.year_op),
    )
    return ds


@pytest.fixture(scope="module")
def transport(coords):
    ds = xr.Dataset(
        data_vars=dict(
            transport=np.array(True, dtype="bool"),
            transport_area_from=(
                ["link"],
                np.array(["area_1", "area_2"], dtype=str),
            ),
            transport_area_to=(
                ["link"],
                np.array(["area_2", "area_1"], dtype=str),
            ),
            transport_early_decommissioning=np.array(False, dtype="bool"),
            transport_annuity_perfect_foresight=np.array(False, dtype="bool"),
            transport_invest_cost=(
                ["transport_tech"],
                np.array([300, 200, 100], dtype="float64"),
            ),
            transport_fixed_cost=np.array(10, dtype="float64"),
            transport_finance_rate=np.array(0, dtype="float64"),
            transport_life_span=np.array(20, dtype="float64"),
            transport_power_capacity_investment_max=np.array(
                np.nan, dtype="float64"
            ),
            transport_power_capacity_investment_min=np.array(
                np.nan, dtype="float64"
            ),
            transport_power_capacity_max=np.array(np.nan, dtype="float64"),
            transport_power_capacity_min=np.array(np.nan, dtype="float64"),
            transport_availability=np.array(np.nan, dtype="float64"),
            transport_ramp_up=np.array(np.nan, dtype="float64"),
            transport_ramp_down=np.array(np.nan, dtype="float64"),
            transport_linepack_capacity=np.array(np.nan, dtype="float64"),
            transport_hurdle_costs=np.array(0.01, dtype="float64"),
            transport_resource=(
                ["transport_tech"],
                np.array(["methane", "methane", "electricity"], dtype=str),
            ),
        ),
        coords=dict(
            link=coords.link,
            transport_tech=coords.transport_tech,
            year_inv=coords.year_inv,
        ),
    )
    ds = ds.assign(
        transport_annuity_cost=ds.transport_invest_cost
        * xr.DataArray(
            square_array_by_diagonals(
                6, {0: 1 / 20, 1: 1 / 10}, fill=np.nan, dtype="float64"
            )[:, 1:],
            coords=[coords.year_dec, coords.year_inv],
        ),
        transport_end_of_life=ds.transport_life_span + coords.year_inv,
    )
    return ds


@pytest.fixture(scope="module")
def parameters_dispatch_invest(
    coords, demand, load_shedding, spillage, conversion, eco, time_slice
):
    return sort_coords_and_vars(
        xr.merge(
            [
                xr.Dataset(coords=coords),
                demand,
                load_shedding,
                spillage,
                conversion,
                eco,
                time_slice,
            ]
        )
    )


@pytest.fixture(scope="module")
def parameters(
    coords,
    demand,
    load_shedding,
    spillage,
    combined,
    conversion,
    storage,
    net_import,
    transport,
    carbon,
    turpe,
    eco,
    time_slice,
):
    return sort_coords_and_vars(
        xr.merge(
            [
                xr.Dataset(coords=coords),
                demand,
                load_shedding,
                spillage,
                combined,
                conversion,
                storage,
                net_import,
                transport,
                carbon,
                time_slice,
                turpe,
                eco,
            ]
        )
    )
