.. _dataset_description:

Dataset Description
===================

Time steps
----------

``time_step_duration`` is the length of each time step in hours (1 by default). The adequacy balance of
each resource is written in energy per time step:
:math:`\Delta t \, (\text{conversion} + \text{storage} + \text{transport} + \text{load shedding} - \text{spillage}) + \text{imports} = \text{demand}`.
Power variables (including load shedding and spillage) are multiplied by :math:`\Delta t`; ``demand``,
``flexdem_demand`` and net imports are energies per step (sums over the step); intensive data
(availabilities, prices, emission factors) are averages over the step.

.. list-table::
    :header-rows: 1

    * - Key
      - Description
    * - time_step_duration
      - Length of each time step [h]. Power variables are multiplied by it in the energy balances. The process module is only valid at 1 h steps (a warning is emitted otherwise).
    * - demand
      - Energy demanded in each time step [MWh per step], the sum over the step of the hourly demand (not a power).
    * - flexdem_demand
      - Baseline flexible demand in each time step [MWh per step], the sum over the step (not a power).
    * - load_shedding_max_capacity
      - Maximum load shedding power [MW]. The energy shed in a step is ``time_step_duration`` x power.
    * - spillage_max_capacity
      - Maximum spillage power [MW]. The energy spilled in a step is ``time_step_duration`` x power.
    * - process_minimum_startup_time, process_minimum_shutdown_time
      - Minimum time a process unit stays on (off) after a start-up (shut-down), counted in time steps: valid at 1 h steps, where it is in hours.

.. list-table::
    :header-rows: 1

    * - Key
      - default
      - index_set
      - type

    * - annuity_full_life
      - 1
      - []
      - bool

    * - carbon_goal
      - nan
      - ['year_op']
      - float64

    * - carbon_goal_area
      - nan
      - ['area', 'year_op']
      - float64

    * - carbon_goal_group_areas
      - 0
      - ['area', 'carbon_goal_group']
      - bool

    * - carbon_goal_group_max
      - nan
      - ['carbon_goal_group', 'year_op']
      - float64

    * - carbon
      - 0
      - []
      - bool

    * - carbon_tax
      - nan
      - ['area', 'year_op']
      - float64

    * - carbon_tax_group_areas
      - 0
      - ['area', 'carbon_tax_group']
      - bool

    * - carbon_tax_group_rate
      - nan
      - ['carbon_tax_group', 'year_op']
      - float64

    * - combined
      - 0
      - []
      - bool

    * - combined_annuity_perfect_foresight
      - 0
      - ['area', 'combined_tech', 'year_inv']
      - bool

    * - combined_annuity_cost
      - nan
      - ['area', 'combined_tech', 'year_dec', 'year_inv']
      - float64

    * - combined_early_decommissioning
      - 0
      - ['area', 'combined_tech', 'year_inv']
      - bool

    * - combined_emission_factor
      - 0.0
      - ['area', 'combined_tech', 'mode', 'year_op']
      - float64

    * - combined_end_of_life
      - 0
      - ['area', 'combined_tech', 'year_inv']
      - int64

    * - combined_factor
      - 0.0
      - ['area', 'combined_tech', 'mode', 'resource', 'year_op']
      - float64

    * - combined_finance_rate
      - 0.0
      - ['area', 'combined_tech', 'year_inv']
      - float64

    * - combined_fixed_cost
      - 0
      - ['area', 'combined_tech', 'year_op']
      - float64

    * - combined_invest_cost
      - 0
      - ['area', 'combined_tech', 'year_inv']
      - float64

    * - combined_life_span
      - nan
      - ['area', 'combined_tech', 'year_inv']
      - float64

    * - combined_power_capacity_investment_max
      - nan
      - ['area', 'combined_tech', 'year_inv']
      - float64

    * - combined_power_capacity_investment_min
      - nan
      - ['area', 'combined_tech', 'year_inv']
      - float64

    * - combined_must_run
      - nan
      - ['area', 'combined_tech', 'mode', 'year_op']
      - float64

    * - combined_power_capacity_max
      - nan
      - ['area', 'combined_tech', 'year_op']
      - float64

    * - combined_power_capacity_min
      - nan
      - ['area', 'combined_tech', 'year_op']
      - float64

    * - combined_max_load
      - 1.0
      - ['area', 'combined_tech', 'year_op']
      - float64

    * - combined_min_load
      - 0.0
      - ['area', 'combined_tech', 'year_op']
      - float64

    * - combined_ramp_down
      - nan
      - ['area', 'combined_tech', 'mode', 'year_op']
      - float64

    * - combined_ramp_up
      - nan
      - ['area', 'combined_tech', 'mode', 'year_op']
      - float64

    * - combined_mode_ramp_down
      - nan
      - ['area', 'combined_tech', 'mode', 'year_op']
      - float64

    * - combined_mode_ramp_up
      - nan
      - ['area', 'combined_tech', 'mode', 'year_op']
      - float64

    * - combined_variable_cost
      - 0
      - ['area', 'combined_tech', 'mode', 'operating_condition', 'year_op']
      - float64

    * - conversion
      - 0
      - []
      - bool

    * - conversion_annuity_perfect_foresight
      - 0
      - ['area', 'conversion_tech', 'year_inv']
      - bool

    * - conversion_annuity_cost
      - nan
      - ['area', 'conversion_tech', 'year_dec', 'year_inv']
      - float64

    * - conversion_availability
      - nan
      - ['area', 'conversion_tech', 'hour', 'operating_condition', 'year_op']
      - float64

    * - conversion_early_decommissioning
      - 0
      - ['area', 'conversion_tech', 'year_inv']
      - bool

    * - conversion_emission_factor
      - 0.0
      - ['area', 'conversion_tech', 'year_op']
      - float64

    * - conversion_end_of_life
      - 0
      - ['area', 'conversion_tech', 'year_inv']
      - int64

    * - conversion_factor
      - 0.0
      - ['area', 'conversion_tech', 'resource', 'year_op']
      - float64

    * - conversion_finance_rate
      - 0.0
      - ['area', 'conversion_tech', 'year_inv']
      - float64

    * - conversion_fixed_cost
      - 0
      - ['area', 'conversion_tech', 'year_op']
      - float64

    * - conversion_invest_cost
      - 0
      - ['area', 'conversion_tech', 'year_inv']
      - float64

    * - conversion_life_span
      - nan
      - ['area', 'conversion_tech', 'year_inv']
      - float64

    * - conversion_power_capacity_max
      - nan
      - ['area', 'conversion_tech', 'year_op']
      - float64

    * - conversion_power_capacity_min
      - nan
      - ['area', 'conversion_tech', 'year_op']
      - float64

    * - conversion_power_capacity_investment_max
      - nan
      - ['area', 'conversion_tech', 'year_inv']
      - float64

    * - conversion_power_capacity_investment_min
      - nan
      - ['area', 'conversion_tech', 'year_inv']
      - float64

    * - conversion_max_yearly_production
      - nan
      - ['area', 'conversion_tech', 'operating_condition', 'year_op']
      - float64

    * - conversion_max_capacity_factor
      - nan
      - ['area', 'conversion_tech', 'year_op']
      - float64

    * - conversion_max_cumulative_production
      - nan
      - ['area', 'conversion_tech']
      - float64

    * - conversion_max_daily_production
      - nan
      - ['area', 'conversion_tech', 'year_op']
      - float64

    * - conversion_must_run
      - nan
      - ['area', 'conversion_tech', 'year_op']
      - float64

    * - conversion_ramp_down
      - nan
      - ['area', 'conversion_tech', 'year_op']
      - float64

    * - conversion_ramp_up
      - nan
      - ['area', 'conversion_tech', 'year_op']
      - float64

    * - conversion_variable_cost
      - 0
      - ['area', 'conversion_tech', 'operating_condition', 'year_op']
      - float64

    * - load_shedding_cost
      - 0.0
      - ['area', 'resource', 'operating_condition', 'year_op']
      - float64

    * - load_shedding_max_capacity
      - nan
      - ['area', 'resource', 'year_op']
      - float64

    * - demand
      - 0.0
      - ['area', 'hour', 'resource', 'operating_condition', 'year_op']
      - float64

    * - discount_factor
      - 1.0
      - ['year_op']
      - float64

    * - discount_rate
      - 0.0
      - ['year_op']
      - float64

    * - net_import
      - 0
      - []
      - bool

    * - net_import_emission_factor
      - 0.0
      - ['area', 'hour', 'resource', 'operating_condition', 'year_op']
      - float64

    * - net_import_import_price
      - 0.0
      - ['area', 'hour', 'resource', 'operating_condition', 'year_op']
      - float64

    * - net_import_export_price
      - 0.0
      - ['area', 'hour', 'resource', 'operating_condition', 'year_op']
      - float64

    * - net_import_max_yearly_energy_export
      - nan
      - ['area', 'resource', 'operating_condition', 'year_op']
      - float64

    * - net_import_max_yearly_energy_import
      - nan
      - ['area', 'resource', 'operating_condition', 'year_op']
      - float64

    * - net_import_total_emission_factor
      - 0.0
      - ['area', 'hour', 'resource', 'operating_condition', 'year_op']
      - float64

    * - operation_year_duration
      - 8760
      - ['year_op']
      - float64

    * - operation_year_weight
      - nan
      - ['year_op']
      - float64

    * - operation_day_duration
      - 24
      - []
      - int64

    * - planning_step
      - 0
      - []
      - int64

    * - spillage_cost
      - 0.0
      - ['area', 'resource', 'operating_condition', 'year_op']
      - float64

    * - spillage_max_capacity
      - nan
      - ['area', 'resource', 'year_op']
      - float64

    * - storage
      - 0
      - []
      - bool

    * - storage_annuity_perfect_foresight
      - 0
      - ['area', 'storage_tech', 'year_inv']
      - bool

    * - storage_annuity_cost_energy
      - nan
      - ['area', 'storage_tech', 'year_dec', 'year_inv']
      - float64

    * - storage_annuity_cost_power
      - nan
      - ['area', 'storage_tech', 'year_dec', 'year_inv']
      - float64

    * - storage_dissipation
      - 0.0
      - ['area', 'storage_tech', 'year_op']
      - float64

    * - storage_early_decommissioning
      - 0
      - ['area', 'storage_tech', 'year_inv']
      - bool

    * - storage_end_of_life
      - 0
      - ['area', 'storage_tech', 'year_inv']
      - int64

    * - storage_factor_in
      - 0.0
      - ['area', 'resource', 'storage_tech', 'year_op']
      - float64

    * - storage_factor_keep
      - 0.0
      - ['area', 'resource', 'storage_tech', 'year_op']
      - float64

    * - storage_factor_out
      - 0.0
      - ['area', 'resource', 'storage_tech', 'year_op']
      - float64

    * - storage_finance_rate
      - 0.0
      - ['area', 'storage_tech', 'year_inv']
      - float64

    * - storage_fixed_cost_energy
      - 0.0
      - ['area', 'storage_tech', 'year_op']
      - float64

    * - storage_fixed_cost_power
      - 0.0
      - ['area', 'storage_tech', 'year_op']
      - float64

    * - storage_invest_cost_energy
      - 0.0
      - ['area', 'storage_tech', 'year_inv']
      - float64

    * - storage_invest_cost_power
      - 0.0
      - ['area', 'storage_tech', 'year_inv']
      - float64

    * - storage_life_span
      - nan
      - ['area', 'storage_tech', 'year_inv']
      - float64

    * - storage_main_resource
      - nan
      - ['area', 'storage_tech']
      - str

    * - storage_discharge_ramp
      - nan
      - ['area', 'storage_tech', 'year_op']
      - float64

    * - storage_charge_ramp
      - nan
      - ['area', 'storage_tech', 'year_op']
      - float64

    * - storage_energy_capacity_max
      - nan
      - ['area', 'storage_tech', 'year_op']
      - float64

    * - storage_energy_capacity_min
      - nan
      - ['area', 'storage_tech', 'year_op']
      - float64

    * - storage_power_capacity_max
      - nan
      - ['area', 'storage_tech', 'year_op']
      - float64

    * - storage_power_capacity_min
      - nan
      - ['area', 'storage_tech', 'year_op']
      - float64

    * - storage_energy_capacity_investment_max
      - nan
      - ['area', 'storage_tech', 'year_inv']
      - float64

    * - storage_power_capacity_investment_max
      - nan
      - ['area', 'storage_tech', 'year_inv']
      - float64

    * - storage_energy_capacity_investment_min
      - nan
      - ['area', 'storage_tech', 'year_inv']
      - float64

    * - storage_power_capacity_investment_min
      - nan
      - ['area', 'storage_tech', 'year_inv']
      - float64

    * - storage_energy_power_ratio
      - nan
      - ['area', 'storage_tech']
      - float64

    * - storage_energy_to_power_min
      - nan
      - ['area', 'storage_tech', 'year_op']
      - float64

    * - time_step_duration
      - 1
      - ['hour']
      - float64

    * - transport
      - 0
      - []
      - bool

    * - transport_annuity_perfect_foresight
      - 0
      - ['link', 'transport_tech', 'year_inv']
      - bool

    * - transport_annuity_cost
      - nan
      - ['link', 'transport_tech', 'year_dec', 'year_inv']
      - float64

    * - transport_area_from
      - None
      - ['link', 'transport_tech']
      - str

    * - transport_area_to
      - None
      - ['link', 'transport_tech']
      - str

    * - transport_early_decommissioning
      - 0
      - ['link', 'transport_tech', 'year_inv']
      - bool

    * - transport_end_of_life
      - 0
      - ['link', 'transport_tech', 'year_inv']
      - int64

    * - transport_finance_rate
      - nan
      - ['link', 'transport_tech', 'year_inv']
      - float64

    * - transport_fixed_cost
      - nan
      - ['link', 'transport_tech', 'year_op']
      - float64

    * - transport_hurdle_costs
      - nan
      - ['link', 'transport_tech', 'operating_condition', 'year_op']
      - float64

    * - transport_invest_cost
      - 0
      - ['link', 'transport_tech', 'year_inv']
      - float64

    * - transport_life_span
      - nan
      - ['link', 'transport_tech', 'year_inv']
      - float64

    * - transport_power_capacity_investment_max
      - nan
      - ['link', 'transport_tech', 'year_inv']
      - float64

    * - transport_power_capacity_investment_min
      - nan
      - ['link', 'transport_tech', 'year_inv']
      - float64

    * - transport_power_capacity_max
      - nan
      - ['link', 'transport_tech', 'year_op']
      - float64

    * - transport_linepack_capacity
      - nan
      - ['link', 'transport_tech', 'year_op']
      - float64

    * - transport_delay
      - 0.0
      - ['link', 'transport_tech']
      - float64

    * - transport_power_capacity_min
      - nan
      - ['link', 'transport_tech', 'year_op']
      - float64

    * - transport_resource
      - nan
      - ['link', 'transport_tech']
      - str

    * - transport_efficiency
      - 1.0
      - ['link', 'transport_tech', 'year_op']
      - float64

    * - transport_fuel_consumption
      - nan
      - ['link', 'transport_tech', 'year_op']
      - float64

    * - transport_fuel_area
      - nan
      - ['link', 'transport_tech']
      - str

    * - transport_fuel_resource
      - nan
      - ['link', 'transport_tech']
      - str

    * - transport_availability
      - nan
      - ['link', 'transport_tech', 'hour', 'operating_condition', 'year_op']
      - float64

    * - transport_ramp_down
      - nan
      - ['link', 'transport_tech', 'year_op']
      - float64

    * - transport_ramp_up
      - nan
      - ['link', 'transport_tech', 'year_op']
      - float64

    * - turpe
      - 0
      - []
      - bool

    * - turpe_calendar
      - nan
      - ['hour', 'year_op']
      - str

    * - turpe_fixed_cost
      - 0.0
      - ['hour_type', 'year_op']
      - float64

    * - turpe_variable_cost
      - 0.0
      - ['hour_type', 'year_op']
      - float64

    * - year_ref
      - 2000
      - []
      - int64

    * - retrofit_conversion
      - 0
      - []
      - bool

    * - retrofit_conversion_tech_from
      - None
      - ['retrofit_conversion_pair']
      - str

    * - retrofit_conversion_tech_to
      - None
      - ['retrofit_conversion_pair']
      - str

    * - retrofit_conversion_life_span
      - 0
      - ['area', 'retrofit_conversion_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_conversion_end_of_life
      - 0
      - ['area', 'retrofit_conversion_pair', 'year_ret', 'year_inv']
      - int64

    * - retrofit_conversion_finance_rate
      - 0.0
      - ['area', 'retrofit_conversion_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_conversion_invest_cost
      - 0
      - ['area', 'retrofit_conversion_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_conversion_annuity_cost
      - nan
      - ['area', 'retrofit_conversion_pair', 'year_dec', 'year_ret', 'year_inv']
      - float64

    * - retrofit_conversion_annuity_perfect_foresight
      - 0
      - ['area', 'retrofit_conversion_pair', 'year_ret', 'year_inv']
      - bool

    * - retrofit_conversion_early_decommissioning
      - 0
      - ['area', 'retrofit_conversion_pair', 'year_ret', 'year_inv']
      - bool

    * - retrofit_conversion_ret_ratio
      - 1.0
      - ['area', 'retrofit_conversion_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_conversion_eligible_share
      - 1.0
      - ['area', 'conversion_tech', 'year_ret', 'year_inv']
      - float64

    * - retrofit_conversion_power_capacity_investment_min
      - nan
      - ['area', 'retrofit_conversion_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_conversion_power_capacity_investment_max
      - nan
      - ['area', 'retrofit_conversion_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_combined
      - 0
      - []
      - bool

    * - retrofit_combined_tech_from
      - None
      - ['retrofit_combined_pair']
      - str

    * - retrofit_combined_tech_to
      - None
      - ['retrofit_combined_pair']
      - str

    * - retrofit_combined_life_span
      - 0
      - ['area', 'retrofit_combined_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_combined_end_of_life
      - 0
      - ['area', 'retrofit_combined_pair', 'year_ret', 'year_inv']
      - int64

    * - retrofit_combined_finance_rate
      - 0.0
      - ['area', 'retrofit_combined_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_combined_invest_cost
      - 0
      - ['area', 'retrofit_combined_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_combined_annuity_cost
      - nan
      - ['area', 'retrofit_combined_pair', 'year_dec', 'year_ret', 'year_inv']
      - float64

    * - retrofit_combined_annuity_perfect_foresight
      - 0
      - ['area', 'retrofit_combined_pair', 'year_ret', 'year_inv']
      - bool

    * - retrofit_combined_early_decommissioning
      - 0
      - ['area', 'retrofit_combined_pair', 'year_ret', 'year_inv']
      - bool

    * - retrofit_combined_ret_ratio
      - 1.0
      - ['area', 'retrofit_combined_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_combined_eligible_share
      - 1.0
      - ['area', 'combined_tech', 'year_ret', 'year_inv']
      - float64

    * - retrofit_combined_power_capacity_investment_min
      - nan
      - ['area', 'retrofit_combined_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_combined_power_capacity_investment_max
      - nan
      - ['area', 'retrofit_combined_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage
      - 0
      - []
      - bool

    * - retrofit_storage_tech_from
      - None
      - ['retrofit_storage_pair']
      - str

    * - retrofit_storage_tech_to
      - None
      - ['retrofit_storage_pair']
      - str

    * - retrofit_storage_life_span
      - 0
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_end_of_life
      - 0
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - int64

    * - retrofit_storage_finance_rate
      - 0.0
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_invest_cost_power
      - 0
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_invest_cost_energy
      - 0
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_annuity_cost_power
      - nan
      - ['area', 'retrofit_storage_pair', 'year_dec', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_annuity_cost_energy
      - nan
      - ['area', 'retrofit_storage_pair', 'year_dec', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_annuity_perfect_foresight
      - 0
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - bool

    * - retrofit_storage_early_decommissioning
      - 0
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - bool

    * - retrofit_storage_ret_ratio_power
      - 1.0
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_ret_ratio_energy
      - 1.0
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_eligible_share
      - 1.0
      - ['area', 'storage_tech', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_power_capacity_investment_min
      - nan
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_power_capacity_investment_max
      - nan
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_energy_capacity_investment_min
      - nan
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_storage_energy_capacity_investment_max
      - nan
      - ['area', 'retrofit_storage_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_transport
      - 0
      - []
      - bool

    * - retrofit_transport_tech_from
      - None
      - ['retrofit_transport_pair']
      - str

    * - retrofit_transport_tech_to
      - None
      - ['retrofit_transport_pair']
      - str

    * - retrofit_transport_life_span
      - 0
      - ['link', 'retrofit_transport_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_transport_end_of_life
      - 0
      - ['link', 'retrofit_transport_pair', 'year_ret', 'year_inv']
      - int64

    * - retrofit_transport_finance_rate
      - 0.0
      - ['link', 'retrofit_transport_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_transport_invest_cost
      - 0
      - ['link', 'retrofit_transport_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_transport_annuity_cost
      - nan
      - ['link', 'retrofit_transport_pair', 'year_dec', 'year_ret', 'year_inv']
      - float64

    * - retrofit_transport_annuity_perfect_foresight
      - 0
      - ['link', 'retrofit_transport_pair', 'year_ret', 'year_inv']
      - bool

    * - retrofit_transport_early_decommissioning
      - 0
      - ['link', 'retrofit_transport_pair', 'year_ret', 'year_inv']
      - bool

    * - retrofit_transport_ret_ratio
      - 1.0
      - ['link', 'retrofit_transport_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_transport_eligible_share
      - 1.0
      - ['link', 'transport_tech', 'year_ret', 'year_inv']
      - float64

    * - retrofit_transport_power_capacity_investment_min
      - nan
      - ['link', 'retrofit_transport_pair', 'year_ret', 'year_inv']
      - float64

    * - retrofit_transport_power_capacity_investment_max
      - nan
      - ['link', 'retrofit_transport_pair', 'year_ret', 'year_inv']
      - float64

    * - process
      - 0
      - []
      - bool

    * - process_annuity_perfect_foresight
      - 0
      - ['area', 'process_tech', 'year_inv']
      - bool

    * - process_annuity_cost
      - nan
      - ['area', 'process_tech', 'year_dec', 'year_inv']
      - float64

    * - process_early_decommissioning
      - 0
      - ['area', 'process_tech', 'year_inv']
      - bool

    * - process_emission_factor
      - 0.0
      - ['area', 'process_tech', 'mode', 'year_op']
      - float64

    * - process_end_of_life
      - 0
      - ['area', 'process_tech', 'year_inv']
      - int64

    * - process_factor
      - 0.0
      - ['area', 'process_tech', 'mode', 'resource', 'year_op']
      - float64

    * - process_finance_rate
      - 0.0
      - ['area', 'process_tech', 'year_inv']
      - float64

    * - process_fixed_cost
      - 0
      - ['area', 'process_tech', 'year_op']
      - float64

    * - process_invest_cost
      - 0
      - ['area', 'process_tech', 'year_inv']
      - float64

    * - process_life_span
      - nan
      - ['area', 'process_tech', 'year_inv']
      - float64

    * - process_power_capacity_investment_max
      - nan
      - ['area', 'process_tech', 'year_inv']
      - float64

    * - process_power_capacity_investment_min
      - nan
      - ['area', 'process_tech', 'year_inv']
      - float64

    * - process_power_capacity_max
      - nan
      - ['area', 'process_tech', 'year_op']
      - float64

    * - process_power_capacity_min
      - nan
      - ['area', 'process_tech', 'year_op']
      - float64

    * - process_unit_size
      - 0
      - ['area', 'process_tech', 'year_op']
      - float64

    * - process_max_load
      - 1.0
      - ['area', 'process_tech', 'year_op']
      - float64

    * - process_min_load
      - 0.0
      - ['area', 'process_tech', 'year_op']
      - float64

    * - process_startup_cost
      - 0
      - ['area', 'process_tech', 'year_op']
      - float64

    * - process_shutdown_cost
      - 0
      - ['area', 'process_tech', 'year_op']
      - float64

    * - process_minimum_startup_time
      - 0
      - ['area', 'process_tech', 'year_op']
      - int64

    * - process_minimum_shutdown_time
      - 0
      - ['area', 'process_tech', 'year_op']
      - int64

    * - process_ramp_down
      - nan
      - ['area', 'process_tech', 'mode', 'year_op']
      - float64

    * - process_ramp_up
      - nan
      - ['area', 'process_tech', 'mode', 'year_op']
      - float64

    * - process_mode_ramp_down
      - nan
      - ['area', 'process_tech', 'mode', 'year_op']
      - float64

    * - process_mode_ramp_up
      - nan
      - ['area', 'process_tech', 'mode', 'year_op']
      - float64

    * - process_no_shutdown
      - 0
      - ['area', 'process_tech', 'year_op']
      - bool

    * - process_variable_cost
      - 0
      - ['area', 'process_tech', 'mode', 'operating_condition', 'year_op']
      - float64

    * - flexdem
      - 0
      - []
      - bool

    * - flexdem_demand
      - 0.0
      - ['area', 'hour', 'resource', 'operating_condition', 'year_op']
      - float64

    * - flexdem_conservation_hrs
      - 0
      - ['area', 'resource', 'year_op']
      - int64

    * - flexdem_window_offset_hrs
      - 0.0
      - []
      - float64

    * - flexdem_shifted_energy_initial
      - 0.0
      - ['area', 'resource', 'year_op']
      - float64

    * - flexdem_ramp_up
      - nan
      - ['area', 'resource', 'year_op']
      - float64

    * - flexdem_ramp_down
      - nan
      - ['area', 'resource', 'year_op']
      - float64

    * - flexdem_maxload_ratio
      - 1.0
      - ['area', 'resource', 'year_op']
      - float64

    * - flexdem_minload_ratio
      - 1.0
      - ['area', 'resource', 'year_op']
      - float64

    * - operating_condition_weight
      - nan
      - ['operating_condition', 'year_op']
      - float64

    * - carbon_goal_expected
      - 0
      - []
      - bool

    * - carbon_goal_area_expected
      - 0
      - []
      - bool

    * - carbon_goal_group_expected
      - 0
      - []
      - bool

    * - conversion_max_yearly_production_expected
      - 0
      - []
      - bool

    * - net_import_max_yearly_energy_import_expected
      - 0
      - []
      - bool

    * - net_import_max_yearly_energy_export_expected
      - 0
      - []
      - bool
