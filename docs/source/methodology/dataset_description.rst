.. _dataset_description:

Dataset Description
===================

.. list-table::
    :header-rows: 1

    * - Key
      - default
      - index_set
      - type

    * - carbon_goal
      - nan
      - ['year_op']
      - float64

    * - carbon
      - 0
      - []
      - bool

    * - carbon_tax
      - nan
      - ['area', 'year_op']
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

    * - combined_ramp_down
      - nan
      - ['area', 'combined_tech', 'mode', 'year_op']
      - float64

    * - combined_ramp_up
      - nan
      - ['area', 'combined_tech', 'mode', 'year_op']
      - float64

    * - combined_variable_cost
      - 0
      - ['area', 'combined_tech', 'mode', 'year_op']
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
      - ['area', 'conversion_tech', 'hour', 'year_op']
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
      - ['area', 'conversion_tech', 'year_op']
      - float64

    * - load_shedding_cost
      - 0.0
      - ['area', 'resource', 'year_op']
      - float64

    * - load_shedding_max_capacity
      - nan
      - ['area', 'resource', 'year_op']
      - float64

    * - demand
      - 0.0
      - ['area', 'hour', 'resource', 'year_op']
      - float64

    * - discount_factor
      - nan
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
      - ['area', 'hour', 'resource', 'year_op']
      - float64

    * - net_import_import_price
      - 0.0
      - ['area', 'hour', 'resource', 'year_op']
      - float64

    * - net_import_max_yearly_energy_export
      - nan
      - ['area', 'resource', 'year_op']
      - float64

    * - net_import_max_yearly_energy_import
      - nan
      - ['area', 'resource', 'year_op']
      - float64

    * - net_import_total_emission_factor
      - 0.0
      - ['area', 'hour', 'resource', 'year_op']
      - float64

    * - operation_year_duration
      - 8760
      - ['year_op']
      - float64

    * - planning_step
      - 0
      - []
      - int64

    * - spillage_cost
      - 0.0
      - ['area', 'resource', 'year_op']
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
      - ['link', 'transport_tech', 'year_op']
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

    * - transport_resource
      - nan
      - ['link', 'transport_tech']
      - str

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

    * - retrofit
      - 0.0
      - []
      - bool

    * - retrofit_tech_from
      - ""
      - ['area']
      - str

    * - retrofit_tech_to
      - ""
      - ['area']
      - str

    * - retrofit_year
      - nan
      - ['year_op']
      - int64

    * - retrofit_factor
      - 1.0
      - ['area','year_inv','retrofit_tech_from','retrofit_tech_to']
      - float64

    * - retrofit_invest_cost
      - 0.0
      - ['area','year_op','retrofit_tech_from','retrofit_tech_to', 'retrofit_year', 'year_dec']
      - float64

    * - retrofit_finance_rate
      - 0.0
      - ['area','retrofit_tech_from','retrofit_tech_to', 'retrofit_year']
      - float64

    * - retrofit_power_capacity_investment_min
      - nan
      - ['retrofit_tech_to', 'retrofit_year']
      - float64

    * - retrofit_power_capacity_investment_max
      - nan
      - ['retrofit_tech_to', 'retrofit_year']
      - float64