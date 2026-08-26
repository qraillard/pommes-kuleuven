.. _nomenclature:

Nomenclature
************

Indices and Index Sets
=======================

.. tabularcolumns:: p{0.132\linewidth}p{0.434\linewidth}p{0.434\linewidth}
.. list-table::
    :header-rows: 1

    * - Symbol
      - POMMES
      - Description

    * - :math:`a \in \mathcal{A}`
      - ``area``
      - Areas

    * - :math:`i \in \mathcal{Y}^{inv}`
      - ``year_inv``
      - Investment years

    * - :math:`y \in \mathcal{Y}^{op}`
      - ``year_op``
      - Operation years

    * - :math:`d \in \mathcal{Y}^{dec}`
      - ``year_dec``
      - Decommissioning years

    * - :math:`t \in \mathcal{T}`
      - ``hour``
      - Operation snapshots

    * - :math:`r \in \mathcal{R}_{a}`
      - ``resource``
      - Resources in area :math:`a`

    * - :math:`c \in \mathcal{C}^{tech}_{a, i}`
      - ``conversion_tech``
      - Conversion technologies in area :math:`a` for year :math:`i`

    * - :math:`s \in \mathcal{S}^{tech}_{a, i}`
      - ``storage_tech``
      - Storage technologies in area :math:`a` for year :math:`i`

    * - :math:`w \in \mathcal{W}^{tech}_{a, i}`
      - ``transport_tech``
      - Transport technologies in area :math:`a` for year :math:`i`

    * - :math:`e \in \mathcal{E}^{tech}_{a, i}`
      - ``net_import_tech``
      - Import/export technologies in area :math:`a` for year :math:`i`

    * - :math:`h \in \mathcal{H}^{type}`
      - ``hour_type``
      - Hour type in TURPE

    * - :math:`j \in \mathcal{J}^{rep}`
      - ``retrofit_year``
      - retrofit years

General Parameters
==================

.. tabularcolumns:: p{0.132\linewidth}p{0.434\linewidth}p{0.434\linewidth}
.. list-table::
    :header-rows: 1

    * - Symbol
      - POMMES
      - Description

    * - :math:`d_{a, r, t, y}`
      - ``demand``
      - Exogenous demand of :math:`r` in area :math:`a` in :math:`y` at :math:`t` [MWh]

    * - :math:`\lambda^{curt}_{a, r, y}`
      - ``load_shedding_cost``
      - Energy curtailed cost [€/MWh]

    * - :math:`\lambda^{spill}_{a, r, y}`
      - ``spillage_cost``
      - Resource spillage cost [€/MWh]

    * - :math:`g^{CO_2}_{a, y}`
      - ``carbon_goal``
      - Maximum :math:`\mathrm{CO_2}` emissions in :math:`y` [kg :math:`\mathrm{CO_2}`]

    * - :math:`t^{CO_2}_{a, y}`
      - ``carbon_tax``
      - Carbon tax value in :math:`y` [€/kg :math:`\mathrm{CO_2}`]

    * - :math:`\tau`
      - ``discount_rate``
      - Discount rate [%]

    * - :math:`\alpha_{c}`
      - ``conversion_finance_rate``
      - Conversion tech finance rate [%]

    * - :math:`\alpha_{c1, c2, j}`
      - ``retrofit_finance_rate``
      - retrofit finance rate [%]

    * - :math:`\alpha_{s}`
      - ``storage_finance_rate``
      - Storage tech finance rate [%]

    * - :math:`y_0`
      - ``year_ref``
      - Reference year for actualisation

    * - :math:`\phi_1(\tau,y, y_0)`
      - ``discount_factor``
      - Discount factor [%]

    * - :math:`\phi_2(\alpha, n)`
      - ``crf``
      - Capital recovery factor [%]

    * - :math:`\Delta i = \Delta j = \Delta y`
      - ``planning_step``
      - Time step for investment [year]

    * - :math:`(\Delta T)_{t}`
      - ``hour``
      - Snapshot length [hour]

Retrofit Parameters
=====================

.. tabularcolumns:: p{0.132\linewidth}p{0.434\linewidth}p{0.434\linewidth}
.. list-table::
    :header-rows: 1

    * - Symbol
      - POMMES
      - Description

    * - :math:`\beta^{rep}_{c_1,c,j}`
      - ``retrofit_invest_cost``
      - Power capacity cost of retrofitting conversion technology :math:`c_1` to :math:`c` in year :math:`j`  [€/MW]

    * - :math:`q^{rep}_{c_1,c}`
      - ``retrofit_factor``
      - Retrofit factor from conversion technology :math:`c_1` to :math:`c`  [%]

    * :math:`p^{c, min, rep}_{j,c}`
      - ``retrofit_power_capacity_investment_min``
      - Minimum retrofitted capacity into :math:`c` [MW]

    * :math:`p^{c, max, rep}_{j,c}`
      - ``retrofit_power_capacity_investment_max``
      - Maximum retrofitted capacity into :math:`c` [MW]

Conversion Parameters
=====================

.. tabularcolumns:: p{0.132\linewidth}p{0.434\linewidth}p{0.434\linewidth}
.. list-table::
    :header-rows: 1

    * - Symbol
      - POMMES
      - Description

    * - :math:`\beta_{c}`
      - ``conversion_invest_cost``
      - Specific overnight cost of :math:`c` [€/MW]

    * - :math:`\omega_{c}`
      - ``conversion_fixed_cost``
      - Fixed OPEX of :math:`c` [€/MW/yr]

    * - :math:`\lambda_{c}`
      - ``conversion_variable_cost``
      - Variable OPEX of :math:`c` [€/MWh]

    * - :math:`a_{y, t, c, i}`
      - ``conversion_availability``
      - Availability of :math:`c` [:math:`0, 1`]

    * - :math:`k_{c, i, r}`
      - ``conversion_factor``
      - Conversion factor for :math:`c` [%]

    * - :math:`\varepsilon^{CO_2}_{c}`
      - ``conversion_emission_factor``
      - :math:`\mathrm{CO_2}` emission rate for :math:`c` [kg :math:`\mathrm{CO_2}/\text{MWh}`]

    * - :math:`l_{c}`
      - ``conversion_life_span``
      - Life length of :math:`c` [year]

    * - :math:`p^{c, min}_{c}`
      - ``conversion_power_capacity_investment_min``
      - Minimum capacity of :math:`c` [MW]

    * - :math:`p^{c, max}_{c}`
      - ``conversion_power_capacity_investment_max``
      - Maximum capacity of :math:`c` [MW]


Storage Parameters
==================

.. tabularcolumns:: p{0.132\linewidth}p{0.434\linewidth}p{0.434\linewidth}
.. list-table::
    :header-rows: 1

    * - Symbol
      - POMMES
      - Description

    * - :math:`\beta_{a, i, s}`
      - ``storage_invest_cost_power``
      - Power capacity cost of :math:`s` [€/MW]

    * - :math:`\sigma_{a, i, s}`
      - ``storage_invest_cost_energy``
      - Energy capacity cost of :math:`s` [€/MWh]

    * - :math:`\omega_{a, i, s}`
      - ``storage_fixed_cost_power``
      - Fixed OPEX of :math:`s` [€/MW/yr]

    * - :math:`\eta_{a, i, s}`
      - ``storage_dissipation``
      - Dissipation losses over 1 hour [%]

    * - :math:`k^{in}_{s, i, r}`
      - ``storage_factor_in``
      - Charging factor for :math:`s` [%]

    * - :math:`k^{keep}_{s, i, r}`
      - ``storage_factor_keep``
      - Consumption factor to keep 1 MWh in :math:`s` [:math:`\text{h}^{-1}`]

    * - :math:`k^{out}_{s, i, r}`
      - ``storage_factor_out``
      - Discharging factor for :math:`s` [%]

    * - :math:`res_{s}`
      - ``storage_main_resource``
      - Main resource stored by :math:`s`

    * - :math:`l_{a, i, s}`
      - ``storage_life_span``
      - Life length of :math:`s` [year]

    * - :math:`p^{s, min}_{a, i, s}`
      - ``storage_power_capacity_investment_min``
      - Minimum storage power capacity [MW]

    * - :math:`p^{s, max}_{a, i, s}`
      - ``storage_power_capacity_investment_max``
      - Maximum storage power capacity [MW]

    * - :math:`s^{s, min}_{a, i, s}`
      - ``storage_energy_capacity_investment_min``
      - Minimum storage energy capacity [MWh]

    * - :math:`s^{s, max}_{a, i, s}`
      - ``storage_energy_capacity_investment_max``
      - Maximum storage energy capacity [MWh]


Transport Parameters
====================

.. tabularcolumns:: p{0.132\linewidth}p{0.434\linewidth}p{0.434\linewidth}
.. list-table::
    :header-rows: 1

    * - Symbol
      - POMMES
      - Description

    * - :math:`\gamma_{y, r, t}`
      - ``TODO``
      - Importation cost of :math:`r` in :math:`y` at :math:`t` from ROW [€/MWh]


Networks Parameters
====================

.. tabularcolumns:: p{0.132\linewidth}p{0.434\linewidth}p{0.434\linewidth}
.. list-table::
    :header-rows: 1

    * - Symbol
      - POMMES
      - Description

    * - :math:`\gamma^{imp}_{y, r, t}`
      - ``net_import_import_cost``
      - Importation cost of :math:`r` in :math:`y` at :math:`t` from ROW [€/MWh]

    * - :math:`\gamma^{exp}_{y, r, t}`
      - ``net_import_export_cost``
      - Exportation price of :math:`r` in :math:`y` at :math:`t` to ROW [€/MWh]

    * - :math:`\varepsilon^{CO_2}_{y, r, t}`
      - ``net_import_emission_factor``
      - Importation emission rate of :math:`r` in :math:`y` at :math:`t` from ROW [kg :math:`\mathrm{CO_2}`/MWh]

    * - :math:`\pi^{max, imp}_{y, r, t}`
      - ``net_import_max_yearly_energy_import``
      - Max yearly importation volume of :math:`r` in :math:`y` from ROW [MWh]

    * - :math:`\pi^{max, exp}_{y, r, t}`
      - ``net_import_max_yearly_energy_export``
      - Max yearly exportation volume of :math:`r` in :math:`y` to ROW [MWh]

    * - :math:`\varphi(t)`
      - ``turpe_calendar``
      - Matching key between operation hour and TURPE hour type :math:`h`

    * - :math:`\psi_{y, h}`
      - ``turpe_fixed_cost``
      - Fixed cost of withdrawal component of TURPE for :math:`h` [€/MW]

    * - :math:`\theta_{y, h}`
      - ``turpe_variable_cost``
      - Variable cost of withdrawal component of TURPE for :math:`h` [€/MWh]

Investment Variables
=====================

.. tabularcolumns:: p{0.132\linewidth}p{0.434\linewidth}p{0.434\linewidth}
.. list-table::
    :header-rows: 1

    * - Symbol
      - POMMES
      - Description

    * - :math:`\mathbf{P}^{c, inv}_{c, i, d}`
      - ``planning_conversion_power_capacity``
      - Capacity of :math:`c` invested for year :math:`i` decommissioned for year :math:`d` [MW]

    * - :math:`\mathbf{P}^{s, inv}_{s, i, d}`
      - ``Planning_storage_power_capacity``
      - Storage power capacity of :math:`s` invested for year :math:`i` decommissioned for year :math:`d` [MW]

    * - :math:`\mathbf{S}^{inv}_{s, i, d}`
      - ``Planning_storage_energy_capacity``
      - Storage energy capacity of :math:`s` invested for year :math:`i` decommissioned for year :math:`d` [MWh]

    * - :math:`\mathbf{P}^{rep}_{c_1, i_1, j, c, d}`
      - ``planning_retrofit_power_capacity``
      - Total retrofit power capacity from conversion technology :math:`c_1` invested in :math:`i_1` to conversion technology :math:`c` in retrofit year :math:`j` and operation in year :math:`y` [MW]

    * - :math:`\bar{P}^{c}_{y, c, i}`
      - ``operation_conversion_power_capacity``
      - Total capacity of :math:`c` invested for :math:`i` in operation in year :math:`y` [MW]

    * - :math:`\bar{P}^{s}_{y, s, i}`
      - ``operation_storage_power_capacity``
      - Total storage power capacity of :math:`s` invested for :math:`i` in operation in year :math:`y` [MW]

    * - :math:`\bar{S}_{y, s, i}`
      - ``operation_storage_energy_capacity``
      - Total storage energy capacity of :math:`s` invested for :math:`i` in operation in year :math:`y` [MWh]




Operation Variables
====================

.. tabularcolumns:: p{0.132\linewidth}p{0.434\linewidth}p{0.434\linewidth}
.. list-table::
    :header-rows: 1

    * - Symbol
      - POMMES
      - Description

    * - :math:`\mathbf{P}^{c}_{y, t, c, i}`
      - ``operation_conversion_power``
      - Power of :math:`(c, i)` in :math:`y` at :math:`t` [MW]

    * - :math:`U^{net, c}_{y, t, r}`
      - ``operation_conversion_net_generation``
      - Resource :math:`r` net generation from conversion technologies in :math:`y` at :math:`t` [MWh]

    * - :math:`\mathbf{P}^{s, in}_{y, t, s, i}`
      - ``operation_storage_power_in``
      - Storage charging power of :math:`(s, i)` in :math:`y` at :math:`t` [MW]

    * - :math:`\mathbf{P}^{s, out}_{y, t, s, i}`
      - ``operation_storage_power_out``
      - Storage discharging power of :math:`(s, i)` in :math:`y` at :math:`t` [MW]

    * - :math:`\mathbf{S}_{y, t, s, i}`
      - ``operation_storage_level``
      - Amount of energy in :math:`(s, i)` in :math:`y` at :math:`t` [MWh]

    * - :math:`U^{net, s}_{y, t, r}`
      - ``operation_storage_net_generation``
      - Resource :math:`r` net generation from storage technologies in :math:`y` at :math:`t` [MWh]

    * - :math:`\mathbf{U}^{curt}_{y, t, r}`
      - ``operation_load_shedding``
      - Curtailed energy for resource :math:`r` in :math:`y` at :math:`t` [MWh]

    * - :math:`U^{spill}_{y, t, r}`
      - ``operation_spillage``
      - Resource spillage for :math:`r` in :math:`y` at :math:`t` [MWh]

    * - :math:`\mathbf{I}_{y, t, r}`
      - ``operation_net_import_import``
      - Importation of :math:`r` in :math:`y` at :math:`t` from ROW [MWh]

    * - :math:`\mathbf{E}_{y, t, r}`
      - ``operation_net_import_export``
      - Exportation of :math:`r` in :math:`y` at :math:`t` to ROW [MWh]

    * - :math:`I^{net}_{y, t, r}`
      - ``operation_net_import_net_generation``
      - Net imports from ROW [MWh]

    * - :math:`R_{y, t, r}`
      - ``operation_net_import_abs``
      - Absolute exchanges from and to ROW [MWh]

    * - :math:`\mathbf{CP}^{w}_{y, h}`
      - ``turpe_TODO !!``
      - Withdrawal contract power for electricity imports from ROW for :math:`h` [MW]

    * - :math:`E^{CO_2}_{y, t}`
      - ``operation_carbon_emissions``
      - :math:`\mathrm{CO_2}` emission at :math:`t` [kg :math:`\mathrm{CO_2}`]


Cost Variables
===============

.. tabularcolumns:: p{0.132\linewidth}p{0.434\linewidth}p{0.434\linewidth}
.. list-table::
    :header-rows: 1

    * - Symbol
      - POMMES
      - Description

    * - :math:`\mathcal{CAP}_{a, y}`
      - ``annualised_totex``
      - Annualised cost of capital of the system in :math:`y`

    * - :math:`\mathcal{FIX}_{a, y}`
      - included in operation costs
      - Fixed operation costs of the system in :math:`y`

    * - :math:`\mathcal{VAR}_{a, y}`
      - included operation _costs
      - Variable (proportional) operation costs of the system in :math:`y`

    * - :math:`\mathcal{A}^{var}_{a, y}`
      - sum of ``operation_load_shedding_costs`` and ``operation_spillage_costs``
      - Adequacy (curtailment and spillage) costs in :math:`y`

    * - :math:`\mathcal{C}^{cap}_{a, y}`
      - ``planning_conversion_costs``
      - Conversion annualised capital cost in :math:`y`

    * - :math:`\mathcal{C}^{fix}_{a, y}`
      - included in ``operation_conversion_costs``
      - Conversion fixed operation cost in :math:`y`

    * - :math:`\mathcal{C}^{var}_{a, y}`
      - included in ``operation_conversion_costs``
      - Conversion variable operation cost in :math:`y`

    * - :math:`\mathcal{S}^{cap}_{a, y}`
      - ``planning_storage_costs``
      - Storage annualised capital cost in :math:`y`

    * - :math:`\mathcal{S}^{fix}_{a, y}`
      - included in ``operation_storage_costs``
      - Storage fixed operation cost in :math:`y`

    * - :math:`\mathcal{I}^{net, var}_{a, y}`
      - ``operation_net_import_costs``
      - Net imports from ROW variable costs in :math:`y`

    * - :math:`\mathcal{R}^{rep}_{a, y}`
      - ``planning_retrofit_costs``
      - Retrofit annualised capital cost in :math:`y`


    * - :math:`\mathcal{T}^{w, fix}_{a, y}`
      - ``TODO``
      - Electricity withdrawal contract power related tax (TURPE) in :math:`y`

    * - :math:`\mathcal{T}^{w, var}_{a, y}`
      - ``TODO``
      - Electricity withdrawal proportional tax (TURPE) in :math:`y`
