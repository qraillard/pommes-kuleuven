.. _model_description:

Model Description
*****************

Here are written the equations of the optimisation problem built by POMMES.
Have a look at the detailed nomenclature if needed.

.. toctree::
    :maxdepth: 1

    nomenclature

Objective function
==================

The objective function of the problem is given by the minimisation of
the actualised costs given in equations :eq:`objective`.
All variables are continuous and positive.

.. math::
    :label: objective

    \min_{
        \mathbf P^{inv},
        \mathbf P^{rep},
        \mathbf P^,
        \mathbf S^{inv},
        \mathbf S,
        \mathbf I,
        \mathbf E,
        \mathbf U
    }
    \sum_{a, y}
        \left[
            \phi_1 \left(
                \tau,y + \frac{\Delta y}{2}, y_0
            \right)
            \left(
                \mathcal{CAP}_{a, y} + \mathcal{FIX}_{a, y} + \mathcal{VAR}_{a, y}
            \right)
       \right]


Where :math:`\phi_1(\tau, y, y_0)` is the discount factor defined in :eq:`discount-factor` to actualise
all the costs to reference year :math:`y_0`.

.. math::
    :label: discount-factor

    \phi_1(\tau, y, y_0) = (1 + \tau)^{-(y - y_0)}


Costs definition
----------------

Annualised capital costs
^^^^^^^^^^^^^^^^^^^^^^^^

Annualised capital costs are given in equation :eq:`capex`,
and details are given in :eq:`capex-conv`, :eq:`capex-storage` and :eq:`capex-transport`.

.. math::
    :label: capex

    \mathcal{CAP}_{a, y} =
        \mathcal C^{cap}_{a, y} +
        \mathcal S^{cap}_{a, y} +
        \mathcal T^{cap}_{a, y} +
        \mathcal R^{cap}_{a, y}
    \quad \forall \, y

Where

.. math::
    :label: capex-conv

    \mathcal C^{cap}_{a, y} =
        \sum_{c, i \leq y, d > y}
            \phi_2(\alpha_{c}, d - i) \,
            \beta_{c} \,
            \mathbf P^{c, inv}_{c, i, d}
    \quad \forall \, y

.. math::
    :label: capex-storage

    \mathcal S^{cap}_{a, y} =
        \sum_{s, i \leq y, d > y}
            \phi_2(\alpha_{a, i, s}, d - i)
            \left(
                \beta_{a, i, s}\, \mathbf P^{s, inv}_{s, i, d} +
                \sigma_{a, i, s}\, \mathbf S^{inv}_{s, i, d}
            \right)
    \quad \forall \, y

.. math::
    :label: capex-retrofit

    \mathcal R^{cap}_{a, y} =
        \sum_{c_1, i, j, c, d, j \leq y < d}
            \phi_2(\alpha_{c_1, c, j}, d - j) \,
            \beta^{rep}_{c_1, c, j} \,
            \mathbf P^{rep}_{c_1, i, j, c, d}
    \quad \forall \, y

.. math::
    :label: capex-transport

    \mathcal{TODO}

The coefficient :math:`\phi_2(\alpha, n)` represents the capital
recovery factor in annualising the costs with a finance rate of
:math:`\alpha` during :math:`n` years with one term per year :eq:`crf`. Payments occur at the end of the year.

.. math::
    :label: crf

    \phi_2(\alpha, n) =
        \frac{\alpha}{1 - (1 + \alpha)^{-n}}

Repayments only occur when technology is available for operation. As a
result, all the CAPEX is considered between the installation and the
decommissioning or retrofit of a technology.

This describes the ``until_decommissioning`` convention. The study
configuration chooses the convention with ``pre_process: annuity_mode``:

- ``full_life`` (default): the annuity of a vintage is fixed at investment,
  :math:`\beta \, \phi_2(\alpha, L)` with :math:`L` the life span, and paid in
  every operation year :math:`y` of the full life :math:`i \leq y < i + L`,
  also after an early decommissioning (sunk cost, applied to conversion,
  combined, process, storage, transport and retrofit). Decommissioning early
  then only saves fixed and variable operation costs. The annuity has no
  decommissioning year dimension and ``annuity_perfect_foresight`` is not used.
  The capacity of a vintage ends at the last decommissioning year not after
  :math:`i + L`, while the annuity runs over the whole life span: when
  :math:`i + L` is not a decommissioning year, the operation years in between
  pay an annuity for capacity that is gone (a warning lists the affected
  entries). The gap does not exist when ``year_dec`` is not listed in the
  ``coords`` of the study configuration: the grid is then derived from the data
  (``derive_year_dec``) as the sorted union of every ``year_inv + life_span``
  (``year_ret + life_span`` for a retrofit; a fractional end of life is rounded
  down to a whole year), ``year_inv``, ``year_op`` and ``year_ret``, so every
  end of life is a decommissioning year and the capacity ends exactly at
  :math:`i + L`. A ``year_dec`` that is listed is used as it is, as before (a
  finer grid, for example yearly, also removes the gap). A parameter read from
  a file cannot be indexed by ``year_dec`` when ``year_dec`` is not listed.
  A dataset written by hand whose ``*_end_of_life`` is not in ``year_dec`` (its
  vintage would have no capacity variable) is completed by ``check_inputs``
  when the annuities are full life: the missing years are added to
  ``year_dec``, with a warning. With ``until_decommissioning`` the annuity
  tables depend on ``year_dec``, so ``check_inputs`` only warns.
- ``until_decommissioning``: the previous behaviour, as in the equations
  above, with annuities that depend on the decommissioning year :math:`d`.

Fixed operation costs
^^^^^^^^^^^^^^^^^^^^^^

Fixed operation costs are given in equation :eq:`fixed-opex` and details are provided in
:math:`fixed-costs-conv`, :math:`fixed-costs-storage`, and :math:`fixed-costs-turpe`.

.. math::
    :label: fixed-opex

    \mathcal{FIX}_{a, y} =
        \mathcal C^{fix}_{a, y} +
        \mathcal S^{fix}_{a, y} +
        \mathcal T^{w, fix}_{a, y}
    \quad \forall \, y

Where

.. math::
    :label: fixed-costs-conv

    \mathcal C^{fix}_{a, y} =
        \sum_{c, i \leq y}
           \omega_{c} \bar P^{c}_{y, c, i}
    \quad \forall \, y

.. math::
    :label: fixed-costs-storage

    \mathcal S^{fix}_{a, y} =
        \sum_{s, i \leq y}
           \omega_{a, i, s} \bar P^{s}_{y, s, i}
    \quad \forall \, y

.. math::
    :label: fixed-costs-turpe

    \mathcal T^{w, fix}_{a, y} =
        \sum_{h}
           \psi_{y, h}
           \left(
               \mathbf{CP}^{w}_{y, h} - \mathbf{CP}^{w}_{y, h - 1}
           \right)
    \quad \forall \, y

One can notice that repurposed conversion technologies fixed costs are
taken into account in conversion fixed costs as
:math:`\bar P^{s}_{y, s, i}` include their capacities.

Moreover, storage fixed costs are only proportional to installed power
capacity, not energy capacity.

As withdrawal contract power grows with the hour type index (see
:ref:`section-turpe-constraints`), the TURPE fixed tax is proportional to
the power increment of the next hour type in equation :eq:`fixed-costs-turpe`. No injection is
allowed to the ROW grid here.

Variable operation costs
^^^^^^^^^^^^^^^^^^^^^^^^

Variable operation costs are given in equation :eq:`var-opex` and details are given in
:eq:`var-opex-conv`, :eq:`var-opex-net-imp`, :eq:`var-opex-turpe`, and
:eq:`var-opex-adequacy`.

.. math::
    :label: var-opex

    \mathcal{VAR}_{a, y} =
       \mathcal C^{var}_{a, y}
       + \mathcal I^{net, var}_{a, y}
       + \mathcal{T}^{w, var}_{a, y}
       + \mathcal A^{var}_{a, y}
    \quad \forall \, y

Where

.. math::
    :label: var-opex-conv

    \mathcal C^{var}_{a, y} =
        \sum_{y, t, c, i}
            (\Delta T)_{t} \times
                \left(
                    t^{CO_2}_{a, y} \varepsilon^{CO_2}_{c} + \lambda_{c}
                \right)
                \mathbf P^{c}_{y, t, c, i}
        \quad \forall \, y

.. math::
    :label: var-opex-net-imp

    \mathcal I^{net, var}_{a, y} =
        \sum_{t, r}
            \left(
                \gamma^{imp}_{r, t, y} \times \mathbf{I}_{r, t, y}
                - \gamma^{exp}_{r, t, y} \times \mathbf{E}_{r, t, y}
            \right)
        + \left(t^{CO_2}_{a, y} \varepsilon^{CO_2}_{y, r, t} \right) \sum{r, t}
            I^{net}_{y, r, t}
    \quad \forall \, y

The TURPE cost structure implemented in this model sets the withdrawal
proportional tax for electricity imports on the hour type (equation :eq:`var-opex-turpe`). No injection is modelled
for electricity.

.. math::
    :label: var-opex-turpe

    \mathcal{T}^{w, var}_{a, y} =
        \sum_{t}
           \theta_{y, h = \varphi(t)} \, \mathbf I^{}_{y, t, r = electricity}
    \quad \forall \, y

The model has no proportional power capacity or energy capacity costs
for storage.

.. math::
    :label: var-opex-adequacy

    \mathcal{A}^{var}_{a, y} =
        \sum_{t, r}
           \left(
               \lambda^{curt}_{a, r, y}  \, \mathbf U^{curt}_{y, t, r}
               + \lambda^{spill}_{a, r, y}  \, U^{spill}_{y, t, r}
           \right)
    \quad \forall \, y

Adequacy constraint
====================

The adequacy is met for each resource at each operation time
:math:`adequacy`. The net generation of the conversion
(resp. storage) technologies are aggregated for each time step in the
:math:`U^{net, c}_{y, t, r}` (resp. :math:`U^{net, s}_{y, t, r}`)
variable defined in equation
:math:`conv-net-gen-def` (resp.
:math:`storage-net-gen-def`).

.. math::
    :label: adequacy

    \boxed{
        d^{}_{a, r, t, y}
        + U^{spill}_{y, t, r} =
            U^{net, c}_{y, t, r}
            + U^{net, s}_{y, t, r}
            + I^{net}_{y, t, r}
            + \mathbf U^{curt}_{y, t, r}
        \quad \forall \, y, t, r
    }

.. math::
    :label: conv-net-gen-def

    U^{net, c}_{y, t, r} =
        (\Delta T)_{t} \times
            \sum_{c, i}
                k^{}_{c, i, r} \, \mathbf P^{c}_{y, t, c, i}
    \quad \forall \, y, t, r

.. math::
    :label: storage-net-gen-def

    U^{net, s}_{y, t, r} =
        (\Delta T)_{t} \times
            \sum_{a, i, s}
                \left(
                    k^{in}_{s, i, r} \, \mathbf P^{s, in}_{y, t, s, i} +
                    k^{keep}_{s, i, r} \, \mathbf S^{}_{y, t, s, i} +
                    k^{out}_{s, i, r} \, \mathbf P^{s, out}_{y, t, s, i}
                \right)
        \quad \forall \, y, t, r

Capacity constraints
====================

The instant power of the conversion technologies is lower than the
available installed capacity :eq:`conv-availability`.

.. math::
    :label: conv-availability

    \mathbf P^{c}_{y, t, c, i} \leq a_{y, t, c, i} \, \bar P^{c}_{y, c, i}
    \quad \forall\, y, t, c, i

As seen above, this model innovates by allowing conversion technologies
to be repurposed. For instance, carbon capture can be added to SMR. The
retrofit factor :math:`q_{c1, i1, j, c2, i2}` links the capacity of
technology :math:`(c2, i2 = j, d2)` derived from the repurposed
technology :math:`(c1, i1, d1 = j)`. As :math:`\beta^{rep} > 0`, the
cost minimisation avoids retrofit when the retrofit factor is
null.

Considering this, the total installed capacity of :math:`(c, i)` for
operation year :math:`y` (:math:`\bar P^{c}_{y, c, i}`) is defined as
the total installed capacity of technology :math:`(c, i)` that is not
yet decommissioned in :math:`y`, plus the sum of the capacities which
were repurposed into :math:`(c, i)` that are not yet decommissioned in
:math:`y` :eq:`pbar-def`.

.. math::
    :label: pbar-def

    \bar P^{c}_{y, c, i} =
        \sum_{d > y}
            \mathbf P^{c, inv}_{c, i, d} +
        \sum_{c1, i1, j = i, d > y}
            q^{}_{c1, i1, j, c, d} \,
            \mathbf P^{c, rep}_{c1, i1, j, c, d}
    \quad \forall \, y, c, i

The total capacity of technology :math:`(c, i)` being repurposed to
other conversion technologies for year :math:`j` is lower than the
installed capacity of :math:`(c, i, d = j)` whether the capacity comes
from direct investment or retrofit :eq:`max-retrofit`. As a
consequence, it is possible to chain retrofit investments.

.. math::
    :label: max-retrofit

    \sum_{c2, d2}
        \mathbf P^{c, rep}_{c, i, j, c2, d2} \leq
    \mathbf P^{c, inv}_{c, i , d = j} +
    \sum_{c1, i1}
        \mathbf P^{c, rep}_{c1, i1, j = i, c, d = j}
    \quad \forall \, c, i, j

The minimum bounds are the invested capacity and the maximum allowed
each year by the decision maker :eq:`conv-inv-bounds`. This constraint
could be related to the deployment rate or to the limited space of the
local area, for example.

.. math::
    :label: conv-inv-bounds

    p^{c, min}_{c} \leq
    \sum_{d}
        \mathbf P^{c, inv}_{c, i, d} \leq
    p^{c, max}_{c}
    \quad \forall\, c, i

The same principle applies to retrofit in equation :eq:`rep-inv-bounds`.

.. math::
    :label: rep-inv-bounds

    p^{c, min,\, rep}_{i, c} \leq
    \sum_{c1, i1, j = i, d}
        \mathbf P^{c, rep}_{c1, i1, j, c, d} \leq
    p^{c, max,\, rep}_{i, c}
    \quad \forall \, i, c

Storage constraints
===================

The total storage power (resp. energy) capacity for operation year
:math:`y` is defined as the total power (resp. energy) capacity of
:math:`(s, i)` that is not yet decommissioned in :math:`y` in equation
:math:`stor-power-tot-def` (resp. :math:`stor-energy-tot-def`).

.. math::
    :label: stor-power-tot-def

    \bar P^{s}_{y, s, i} =
        \sum_{d > y}
            \mathbf P^{s, inv}_{s, i, d}
    \quad \forall \, y, s, i

.. math::
    :label: stor-energy-tot-def

    \bar S^{}_{y, s, i} =
        \sum_{d > y}
            \mathbf S^{inv}_{s, i, d}
    \quad \forall \, y, s, i

Any resource of the model can be stored in the right storage technology
:math:`s` is invested in. For all time steps :math:`t`, the input rate
of storage :math:`P^{s, in/out}_{y, t, s, i}` is bounded by the total
invested power capacity :math:`\bar P^{s}_{y, s, i}`:

.. math::
    :label: stor-power-capa

    \mathbf P^{s, in/out}_{y, t, s, i} \leq \bar P^{s}_{y, s, i}
    \quad \forall \, y, t, s

The total amount of energy in storage :math:`S_{y, t, s, i}` should
remain lower than the total invested energy capacity :math:`\bar S_{y, s, i}`:

.. math::
    :label: stor-energy-capa

    \mathbf S^{}_{y, t, s, i} \leq \bar S^{}_{y, s, i}
    \quad \forall \, y, t, s, i

Invested power and energy capacities are bounded by the minimum and
maximum allowed values in equations :eq:`stor-power-bounds` and
:math:`stor-energy-bounds`.

.. math::
    :label: stor-power-bounds

    p^{s, min}_{a, i, s} \leq
        \sum_{d}
            \mathbf P^{s, inv}_{s, i, d}
        \leq  p^{s, max}_{a, i, s}
    \quad \forall\, s, i, d

.. math::
    :label: stor-energy-bounds

    s^{s, min}_{a, i, s} \leq
        \sum_{d}
            \mathbf S^{inv}_{s, i, d}
        \leq s^{s, max}_{a, i, s}
    \quad \forall \, s, i, d

The total amount of energy in the storage at each time step
:math:`S_{y, t, s, i}` is defined as the total amount of energy in the
storage at the previous time step minus the dissipation plus the loaded
energy minus the discharged energy :eq:`stor-level`. The constraint is cyclic to avoid side effects.

Note that :math:`P^{s, in}_{y, t, s, i}` is the power to load 1 MWh into
the storage and :math:`P^{s, out}_{y, t, s, i}` is discharge power to
lower the storage level of 1 MWh.

.. math::
    :label: stor-level

    \mathbf S_{y, t, s, i} =
        \left(
            1 - \eta_{a, i, s}
        \right)^{\frac{(\Delta T)_{t}}{1[h]}} \, \mathbf S^{}_{y, t - 1, s, i}
        + (\Delta T)_{t}
        \times (
            \mathbf P^{s, in}_{y, t, s, i}
            - \mathbf P^{s, out}_{y, t, s, i}
        )
    \quad \forall \ y, t, s, i

Carbon related constraints
==========================

Equation :eq:`emission-def` defines the operation emission variable.

.. math::
    :label: emission-def

    E^{\ce{CO2}}_{y, t} =
        (\Delta T)_{t} \times
        \sum_{c, i}
            \varepsilon^{\ce{CO2}}_{c} \, \mathbf P^{c}_{y, t, c, i}
        + \sum_{r}
            \varepsilon^{\ce{CO2}}_{y, r, t} \, I^{net}_{y, r, t}
    \quad \forall \ y, t

The total emission constraint is in equation :eq:`emission-max`.

.. math::
    :label: emission-max

    \sum_{t}
        E^{\ce{CO2}}_{y, t}
    \leq g^{\ce{CO2}}_{a, y}
    \quad \forall \ y

Imports constraints
===================

Imports bound is defined in equation :eq:`imports`. Exports have not
been implemented yet. Variables are in energy units (MWh).

.. math::
    :label: net-imports-def

    I^{net}_{y, r, t} =
        \mathbf I^{}_{y, r, t} - \mathbf E^{}_{y, r, t}
    \quad \forall \, y, r, t

.. math::
    :label: abs-exchanges-def

    R_{y, r, t} =
        \mathbf I^{}_{y, r, t} + \mathbf E^{}_{y, r, t}
    \quad \forall \, y, r, t

.. math::
    :label: imports

    \mathbf I^{}_{y, r, t} \leq \pi^{max, imp}_{y, r, t}
    \quad \forall \, y, r, t

.. math::
    :label: exports

    \mathbf E^{}_{y, r, t} \leq \pi^{max, exp}_{y, r, t}
    \quad \forall \, y, r, t

.. _section-turpe-constraints:

TURPE constraints
=================

The constraints to implement the costs of the Tarif d’Utilisation du
Réseau Public d’Electricité (TURPE) are listed below.

First, the withdrawal power from the ROW grid is bound by the contract
power :eq:`max-withdraw-turpe`.

.. math::
    :label: max-withdraw-turpe

    \mathbf I^{}_{y, t, r = electricity} \leq
        \mathbf{CP}^{w}_{y, \varphi(t)}
    \quad \forall \, y, t

The function :math:`\varphi` links the hour of the operation with the
TURPE hour type (heure pleine, heure creuse, etc.)

.. _tab-turpe-hour-types:

.. table:: TURPE hour types

   ================== =========================== ===========
   TURPE hour type    :math:`h` (hour type index) designation
   ================== =========================== ===========
   heure creuse été   1                           HCE
   heure pleine été   2                           HPE
   heure creuse hiver 3                           HCH
   heure pleine hiver 4                           HPH
   pointe             5                           P
   ================== =========================== ===========

The hour type index indicates the criticality of the hour on the grid
(see :ref:`tab-turpe-hour-types`), and progressive taxes are in
place. The regulation imposes that the withdrawal contract power grows
with the criticality of the hour: equation :eq:`progressive-turpe`.

.. math::
    :label: progressive-turpe

    \mathbf{CP}^{w}_{y, h} \geq \mathbf{CP}^{w}_{y, h - 1}
    \quad \forall \, y, h

.. _section-time-blocks:

Time blocks (multi-resolution time steps)
=========================================

By default every operation variable has one value per time step. Four optional parameters let a conversion or a
combined technology, and a resource balance, run on **time blocks** of fixed length: one decision and one balance row
per block instead of one per step. Process, storage and transport stay at the step resolution. Without the parameters
the model is exactly that of 0.6.1+kul (same variables, constraints, objective and LP file).

Parameters (all ``nan`` by default): ``conversion_time_block_hrs`` and ``combined_time_block_hrs`` (block length
:math:`L` in whole hours of a technology), ``resource_time_block_hrs`` (forces the length of a resource balance) and
``conversion_time_block_rule`` (``"min"`` by default, or ``"mean"``: how the availability becomes a block value).
None of them has an ``area``, ``year_op``, ``operating_condition`` or ``hour`` coordinate: the block structure is
shared by the areas, years and conditions.

Blocks and anchoring
--------------------

With :math:`(\Delta T)_{t}` the length of step :math:`t`, the start of a step is
:math:`\tau_t = \sum_{t' < t} (\Delta T)_{t'}` and its block is :math:`b_L(t) = \lfloor \tau_t / L \rfloor`, so blocks
are anchored at the first step of the modelled horizon (the start of the year in a full-year run, the first step of a
time slice). A block :math:`b` has duration :math:`D_L(b) = \sum_{t \in b} (\Delta T)_{t}` and the duration of its
first step is :math:`\Delta^0_L(b)`. The last block may be shorter than :math:`L` (8760 h gives 52 weekly blocks and a
last block of 24 h). A step must not cross a block boundary (tolerance :math:`10^{-9}` h). The block dimension of
length :math:`L` is ``time_block_{L}h``, with values ``0 ... B_L - 1``.

Balance length (derivation)
---------------------------

Each resource balance has a length: hourly (the step grid) or one block length. A *counted contributor* of a resource
is a technology with a variable that can move that resource: a conversion or combined technology with a finite non-zero
factor, a storage, a transport (or its fuel), a process. Net imports, flexible demand, demand, load shedding and
spillage are not counted: they follow the balance. The length of the balance of a resource is

1. hourly if it has a counted hourly contributor (storage, transport, process, or a technology without a block length);
2. otherwise the smallest block length among its counted block contributors (every other one must be a multiple of it);
3. otherwise hourly.

``resource_time_block_hrs`` replaces the derived length when it is finite (a **forced** coarse balance); storage,
transport and process technologies then stay hourly and their step energies are summed into the block row. When some
resource has a block contributor but is derived hourly, ``build_model`` warns once and names the first hourly
contributor that decided it (for example ``steel: hourly because of storage tech steel_tank``).

The balance of resource :math:`r` of block length :math:`L` is written once per block, in energy per block, instead of
the hourly equation of the adequacy constraint:

.. math::
    :label: block-adequacy

    \sum_{t \in b} \Big( \text{hourly contributions}_{t} \Big)
    + \sum_{k} \text{block contributions}_{k, b}
    + D_L(b) \, \big( U^{curt}_{y, b, r} - U^{spill}_{y, b, r} \big)
    = d_{a, r, b} \quad \forall \, b

where the demand of the block is the sum of the step demands, a block technology of length :math:`L` contributes the
block energy :math:`D_L(b) \, \text{factor} \, P_{b}`, a technology of a finer resolution is summed over the block
and a technology of a coarser resolution is spread flat over the steps of its block. Its dual, for a coarse resource,
is one per block (the hourly adequacy dual is ``nan`` on coarse resources).

Block rows of a technology
--------------------------

Conversion and combined power variables of a group of length :math:`L` are
``operation_conversion_block_{L}h_power`` and ``operation_combined_block_{L}h_power``, on ``time_block_{L}h``; their
net generations (energies per block) are ``..._block_{L}h_net_generation``. Every block row has the name of the hourly
row with ``operation_<module>_`` replaced by ``operation_<module>_block_{L}h_``. Capacities, planning and annuity rows
are unchanged (one per unit).

- **Availability.** One block availability :math:`\bar A_b` per unit, block, year and condition (missing values read
  as 1 before aggregating): the minimum of the step availabilities in the block (rule ``min``: power maximum is exact
  against the hourly model with constant power in the block) or their duration-weighted mean (rule ``mean``: a
  relaxation). Power maximum: :math:`P_b \leq \bar A_b \, \text{cap}`. Must-run uses the same :math:`\bar A_b`:
  :math:`P_b = \bar A_b \, \text{cap}` for a must-run of 1 and :math:`P_b \geq m \, \bar A_b \, \text{cap}` for
  :math:`0 < m < 1`. On an availability that varies inside a block this is a relaxation of the hourly must-run rows.
- **Constant operation and ramps.** :math:`P_b - P_{b-1} = 0`, and for a ramp
  :math:`P_b - P_{b-1} \leq \rho \, \text{cap} \, \Delta^0_L(b)`, on blocks other than the first; the divisor is the
  duration of the first step of the block, which matches the hourly model where the jump happens at the first step of
  the block.
- **Annual quantities** (yearly production maximum, capacity factor, cumulative production, variable costs) use
  :math:`\sum_b D_L(b) \, P_b`, exact. The block net generation definition is
  :math:`NG_b = D_L(b) \, \text{factor} \, P_b`.
- **Daily production maximum.** The block length must divide the day length or be a multiple of it; the day energy of a
  flat block is then exact.
- **Flexible demand** on a coarse resource has a block variable ``operation_flexdem_block_{L}h_demand``, an energy per
  block. Maximum and minimum ratios, the global conservation and the conservation windows are written on block
  energies (the window length and offset must be multiples of :math:`L`). Ramps are written on the **mean powers** of the
  blocks, :math:`\bar f_b = F_b / D_L(b)`: :math:`\bar f_b - \bar f_{b-1} \leq \rho^{up} \, FD_b` (and the mirror for
  ramp down), with :math:`FD_b` the baseline demand energy of the current block.
- **Net imports** on a coarse resource are block energies (``operation_net_import_block_{L}h_import``, ``_export``,
  ``_abs`` and ``_net_generation``). Prices and emission factors are their duration-weighted block means, the yearly
  maxima are sums over blocks and the costs variable stays single.
- **Load shedding and spillage** on a coarse resource are block powers
  (``operation_load_shedding_block_{L}h_power``, ``operation_spillage_block_{L}h_power``) with the same caps and costs
  per unit of energy.
- **Carbon.** The emissions of a block technology are spread flat over the steps of its block (each step gets
  :math:`(\Delta T)_{t} \, \text{ef} \, P_b`), block net-import emissions likewise (:math:`(\Delta T)_{t}/D_L(b)` of the
  block emission). For block technologies the yearly sums, goals, taxes and the objective are therefore equal to those
  of the flat model; the emission factors of coarse net imports are block means, so their yearly sums are equal only
  against flat imports.

Refused inputs
--------------

``check_inputs`` and ``build_model`` refuse (with the same checks):

- R1 a block length that is not a positive whole number of hours;
- R2 a step that crosses a block boundary (this includes a length shorter than a step);
- R3 a non-positive or non-finite ``time_step_duration`` when a block length is given;
- R4 non-nested blocks in a balance (derived: a contributor whose length is not a multiple of the balance length;
  forced: lengths that do not divide one another);
- R5 a coarse electricity balance (forced or derived) while ``turpe`` is on;
- R6 a daily production maximum on a block technology whose length neither divides nor is a multiple of the day length;
- R8 a ``conversion_time_block_rule`` other than ``"min"`` or ``"mean"`` (``"nan"``, ``""`` and ``nan`` read as
  ``"min"``);
- R9 a flexible-demand conservation length or window offset that is not a multiple of the length of a coarse balance
  with active flexible demand;
- R10 every resource balance coarse (at least one resource must stay hourly).

Must-run on blocks is never refused. A length longer than the horizon gives one block.

Exactness and relaxations
-------------------------

The reference is the *flat twin*: the hourly model plus, for every block technology (per mode for a combined one), an
equality forcing the power to be constant inside each block.

.. list-table::
    :header-rows: 1

    * - Feature
      - Status
    * - Block technology, every resource hourly
      - exact against the flat twin (power maximum with rule ``min``, must-run where the availability is constant in
        each block, ramps, constant operation, annual sums, daily rule)
    * - Must-run on an availability that varies inside a block
      - relaxation (the block rows use :math:`\bar A_b`; the block objective is at most the twin's)
    * - Availability rule ``mean``
      - relaxation
    * - Coarse resource balance (derived or forced)
      - relaxation of the hourly balance where net import prices and emission factors are constant inside each block (or
        net import is absent): the block holds the energy of the block, not of each step; otherwise it has no sign
        against the hourly balance. Exact against the model run with steps of :math:`L` hours when every contributor has the
        balance's length and the horizon is a multiple of :math:`L`
    * - Load shedding and spillage on a coarse resource
      - exact
    * - Flexible demand on a coarse balance
      - equal to the model run with steps of :math:`L` hours; against an hourly flexible demand it has no sign when
        ramps bind (the ramp allowance is :math:`\rho \, FD_b` of the current block)
    * - Net imports on a coarse balance
      - exact against hourly imports held flat inside each block; equal to summed hourly imports when prices (and
        emission factors if carbon is on) are constant inside each block, otherwise a restriction of it
    * - Carbon of block technologies
      - exact against the flat twin, step by step

Outputs
-------

Block variables are written as built, with the names above. Because the hourly and block variables of a module cover
different technologies, ``model.solution`` joins them with ``nan`` outside their group (linopy raises a
``UserWarning``, expected for every block model). ``calculate_summaries`` sums the ``time_block_*`` dimensions like
``hour`` and combines the groups of a module by label under the module name (the group suffix ``_block_{L}h`` is
dropped). ``utils.expand_to_hours(da, tb, kind)`` maps a block array to the steps, ``kind="energy"`` multiplying by
:math:`(\Delta T)_{t}/D_L(b)` and ``kind="intensive"`` repeating the value; ``post_process.module_hourly`` combines the
hourly variable of a module (when it exists) and the expanded block variables, and ``get_net_generation`` and the plots
use it, including the block load shedding and spillage.

.. _section-operating-conditions:

Operating conditions (weighted weather years)
=============================================

A dataset may carry an optional coordinate ``operating_condition`` (index :math:`\kappa \in \mathcal{K}`,
free labels, int or str), for instance weather years, and a weight
:math:`\varpi_{y, \kappa}` (``operating_condition_weight``): the probability of a condition given its
operation year. Without the coordinate the model is exactly the model described above: every helper
returns its input unchanged and no variable or constraint is added.

Weights
-------

- Without the ``year_op`` dimension a weight applies to every year.
- The finite weights of a year sum to 1 (tolerance :math:`10^{-9}`), otherwise ``check_inputs`` raises.
  A weight must be finite or ``nan``, and not negative.
- ``nan``: the condition does not exist in that year. No variable and no constraint is written for the pair.
- ``0``: the condition is kept. Its constraints hold and its costs count for nothing. In particular its load
  shedding costs nothing: with load shedding allowed, a condition of weight 0 imposes no capacity; it imposes
  capacity only where shedding is not allowed.
- A dataset with the coordinate must carry the weight when it has several conditions; a single condition
  without weight gets weight 1.
- The weight composes with ``operation_year_weight``, whose meaning is unchanged (it scales the cumulative
  production only and does not enter the objective). The objective weighs a year by its discount factor.

Only the following parameters may carry the coordinate; ``check_inputs`` refuses it on any other parameter
and names it. Hourly profiles: ``conversion_availability``, ``transport_availability``, ``demand``,
``flexdem_demand``, ``net_import_import_price``, ``net_import_export_price``, ``net_import_emission_factor``,
``net_import_total_emission_factor``. Variable costs: ``conversion_variable_cost``,
``combined_variable_cost``, ``process_variable_cost``, ``transport_hurdle_costs``, ``load_shedding_cost``,
``spillage_cost``. Annual operational ceilings: ``conversion_max_yearly_production``,
``net_import_max_yearly_energy_import``, ``net_import_max_yearly_energy_export``. The time grid, the
capacities and their costs and bounds, the life spans, the technologies and the policy (carbon tax and goals,
TURPE tariffs) are shared by the conditions.

What takes the dimension
------------------------

Every operational variable (all the variables indexed by ``hour``) and their annual operational aggregates
(operational costs and emissions per year) carry the dimension :math:`\kappa`. The planning variables
(capacities, investments, decommissioning, retrofit), ``annualised_totex`` and the objective never do: the
capacities are shared by the conditions (one capacity covers the most demanding condition), the dispatch is
chosen per condition. The size of the operational part of the model is multiplied by the number of conditions;
the planning part is not.

Costs and objective
-------------------

Each operational cost is written per condition and enters the annualised totex through its expectation over
the conditions, :math:`\sum_{\kappa} \varpi_{y, \kappa} \, (\cdot)_{y, \kappa}` (a ``nan`` weight counts as 0).
Costs shared by the conditions (capital and fixed costs, the TURPE fixed cost) come back exactly once because the
weights of a year sum to 1. The objective keeps its form (equation :eq:`objective`): investment, fixed costs and
expected operation, with the discount factor of each year.

Operational constraints
-----------------------

The following hold in each condition: adequacy, capacity limits of the dispatch, ramps and constant operation,
transport (the ``transport_delay`` shift runs along ``hour`` inside each condition), flexible demand windows and
horizon conservation, storage level (the cycle is closed within each condition: no energy moves from one
condition to another), process unit commitment and minimum times (the integer variables are per condition: the
MILP grows with the number of conditions), ``conversion_max_capacity_factor`` and
``conversion_max_daily_production`` (no flag).

TURPE: the contract power is subscribed once per year (it has no condition) and covers the withdrawal of every
condition; its fixed cost is counted once and its variable cost in expectation. A condition of weight 0 imposes
contract power only if its dispatch needs it, its dispatch being free otherwise.

Annual constraints
------------------

One boolean flag per family (default ``False``) chooses the mode of an annual ceiling or goal. With ``False``
the constraint holds in each condition (one row per existing year and condition, and the ceiling may carry the
coordinate). With ``True`` the annual quantity goes through the expectation and there is one row per year; a
ceiling that carries the coordinate is then refused by ``check_inputs``. A condition of weight 0 still has its row
with ``False`` and counts for 0 in the expectation with ``True``.

.. list-table::
    :header-rows: 1

    * - Constraint
      - Flag
      - Summed quantity
    * - ``operation_carbon_goal_constraint``, ``operation_carbon_goal_area_constraint``,
        ``operation_carbon_goal_group_constraint``
      - ``carbon_goal_expected``, ``carbon_goal_area_expected``, ``carbon_goal_group_expected``
      - annual emissions times ``operation_year_normalization`` (unchanged)
    * - ``operation_conversion_yearly_production_max_constraint``
      - ``conversion_max_yearly_production_expected``
      - annual production
    * - ``operation_net_import_import_yearly_max_constraint``,
        ``operation_net_import_export_yearly_max_constraint``
      - ``net_import_max_yearly_energy_import_expected``, ``net_import_max_yearly_energy_export_expected``
      - annual energy
    * - ``operation_conversion_cumulative_production_max_constraint``
      - none
      - sum over the years of the expected annual production

The cumulative production spans several years whose conditions may differ, so "in each condition" has no meaning
there: it always uses the expectation. The carbon tax of an area and of a tax group is paid in each condition (floored
at 0, as without conditions) and counted in expectation: with emissions of +10 and -10 in two equally weighted
conditions, the tax is paid on 5, not on 0. ``combined`` has no annual production ceiling.

Retrofit
--------

Retrofit has no hourly variable: its decisions change capacities only. They are planning decisions shared by the
conditions, the retrofitted capacity is available in every condition and its costs are counted once.

Outputs
-------

- ``model.solution``: the operational variables keep the ``operating_condition`` dimension. ``save_solution`` and
  ``read_solution`` are unchanged: one more column in the CSV files, one more dimension in the NetCDF files.
  Selecting one condition needs a list, ``sel(operating_condition=["a"])``.
- Prices: the raw dual of an operational constraint is the weight of its condition times the price. Divide it by
  ``condition_dual_scale(p)`` (the weight, 1 without conditions) to get the price. It applies only to constraints that
  carry ``operating_condition``: the one-row-per-year constraints of the ``_expected`` flags already give the price of
  the year. In a condition of weight 0 the weight is 0: there is no
  price there (the division gives nan or inf).
- Summaries (``calculate_summaries.py``, ``post_process.py``) never sum over ``operating_condition``: energy,
  emissions and costs come per condition. Capacities, shared by the conditions, have no condition. The planning
  costs, which have none, are repeated in each condition, so the cost of a condition is its planning costs plus its
  operation costs, and the total is the weighted mean ``expected_over_conditions(...)`` (the weights of a year sum
  to 1), never a sum over the conditions. ``annualised_totex`` is already an expectation.
- ``expected_over_conditions`` on a result skips nan: an absent pair, or a variable masked in one condition, counts as 0,
  and an entry that is nan in every condition gives 0, not nan.
- In a condition of weight 0 the cost variables defined by inequalities (``operation_carbon_costs``,
  ``operation_carbon_tax_group_costs``) are not tight, since nothing pushes them down: their values there are
  bounds, not costs.

Out of scope: a rolling horizon with operating conditions (``flexdem_shifted_energy_initial`` cannot carry the
coordinate), conditions on capacities or policy parameters, and a different set of weights per area.
