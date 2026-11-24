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
        + \sum_{l, \tau, a = a^{to}_{l, \tau}}
            \varepsilon^{\ce{CO2}, fuel}_{y, l, \tau} \, \kappa_{y, l, \tau} \,
            (\Delta T)_{t} \, \mathbf P^{in}_{y, t, l, \tau}
    \quad \forall \ y, t

Transport. A transport technology :math:`\tau` on a link :math:`l` may burn fuel, :math:`\kappa` per MWh injected
(``transport_fuel_consumption``, drawn at ``transport_fuel_area`` or at the departure area). The user chooses how its
emissions are included, with two optional parameters in tCO2 per MWh of fuel burnt (default 0): ``transport_fuel_emission_factor``
(:math:`\varepsilon^{\ce{CO2}, fuel}`, the last term above) enters the emissions that the carbon tax and the carbon tax
groups see, ``transport_fuel_total_emission_factor`` enters the emissions that the carbon goals see (system, per area,
groups of areas), through the same term with its own factor. A study may tax them, count them in the goals without taxing
them (an area that taxes its direct emissions only), do both, or neither. They are counted in the arrival area of the
link (``transport_area_to``), whatever ``transport_fuel_area`` is, at the step the fuel is burnt (the injection step,
also with a ``transport_delay``). A term is added only where a (link, technology) pair that exists burns fuel with a
non-zero factor, so without the parameters (or at 0, or NaN) the model is unchanged. If the supplier of the fuel already
carries its emissions (an emission factor on the conversion that produces it, or on its import), leave these factors at
0, or the emissions are counted twice. When ``transport_fuel_resource`` is not a resource of the model, no balance is
drawn from and the factors still apply to the consumption, so a study can count the emissions of a fuel it does not
model. A block transport technology spreads its fuel emissions over the steps of its block (see the Time blocks
section).

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

.. _section-transport-modes:

Transport technologies with modes
=================================

A transport technology can carry several resources on one link, for instance a methane pipe that may carry a blend of
methane and hydrogen. The technology has **modes**: a mode is a composition (how many MWh of each resource leave the
departure area per MWh of flow). The model chooses, hour by hour, how much flow goes in each mode. One capacity is shared
by all the modes. Without a modal technology (no coordinate ``transport_mode``, or no factor above 0) the model is
exactly that of 0.8.0+kul: no variable and no constraint is added, and the LP file is the same byte for byte.

Inputs
------

- The optional coordinate ``transport_mode`` (labels of the modes of all the technologies), declared in the study config
  under ``coords`` like ``operating_condition``. **The order of the labels is the order of the modes** (see below).
- ``transport_mode_factor[transport_tech, transport_mode, resource]`` (default 0, ``nan`` reads as 0, values
  :math:`\geq 0`): :math:`f_{\tau, m, r}`, MWh of resource :math:`r` that leave the "from" area per MWh of injected flow
  :math:`P^{in}` of technology :math:`\tau` in mode :math:`m`. The "to" area receives :math:`\eta` times that
  (``transport_efficiency``). A factor has no link, area, year or hour: the composition of a mode is the same on every
  link where the technology exists. A dimension may be missing from the factor (it is broadcast); another one is refused.
- ``transport_mode_ramp_up[link, transport_tech, year_op]`` and ``transport_mode_ramp_down[link, transport_tech,
  year_op]`` (default ``nan``, no constraint): :math:`\rho^{sw, up}`, :math:`\rho^{sw, down}` (``sw``: switch), the
  largest change per hour, as a fraction of the capacity, of the difference between the flows of two modes. Read only for
  modal technologies.
- ``transport_resource`` stays the *main* resource of a link: it decides where the (link, technology) pair exists and
  must have a positive factor in some mode (refusal R17).

In the study config, the coordinate and the factor are declared as follows (the files are CSV files of the study; the
ramps, when used, are declared in the same way, with ``fill: nan`` and ``index_input: [link, transport_tech, year_op]``):

.. code-block:: yaml

    coords:
      transport_mode:
        type: str
        values: ["blend", "methane"]

    input:
      parameters:
        transport_mode_factor:
          file: transport_mode.csv
          column: transport_mode_factor
          type: float64
          fill: 0.0
          index_input: [transport_tech, transport_mode, resource]

A mode :math:`m` is *defined* for :math:`\tau` when one of its factors is above 0, and :math:`\tau` is *modal*
(:math:`\tau \in \mathcal{W}^{mod} \subset \mathcal{W}^{tech}`) when it has a defined mode. One defined mode alone is
allowed (a fixed blend). A mode may be defined for some technologies and not for others.

Variable and constraints
------------------------

For a modal technology :math:`\tau`, the variable :math:`\mathbf X_{y, t, l, \tau, m}`
(``operation_transport_mode_power_in``, MW of injected flow, :math:`\geq 0`) is the power injected at the "from" end of
link :math:`l` in mode :math:`m`. It lives on the modal technologies only and on the existing pairs and defined modes.
:math:`\mathbf{P}`, :math:`\mathbf{P}^{in}` and :math:`\mathbf{NG}` are ``operation_transport_power`` (withdrawn at the
"to" end), ``operation_transport_power_in`` and ``operation_transport_net_generation``; :math:`\mathrm{cap}` is
``operation_transport_power_capacity``.

The modes are tied to the total injected flow (``operation_transport_mode_power_in_def``):

.. math::
    :label: transport-mode-tie

    \sum_{m} \mathbf X_{y, t, l, \tau, m} - \mathbf P^{in}_{y, t, l, \tau} = 0
    \quad \forall \, y, t, l, \tau \in \mathcal{W}^{mod}

The net generation of every resource that the technology carries (a resource is *carried* when one of its factors is above
0) is defined from the modes (``operation_transport_mode_net_generation_def``). It replaces
``operation_transport_net_generation_def`` for the modal technologies:

.. math::
    :label: transport-mode-net-generation

    \mathbf{NG}_{y, t, a, \tau, r} = \Delta t_t \sum_{m} f_{\tau, m, r}
    \Big( \sum_{l : to(l) = a} \eta_{y, l, \tau} \, \mathbf X_{y, t, l, \tau, m}
        - \sum_{l : from(l) = a} \mathbf X_{y, t, l, \tau, m} \Big)
    \quad \forall \, y, t, a, \tau \in \mathcal{W}^{mod}, r

For each pair of modes :math:`(m, m')`, the first one before the second one in the coordinate, both defined for
:math:`\tau`, with :math:`d_{y, t, l, \tau} = \mathbf X_{y, t, l, \tau, m} - \mathbf X_{y, t, l, \tau, m'}` and on the hours
other than the first (the ramp is not cyclic), where the ramp is finite:

.. math::
    :label: transport-mode-ramp-up

    \frac{d_{y, t} - d_{y, t-1}}{\Delta t_t} \leq \rho^{sw, up}_{y, l, \tau} \, \mathrm{cap}_{y, l, \tau}

.. math::
    :label: transport-mode-ramp-down

    \frac{d_{y, t-1} - d_{y, t}}{\Delta t_t} \leq \rho^{sw, down}_{y, l, \tau} \, \mathrm{cap}_{y, l, \tau}

The names of the two ramps are ``operation_transport_mode_ramp_up_constraint_{i}_{j}`` and
``operation_transport_mode_ramp_down_constraint_{i}_{j}``, where :math:`i < j` are the **zero-based positions** of the
two modes in the coordinate ``transport_mode`` (never their labels): position 0 is the first label. Only pairs of modes
that are defined for a technology are written, and a pair with no live cell is not written. The ramps are in capacity
fractions per hour of the *injected* flow; at :math:`\eta = 1` this is the withdrawn flow.

What the mode ramp bounds
^^^^^^^^^^^^^^^^^^^^^^^^^

The ramps bound the hourly change of the **difference** of the flows of two modes, in MW, by :math:`\rho^{sw} \,
\mathrm{cap}`. They do not bound the share of a mode: a ramp on the share :math:`X_{m} / (X_{m} + X_{m'})` is a ratio of two
variables and is not linear, so it is not written. The ramp is a ramp on the split between two modes only while the
total flow is constant. With :math:`\mathrm{cap} = 20` MW and :math:`\rho^{sw} = 0.5` (10 MW per hour), flows written
as (blend, methane):

.. list-table::
    :header-rows: 1

    * - Case
      - Flows
      - Change of the difference
      - Bounded
    * - Constant total, the split switches
      - (0, 10) then (10, 0)
      - 20
      - yes: two hours
    * - One mode alone, the total rises
      - (0, 0) then (0, 20)
      - 20
      - yes: two hours, with no change of mode
    * - Both modes rise by the same amount
      - (5, 5) then (10, 10)
      - 0
      - no

Two consequences. A complete switch at a constant flow :math:`Q` changes the difference by :math:`2 Q`: on a full pipe,
:math:`\rho^{sw} = 2`, not 1, leaves the switch free within the hour. And the mode ramp also slows a pipe that starts or stops
in a single mode.

The order of the modes is yours
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``transport_mode_ramp_up`` and ``transport_mode_ramp_down`` follow the order of the coordinate ``transport_mode``: ``up``
bounds how fast the first mode minus the second one can grow, ``down`` how fast it can fall. The model never sorts the
modes, and the two inputs stay separate: you order the modes and you choose the two values. With the order
``[blend, methane]``, the rise of ``blend`` alone is bounded by ``up`` and the rise of ``methane`` alone by ``down``. When
no asymmetry is wanted, **give the same value to** ``transport_mode_ramp_up`` **and** ``transport_mode_ramp_down``. The
order read from a config file is the order of its ``coords`` entry.

What stays on the total flow
----------------------------

Everything else is written as without modes, on the total flows :math:`\mathbf P` and :math:`\mathbf P^{in}`, for a modal
and a non-modal technology alike: the capacity shared by the modes (``operation_transport_power_max_constraint``:
:math:`\mathbf P \leq \mathrm{cap} \cdot` availability), ``operation_transport_power_in_def`` (:math:`\mathbf P^{in}
\eta = \mathbf P`), the ramp on the total flow (``transport_ramp_up``, ``transport_ramp_down``), the capacity, its bounds,
the planning and the annuities, the fixed cost on the capacity, the hurdle cost on :math:`\mathbf P`, the fuel and its
emissions on :math:`\mathbf P^{in}`, the retrofit and the adequacy (:math:`\mathbf{NG}` has one cell per carried
resource). Only the existing net generation equation excludes the modal technologies, which the mode equation replaces.
A modal technology is an hourly technology.

Meaning of a flow
-----------------

The factors are free: they are not shares. For a modal pair, :math:`\mathbf X_m` is a flow of injected power, in MW of
flow, not MW of energy. Per MW of :math:`\mathbf X_m`:

- :math:`f_{\tau, m, r}` MWh of resource :math:`r` leave the "from" area and :math:`\eta f_{\tau, m, r}` arrive at the
  "to" area;
- the capacity and the hurdle cost count the withdrawn flow :math:`\eta \sum_m \mathbf X_m` (per MWh of injected flow the
  capacity used is :math:`\eta`), and the fuel counts the injected flow :math:`\sum_m \mathbf X_m`;
- :math:`F_{\tau, m} = \sum_r f_{\tau, m, r}` is the energy that leaves the "from" area per unit of injected flow.
  :math:`F = 1` is a mode of shares. :math:`F < 1` is a loss of energy written in the matrix: the pipe moves less
  energy than its flow, no resource holds the missing :math:`1 - F`, nothing is created and nothing is stored.
  :math:`F > 1` is allowed and means more energy than flow. A mode that does not sum to 1 raises one warning (W1, below).

*Worked example.* A methane pipe whose capacity is read in MW of methane. Mode ``methane``: :math:`f = \{methane: 1.00\}`.
Mode ``blend``: :math:`f = \{methane: 0.90, hydrogen: 0.03\}`, so :math:`F = 0.93` (a blend of 10 % hydrogen by volume at
equal pressure carries about 0.90 of the energy of methane in methane and 0.03 in hydrogen per unit of volume, because
hydrogen has about 3 kWh/Nm3 against about 10 for methane). With a capacity of 20 MW, :math:`\eta = 1` and
:math:`\mathbf X_{blend} = 10` MW, the pipe takes 10 MW of capacity and pays 10 MW of hurdle cost and fuel; it takes 9.0
MWh/h of methane and 0.3 MWh/h of hydrogen from the departure area and delivers the same; it carries 9.3 MWh/h of energy,
of which 3.23 % (:math:`0.03 / 0.93`) is hydrogen. To deliver 10 MWh/h of energy in this mode alone one needs
:math:`\mathbf X = 10 / 0.93 = 10.7527` MW of flow, which still fits the 20 MW. With a capacity of 10 MW, at most 9.3
MWh/h could be delivered.

The share cap
-------------

For a modal pair at a step with flows :math:`X_m \geq 0`, the energy share of resource :math:`r` in what the pipe carries
is :math:`s_r = \sum_m f_{m, r} X_m / \sum_m F_m X_m`. By the mediant inequality :math:`\min_m f_{m, r} / F_m \leq s_r \leq
\max_m f_{m, r} / F_m` (over the modes with :math:`X_m > 0`), and each bound is reached by flowing in one mode only. So the
hydrogen share is at most that of the richest mode (:math:`f_{m, h}` when :math:`F = 1`), with no constraint written for
it. The same holds for the energy delivered, as :math:`\eta` is common.

Refusals and warning
--------------------

``check_inputs`` and ``build_model`` refuse (``ValueError``), only when the ``transport`` flag is on. A dataset that
breaks several rules gets one message. R18 and the refusal of a factor with another dimension can be raised while the
time blocks are derived, before the others; apart from that the refusals are checked in this order, before R11 and
R12 of the block technologies:

- single resource: a transport technology that carries more than one resource of the model through
  ``transport_resource`` on its links: ``a transport tech must carry a single resource on all its links`` (this refusal
  also applies without modes: the dense engine used to build a wrong model in this case);
- R18 ``transport_mode`` is a dimension of the dataset and not a coordinate: ``transport_mode is a dimension without a
  coordinate: call assign_coords(transport_mode=[...]) first.`` (a dataset built by hand; a config that does not declare
  the mode under ``coords`` raises a ``KeyError``);
- R16 a factor that is negative or infinite (``nan`` reads as 0), the first one in the order technology, mode, resource:
  ``{tech}: transport_mode_factor[{mode}, {resource}] = {value}: a factor is finite and >= 0 (NaN reads as 0).``;
- R17 a modal technology with an existing pair whose ``transport_resource`` has no positive factor in any mode:
  ``{tech}: transport_resource {resource} on link {link} has no positive transport_mode_factor in any mode.``;
- R15 a modal technology with a block length (``transport_time_block_hrs``): ``{tech}: transport_time_block_hrs = {L} h:
  a transport technology with modes cannot be a block technology.``;
- R13 a modal technology with an existing pair whose ``transport_delay`` is not 0 (``nan`` reads as 0):
  ``{tech}: transport_delay = {value} h on link {link}: a transport technology with modes takes no delay.``;
- R14 a modal technology with an existing pair with a finite ``transport_linepack_capacity`` in some year:
  ``{tech}: transport_linepack_capacity is finite on link {link}: a transport technology with modes takes no linepack.``

R13, R14 and R17 name the first offending link. A modal technology is read after R16, with ``nan`` as 0.

**W1, a warning.** One ``UserWarning`` lists every defined mode of a modal technology whose factors do not sum to 1
(tolerance :math:`10^{-9}`), in the order technology, mode: ``Transport modes whose factors do not sum to 1: {tech} {mode}:
{F:g}; {tech} {mode}: {F:g}. A flow of 1 MWh in such a mode carries that many MWh of resources.`` It also catches a resource
name that the CSV reader dropped because it is not in the coordinate ``resource``, but only when the remaining factors of
the mode no longer sum to 1. The CSV reader drops silently a resource, a mode or a technology label that is not listed in
its coordinate: a mistyped mode label can therefore leave a technology with fewer modes, or plain, and W1 sees it only
when the remaining factors of a defined mode no longer sum to 1.

**Not refused:** factors whose sum is not 1 (W1 only); a mode defined for some technologies and not for others; two modes
with the same factors; a modal technology with no existing pair (it builds no flow, and its net generation cells are
pinned to 0); a delay or a linepack on a pair that does not exist, on a non-modal technology or on a technology whose
factors are all 0; a block length on a non-modal technology of a dataset that has a modal one; a finite
``transport_mode_ramp_*`` on a non-modal technology (ignored) and a negative one (unchecked, as ``transport_ramp_up``);
modal data with ``transport`` off; a modal technology with ``transport_availability``, ``transport_ramp_*``,
``transport_efficiency``, fuel, hurdle cost or retrofit (they act on the total flow).

A linepack with modes
---------------------

A linepack on a modal technology is refused (R14): the stock of a blend cannot be written linearly on the link, since what
leaves the pipe would have the composition of the stock. To model a linepack, add a **buffer area** between the two areas
and link them with two modal technologies without linepack (departure area to buffer, buffer to arrival area), and put in
the buffer area a storage of each carried resource (the stock of each resource is then a storage level).

Outputs
-------

The variable ``operation_transport_mode_power_in`` is in the solution and written by ``save_solution`` (CSV with a
``transport_mode`` column, NetCDF): a user who wants the mode of each flow reads it by name. There is no new summary.
``get_net_generation`` lists a modal technology under each resource it carries. The plot of the costs by resource lists a
modal technology under its main resource only. The variable lives on the modal technologies only: when some transport
technology is not modal, ``model.solution`` has ``transport_tech`` outer-joined for every transport variable, with the
``UserWarning`` "Coordinates across variables not equal" (``save_solution`` silences it). When every transport technology
is modal there is no outer join and no warning.

The order that fixes the sense of ``transport_mode_ramp_up`` and ``transport_mode_ramp_down`` is the order of the
coordinate ``transport_mode`` in the input dataset, and ``save_solution`` keeps it (NetCDF and CSV rows). ``read_solution``
returns every coordinate it reads from CSV in alphabetical order, ``transport_mode`` included, with the values right by
label. The helper ``sort_coords_and_vars`` sorts ``transport_mode`` too: do not apply it to a dataset before building the
model when the two ramps differ.

Why we diverge from the official repository
-------------------------------------------

The official repository (pommes 0.11.0, "Transport capacity shared between resources") lets one transport technology carry
several resources: the flow gets a ``resource`` dimension, the resources are listed in the ``transport_resource`` string
separated by "-", and the capacity bounds the sum of the flows. We keep its idea (one capacity, free mix) as a special
case and diverge on seven points, each for a reason:

1. **Modes instead of a string.** A mode table (``transport_mode_factor``) replaces the split of a string. The official
   model has to forbid "-" in resource names and to guard a silent truncation of the string: a resource that contained "-"
   would split into two unknown names and the pipe would carry nothing, without a message. With a table there is no string
   to split, so there is no restriction on resource names.
2. **A composition per mode, so a cap on a share.** The official flow is free per resource: any mix, so no cap on the
   hydrogen share, and a blend with at most 10 % hydrogen cannot be written. Here the blend is a mode and the share cap is
   that of the richest mode, with no extra constraint. The free mix is the case of pure modes (below).
3. **A ramp on mode change.** The official model states that nothing prevents the mix from changing from hour to hour, and
   that a pipe to be purged is not represented. Here an optional ramp bounds the difference of two mode flows (default:
   none), as the combined module does.
4. **Free factors.** The official model counts 1 MW per MW for every resource. Here that holds when the factors of a mode
   sum to 1; when they sum to less, the pipe carries less energy than its flow: the loss is in the matrix. A capacity
   weight per mode is not needed. A mode that does not sum to 1 raises one warning (W1).
5. **Efficiency, fuel, hurdle, availability, ramp, delay, linepack stay on the total flow.** The official model has none of
   them; this model already has them and they keep their meaning for a modal technology. Delay, linepack and time blocks
   are refused with modes until a per-mode form exists.
6. **The flow variable.** The official model changed ``operation_transport_power`` for everybody (a ``resource``
   dimension). Here the total keeps its name and ``operation_transport_mode_power_in`` is added for modal technologies
   only, so that a dataset without modes keeps its LP byte for byte, and the sparse engine builds the same model.
7. **One resource per technology on its links.** A technology that carries different resources on different links is
   refused (the dense engine now, as the sparse engine since 0.4.1): the dense model would count the flow of every link at
   every resource. In the official model different links may list different strings.

The net generation stays a variable here and is an expression in the official model: a construction difference, not a
formulation one.

**The official model as pure modes.** With, for each carried resource :math:`r` of a technology, one pure mode
(:math:`f_{m_r, r} = 1`, 0 elsewhere), :math:`\eta = 1`, no availability and no mode ramp, the capacity constraint is
:math:`\sum_r X_{m_r} \leq \mathrm{cap}`, the net generation is the official expression, and the hurdle cost acts on the
sum of the flows: the feasible set of the official model (any mix, hour by hour) is the set of :math:`X \geq 0` with
:math:`\sum X \leq \mathrm{cap}`. The two models then give the same objective and the same flows.

**Mapping an official** ``"a-b"`` **string to pure modes** (there is no converter). For each technology whose string is
``"a-b"`` (the same on all its links), declare the modes ``pure_a`` and ``pure_b``, set ``transport_mode_factor[tech,
pure_a, a] = 1`` and ``[tech, pure_b, b] = 1``, set ``transport_resource`` to ``a`` on every link of the technology and
leave the ramps empty. If the strings of one technology differ by link (``"a-b"`` on one, ``"a"`` on another), make one
technology per distinct string, since a mode belongs to a technology and not to a link. A string without "-" is a plain
technology: do not turn it into a one-mode technology.

.. _section-time-blocks:

Time blocks (multi-resolution time steps)
=========================================

By default every operation variable has one value per time step. Five optional parameters let a conversion, a
combined or a transport technology, and a resource balance, run on **time blocks** of fixed length: one decision and
one balance row per block instead of one per step. Process and storage stay at the step resolution. Without the
parameters the model is exactly that of 0.7.1+kul (same variables, constraints, objective and LP file).

Parameters (all ``nan`` by default): ``conversion_time_block_hrs``, ``combined_time_block_hrs`` and
``transport_time_block_hrs`` (block length :math:`L` in whole hours of a technology), ``resource_time_block_hrs``
(forces the length of a resource balance) and ``conversion_time_block_rule`` (``"min"`` by default, or ``"mean"``: how
the availability becomes a block value). None of them has an ``area``, ``year_op``, ``operating_condition`` or ``hour``
coordinate: the block structure is shared by the areas, years and conditions, and all the links of a transport
technology share its length. A transport technology has no rule parameter: its availability is always aggregated by
the minimum. A valid length on a transport technology is ignored when the ``transport`` flag is off (its value
and its coordinates are checked all the same).

Blocks and anchoring
--------------------

With :math:`(\Delta T)_{t}` the length of step :math:`t`, the start of a step is
:math:`\tau_t = \sum_{t' < t} (\Delta T)_{t'}` and its block is
:math:`b_L(t) = \lfloor (\tau_t + 10^{-9}) / L \rfloor` (a tolerance of :math:`10^{-9}` h), so blocks are anchored
at the first step of the modelled horizon (the start of the year in a full-year run, the first step of a
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

1. hourly if it has a counted hourly contributor (storage, process, or a technology without a block length; a transport
   technology is hourly only without a block length; a modal transport technology, which cannot have a block length, is an
   hourly contributor of every resource it carries);
2. otherwise the smallest block length among its counted block contributors (every other one must be a multiple of it);
3. otherwise hourly.

``resource_time_block_hrs`` replaces the derived length when it is finite (a **forced** coarse balance); storage and
process technologies then stay hourly and their step energies are summed into the block row. A block transport
technology of length :math:`L_T` enters a balance of length :math:`L_r` (the same balance at both ends of a link, and
for its fuel the balance of the fuel resource) through the same conversion of resolution as any block technology:
identity if :math:`L_T = L_r`, spread by duration share if :math:`L_T > L_r`, summed over the blocks if
:math:`L_T < L_r`. A transport technology without a block length keeps its step flows even when conversion has blocks.
When some
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
net generations (energies per block) are ``..._block_{L}h_net_generation``. The transport variables of a group are
``operation_transport_block_{L}h_power`` (power withdrawn at the "to" end), ``..._power_in`` (power injected at the
"from" end) and ``..._net_generation``, on ``(link, transport_tech, time_block_{L}h)``. Every block row has the name of
the hourly row with ``operation_<module>_`` replaced by ``operation_<module>_block_{L}h_``. Capacities, planning,
annuity and retrofit rows are unchanged (one per unit, or per link and transport technology).

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
- **Transport.** One flow decision per link, technology and block, and one row per block for each row family of the
  hourly model, on the existing (link, technology) pairs. The block availability is the minimum of the step
  availabilities in the block (missing values read as 1 first; no other rule): power maximum
  :math:`P_b \leq \bar A_b \, \text{cap}`. Efficiency :math:`P^{in}_b \, \eta = P_b`. Net generation
  :math:`NG_b = D_L(b) \, (\sum P_b \text{ at the "to" end} - \sum P^{in}_b \text{ at the "from" end})`. The fuel
  drawn at the fuel area is :math:`D_L(b) \, \varphi \, P^{in}_b`, and the hurdle costs :math:`\sum_b D_L(b) \, P_b`.
  Ramps :math:`P_b - P_{b-1} \leq \rho \, \text{cap} \, \Delta^0_L(b)` (and the mirror), on blocks other than the first
  and where the ramp is finite, with the divisor of the combined module (the duration of the first step of the block).
  A block transport technology has no delay set, no linepack and no level variable (refused inputs R11 and R12). A
  ``nan`` ``transport_delay`` reads as 0. Its fuel emissions (``transport_fuel_emission_factor`` and
  ``transport_fuel_total_emission_factor``) are counted in the arrival area and, like every block emission, spread over
  the steps of each block by their duration share :math:`(\Delta T)_{t}/D_L(b)`: the emission rows stay hourly and
  equal the flat twin's step by step. A transport technology with modes cannot be a block technology (R15, see
  :ref:`section-transport-modes`).
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
- R10 every resource balance coarse (at least one resource must stay hourly);
- R11 a transport technology with a block length that has an existing (link, technology) pair with a ``transport_delay``
  other than 0 (``nan`` reads as 0): ``{tech}: transport_delay = {value} h on link {link}: a block transport technology
  ({L} h) takes no delay.`` Delays by whole blocks are left for a later release;
- R12 a transport technology with a block length that has an existing pair with a finite
  ``transport_linepack_capacity`` in some year: ``{tech}: transport_linepack_capacity is finite on link {link}: a block
  transport technology ({L} h) takes no linepack.`` Both messages name the first offending link;
- R15 a transport technology with modes (a modal technology, see :ref:`section-transport-modes`, which also lists R13 and
  R14) that has a block length: ``{tech}: transport_time_block_hrs = {L} h: a transport technology with modes cannot be a
  block technology.``

A delay or a linepack on a pair that does not exist (it carries no resource), a delay on an hourly transport technology
and a block length on a technology with no existing pair are not refused. A delay on a pair that does not exist is not
refused by R11, but the model still needs uniform time steps as soon as any ``transport_delay`` is positive, as before
(the check reads every cell). R4 names the transport technology
(``transport tech {name}``). R5, R9 and R10 also apply to balances that are coarse only because of block transport
technologies.

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
        each block, ramps, constant operation, annual sums, daily rule; for a transport technology: power maximum with
        rule ``min``, efficiency, fuel, hurdle costs, ramps; for a transport technology only with an efficiency
        above 0, since at 0 the block model is a restriction (the injected flow is held flat), and ramps of 0 or more)
    * - Must-run on an availability that varies inside a block
      - relaxation (the block rows use :math:`\bar A_b`; the block objective is at most the twin's)
    * - Availability rule ``mean``
      - relaxation
    * - Coarse resource balance (derived or forced)
      - relaxation of the hourly balance where net import prices and emission factors are constant inside each block (or
        net import is absent) and no flexible demand ramp is finite (with binding ramps the coarse model can be
        dearer): the block holds the energy of the block, not of each step; otherwise it has no sign against the hourly
        balance. Exact against the model run with steps of :math:`L` hours when every contributor has the balance's
        length, the horizon is a multiple of :math:`L`, there is no conversion, combined or transport ramp (mode-pair ramps
        included) and no daily production maximum (the block ramp divides by the duration of the block's first step and
        the daily row counts days), and, under rule ``mean``, the reference's availability is set to the
        duration-weighted mean per step (for transport, which has no rule, the reference's availability is the minimum
        per step); with net import, the reference's two prices and two emission factors are set
        to the duration-weighted mean per step too, and carbon is then equal on yearly totals only (the block model's
        emission rows stay hourly)
    * - Load shedding and spillage on a coarse resource
      - exact
    * - Flexible demand on a coarse balance
      - equal to the model run with steps of :math:`L` hours when the horizon is a multiple of :math:`L` (with a short
        last block the reference's ramp is the energy-difference row); against an hourly flexible demand it has no sign
        when ramps bind (the ramp allowance is :math:`\rho \, FD_b` of the current block)
    * - Net imports on a coarse balance
      - exact against hourly imports held flat inside each block; equal to summed hourly imports when prices (and
        emission factors if carbon is on) are constant inside each block, otherwise a restriction of it
    * - Delay or linepack of a block transport technology
      - refused (R11, R12)
    * - Carbon of block technologies
      - exact against the flat twin, step by step (the fuel emissions of a block transport technology are spread over the
        hours of each block, as its flow is, under the same condition of an efficiency above 0)

Outputs
-------

Block variables are written as built, with the names above. Because the hourly and block variables of a module cover
different technologies, ``model.solution`` joins them with ``nan`` outside their group (linopy raises a
``UserWarning``, expected for every block model). ``calculate_summaries`` sums the ``time_block_*`` dimensions like
``hour`` and combines the groups of a module by label under the module name (the group suffix ``_block_{L}h`` is
dropped). ``utils.expand_to_hours(da, tb, kind)`` maps a block array to the steps, ``kind="energy"`` multiplying by
:math:`(\Delta T)_{t}/D_L(b)` and ``kind="intensive"`` repeating the value; ``post_process.module_hourly`` combines the
hourly variable of a module (when it exists) and the expanded block variables, and ``get_net_generation`` and the plots
use it, including the block load shedding and spillage and the block transport technologies (``get_net_generation``
reads ``module_hourly(s, "transport", "net_generation", p)``; a user who sets a transport length reads the powers with
``module_hourly(s, "transport", "power", p)``, the hourly variable ``operation_transport_power`` holding only the
technologies without a length). The outer-join ``UserWarning`` of ``model.solution`` then also arises on
``transport_tech``. In a block model ``get_net_generation`` does not list a technology that has no existing pair. The
variable ``operation_transport_mode_power_in`` of the transport technologies with modes is hourly and written by
``save_solution``; ``get_net_generation`` lists a modal technology under each resource it carries (see
:ref:`section-transport-modes`). The dual of ``operation_transport_block_{L}h_power_max_constraint`` is one value per
link, technology, block and year: it is the sum of the flat twin's hourly duals over the
block (with one binding hour, the dual of that hour), in EUR per MW of the block's power, and it is not an hourly
price, so it is not expanded to the steps. ``module_hourly``, and so ``get_net_generation`` and
``plot_power`` on block models, take a dense solution: a solution of the sparse engine (pommes-kul) must be densified
first.

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
transport (the ``transport_delay`` shift of an hourly technology runs along ``hour`` inside each condition; a block
transport technology takes no delay), flexible demand windows and
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
