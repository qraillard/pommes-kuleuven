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
