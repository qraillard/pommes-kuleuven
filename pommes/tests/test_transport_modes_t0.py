"""T0 of the transport modes: without a modal technology the model is the 0.8.0+kul model, LP file byte for byte."""

import linopy
import numpy as np
import pytest

from pommes.model.build_model import build_model
from pommes.model.data_validation import ref_inputs
from pommes.tests.test_time_blocks_t0 import HOURLY_TRANSPORT_ORDER, _transport_fuel_dataset
from pommes.tests.time_blocks_helper import (
    lp_text,
    transport_every_row_dataset,
    transport_toy,
    with_blocks,
)
from pommes.utils import _contributors, time_blocks

NEW_NAMES = ["transport_mode_factor", "transport_mode_ramp_up", "transport_mode_ramp_down"]
MODES = ["blend", "methane"]
CASES = ["parameters", "every_row", "pairless", "transport_fuel"]


def _pairless_toy():
    """Ships ``Z, A, H`` (not alphabetical); ``A`` carries nothing, so it has no pair."""
    plain = transport_toy([1.0] * 12, techs=["Z", "A", "H"])
    for link in plain.link.values:
        resource = plain.transport_resource.copy()
        resource.loc[{"link": link, "transport_tech": "A"}] = "nan"
        plain = plain.assign(transport_resource=resource)
    return plain


def _dataset(case, request):
    if case == "parameters":
        return request.getfixturevalue("parameters")
    if case == "every_row":
        return transport_every_row_dataset()
    if case == "pairless":
        return _pairless_toy()
    return _transport_fuel_dataset(request.getfixturevalue("parameters"))


def _bare(p):
    return p.drop_vars(NEW_NAMES, errors="ignore")


def _with_coordinate(p, variant):
    """``p`` (without the new names) with the coordinate ``transport_mode`` and a factor that makes no technology modal."""
    q = _bare(p).assign_coords(transport_mode=MODES)
    shape = (q.sizes["transport_tech"], len(MODES), q.sizes["resource"])
    dims = ("transport_tech", "transport_mode", "resource")
    if variant == "zero":
        return q.assign(transport_mode_factor=(dims, np.zeros(shape)))
    if variant == "nan":
        return q.assign(transport_mode_factor=(dims, np.full(shape, np.nan)))
    if variant == "default":
        return q.assign(transport_mode_factor=np.array(0.0))
    return q  # absent


class _Spy:
    """The names given to ``add_variables`` and ``add_constraints``."""

    def __init__(self, monkeypatch):
        self.names = []
        for method in ("add_variables", "add_constraints"):
            original = getattr(linopy.Model, method)

            def spy(model, *args, _original=original, **kwargs):
                self.names.append(kwargs.get("name"))
                return _original(model, *args, **kwargs)

            monkeypatch.setattr(linopy.Model, method, spy)


@pytest.mark.parametrize("case", CASES)
def test_t0_new_parameters_at_default(case, request):
    """The three new parameters at their defaults, as check_inputs adds them, change nothing."""
    bare = _bare(_dataset(case, request))
    defaults = {n: np.array(ref_inputs[n]["default"], dtype=ref_inputs[n]["type"]) for n in NEW_NAMES}
    assert lp_text(build_model(bare.assign(defaults))) == lp_text(build_model(bare))


@pytest.mark.parametrize("variant", ["zero", "nan", "absent", "default"])
@pytest.mark.parametrize("case", CASES)
def test_t0_coordinate_without_modal_technology(case, variant, request, monkeypatch):
    """The coordinate alone, or with a factor that defines no mode, builds the model without it."""
    p = _dataset(case, request)
    expected = lp_text(build_model(_bare(p)))
    spy = _Spy(monkeypatch)
    m = build_model(_with_coordinate(p, variant))
    assert spy.names and not [n for n in spy.names if "transport_mode" in str(n)]
    assert not [n for n in [*m.variables, *m.constraints] if "transport_mode" in n]
    assert lp_text(m) == expected


def test_t0_statement_order_with_the_coordinate():
    p = _with_coordinate(transport_every_row_dataset(), "zero")
    m = build_model(p)
    assert [n for n in m.constraints if "transport" in n] == HOURLY_TRANSPORT_ORDER


def _contributor_table(p, tb):
    """``_contributors`` with the NaN lengths of the hourly contributors as None (NaN is not equal to itself)."""
    found = _contributors(p, tb.tech_length)
    return {r: [(kind, name, None if np.isnan(L) else L) for kind, name, L in rows] for r, rows in found.items()}


def test_t0_contributors_without_modal_technology():
    """The OR of the contributors needs a modal technology: a factor 0 gives the contributors of 0.8.0."""
    plain = with_blocks(_pairless_toy(), conversion={"mill": 4})
    with_zero = _with_coordinate(plain, "zero")
    tb, tb_zero = time_blocks(plain), time_blocks(with_zero)
    assert tb.active
    assert _contributor_table(with_zero, tb_zero) == _contributor_table(plain, tb)
    assert tb_zero.resource_length.equals(tb.resource_length)
    assert all(tb_zero.tech_length[k].equals(tb.tech_length[k]) for k in tb.tech_length)


def test_modal_names_order():
    """The transport names: the hourly ones, the modal ones with the ramps, then the block ones (section 8.1)."""
    import xarray as xr

    from pommes.tests.time_blocks_helper import transport_every_row_dataset, with_blocks, with_modes

    p = transport_every_row_dataset()
    ship = p.sel(transport_tech=["power_line"]).assign_coords(transport_tech=["ship"])
    q = xr.concat([p, ship], dim="transport_tech", data_vars="minimal", coords="minimal", compat="override")
    for name, fill in (("transport_delay", 0.0), ("transport_linepack_capacity", np.nan), ("transport_ramp_up", np.nan), ("transport_ramp_down", np.nan)):
        q[name] = q[name].where(q.transport_tech != "ship", fill)
    # power_line has no delay and no linepack: it is the modal technology; the ship is on 2 h blocks
    modes = {"power_line": {"a": {"electricity": 1.0}, "b": {"electricity": 1.0}}}
    q = with_modes(q, modes, ramp={"power_line": (0.5, 0.5)})
    names = [n for n in build_model(with_blocks(q, transport={"ship": 2})).constraints if "transport" in n]
    modal = [
        "operation_transport_mode_power_in_def",
        "operation_transport_mode_net_generation_def",
        "operation_transport_mode_ramp_up_constraint_0_1",
        "operation_transport_mode_ramp_down_constraint_0_1",
    ]
    blocks = [n for n in names if n.startswith("operation_transport_block_")]
    assert blocks
    assert names == HOURLY_TRANSPORT_ORDER + modal + blocks
