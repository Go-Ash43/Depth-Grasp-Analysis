"""eval/invariants.py protocol_a_check on the tiny synthetic cache."""
import numpy as np
import pytest
import torch

from data.dataset import GraspDataset
from eval.invariants import protocol_a_check, seeded_slots
from models.grconv_lite import build_model
from tests.test_dataset import DCFG, SPLITS, cache  # noqa: F401  (fixture)


class Probe(torch.nn.Module):
    """Deterministic, any channel count; poison=True changes a buffer in forward."""

    def __init__(self, poison=False):
        super().__init__()
        self.register_buffer("b", torch.zeros(1))
        self.poison = poison

    def forward(self, x):
        if self.poison:
            self.b += 1
        s = x.sum(1, keepdim=True)
        return {"q": s, "cos2": 0.5 * s, "sin2": 0.25 * s, "width": 2.0 * s}


def _slots(out):
    return [int(s) for s in GraspDataset(out, "train", "M", mode="eval", splits=SPLITS, data_cfg=DCFG).slots]


def _check(out, models, **kw):
    return protocol_a_check(models, out, "train", _slots(out), splits=SPLITS, data_cfg=DCFG, batch_size=2, **kw)


def test_real_models_pass_and_edit_is_counted(cache):  # noqa: F811
    models = {n: build_model(n, 0) for n in ("R", "A0", "M")}
    for m in models.values():
        m.train()
    rep = _check(cache, models)
    assert rep["verdict"] == "PASS", rep["failures"]
    n = len(_slots(cache))
    for cond in ("none", "mid"):
        m = rep["conditions"][cond]["M"]
        assert m["flag_changed_px"]["edit"] == 2304 * n and m["scenes_flag_changed"]["edit"] == n
        assert isinstance(m["oracle_vacuous"], bool)
        for name in ("R", "A0"):
            r = rep["conditions"][cond][name]
            assert all(r["x_identical"].values()) and all(r["outputs_identical"].values())
    assert all(m.training for m in models.values())       # modes restored


def test_m_ignoring_the_flag_is_not_a_failure(cache):  # noqa: F811
    m = build_model("M", 0)
    with torch.no_grad():
        m.conv1.weight[:, 4] = 0.0                         # M blind to its flag channel
    rep = _check(cache, {"M": m})
    assert rep["verdict"] == "PASS"
    assert rep["conditions"]["mid"]["M"]["top1_cell_changes"]["edit"] == 0


def test_flag_leak_into_flag_blind_input_fails(cache, monkeypatch):  # noqa: F811
    real = GraspDataset._x

    def leaky(self, planes):
        if self.model == "A0":
            planes = dict(planes, depth=planes["depth"] + 1000.0 * planes["flag"])
        return real(self, planes)

    monkeypatch.setattr(GraspDataset, "_x", leaky)
    rep = _check(cache, {"A0": Probe()}, conditions=("none",))
    assert rep["verdict"] == "FAIL"
    assert any("flag-blind input x changed under the edit" in f for f in rep["failures"])


def test_buffer_change_fails(cache):  # noqa: F811
    rep = _check(cache, {"R": Probe(poison=True)}, conditions=("none",))
    assert rep["verdict"] == "FAIL" and any("buffer" in f for f in rep["failures"])


class NanProbe(Probe):
    def forward(self, x):
        out = super().forward(x)
        out["width"] = out["width"] * float("nan")
        return out


def test_nan_outputs_fail_even_for_m(cache):  # noqa: F811
    rep = _check(cache, {"M": NanProbe()}, conditions=("none",))
    assert rep["verdict"] == "FAIL" and any("non-finite" in f for f in rep["failures"])


@pytest.mark.parametrize("kw", [{"slots": []}, {"conditions": ()}, {"batch_size": 0}, {"batch_size": -1}])
def test_nothing_to_evaluate_is_refused(cache, kw):  # noqa: F811
    args = {"slots": _slots(cache), "conditions": ("none",), "batch_size": 2, **kw}
    with pytest.raises(ValueError, match="need slots"):
        protocol_a_check({"A0": Probe()}, cache, "train", args["slots"], conditions=args["conditions"],
                         batch_size=args["batch_size"], splits=SPLITS, data_cfg=DCFG)


def test_every_variant_counts_every_slot(cache):  # noqa: F811
    rep = _check(cache, {"A0": Probe(), "M": Probe()}, conditions=("none",))
    n = len(_slots(cache))
    for name in ("A0", "M"):
        assert rep["conditions"]["none"][name]["n_evaluated"] == {"base": n, "oracle": n, "edit": n}


def test_astar_refused(cache):  # noqa: F811
    with pytest.raises(ValueError):
        _check(cache, {"Astar": Probe()})


def test_seeded_slots():
    s = np.arange(100)
    a, b = seeded_slots(s, 20, 5), seeded_slots(s, 20, 5)
    assert a == b == sorted(a) and len(set(a)) == 20
    assert seeded_slots(s, 500, 5) == list(range(100))
