"""eval/val_epoch.py: miss categories (forced predicted cell), run_val mode / buffer / coverage guards."""
import numpy as np
import pytest
import torch

from data.labels import load_labels_cfg, targets_from_cfg
from eval.val_epoch import MISS_CATEGORIES, classify_sample, rows_to_csv_bytes, run_val

CFG = load_labels_cfg()
# one grasp at the image centre (1024 frame): x, y, theta, opening, jaw; jaw = opening / 2 (decoder rule)
GT = np.array([[512.0, 512.0, 0.0, 20.0, 10.0]])
TG = targets_from_cfg(GT, CFG)
CENTRE = (112, 112)                                   # (i, j): the target's own Q maximum


def _tmaps():
    return {k: TG[k].astype(np.float32) if k != "void" else TG["void"] for k in ("q", "cos2", "sin2", "width", "void")}


def _pred(q_cell=None, theta_deg=None, width=None):
    t = _tmaps()
    p = {k: t[k].copy() for k in ("q", "cos2", "sin2", "width")}
    if q_cell is not None:
        p["q"] = np.zeros_like(p["q"])
        p["q"][q_cell] = 1.0
    if theta_deg is not None:
        p["cos2"][:] = np.cos(np.deg2rad(2 * theta_deg))
        p["sin2"][:] = np.sin(np.deg2rad(2 * theta_deg))
    if width is not None:
        p["width"][:] = width
    return p


def test_target_argmax_is_the_centre_cell_and_exact_targets_match():
    assert np.unravel_index(np.argmax(TG["q"]), TG["q"].shape) == CENTRE
    r = classify_sample(_pred(), _tmaps(), GT, CFG)
    assert r["match_primary"] and r["match_strict"]
    assert r["cat_primary"] == r["cat_strict"] == ""


def test_invalid_decode():
    r = classify_sample(_pred(width=-0.1), _tmaps(), GT, CFG)
    assert not r["valid"]
    assert r["cat_primary"] == r["cat_strict"] == "invalid_decode"


def test_void_cell():
    assert TG["void"][10, 10] == 1
    # valid angle / width everywhere (target maps are 0 off-window, which would decode as invalid)
    r = classify_sample(_pred(q_cell=(10, 10), theta_deg=0.0, width=0.2), _tmaps(), GT, CFG)
    assert r["valid"] and r["target_void"]
    assert r["cat_primary"] == r["cat_strict"] == "void_cell"


def test_off_window_cell_with_target_maps_is_invalid_not_void():
    r = classify_sample(_pred(q_cell=(10, 10)), _tmaps(), GT, CFG)   # cos = sin = width = 0 there
    assert not r["valid"] and r["cat_primary"] == "invalid_decode"   # invalid is checked before void


def test_regression_miss():
    r = classify_sample(_pred(q_cell=CENTRE, theta_deg=60.0), _tmaps(), GT, CFG)
    assert not r["match_primary"] and r["cf_match_primary"]
    assert r["cat_primary"] == r["cat_strict"] == "regression_miss"


def test_target_at_cell_miss_uses_the_predicted_cell_not_the_target_argmax():
    cell = (112, 116)                                 # 4.4 px right of the centre at 224: supervised
    assert TG["void"][cell] == 0
    r = classify_sample(_pred(q_cell=cell), _tmaps(), GT, CFG)
    assert (r["cell_i"], r["cell_j"]) == cell
    assert not r["match_primary"] and r["cf_match_primary"] is False
    # decoding the full target maps would pick CENTRE and match, turning this into a regression_miss
    assert r["cat_primary"] == "target_at_cell_miss"


class Stub(torch.nn.Module):
    """Returns the exact targets of each sample (looked up by the batch order), has a BatchNorm."""

    def __init__(self, maps, poison=False):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.maps, self.poison, self.calls = maps, poison, 0

    def forward(self, x):
        if self.poison:
            self.bn.running_mean += 1.0
        self.bn(x[:, :1])
        out = self.maps[self.calls]
        self.calls += 1
        return {k: torch.from_numpy(v) for k, v in out.items()}


def _loader(slots, invalid_last=False):
    batches, preds = [], []
    for s in slots:
        t = _tmaps()
        batches.append({"x": torch.zeros(1, 1, 224, 224), "slot": torch.tensor([s]), "severity": ["clean"],
                        "hole_frac": torch.tensor([0.0]), "covered_lie_frac": torch.tensor([0.0]),
                        "gt_grasps": [GT],
                        **{k: torch.from_numpy(np.asarray(t[k]))[None, None] for k in ("q", "cos2", "sin2", "width")},
                        "void": torch.from_numpy(TG["void"].astype(bool))[None, None]})
        p = _pred(width=-1.0) if (invalid_last and s == slots[-1]) else _pred()
        preds.append({k: v[None, None] for k, v in p.items()})
    return batches, preds


def test_run_val_counts_invalid_in_denominator_and_restores_train_mode():
    batches, preds = _loader([3, 5, 7], invalid_last=True)
    m = Stub(preds)
    m.train()
    rows, s = run_val(m, batches, [3, 5, 7], CFG, "cpu", "none")
    assert m.training
    assert s["n"] == 3 and s["invalid_decodes"] == 1
    assert s["top1_primary"] == pytest.approx(2 / 3)
    assert set(s["misses_primary"]) == set(MISS_CATEGORIES) and s["misses_primary"]["invalid_decode"] == 1
    assert rows_to_csv_bytes(rows).count(b"\n") == 4


def test_run_val_buffer_change_raises():
    batches, preds = _loader([1])
    with pytest.raises(AssertionError, match="buffer"):
        run_val(Stub(preds, poison=True), batches, [1], CFG, "cpu", "none")


@pytest.mark.parametrize("expected", [[1, 2, 9], [1]])
def test_run_val_slot_coverage(expected):
    batches, preds = _loader([1, 2])
    with pytest.raises(AssertionError, match="slots differ"):
        run_val(Stub(preds), batches, expected, CFG, "cpu", "none")


def test_run_val_duplicate_slot():
    batches, preds = _loader([1, 1])
    with pytest.raises(AssertionError, match="more than once"):
        run_val(Stub(preds), batches, [1], CFG, "cpu", "none")


def test_decoder_gets_2d_maps_at_batch_size_one():
    with pytest.raises(ValueError, match="2D"):
        classify_sample({k: v[None] for k, v in _pred().items()}, _tmaps(), GT, CFG)
