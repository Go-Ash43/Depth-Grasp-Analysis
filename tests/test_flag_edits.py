"""data/flag_edits.py and the dataset's flag_edit path (step 5 proposal rev 3 section 5)."""
import numpy as np
import pytest
import torch

import data.flag_edits as fe
from data.dataset import GraspDataset
from tests.test_dataset import DCFG, SPLITS, cache  # noqa: F401  (fixture)

EDIT = "centre_box48_flip"


def test_edit_returns_a_new_mask_and_leaves_the_input():
    f = np.zeros((224, 224), bool)
    f[0, 0] = True
    before = f.copy()
    g = fe.centre_box48_flip(f)
    assert g is not f and np.array_equal(f, before)
    assert int((g != f).sum()) == fe.EDIT_PIXELS[EDIT] == 2304
    assert g[88:136, 88:136].all() and not g[87, 88] and not g[136, 135]


def test_edit_flips_both_ways():
    f = np.ones((224, 224), np.uint8)
    g = fe.centre_box48_flip(f)
    assert g.dtype == bool and int((~g).sum()) == 2304


def _ds(out, model, **kw):
    return GraspDataset(out, "train", model, mode="eval", splits=SPLITS, data_cfg=DCFG, **kw)


@pytest.mark.parametrize("model", ["R", "A0", "M"])
def test_edit_executes_for_every_flag_capable_model(cache, model, monkeypatch):  # noqa: F811
    calls = []
    real = fe.FLAG_EDITS[EDIT]
    monkeypatch.setitem(fe.FLAG_EDITS, EDIT, lambda f: calls.append(1) or real(f))
    d = _ds(cache, model, holes="mid", flag_edit=EDIT)
    d[(0, int(d.slots[0]))]
    assert calls == [1]


@pytest.mark.parametrize("model", ["R", "A0"])
def test_flag_blind_x_unchanged_by_the_edit(cache, model):  # noqa: F811
    a, b = _ds(cache, model, holes="mid"), _ds(cache, model, holes="mid", flag_edit=EDIT)
    for s in a.slots:
        assert torch.equal(a[(0, int(s))]["x"], b[(0, int(s))]["x"])


def test_m_flag_channel_changes_by_exactly_the_box(cache):  # noqa: F811
    a, b = _ds(cache, "M", holes="mid"), _ds(cache, "M", holes="mid", flag_edit=EDIT)
    for s in a.slots:
        xa, xb = a[(0, int(s))]["x"], b[(0, int(s))]["x"]
        assert torch.equal(xa[:4], xb[:4])
        assert int((xa[4] != xb[4]).sum()) == 2304


def test_train_mode_astar_and_unknown_edits_rejected(cache):  # noqa: F811
    with pytest.raises(ValueError, match="flag_edit"):
        GraspDataset(cache, "train", "M", mode="train", seed=0, splits=SPLITS, data_cfg=DCFG, flag_edit=EDIT)
    with pytest.raises(ValueError, match="flag_edit"):
        _ds(cache, "Astar", flag_edit=EDIT)
    with pytest.raises(ValueError, match="flag_edit"):
        _ds(cache, "M", flag_edit="nope")
