"""data/dataset.py on a tiny synthetic cache: whitelist / leak guard, Protocol A, pairing, random
streams, epoch propagation, eval corruption seed, flips, frozen normalisation (docs/proj_v3.md)."""
import copy
import multiprocessing
import pickle

import numpy as np
import pytest
import torch

from data.cache_layout import open_cache
from data.dataset import (BATCH_KEYS, MODELS, EpochSampler, GraspDataset, collate, load_data_cfg,
                          load_frozen)
from data.mask_ops import complete_protocol_a
from tests.test_build_cache import EXCL_SCENES, _build, _splits_for

SPLITS = _splits_for(EXCL_SCENES)
FROZEN = load_frozen()
DCFG = load_data_cfg()
DCFG_V = copy.deepcopy(DCFG)                     # a hypothetical flip config, for the flip machinery only
DCFG_V["augment"]["train"] = "flip"
DCFG_V["augment"]["flip_axis"] = "vertical"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    out, _ = _build(tmp_path, monkeypatch, "ds", scenes=EXCL_SCENES)
    return out


def ds(out, model="M", mode="train", seed=0, **kw):
    kw.setdefault("data_cfg", DCFG)
    return GraspDataset(out, "train", model, mode=mode, seed=seed, splits=SPLITS, **kw)


def _arrays(out):
    return open_cache(out, "train", splits=SPLITS)


def _holed_item(d, slots, epochs=range(30)):
    """First (epoch, slot) whose sample has synthetic holes."""
    for e in epochs:
        for s in slots:
            it = d[(e, int(s))]
            if it["m_synth"].any():
                return e, int(s), it
    raise AssertionError("no holed sample found")


# ------------------------------------------------------------------ whitelist and leak guard
@pytest.mark.parametrize("model, n", [("R", 3), ("A0", 4), ("M", 5), ("Astar", 4)])
def test_channels_and_keys(cache, model, n):
    d = ds(cache, model, holes="none" if model == "Astar" else "train")
    for s in d.slots:
        it = d[(0, int(s))]
        assert set(it) == BATCH_KEYS                              # no dstar / lie / obj keys, ever
        assert it["x"].shape == (n, 224, 224) and it["x"].dtype == torch.float32
    assert MODELS[model][0] == "rgb"


def test_flag_is_exactly_the_true_invalid_set_unless_oracle(cache):
    c = _arrays(cache)
    A = c["arrays"]
    dep = ds(cache, "M", holes="train")
    orc = ds(cache, "M", holes="train", flag_setting="oracle")
    e, s, it = _holed_item(dep, dep.slots)
    ms = A["m_sensor"][s].astype(bool)
    synth = it["m_synth"].numpy().astype(bool)
    np.testing.assert_array_equal(it["x"][4].numpy().astype(bool), ms | synth)       # no lie cell
    lie = A["lie_any_t12"][s].astype(bool)
    np.testing.assert_array_equal(orc[(e, s)]["x"][4].numpy().astype(bool), ms | synth | lie)


def test_astar_sees_dstar_and_no_holes_or_flag(cache):
    c = _arrays(cache)
    d = ds(cache, "Astar")
    s = int(d.slots[0])
    m, sd = FROZEN["normalisation"]["dstar_mm"]["median"], FROZEN["normalisation"]["dstar_mm"]["robust_sigma"]
    want = ((c["arrays"]["dstar"][s].astype(np.float64) - m) / sd).astype(np.float32)
    np.testing.assert_array_equal(d[(0, s)]["x"][3].numpy(), want)
    with pytest.raises(ValueError, match="Astar"):
        ds(cache, "Astar", holes="train")
    with pytest.raises(ValueError, match="Astar"):
        ds(cache, "Astar", flag_setting="oracle")


def test_a0_depth_is_not_dstar(cache):
    d = ds(cache, "A0", holes="none")
    a = ds(cache, "Astar")
    s = int(d.slots[0])
    assert not torch.equal(d[(0, s)]["x"][3], a[(0, s)]["x"][3])


# ------------------------------------------------------------------ Protocol A
def test_cached_ds_comp_equals_protocol_a_bit_for_bit(cache):
    c = _arrays(cache)
    A = c["arrays"]
    for s, r in enumerate(c["index"]):
        if r["status"] == "ok":
            got = complete_protocol_a(A["ds"][s], A["m_sensor"][s], FROZEN["telea_radius"])
            np.testing.assert_array_equal(got, A["ds_comp"][s])


def test_garbage_under_m_synth_leaves_x_bit_identical(cache):
    d = ds(cache, "M", mode="eval", seed=None, holes="high")
    s = None
    for slot in d.slots:
        if d[(0, int(slot))]["m_synth"].any():
            s = int(slot)
            break
    assert s is not None
    before = d[(0, s)]
    synth = before["m_synth"].numpy().astype(bool)
    p = cache / "train" / "ds.npy"
    a = np.load(p)
    a[s][synth] = 99999.0                                          # the true depth under the hole
    np.save(p, a)
    after = ds(cache, "M", mode="eval", seed=None, holes="high")[(0, s)]
    assert torch.equal(before["x"], after["x"])


def test_flag_setting_never_changes_depth_or_holes(cache):
    dep = ds(cache, "M", holes="train", seed=4)
    orc = ds(cache, "M", holes="train", seed=4, flag_setting="oracle")
    for e in range(4):
        for s in dep.slots:
            a, b = dep[(e, int(s))], orc[(e, int(s))]
            assert torch.equal(a["x"][:4], b["x"][:4]) and torch.equal(a["m_synth"], b["m_synth"])


# ------------------------------------------------------------------ holes ignore M_lie (holes decision, notes/holes_decision_v1.md)
def test_holes_do_not_depend_on_lies(cache):
    """Condition 1: with fixed random draws, replacing every lie mask by zeros leaves M_synth identical."""
    rng = np.random.default_rng(0)                    # dense known lies first: the synthetic scenes have
    for p in (cache / "train").glob("lie_*.npy"):     # none under their holes at tau 12, which would
        a = np.load(p)                                # make this test vacuous
        np.save(p, (rng.random(a.shape) < 0.3).astype(a.dtype))
    d = ds(cache, "M", mode="eval", seed=None, holes="mid")
    before = {int(s): d[(0, int(s))] for s in d.slots}
    assert any(it["covered_lie_frac"] > 0 for it in before.values())   # not vacuous: lies under holes
    for p in (cache / "train").glob("lie_*.npy"):
        np.save(p, np.zeros_like(np.load(p)))
    d2 = ds(cache, "M", mode="eval", seed=None, holes="mid")
    for s, it in before.items():
        after = d2[(0, s)]
        assert torch.equal(it["m_synth"], after["m_synth"])
        assert after["covered_lie_frac"] == 0.0


def test_covered_lie_frac_is_lie_any_t12_inside_the_hole(cache):
    A = _arrays(cache)["arrays"]
    d = ds(cache, "M", mode="eval", seed=None, holes="high")
    for s in d.slots:
        it = d[(0, int(s))]
        h = it["m_synth"].numpy().astype(bool)
        want = float((h & A["lie_any_t12"][int(s)].astype(bool)).sum() / h.sum()) if h.any() else 0.0
        assert it["covered_lie_frac"] == want


# ------------------------------------------------------------------ pairing and streams
def test_r_a0_m_are_paired_within_a_seed(cache):
    dd = {m: ds(cache, m, holes="train", seed=7) for m in ("R", "A0", "M")}
    for e in range(4):
        for s in dd["M"].slots:
            its = {m: d[(e, int(s))] for m, d in dd.items()}
            assert torch.equal(its["R"]["m_synth"], its["M"]["m_synth"])
            assert torch.equal(its["A0"]["m_synth"], its["M"]["m_synth"])
            assert torch.equal(its["A0"]["x"], its["M"]["x"][:4])
            assert torch.equal(its["R"]["x"], its["M"]["x"][:3])


def test_item_is_independent_of_order_and_of_pickling(cache):
    d = ds(cache, "M", holes="train", seed=2)
    keys = [(e, int(s)) for e in range(3) for s in d.slots]
    first = {k: d[k] for k in keys}
    d2 = pickle.loads(pickle.dumps(d))                              # what a worker receives
    assert d2._cache is None
    for k in reversed(keys):
        assert torch.equal(first[k]["x"], d2[k]["x"]) and torch.equal(first[k]["m_synth"], d2[k]["m_synth"])


def test_epochs_change_holes(cache):
    d = ds(cache, "M", holes="mid", seed=1, mode="train")
    s = int(d.slots[0])
    assert any(not torch.equal(d[(0, s)]["m_synth"], d[(e, s)]["m_synth"]) for e in range(1, 6))


def test_run_seed_changes_training_holes_but_not_eval_holes(cache):
    s = None
    tr = [ds(cache, "M", holes="mid", seed=k) for k in (1, 2)]
    for slot in tr[0].slots:
        if not torch.equal(tr[0][(0, int(slot))]["m_synth"], tr[1][(0, int(slot))]["m_synth"]):
            s = int(slot)
    assert s is not None
    ev = [ds(cache, "M", mode="eval", seed=k, holes="mid") for k in (1, 2)]
    for slot in ev[0].slots:
        assert torch.equal(ev[0][(0, int(slot))]["m_synth"], ev[1][(0, int(slot))]["m_synth"])
    other = copy.deepcopy(DCFG)
    other["corruption_seed"] = DCFG["corruption_seed"] + 1
    ev2 = ds(cache, "M", mode="eval", holes="mid", data_cfg=other)
    assert any(not torch.equal(ev[0][(0, int(x))]["m_synth"], ev2[(0, int(x))]["m_synth"]) for x in ev2.slots)


def test_split_guards(cache):
    with pytest.raises(ValueError, match="train split only"):
        GraspDataset(cache, "val", "M", mode="train", seed=0, splits=SPLITS, data_cfg=DCFG)
    with pytest.raises(ValueError, match="allow_sealed"):
        GraspDataset(cache, "test_sealed", "M", mode="eval", splits=SPLITS, data_cfg=DCFG)


def test_bare_index_and_eval_epoch_are_refused(cache):
    d = ds(cache, "M", holes="train")
    with pytest.raises(TypeError, match="epoch"):
        d[0]
    e = ds(cache, "M", mode="eval", holes="mid")
    with pytest.raises(ValueError, match="epoch 0"):
        e[(1, int(e.slots[0]))]
    with pytest.raises(ValueError):
        ds(cache, "M", mode="eval", holes="train")


def test_sampler_carries_the_epoch():
    smp = EpochSampler([3, 5, 9], shuffle=True, seed=0)
    smp.set_epoch(4)
    pairs = list(smp)
    assert {p[0] for p in pairs} == {4} and sorted(p[1] for p in pairs) == [3, 5, 9]
    smp.set_epoch(5)
    assert {p[0] for p in smp} == {5}


def test_split_is_part_of_the_stream_key():
    from data.dataset import stream_rng
    a = stream_rng(1, 0, 0, 5, "val").random(4)
    b = stream_rng(1, 0, 0, 5, "test_module").random(4)
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a, stream_rng(1, 0, 0, 5, "val").random(4))
    with pytest.raises(ValueError):
        stream_rng(1, 0, 0, 5, "nope")


def test_strict_sampler_refuses_a_repeated_epoch():
    smp = EpochSampler([3, 5], shuffle=True, seed=0, strict=True)
    iter(smp), iter(smp)                                        # creating iterators is not reading
    list(smp)
    with pytest.raises(RuntimeError, match="set_epoch"):
        list(smp)                                               # forgot set_epoch
    smp.set_epoch(1)
    assert {p[0] for p in smp} == {1}


@pytest.mark.skipif(multiprocessing.get_start_method() != "fork", reason="forked workers read the tmp cache")
def test_epoch_reaches_persistent_workers(cache):
    d = ds(cache, "M", holes="train", seed=0)
    smp = d.sampler(shuffle=True)
    dl = torch.utils.data.DataLoader(d, batch_size=2, sampler=smp, num_workers=2, persistent_workers=True,
                                     collate_fn=collate)
    seen = {}
    for e in (0, 1, 2):
        smp.set_epoch(e)
        seen[e] = {int(x) for b in dl for x in b["epoch"]}
    assert seen == {0: {0}, 1: {1}, 2: {2}}


# ------------------------------------------------------------------ augmentation
def test_augment_needs_a_frozen_axis(cache):
    with pytest.raises(ValueError, match="augment.train"):
        ds(cache, "M", holes="train", augment=True)                # the frozen config says "none"
    axis_only = copy.deepcopy(DCFG)
    axis_only["augment"]["flip_axis"] = "vertical"                 # an axis alone must not switch it on
    with pytest.raises(ValueError, match="augment.train"):
        ds(cache, "M", holes="train", augment=True, data_cfg=axis_only)
    flip_no_axis = copy.deepcopy(DCFG)
    flip_no_axis["augment"]["train"] = "flip"
    with pytest.raises(ValueError, match="flip_axis"):
        ds(cache, "M", holes="train", augment=True, data_cfg=flip_no_axis)
    with pytest.raises(ValueError):
        ds(cache, "M", mode="eval", holes="mid", augment=True, data_cfg=DCFG_V)


def test_flipped_sample_flips_inputs_targets_and_gt(cache):
    d = ds(cache, "M", holes="train", seed=3, augment=True, data_cfg=DCFG_V)
    e, s, _ = _holed_item(d, d.slots)
    a, b = d.sample(e, s, force_flip=False), d.sample(e, s, force_flip=True)
    assert torch.equal(b["x"], torch.flip(a["x"], [1]))
    assert torch.equal(b["m_synth"], torch.flip(a["m_synth"], [0]))
    for k in ("q", "cos2", "width", "void"):
        assert torch.equal(b[k], torch.flip(a[k], [1])), k
    assert torch.equal(b["sin2"], -torch.flip(a["sin2"], [1]) + 0.0)
    ga, gb = a["gt_grasps"], b["gt_grasps"]
    np.testing.assert_array_equal(gb[:, 1], 1023 - ga[:, 1])
    from data.labels import canonical_theta
    np.testing.assert_array_equal(gb[:, 2], canonical_theta(-ga[:, 2]) + 0.0)
    assert b["flip"] == 1 and a["flip"] == 0


def test_augment_flip_comes_from_its_own_stream(cache):
    d = ds(cache, "M", holes="train", seed=3, augment=True, data_cfg=DCFG_V)
    n = ds(cache, "M", holes="train", seed=3, augment=False)
    flips = []
    for e in range(8):
        for s in d.slots:
            a, b = d[(e, int(s))], n[(e, int(s))]
            flips.append(a["flip"])
            want = torch.flip(b["m_synth"], [0]) if a["flip"] else b["m_synth"]
            assert torch.equal(a["m_synth"], want)                  # same holes whether or not flipped
    assert 0 < sum(flips) < len(flips)


# ------------------------------------------------------------------ normalisation and collate
def test_frozen_normalisation_is_used(cache):
    c = _arrays(cache)
    d = ds(cache, "A0", holes="none")
    s = int(d.slots[0])
    x = d[(0, s)]["x"].numpy()
    m, sd = FROZEN["normalisation"]["ds_comp_mm"]["median"], FROZEN["normalisation"]["ds_comp_mm"]["robust_sigma"]
    np.testing.assert_array_equal(x[3], ((c["arrays"]["ds_comp"][s].astype(np.float64) - m) / sd).astype(np.float32))
    rgb = c["arrays"]["rgb"][s].astype(np.float32)
    np.testing.assert_array_equal(x[:3], ((rgb / 255.0 - 0.5) / 0.25).transpose(2, 0, 1))


def test_collate(cache):
    d = ds(cache, "M", holes="train")
    b = collate([d[(0, int(s))] for s in d.slots])
    assert b["x"].shape == (len(d.slots), 5, 224, 224) and b["void"].dtype == torch.bool
    assert isinstance(b["gt_grasps"], list) and len(b["severity"]) == len(d.slots)
