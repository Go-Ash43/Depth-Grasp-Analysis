"""training/runcfg.py guards: config validation, W_NORM rule, clean tree, cache re-hash, precision."""
import copy
import json

import numpy as np
import pytest
import torch

import training.runcfg as rc
from data.labels import load_labels_cfg
from tests.test_dataset import SPLITS, cache  # noqa: F401  (fixture)

CFG, _ = rc.load_train_cfg()


@pytest.fixture
def restore_precision():
    prev = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.fp32_precision, torch.backends.cudnn.conv.fp32_precision)
    yield
    torch.use_deterministic_algorithms(prev[0])
    torch.backends.cudnn.benchmark = prev[1]
    torch.backends.cuda.matmul.fp32_precision = prev[2]
    torch.backends.cudnn.conv.fp32_precision = prev[3]


def test_real_config_validates():
    rc.validate_train_cfg(copy.deepcopy(CFG))


def test_draft_config_with_null_epochs_validates():
    rc.validate_train_cfg(dict(copy.deepcopy(CFG), status="draft", epochs=None))


def test_require_versions():
    good = {"versions": {"ok": True, "python": "3.11.9", "python_ok": True, "mismatches": {}}}
    rc.require_versions(good)
    bad = {"versions": {"ok": False, "python": "3.11.9", "python_ok": True,
                        "mismatches": {"numpy": {"locked": "2.4.6", "installed": "2.3.0"}}}}
    with pytest.raises(SystemExit, match="numpy"):
        rc.require_versions(bad)


@pytest.mark.parametrize("path,value", [
    (("precision", "fp32_precision"), {"cuda.matmul": "tf32", "cudnn.conv": "ieee"}),
    (("precision", "amp"), True),
    (("selection",), "best_val"),
    (("effective_holes",), {"R": "train", "A0": "train", "M": "train", "Astar": "train"}),
    (("drop_last_train",), True),
    (("model", "init"), "kaiming_normal"),
    (("loss", "aw_reduction"), "mean of per-image means"),
    (("loss", "q_reduction"), "per-image mean"),
    (("schedule", "step"), "one scheduler step per epoch"),
    (("schedule", "horizon"), "epochs"),
    (("overfit", "schedule"), "cosine"),
    (("cache_verify",), "report only"),
])
def test_config_policies_must_match_the_code(path, value):
    c = copy.deepcopy(CFG)
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    with pytest.raises(ValueError):
        rc.validate_train_cfg(c)


def test_frozen_needs_epochs():
    c = dict(copy.deepcopy(CFG), status="frozen", epochs=None)   # independent of the live config
    with pytest.raises(ValueError):
        rc.validate_train_cfg(c)
    c["epochs"] = 20
    rc.validate_train_cfg(c)


def test_effective_holes_and_seeds():
    c = dict(copy.deepcopy(CFG), seeds=[1, 2, 3, 4, 5], astar_seeds=[1])
    assert [rc.effective_holes(c, m) for m in ("R", "A0", "M", "Astar")] == ["train"] * 3 + ["none"]
    assert rc.allowed_seeds(c, "Astar") == [1] and rc.allowed_seeds(c, "M") == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("mx,want", [(93.1875, 100.0), (90.0, 90.0), (90.0001, 100.0), (0.3, 10.0)])
def test_w_norm_rule(mx, want):
    assert rc.w_norm_rule(mx) == want


def test_check_w_norm_on_cache(cache):  # noqa: F811
    lab = load_labels_cfg()
    g = np.load(cache / "train" / "grasps_1024.npy")
    want = rc.w_norm_rule(float((g[:, 3] * 224 / 1024).max()))
    ok = copy.deepcopy(lab)
    ok["targets"]["w_norm_px_224"] = want
    rec = rc.check_w_norm(cache, ok, splits=SPLITS)
    assert rec["rule_value"] == want
    bad = copy.deepcopy(lab)
    bad["targets"]["w_norm_px_224"] = want + 10.0
    with pytest.raises(SystemExit, match="stale"):
        rc.check_w_norm(cache, bad, splits=SPLITS)


def test_clean_tree(monkeypatch):
    monkeypatch.setattr(rc, "git_state", lambda: ("abc", True))
    with pytest.raises(SystemExit, match="dirty"):
        rc.require_clean_tree(False)
    assert rc.require_clean_tree(True) == {"git_commit": "abc", "git_dirty": True}
    monkeypatch.setattr(rc, "git_state", lambda: (None, None))
    with pytest.raises(SystemExit):
        rc.require_clean_tree(True)


def write_manifest(cache_dir, path, splits=("train",)):
    from scripts import cache_manifest as cm
    rels = cm.list_files(cache_dir, list(splits))
    got, total = cm.hash_all(cache_dir, rels, 2)
    path.write_text(json.dumps({"version": 1, "splits": list(splits), "total_bytes": total,
                                "files": [{"path": r, "bytes": got[r][0], "sha256": got[r][1]} for r in rels]}))
    return path


def test_verify_cache_full_rehash_detects_payload_change(cache, tmp_path):  # noqa: F811
    man = write_manifest(cache, tmp_path / "m.json")
    rec = rc.verify_cache(cache, man, {"train"}, 2)
    assert rec["identical"] and rec["manifest_sha256"] == rc.file_sha256(man)
    with pytest.raises(SystemExit, match="does not cover"):
        rc.verify_cache(cache, man, {"train", "val"}, 2)
    p = cache / "train" / "rgb.npy"
    b = bytearray(p.read_bytes())
    b[-1] ^= 1                                      # same size, different payload
    p.write_bytes(bytes(b))
    with pytest.raises(SystemExit, match="differs"):
        rc.verify_cache(cache, man, {"train"}, 2)


def test_precision_ieee_and_deterministic(monkeypatch, restore_precision):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", rc.CUBLAS)
    eff = rc.setup_precision(CFG)
    assert eff["cuda.matmul.fp32_precision"] == "ieee" and eff["cudnn.conv.fp32_precision"] == "ieee"
    assert eff["deterministic_algorithms"] and not eff["cudnn.benchmark"]
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(SystemExit):
        rc.setup_precision(CFG)


def test_device_record():
    assert rc.device_record("cpu") == {"type": "cpu", "index": None, "name": None}
    if torch.cuda.is_available():
        r = rc.device_record("cuda")
        assert r["type"] == "cuda" and r["index"] == torch.cuda.current_device() and r["name"]


def test_loader_generators_keyed():
    a, b, c = rc.loader_generator(1, 0), rc.loader_generator(1, 0), rc.loader_generator(1, 1)
    assert torch.equal(a.get_state(), b.get_state()) and not torch.equal(a.get_state(), c.get_state())


def test_env_key_shape():
    k = rc.env_key(rc.env_record())
    assert {"python", "version_mismatches", "gpus", "torch", "torch_cuda", "cudnn", "driver", "cv2",
            "numpy", "cpus", "platform", "torch_threads"} == set(k)
