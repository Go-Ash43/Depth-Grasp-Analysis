"""Run configuration and guards for scripts/train.py and the gates (step 5 proposal rev 2/3).

- load / validate configs/train_v1.json (fixed policies must read exactly as approved);
- effective hole policy (R / A0 / M: training-hole mixture; Astar: none) and seed lists;
- precision (one API: fp32_precision = "ieee" for CUDA matmul and cuDNN conv, read back and logged;
  deterministic algorithms; cuDNN benchmark off; CUBLAS_WORKSPACE_CONFIG set before torch is imported);
- the git-clean check, the W_NORM consistency check, the full cache-manifest re-hash, the environment
  record (training/env.py versions + hardware) and its comparison key.
"""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from data.cache_layout import open_cache
from data.geometry import scale_grasps
from data.provenance import git_state

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "train_v1.json"
CUBLAS = ":4096:8"
KINDS = ("smoke", "pilot", "production")
MODEL_NAMES = ("R", "A0", "M", "Astar")

FIXED = {   # policies the code implements; the config must state exactly these
    ("model", "arch"): "grconv_lite",
    ("loss", "type"): "smooth_l1",
    ("optim", "name"): "adam",
    ("schedule", "name"): "cosine_per_step",
    ("drop_last_train",): False,
    ("effective_holes",): {"R": "train", "A0": "train", "M": "train", "Astar": "none"},
    ("flag_setting_train",): "deployed",
    ("precision", "dtype"): "float32",
    ("precision", "amp"): False,
    ("precision", "fp32_precision"): {"cuda.matmul": "ieee", "cudnn.conv": "ieee"},
    ("precision", "deterministic"): True,
    ("precision", "cudnn_benchmark"): False,
    ("precision", "cublas_workspace"): CUBLAS,
    ("selection",): "final_epoch",
    ("model", "init"): "xavier_uniform gain 1 on Conv2d/ConvTranspose2d weights; default biases and BN",
    ("loss", "aw_reduction"): "sum over void==0 batch pixels / count (pooled, not per-image)",
    ("schedule", "step"): "one scheduler step after every optimizer step",
    ("schedule", "horizon"): "epochs * ceil(n_train / batch_size), fixed at run creation",
    ("overfit", "schedule"): "constant",
    ("cache_verify",): "full manifest re-hash (train, val) at run creation and at every resume",
}


# loss.q_reduction: the config states the policy in words; exactly these two are implemented (proj_v4 5.13).
# configs/train_v1.json keeps the "all" wording byte-identical, so every earlier run stays re-checkable.
Q_REDUCTION_TEXT = {
    "sum over all batch pixels / count": "all",
    "split: sum over void==0 batch pixels / count + sum over void==1 batch pixels / count (pooled)": "split",
}


def q_reduction(cfg):
    """The grasp_loss q_reduction ("all" | "split") a validated config states."""
    return Q_REDUCTION_TEXT[cfg["loss"]["q_reduction"]]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _get(cfg, keys):
    v = cfg
    for k in keys:
        v = v[k]
    return v


def load_train_cfg(path=None):
    p = Path(path or DEFAULT_CONFIG)
    cfg = json.loads(p.read_text())
    validate_train_cfg(cfg)
    return cfg, p


def validate_train_cfg(cfg):
    if cfg.get("status") not in ("draft", "frozen"):
        raise ValueError(f"status must be draft or frozen, got {cfg.get('status')!r}")
    for keys, want in FIXED.items():
        try:
            got = _get(cfg, keys)
        except (KeyError, TypeError):
            raise ValueError(f"config lacks {'.'.join(keys)}") from None
        if got != want:
            raise ValueError(f"config {'.'.join(keys)} = {got!r}; the code implements {want!r}")
    if cfg.get("loss", {}).get("q_reduction") not in Q_REDUCTION_TEXT:
        raise ValueError(f"config loss.q_reduction = {cfg.get('loss', {}).get('q_reduction')!r}; the code "
                         f"implements exactly {sorted(Q_REDUCTION_TEXT)}")
    if set(cfg["loss"]["weights"]) != {"q", "cos2", "sin2", "width"}:
        raise ValueError("loss.weights must name q, cos2, sin2, width")
    if not (isinstance(cfg["batch_size"], int) and cfg["batch_size"] > 0):
        raise ValueError("batch_size must be a positive integer")
    ep = cfg["epochs"]
    if ep is not None and not (isinstance(ep, int) and ep > 0):
        raise ValueError("epochs must be null (draft) or a positive integer")
    if cfg["status"] == "frozen" and ep is None:
        raise ValueError("a frozen config needs epochs")
    if "none" not in cfg["val_conditions"]:
        raise ValueError("val_conditions must include the clean condition 'none'")
    return cfg


def effective_holes(cfg, model):
    if model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {model!r}")
    return cfg["effective_holes"][model]


def allowed_seeds(cfg, model):
    return list(cfg["astar_seeds"] if model == "Astar" else cfg["seeds"])


# ------------------------------------------------------------------ precision
def check_cublas_env():
    got = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if got != CUBLAS:
        raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG must be {CUBLAS!r} before torch initialises CUDA, got {got!r}")


def setup_precision(cfg):
    """Deterministic FP32 via the fp32_precision API only (no legacy allow_tf32 switches).
    Returns the effective values read back."""
    check_cublas_env()
    p = cfg["precision"]["fp32_precision"]
    torch.backends.cuda.matmul.fp32_precision = p["cuda.matmul"]
    torch.backends.cudnn.conv.fp32_precision = p["cudnn.conv"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    eff = {"cuda.matmul.fp32_precision": torch.backends.cuda.matmul.fp32_precision,
           "cudnn.conv.fp32_precision": torch.backends.cudnn.conv.fp32_precision,
           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "cudnn.benchmark": torch.backends.cudnn.benchmark,
           "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if eff["cuda.matmul.fp32_precision"] != p["cuda.matmul"] or eff["cudnn.conv.fp32_precision"] != p["cudnn.conv"] \
            or not eff["deterministic_algorithms"] or eff["cudnn.benchmark"]:
        raise RuntimeError(f"precision settings did not take effect: {eff}")
    return eff


# ------------------------------------------------------------------ guards
def require_clean_tree(allow_dirty):
    head, dirty = git_state()
    if head is None:
        raise SystemExit("git state unknown (not a repo, or git missing): refusing")
    if dirty and not allow_dirty:
        raise SystemExit("working tree is dirty (untracked files count): commit first")
    return {"git_commit": head, "git_dirty": bool(dirty)}


def w_norm_rule(max_opening_224):
    """configs/labels.json rule: max train opening at 224 rounded UP to the next multiple of 10 px."""
    return float(math.ceil(float(max_opening_224) / 10.0) * 10.0)


def check_w_norm(cache_dir, labels_cfg, split="train", splits=None):
    """Recompute the train-wide max opening at the configured size; refuse a stale W_NORM."""
    c = open_cache(cache_dir, split, splits=splits)
    g = np.load(Path(cache_dir).expanduser() / split / "grasps_1024.npy", mmap_mode="r")
    size = int(labels_cfg["size"])
    op = scale_grasps(np.c_[np.zeros((len(g), 3)), np.asarray(g[:, 3]), np.ones(len(g))],
                      (1024, 1024), (size, size))[:, 3]
    if len(op) == 0 or not np.isfinite(op).all() or (op <= 0).any():
        raise SystemExit(f"{split}: train openings empty, non-finite or <= 0")
    mx = float(op.max())
    want = w_norm_rule(mx)
    got = labels_cfg["targets"]["w_norm_px_224"]
    rec = {"max_opening_224": mx, "n_rows": int(len(op)), "rule_value": want, "configured": got,
           "n_ok_slots": sum(r["status"] == "ok" for r in c["index"])}
    if got is None or float(got) != want:
        raise SystemExit(f"W_NORM stale: configured {got}, train rule gives {want} (max opening {mx:.4f} px)")
    return rec


def verify_cache(cache_dir, manifest_path, need_splits, workers=8):
    """Full manifest re-hash (scripts/cache_manifest.py) of every file of the manifest's splits."""
    from scripts import cache_manifest as cm
    t0 = time.time()
    mpath = Path(manifest_path).expanduser()
    manifest = json.loads(mpath.read_text())
    missing = sorted(set(need_splits) - set(manifest["splits"]))
    if missing:
        raise SystemExit(f"manifest {mpath} does not cover splits {missing}")
    rels = cm.list_files(cache_dir, manifest["splits"])
    got, total = cm.hash_all(cache_dir, rels, workers)
    res = cm.compare(manifest, got)
    rec = {"manifest": str(mpath), "manifest_sha256": file_sha256(mpath), "splits": manifest["splits"],
           "n_files": len(rels), "total_bytes": total, "identical": res["identical"],
           "seconds": round(time.time() - t0, 1),
           "differences": {k: res[k][:10] for k in ("missing", "extra", "resized", "changed")}}
    if not res["identical"]:
        raise SystemExit(f"cache differs from the manifest: {rec['differences']}")
    return rec


def env_record():
    from training import env
    return {"versions": env.check_versions(env.parse_lock(), "3.11"),
            "hardware": env.hardware(), "torch_threads": torch.get_num_threads()}


def device_record(device):
    """Resolved execution device: type, index and (CUDA) the selected GPU's name."""
    d = torch.device(device)
    if d.type == "cuda":
        idx = d.index if d.index is not None else torch.cuda.current_device()
        return {"type": "cuda", "index": int(idx), "name": torch.cuda.get_device_name(idx)}
    return {"type": d.type, "index": None, "name": None}


def require_versions(rec):
    """Refuse to train unless every pin of requirements-lock.txt and Python 3.11 match."""
    v = rec["versions"]
    if not v["ok"]:
        raise SystemExit(f"environment does not match requirements-lock.txt: python {v['python']} "
                         f"(ok={v['python_ok']}), mismatches {v['mismatches']}")


def env_key(rec):
    """The part of the environment record an exact continuation depends on."""
    v, h = rec["versions"], rec["hardware"]
    return {"python": v["python"], "version_mismatches": v["mismatches"],
            "gpus": [g["name"] for g in h["gpus"]], "torch": h["torch"], "torch_cuda": h["torch_cuda"],
            "cudnn": h["cudnn"], "driver": h["driver"], "cv2": h["cv2"], "numpy": h["numpy"],
            "cpus": h["cpus"], "platform": h["platform"], "torch_threads": rec["torch_threads"]}


def loader_generator(seed, stream):
    """Dedicated DataLoader generator keyed by (seed, stream): 0 train loader, 1 val loader."""
    s = int(np.random.SeedSequence([int(seed), 7919, int(stream)]).generate_state(1, np.uint32)[0])
    g = torch.Generator()
    g.manual_seed(s)
    return g
