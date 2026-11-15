#!/usr/bin/env python3
"""Protocol A gate on real checkpoints (SPEC_trajectory "Implementation assertions"; G1 exit gate:
Delta_mask = 0 for R and A0 under flag-only edits). The check itself is eval/invariants.py
protocol_a_check, reused by every later evaluation.

Loads any of --ckpt-R / --ckpt-A0 / --ckpt-M (checkpoints from scripts/train.py; the checkpoint's model
name must match), draws --n seeded slots of --split, and runs clean ("none") and "mid" hole conditions
with flag variants deployed / oracle / centre_box48_flip. Exit 0 PASS, 1 FAIL.

Usage:
  python scripts/gate_protocol_a.py --ckpt-A0 /workspace/runs/A0_s1/ckpt/epoch_019.pt \
      --ckpt-M /workspace/runs/M_s1/ckpt/epoch_019.pt --out /workspace/runs/protocol_a_v1 --device cuda
"""
import os

_CUBLAS = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", _CUBLAS) != _CUBLAS:
    raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG is {os.environ['CUBLAS_WORKSPACE_CONFIG']!r}; unset it or use {_CUBLAS!r}")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import training.runcfg as rc  # noqa: E402
from data.dataset import GraspDataset  # noqa: E402
from data.paths import load_paths, paths_source  # noqa: E402
from data.provenance import git_fields, refuse_existing, refuse_inside  # noqa: E402
from eval.invariants import protocol_a_check, seeded_slots  # noqa: E402
from models.grconv_lite import build_model  # noqa: E402
from training.checkpoint import atomic_write_json, load_checkpoint  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for m in ("R", "A0", "M"):
        p.add_argument(f"--ckpt-{m}", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="val", choices=("val", "train"))   # never a test shard
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--slot-seed", type=int, default=20261112)
    p.add_argument("--conditions", default="none,mid")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--config", default=str(rc.DEFAULT_CONFIG))
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv=None, *, splits=None):
    a = parse_args(argv)
    cfg, _ = rc.load_train_cfg(a.config)
    paths = load_paths()
    cache_dir = Path(a.cache_dir or paths["cache_dir"]).expanduser()
    out = refuse_inside(a.out, paths["data_root"], cache_dir)
    refuse_existing([out], overwrite=False)
    precision = rc.setup_precision(cfg)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but no CUDA device")
    models, sources = {}, {}
    for name in ("R", "A0", "M"):
        path = getattr(a, f"ckpt_{name}")
        if path is None:
            continue
        st = load_checkpoint(path)
        if st["meta"].get("model") != name:
            raise SystemExit(f"{path}: checkpoint is model {st['meta'].get('model')!r}, not {name}")
        m = build_model(name, 0, cfg["model"]["c"], cfg["model"]["n_res"])
        m.load_state_dict(st["model"])
        models[name] = m.to(device)
        sources[name] = {"path": str(path), "meta": st["meta"]}
    if not models:
        raise SystemExit("give at least one of --ckpt-R / --ckpt-A0 / --ckpt-M")
    probe = GraspDataset(cache_dir, a.split, "M", mode="eval", splits=splits)
    slots = seeded_slots(probe.slots, a.n, a.slot_seed)
    t0 = time.time()
    rep = protocol_a_check(models, cache_dir, a.split, slots,
                           conditions=tuple(c for c in a.conditions.split(",") if c),
                           batch_size=a.batch_size, device=device, splits=splits)
    rep.update({"seconds": round(time.time() - t0, 1), "slots": slots, "checkpoints": sources,
                "args": vars(a), "precision": precision, **git_fields(), "paths_source": paths_source()})
    out.mkdir(parents=True)
    atomic_write_json(out / "protocol_a.json", rep)
    print(f"{rep['verdict']} ({len(slots)} slots, {rep['seconds']} s)")
    for f in rep["failures"]:
        print(f"  - {f}")
    for cond, c in rep["conditions"].items():
        if "M" in c:
            mm = c["M"]
            print(f"  {cond}/M: oracle flag px changed {mm['flag_changed_px']['oracle']}"
                  f"{' (VACUOUS)' if mm['oracle_vacuous'] else ''}, edit px {mm['flag_changed_px']['edit']}, "
                  f"top-1 cell changes oracle {mm['top1_cell_changes']['oracle']} edit {mm['top1_cell_changes']['edit']}")
    print(f"report: {out / 'protocol_a.json'}")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
