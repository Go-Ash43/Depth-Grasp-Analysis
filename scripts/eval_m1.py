#!/usr/bin/env python3
"""Module 1 evaluation: the nine selected final checkpoints on val or test_module (P-E1 rev 4;
docs/proj_v7.md sections 13-15; SPEC_trajectory Module 1 work item 4, "Test discipline across modules").

Conditions none / low / mid / high: eval-mode GraspDataset (corruption seed of configs/train_data.json,
epoch 0, deployed = truthful flag; none = natural stereo holes only, completed and flagged). Reuses
eval.val_epoch.run_val (inference_mode, every eligible slot exactly once, invalid decodes in the
denominator, BatchNorm buffers unchanged) and its CSV format. All nine checkpoints on ONE device; this
evaluation supplies the authoritative H1 rows.

Checks before inference: selected_runs.json (scripts/m1_runs.py) re-read and every final checkpoint
re-hashed; config status frozen and the CURRENT config / labels / train_data / frozen files hash to the runs'
recorded values (labels / train_data / frozen: or are bridged to them, data.provenance.config_matches); clean git tree; frozen precision (rc.setup_precision; CUBLAS set before torch), batch size =
config batch_size; environment record (lock-file pins). Manifest: val -> the given manifest must hash to the
runs' manifest_sha256; test_module -> it must hash to --expect-manifest-sha256 (written in docs/proj_v7.md
before the test). Full re-hash of the split's cache against it.

Per (run, condition) a sidecar CSV: slot, n_synth_cells, m_synth_sha256, flag_sha256 (flag = cached
M_sensor | m_synth, the deployed definition; for M also asserted equal to its input flag channel),
severity, hole_frac, covered_lie_frac. Computed in the same pass by a wrapper around the loader.

Val only: rows of none / mid are compared with each run's own val/epoch_<E-1>.csv. Corruption metadata must
match exactly. A run trained on THIS device and software (same gpus / driver / torch / cudnn) must agree
exactly row for row, else refused (resolve before the freeze); other runs get a report: label agreement,
accuracy change, max prediction deltas.

test_module: one look, enforced by a fixed ledger <runs_dir>/m1_test_ledger.json (runs_dir from
data.paths.load_paths, not --out) and an exclusive flock on <ledger>.lock for the whole run:
  absent            -> created with O_EXCL (status "started", identities), fsynced, then inference
  started, lock free-> one retry only, identical identities required; retries is incremented and fsynced
                       BEFORE inference; the crashed attempt's .partial is archived unread
  completed         -> refused, except --publish-only (no inference; re-verifies output hashes)
Outputs go to <out>/.partial, are fsynced and hashed, renamed to <out>/results, the ledger is marked
"completed" (fsynced) and ONLY THEN summaries are computed and printed (<out>/summary.json).

Usage (laptop, from the analysis commit):
  python scripts/eval_m1.py --selected $R/m1_selected/selected_runs.json --split val \
      --manifest $R/cache_manifest_v1/manifest.json --out $R/m1_eval_val --num-workers 2
  python scripts/eval_m1.py --selected ... --split test_module \
      --manifest $R/cache_manifest_test_module/manifest.json --expect-manifest-sha256 <sha> --out $R/m1_eval_test
"""
import os

_CUBLAS = ":4096:8"                                  # must be set before torch initialises CUDA
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", _CUBLAS) != _CUBLAS:
    raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG is {os.environ['CUBLAS_WORKSPACE_CONFIG']!r}; unset it or use {_CUBLAS!r}")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS

import argparse  # noqa: E402
import csv  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import training.runcfg as rc  # noqa: E402
from data.dataset import EpochSampler, GraspDataset, collate  # noqa: E402
from data.labels import load_labels_cfg  # noqa: E402
from data.paths import load_paths  # noqa: E402
from data.provenance import config_matches  # noqa: E402
from data.workers import worker_init  # noqa: E402
from eval.val_epoch import CSV_FIELDS, run_val, rows_to_csv_bytes  # noqa: E402
from models.grconv_lite import build_model  # noqa: E402
from scripts.m1_runs import CURRENT, sha256  # noqa: E402
from training.checkpoint import atomic_write_bytes, atomic_write_json, fsync_dir, load_checkpoint  # noqa: E402

CONDITIONS = ("none", "low", "mid", "high")
SIDE_FIELDS = ("slot", "n_synth_cells", "m_synth_sha256", "flag_sha256", "severity", "hole_frac", "covered_lie_frac")
FLAG_CH = 4                                          # M = rgb(3) + depth(1) + flag(1)
EXACT_KEYS = ("gpus", "driver", "torch", "torch_cuda", "cudnn")
PRED_FIELDS = ("x", "y", "theta", "opening", "q_top")
MAX_RETRIES = 1
LEDGER_NAME = "m1_test_ledger.json"


class EvalError(RuntimeError):
    pass


def _need(ok, msg):
    if not ok:
        raise EvalError(msg)


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.uint8).tobytes()).hexdigest()


# ------------------------------------------------------------------ inputs
def load_selected(path, cfg_path):
    sel = json.loads(Path(path).read_text())
    cfg, cfg_path = rc.load_train_cfg(cfg_path)
    _need(cfg["status"] == "frozen", f"{cfg_path}: status {cfg['status']!r}; evaluation needs the frozen config")
    sh = sel["shared"]
    _need(rc.file_sha256(cfg_path) == sh["config_sha256"], f"{cfg_path} differs from the runs' config_sha256")
    for k, p in CURRENT.items():
        _need(config_matches(p, sh[k]), f"{p} differs from the runs' {k}")
    ids = sorted(sel["runs"])
    _need(ids == sorted(f"{m}:{s}" for m in ("R", "A0", "M") for s in (1, 2, 3)), f"selected ids {ids}")
    for k, r in sel["runs"].items():
        _need(sha256(r["final_ckpt"]["path"]) == r["final_ckpt"]["sha256"], f"{k}: final checkpoint changed")
    return sel, cfg


class HashingLoader:
    """Pass-through over a loader feeding run_val; records the per-slot corruption identity."""

    def __init__(self, loader, m_sensor, model):
        self.loader, self.m_sensor, self.model = loader, m_sensor, model
        self.side = []

    def __iter__(self):
        for batch in self.loader:
            for b in range(batch["x"].shape[0]):
                slot = int(batch["slot"][b])
                synth = batch["m_synth"][b].numpy().astype(bool)
                flag = np.asarray(self.m_sensor[slot]).astype(bool) | synth
                if self.model == "M":
                    _need(np.array_equal(batch["x"][b, FLAG_CH].numpy() == 1.0, flag)
                          and np.isin(batch["x"][b, FLAG_CH].numpy(), (0.0, 1.0)).all(),
                          f"slot {slot}: M's flag channel != M_sensor | m_synth")
                self.side.append({"slot": slot, "n_synth_cells": int(synth.sum()), "m_synth_sha256": _h(synth),
                                  "flag_sha256": _h(flag), "severity": batch["severity"][b],
                                  "hole_frac": float(batch["hole_frac"][b]),
                                  "covered_lie_frac": float(batch["covered_lie_frac"][b])})
            yield batch


def side_bytes(side):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SIDE_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in sorted(side, key=lambda r: r["slot"]):
        w.writerow(r)
    return buf.getvalue().encode()


def eval_one(rec, split, cond, cache_dir, cfg, labels_cfg, device, num_workers, splits=None):
    """-> (CSV bytes, sidecar bytes) for one run x condition."""
    model_name, seed = rec["model"], int(rec["seed"])
    ds = GraspDataset(cache_dir, split, model_name, mode="eval", holes=cond, splits=splits)
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), sampler=EpochSampler(ds.slots, shuffle=False, seed=0),
                        collate_fn=collate, num_workers=num_workers,
                        worker_init_fn=worker_init if num_workers > 0 else None,
                        pin_memory=num_workers > 0 and torch.device(device).type == "cuda")
    st = load_checkpoint(rec["final_ckpt"]["path"], rec["final_ckpt"]["sha256"])
    _need((st["meta"]["model"], st["meta"]["seed"]) == (model_name, seed), f"{rec['path']}: checkpoint meta model/seed")
    model = build_model(model_name, seed, cfg["model"]["c"], cfg["model"]["n_res"])
    model.load_state_dict(st["model"])
    model.to(device)
    hl = HashingLoader(loader, ds._c()["arrays"]["m_sensor"], model_name)
    rows, _ = run_val(model, hl, ds.slots, labels_cfg, device, cond)      # summary withheld (publication later)
    _need(sorted(r["slot"] for r in hl.side) == sorted(int(s) for s in ds.slots), f"{cond}: sidecar slots incomplete")
    return rows_to_csv_bytes(rows), side_bytes(hl.side)


def parse_rows(b, condition=None):
    rows = list(csv.DictReader(io.StringIO(b.decode() if isinstance(b, bytes) else b)))
    return {int(r["slot"]): r for r in rows if condition is None or r["condition"] == condition}


def compare_training(new_b, train_csv, cond, exact):
    """New eval rows of one condition vs the run's own training-time final val CSV."""
    new, old = parse_rows(new_b, cond), parse_rows(Path(train_csv).read_bytes(), cond)
    _need(sorted(new) == sorted(old), f"{cond}: slot sets differ from the training-time CSV")
    meta_ok = all(new[s][k] == old[s][k] for s in new for k in ("severity", "hole_frac", "covered_lie_frac"))
    _need(meta_ok, f"{cond}: corruption metadata differs from the training-time CSV")
    rep = {"condition": cond, "n": len(new), "rows_identical": all(new[s] == old[s] for s in new)}
    for m in ("primary", "strict"):
        agree = sum(new[s][f"match_{m}"] == old[s][f"match_{m}"] for s in new)
        an = sum(new[s][f"match_{m}"] == "True" for s in new)
        ao = sum(old[s][f"match_{m}"] == "True" for s in new)
        rep[m] = {"label_agreement": agree, "acc_new": an / len(new), "acc_train": ao / len(new),
                  "acc_change_pp": 100.0 * (an - ao) / len(new)}
    deltas = {}
    for f in PRED_FIELDS:
        d = [abs(float(new[s][f]) - float(old[s][f])) for s in new if new[s][f] not in ("", "nan") and old[s][f] not in ("", "nan")]
        deltas[f] = max(d) if d else 0.0
    rep["max_abs_pred_delta"] = deltas
    if exact:
        _need(rep["rows_identical"], f"{cond}: run trained on this device/software but rows differ from its "
                                     f"training-time CSV (resolve before the freeze): {rep}")
    return rep


def summary_from_csv(b):
    rows = list(csv.DictReader(io.StringIO(b.decode())))
    n = len(rows)
    return {"n": n, "top1_primary": sum(r["match_primary"] == "True" for r in rows) / n,
            "top1_strict": sum(r["match_strict"] == "True" for r in rows) / n,
            "invalid_decodes": sum(r["valid"] == "False" for r in rows)}


# ------------------------------------------------------------------ ledger (test split)
def ledger_dir():
    return Path(load_paths()["runs_dir"]).expanduser()


def _fsync_write(path, obj, exclusive=False):
    data = (json.dumps(obj, indent=1, default=str) + "\n").encode()
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        dfd = os.open(str(Path(path).parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    else:
        atomic_write_bytes(path, data)


def verified_outputs(res, identities):
    """A complete results dir: eval_record.json present, every listed file hash-equal, same identities.
    -> the outputs map (files + eval_record.json) the ledger records. Refuses anything else."""
    rp = res / "eval_record.json"
    _need(rp.is_file(), f"{res}: no eval_record.json; stop for a manual check")
    rec = json.loads(rp.read_text())
    _need(rec.get("identities") == identities, f"{res}: eval_record identities differ from this run's")
    for rel, h in rec["files"].items():
        _need((res / rel).is_file() and sha256(res / rel) == h, f"{res / rel}: missing or sha256 differs")
    return {**rec["files"], "eval_record.json": sha256(rp)}


def _crash_hook(stage):
    """No-op; tests replace it to simulate a crash at a stage boundary."""


def acquire(lock_path):
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise SystemExit(f"{lock_path} is held by another process: a test run is in progress") from None
    return fd


def ledger_begin(ledger, identities, out, publish_only):
    """-> "run" or "publish". Caller holds the lock."""
    if not ledger.exists():
        _need(not publish_only, "--publish-only but no ledger exists")
        _fsync_write(ledger, {"status": "started", "retries": 0, "identities": identities,
                              "started": time.strftime("%Y-%m-%d %H:%M:%S")}, exclusive=True)
        return "run"
    L = json.loads(ledger.read_text())
    if L["status"] == "completed":
        _need(publish_only, f"{ledger}: the module test run is already completed (one look); only --publish-only")
        return "publish"
    _need(L["status"] == "started", f"{ledger}: unknown status {L['status']!r}")
    _need(L["identities"] == identities, f"{ledger}: a retry needs identical identities; differs in "
          f"{sorted(k for k in set(L['identities']) | set(identities) if L['identities'].get(k) != identities.get(k))}")
    if (out / "results").exists():                   # review fix 2: crashed after the rename, before "completed"
        outputs = verified_outputs(out / "results", identities)
        L.update(status="completed", outputs=outputs, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                 recovered="completed from a fully written results dir (crash between rename and ledger update)")
        _fsync_write(ledger, L)
        return "publish"                             # no inference, no retry consumed
    _need(not publish_only, f"{ledger}: run not completed; nothing to publish")
    _need(L["retries"] < MAX_RETRIES, f"{ledger}: retry limit {MAX_RETRIES} reached; stop for a manual check")
    L["retries"] += 1
    L.setdefault("retry_log", []).append(time.strftime("%Y-%m-%d %H:%M:%S"))
    _fsync_write(ledger, L)                          # durable BEFORE inference restarts
    part = out / ".partial"
    if part.exists():                                # archived unread
        os.replace(part, out / f".crashed_attempt_{L['retries']}")
    return "run"


# ------------------------------------------------------------------ main
def parse_args(argv=None, allowed=("val", "test_module")):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected", required=True)
    p.add_argument("--split", required=True, choices=allowed)
    p.add_argument("--manifest", required=True)
    p.add_argument("--expect-manifest-sha256", default=None, help="test_module: the sha256 recorded in proj_v7")
    p.add_argument("--config", default=str(rc.DEFAULT_CONFIG))
    p.add_argument("--cache-dir", default=None, help="default: load_paths()['cache_dir']")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--hash-workers", type=int, default=8)
    p.add_argument("--publish-only", action="store_true", help="test_module: finish publication after a crash")
    p.add_argument("--seed", type=int, default=0, help="unused (all randomness is keyed); recorded by convention")
    return p.parse_args(argv)


def main(argv=None, *, splits=None, allowed=("val", "test_module"), ledger_splits=("test_module",), _ledger_dir=None):
    a = parse_args(argv, allowed)
    try:
        return _main(a, splits, ledger_splits, _ledger_dir)
    except EvalError as e:
        raise SystemExit(f"eval_m1 refused: {e}")


def _main(a, splits, ledger_splits, _ledger_dir):
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    sel_sha = sha256(a.selected)
    sel, cfg = load_selected(a.selected, a.config)
    git = rc.require_clean_tree(False)
    paths = load_paths()
    cache_dir = Path(a.cache_dir or paths["cache_dir"]).expanduser()
    out = Path(a.out).expanduser().resolve()
    man_sha = rc.file_sha256(a.manifest)
    if a.split in ledger_splits:
        _need(a.expect_manifest_sha256 is not None, "test_module needs --expect-manifest-sha256 (from proj_v7)")
        _need(man_sha == a.expect_manifest_sha256, f"manifest sha256 {man_sha} != expected {a.expect_manifest_sha256}")
    else:
        _need(man_sha == sel["shared"]["manifest_sha256"], "manifest differs from the one the runs were verified against")
    precision = rc.setup_precision(cfg)
    device = torch.device(a.device)
    _need(device.type != "cuda" or torch.cuda.is_available(), "--device cuda but no CUDA device")
    env = rc.env_record()
    rc.require_versions(env)
    ek = {**rc.env_key(env), "execution": rc.device_record(device)}
    identities = {"analysis_commit": git["git_commit"], "selected_sha256": sel_sha, "manifest_sha256": man_sha,
                  "split": a.split, "conditions": list(CONDITIONS), "env_key": ek, "precision": precision,
                  "batch_size": int(cfg["batch_size"]), "out": str(out)}
    ledger = lock_fd = None
    mode = "run"
    if a.split in ledger_splits:
        ld = Path(_ledger_dir) if _ledger_dir is not None else ledger_dir()
        ledger = ld / LEDGER_NAME
        lock_fd = acquire(str(ledger) + ".lock")
    try:
        if ledger is not None:
            mode = ledger_begin(ledger, identities, out, a.publish_only)
        elif a.publish_only:                         # val: finish publication of a completed results dir
            _need((out / "results" / "eval_record.json").is_file(), f"{out}/results incomplete; nothing to publish")
            mode = "publish"
        else:
            _need(not out.exists(), f"{out} exists; use a fresh --out")
        if mode == "run":
            _need(not (out / "results").exists(), f"{out}/results exists")
            cache_rec = rc.verify_cache(cache_dir, a.manifest, {a.split}, a.hash_workers)
            labels_cfg = load_labels_cfg()
            part = out / ".partial"
            files, comparisons = {}, {}
            for key in sorted(sel["runs"]):
                rec = sel["runs"][key]
                tag = f"{rec['model']}_s{rec['seed']}"
                exact = all(rec["env_key"].get(k) == ek.get(k) for k in EXACT_KEYS) and \
                    rec["env_key"].get("execution", {}).get("name") == ek["execution"]["name"]
                for cond in CONDITIONS:
                    t0 = time.time()
                    b, sb = eval_one(rec, a.split, cond, cache_dir, cfg, labels_cfg, a.device, a.num_workers, splits)
                    files[f"{tag}/{cond}.csv"] = atomic_write_bytes(part / tag / f"{cond}.csv", b)
                    files[f"{tag}/{cond}.sidecar.csv"] = atomic_write_bytes(part / tag / f"{cond}.sidecar.csv", sb)
                    if ledger is None and cond in sel["shared"]["val_conditions"]:
                        comparisons.setdefault(key, {"exact_required": exact})[cond] = \
                            compare_training(b, rec["final_csv"]["path"], cond, exact)
                    print(f"  {tag} {cond}: {time.time() - t0:.0f} s", flush=True)   # timing only, never accuracy
            record = {"split": a.split, "identities": identities, "cache_verify": cache_rec, "selected": a.selected,
                      "selected_sha256": sel_sha, "m3": sel["m3"], "metric_decision": sel.get("metric_decision"),
                      "git": git, "env": env, "paths": paths, "args": vars(a), "started": started,
                      "conditions": list(CONDITIONS), "csv_fields": list(CSV_FIELDS), "files": files,
                      "training_comparison": comparisons}
            files["eval_record.json"] = atomic_write_json(part / "eval_record.json", record)
            os.replace(part, out / "results")
            fsync_dir(out)                           # review fix 2: the rename itself is durable
            _crash_hook("after_rename")
            if ledger is not None:
                L = json.loads(ledger.read_text())
                L.update(status="completed", outputs=files, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
                _fsync_write(ledger, L)              # durable completion BEFORE any summary
        # ---------------- publication (summaries only after completion)
        res = out / "results"
        if ledger is not None:
            want = json.loads(ledger.read_text())["outputs"]
        else:
            want = json.loads((res / "eval_record.json").read_text())["files"]
            want["eval_record.json"] = sha256(res / "eval_record.json")
        for rel, h in want.items():
            _need(sha256(res / rel) == h, f"{res / rel}: sha256 differs from the record")
        summ = {}
        for rel in sorted(want):
            if rel.endswith(".csv") and not rel.endswith(".sidecar.csv"):
                summ[rel[:-4]] = summary_from_csv((res / rel).read_bytes())
        atomic_write_json(out / "summary.json", {"split": a.split, "summary": summ})
        for k, v in summ.items():
            print(f"  {k}: primary {v['top1_primary']:.4f} strict {v['top1_strict']:.4f} invalid {v['invalid_decodes']}")
        print(f"done: {res}")
        return 0
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


if __name__ == "__main__":
    sys.exit(main())
