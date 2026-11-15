#!/usr/bin/env python3
"""Train one detector (R / A0 / M / Astar) on the 224-px cache (SPEC_trajectory Module 1 item 3; step 5
proposal rev 2 + rev 3, approved; configs/train_v1.json).

Run kinds
  smoke       quick end-to-end check: --epochs (default 1), --limit-train N seeded train slots,
              --batch-size override, --allow-dirty permitted (all recorded). Never used for results.

The loss's Q reduction comes from the config (training.runcfg.q_reduction): configs/train_v1.json = "all"
(config loss), configs/train_v1_sq.json = "split" (SQ); the two configs differ in that field only.
Console output from run creation / resume onward is appended to <out>/console.log; a progress line is
printed every --progress-every steps.
  pilot       the production settings with --epochs given (config epochs still null): measures the
              real epoch wall time that sets the frozen budget. Clean tree required.
  production  config status "frozen", epochs from the config, seed from the config's seed list,
              clean tree, no overrides. Always a fresh initialisation (or a resume of the same run).

Every run: data.paths.load_paths() (DEPTH_LIES_PATHS override honoured, source logged); full cache
manifest re-hash at creation and at every resume (train + val split); W_NORM re-checked against the train
openings; deterministic FP32 (fp32_precision "ieee"); environment record from training/env.py.
Per epoch: train (one scheduler step per optimizer step), validate every val condition (decode + match +
miss categories, eval + inference_mode, BN buffers asserted unchanged), then commit:
  val/epoch_eee.csv -> ckpt/epoch_eee.pt -> fsynced log line (training/checkpoint.py).
Selection is the final epoch (config "selection"); both metrics are always logged.
A fresh run or resume holds an exclusive flock on <out>/.lock for its whole life. flock is reliable on
one machine only: use ONE pod per run directory on the network volume.
Timing: epoch_wall_s covers train, val, CSV and checkpoint writes; the log append is reported in the
next record (prev_log_commit_s) and included in projected_run_s once known.

Usage (pod; paths from $DEPTH_LIES_PATHS):
  python scripts/train.py --kind pilot --model A0 --seed 1 --epochs 2 \
      --manifest /workspace/cache_manifest_v1/manifest.json --num-workers 8 --out /workspace/runs/pilot_A0_s1
  python scripts/train.py --kind pilot ... --resume          # continue the same run directory
"""
import os

_CUBLAS = ":4096:8"                                  # must be set before torch initialises CUDA
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", _CUBLAS) != _CUBLAS:
    raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG is {os.environ['CUBLAS_WORKSPACE_CONFIG']!r}; unset it or use {_CUBLAS!r}")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import training.runcfg as rc  # noqa: E402
from data.dataset import EpochSampler, GraspDataset, collate, load_data_cfg  # noqa: E402
from data.labels import load_labels_cfg  # noqa: E402
from data.paths import load_paths, paths_source  # noqa: E402
from data.provenance import config_matches, refuse_existing, refuse_inside  # noqa: E402
from data.workers import worker_init  # noqa: E402
from eval.invariants import seeded_slots  # noqa: E402
from eval.val_epoch import rows_to_csv_bytes, run_val  # noqa: E402
from models.grconv_lite import build_model, n_params  # noqa: E402
from training.checkpoint import (acquire_run_lock, append_log, atomic_write_bytes,  # noqa: E402
                                 atomic_write_json, capture_state, ckpt_path, csv_path, load_checkpoint,
                                 read_log, recover, release_run_lock, restore_state, save_checkpoint)
from training.loop import sync, train_epoch  # noqa: E402

BUDGET_S = 90 * 60
IMMUTABLE = ("kind", "model", "seed", "holes", "flag_setting", "epochs", "batch_size", "steps_per_epoch",
             "horizon", "n_train", "train_slots_sha256", "limit_train", "val_split", "val_conditions",
             "n_val", "config_sha256", "labels_sha256", "train_data_sha256", "frozen_sha256", "git_commit",
             "manifest_sha256")
META_KEYS = ("model", "seed", "holes", "horizon", "config_sha256")
CONFIG_FILES = {"labels_sha256": REPO / "configs" / "labels.json",          # hashes a resume may bridge
                "train_data_sha256": REPO / "configs" / "train_data.json",  # (data.provenance.config_matches)
                "frozen_sha256": REPO / "configs" / "frozen_day1.json"}


def _same_setting(k, old, now):
    """Resume check for one IMMUTABLE key: equal, or a config hash the bridge maps to the current file."""
    return old == now or (k in CONFIG_FILES and config_matches(CONFIG_FILES[k], old))


def _crash_hook(stage, epoch):
    """No-op; tests replace it to simulate an interruption between commit stages."""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", required=True, choices=rc.KINDS)
    p.add_argument("--model", required=True, choices=rc.MODEL_NAMES)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True, help="run directory (fresh run: must not exist)")
    p.add_argument("--manifest", required=True, help="cache manifest from scripts/cache_manifest.py write")
    p.add_argument("--config", default=str(rc.DEFAULT_CONFIG))
    p.add_argument("--cache-dir", default=None, help="default: load_paths()['cache_dir']")
    p.add_argument("--val-split", default="val")
    p.add_argument("--epochs", type=int, default=None, help="pilot / smoke only")
    p.add_argument("--limit-train", type=int, default=0, help="smoke only: N seeded train slots")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-dirty", action="store_true", help="smoke only")
    p.add_argument("--allow-env-change", action="store_true",
                   help="resume in a different environment (marks continuation_exact false)")
    p.add_argument("--sync-timing", action="store_true",
                   help="diagnostic: synchronise after every step (budget_valid false); not production")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--hash-workers", type=int, default=8, help="threads for the manifest re-hash")
    p.add_argument("--batch-size", type=int, default=None,
                   help="smoke only: override the config's batch_size (recorded in run_config.json)")
    p.add_argument("--progress-every", type=int, default=100, help="progress line every N steps (0 = off)")
    return p.parse_args(argv)


class _Tee:
    """Mirror a stream into <run>/console.log (append, line-flushed); restores nothing itself."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False


def _start_console_log(out, state):
    fh = open(Path(out) / "console.log", "a", buffering=1)
    fh.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv)}\n")
    state["console"] = (fh, sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = _Tee(sys.stdout, fh), _Tee(sys.stderr, fh)


def _stop_console_log(state):
    if state.get("console"):
        fh, so, se = state.pop("console")
        sys.stdout, sys.stderr = so, se
        fh.close()


def _positive_epochs(ep):
    if not (isinstance(ep, int) and ep > 0):
        raise SystemExit(f"epochs must be a positive integer, got {ep!r}")
    return ep


def check_kind(a, cfg):
    if a.val_split in ("test_module", "test_sealed"):
        raise SystemExit("test shards are never a training-time validation split")
    if a.kind in ("pilot", "production") and a.val_split != "val":
        raise SystemExit(f"{a.kind} validates on 'val' only (got {a.val_split!r})")
    if a.kind == "production":
        if cfg["status"] != "frozen":
            raise SystemExit("production needs configs/train_v1.json status 'frozen'")
        if a.epochs is not None or a.limit_train or a.sync_timing or a.allow_dirty or a.batch_size is not None:
            raise SystemExit("production takes no --epochs / --limit-train / --sync-timing / --allow-dirty / "
                             "--batch-size")
        if a.seed not in rc.allowed_seeds(cfg, a.model):
            raise SystemExit(f"seed {a.seed} not in the config's seeds for {a.model}: {rc.allowed_seeds(cfg, a.model)}")
        return int(cfg["epochs"])
    if a.allow_dirty and a.kind != "smoke":
        raise SystemExit("--allow-dirty is for smoke runs only")
    if a.batch_size is not None and (a.kind != "smoke" or a.batch_size <= 0):
        raise SystemExit("--batch-size is a positive override for smoke runs only")
    if a.limit_train and a.kind != "smoke":
        raise SystemExit("--limit-train is for smoke runs only")
    if a.limit_train < 0:
        raise SystemExit("--limit-train must be >= 0")
    if a.kind == "pilot":
        ep = a.epochs if a.epochs is not None else cfg["epochs"]
        if ep is None:
            raise SystemExit("pilot needs --epochs while the config's epochs is null")
        return _positive_epochs(int(ep))
    return _positive_epochs(int(a.epochs if a.epochs is not None else 1))


def _loader(ds, sampler, cfg, a, gen):
    nw = int(a.num_workers)
    return DataLoader(ds, batch_size=int(cfg["batch_size"]), sampler=sampler, collate_fn=collate,
                      num_workers=nw, worker_init_fn=worker_init if nw > 0 else None, generator=gen,
                      drop_last=bool(cfg["drop_last_train"]), pin_memory=not a.no_pin_memory and nw > 0,
                      persistent_workers=nw > 0, prefetch_factor=int(a.prefetch_factor) if nw > 0 else None)


def main(argv=None, *, splits=None):
    """splits: None in production (configs/splits.json, hash-checked by open_cache); a dict for tests.
    The run-directory lock is released in `finally` on success, exceptions and early exits alike."""
    lock = {"fd": None}
    try:
        return _main(argv, splits, lock)
    except SystemExit as e:                            # refusals after the log started go into it too
        if lock.get("console") and isinstance(e.code, str):
            lock["console"][0].write(f"SystemExit: {e.code}\n")
        raise
    except BaseException:                              # crash cause into console.log before it closes
        if lock.get("console"):
            traceback.print_exc(file=lock["console"][0])
        raise
    finally:
        _stop_console_log(lock)
        if lock["fd"] is not None:
            release_run_lock(lock["fd"])


def _main(argv, splits, lock):
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    a = parse_args(argv)
    cfg, cfg_path = rc.load_train_cfg(a.config)
    epochs = check_kind(a, cfg)
    if a.batch_size is not None:                       # smoke only (check_kind); recorded via run / args
        cfg = {**cfg, "batch_size": int(a.batch_size)}
    git = rc.require_clean_tree(a.allow_dirty)
    paths = load_paths()
    cache_dir = Path(a.cache_dir or paths["cache_dir"]).expanduser()
    out = refuse_inside(a.out, paths["data_root"], cache_dir)
    if a.resume:
        if not (out / "run_config.json").is_file():
            raise SystemExit(f"--resume: {out}/run_config.json not found")
        lock["fd"] = acquire_run_lock(out)            # held through recovery and every epoch
        _start_console_log(out, lock)                 # appended on every resume
    else:
        refuse_existing([out], overwrite=False)
    precision = rc.setup_precision(cfg)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but no CUDA device")
    execution = rc.device_record(device)
    labels_cfg, data_cfg = load_labels_cfg(), load_data_cfg()

    print("verifying the cache against the manifest (full re-hash) ...", flush=True)
    cache_rec = rc.verify_cache(cache_dir, a.manifest, {"train", a.val_split}, a.hash_workers)
    print(f"  identical: {cache_rec['n_files']} files, {cache_rec['total_bytes'] / 1e9:.2f} GB, "
          f"{cache_rec['seconds']} s", flush=True)
    wnorm = rc.check_w_norm(cache_dir, labels_cfg, splits=splits)
    env = rc.env_record()
    rc.require_versions(env)                           # fresh runs and resumes alike
    ek = {**rc.env_key(env), "execution": execution}  # a device change breaks exact continuation
    holes = rc.effective_holes(cfg, a.model)

    # ---------------------------------------------------------------- data
    train_ds = GraspDataset(cache_dir, "train", a.model, mode="train", seed=a.seed, holes=holes,
                            flag_setting=cfg["flag_setting_train"], splits=splits)
    train_slots = (seeded_slots(train_ds.slots, a.limit_train, a.seed) if a.limit_train
                   else [int(s) for s in train_ds.slots])
    sampler = EpochSampler(train_slots, shuffle=True, seed=a.seed, strict=True)
    gens = {"train_loader": rc.loader_generator(a.seed, 0), "val_loader": rc.loader_generator(a.seed, 1)}
    train_loader = _loader(train_ds, sampler, cfg, a, gens["train_loader"])
    val_sets = {}
    for cond in cfg["val_conditions"]:
        if a.model == "Astar" and cond != "none":
            continue                                   # Astar: no synthetic holes (D4)
        vds = GraspDataset(cache_dir, a.val_split, a.model, mode="eval", holes=cond, splits=splits)
        vl = _loader(vds, EpochSampler(vds.slots, shuffle=False, seed=0), cfg, a, gens["val_loader"])
        val_sets[cond] = (vds, vl)
    n_val = len(next(iter(val_sets.values()))[0])
    spe = len(train_loader)
    horizon = epochs * spe

    # ---------------------------------------------------------------- model
    model = build_model(a.model, a.seed, cfg["model"]["c"], cfg["model"]["n_res"]).to(device)
    o = cfg["optim"]
    opt = torch.optim.Adam(model.parameters(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                           weight_decay=o["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=horizon, eta_min=cfg["schedule"]["eta_min"])

    run = {"kind": a.kind, "model": a.model, "seed": a.seed, "holes": holes,
           "flag_setting": cfg["flag_setting_train"], "epochs": epochs, "batch_size": int(cfg["batch_size"]),
           "steps_per_epoch": spe, "horizon": horizon, "n_train": len(train_slots),
           "train_slots_sha256": hashlib.sha256(np.asarray(train_slots, np.int64).tobytes()).hexdigest(),
           "limit_train": int(a.limit_train), "val_split": a.val_split, "val_conditions": sorted(val_sets),
           "n_val": n_val, "config_sha256": rc.file_sha256(cfg_path),
           "labels_sha256": rc.file_sha256(REPO / "configs" / "labels.json"),
           "train_data_sha256": rc.file_sha256(REPO / "configs" / "train_data.json"),
           "frozen_sha256": rc.file_sha256(REPO / "configs" / "frozen_day1.json"),
           "git_commit": git["git_commit"], "manifest_sha256": cache_rec["manifest_sha256"]}
    continuation_exact = True
    start, global_step = 0, 0
    if not a.resume:
        run_id = uuid.uuid4().hex
        out.mkdir(parents=True)                       # no exist_ok: a concurrent fresh run fails here
        lock["fd"] = acquire_run_lock(out)
        _start_console_log(out, lock)                 # lines before this point (cache re-hash) are in run_config
        run_cfg = {"run_id": run_id, **run, "git_dirty": git["git_dirty"], "config_path": str(cfg_path),
                   "config": cfg, "n_params": n_params(model), "precision": precision, "w_norm_check": wnorm,
                   "cache_verify": cache_rec, "env": env, "env_key": ek, "execution": execution,
                   "budget_valid": not a.sync_timing, "cache_dir": str(cache_dir), "paths": paths,
                   "paths_source": paths_source(), "args": vars(a), "started": started,
                   "torch_threads": torch.get_num_threads()}
        atomic_write_json(out / "run_config.json", run_cfg)
    else:
        old = json.loads((out / "run_config.json").read_text())
        run_id = old["run_id"]
        diff = {k: (old.get(k), run[k]) for k in IMMUTABLE if not _same_setting(k, old.get(k), run[k])}
        if diff:
            raise SystemExit(f"--resume refused: immutable settings differ (old, now): {diff}")
        rp = out / "recovery.json"
        hist = json.loads(rp.read_text()) if rp.exists() else []
        prior = [h.get("continuation_exact", True) for h in hist] + \
                [r.get("continuation_exact", True) for r in read_log(out)[0]]
        continuation_exact = all(prior)                # once False, False for every later epoch
        if ek != old["env_key"]:
            if not a.allow_env_change:
                raise SystemExit(f"--resume refused: environment differs from the run's record "
                                 f"(pass --allow-env-change to continue without the exactness guarantee)\n"
                                 f"old {old['env_key']}\nnow {ek}")
            continuation_exact = False
        if a.sync_timing != (not old["budget_valid"]):
            raise SystemExit("--resume: --sync-timing must match the original run")
        k, rep = recover(out, run_id, {m: run[m] for m in META_KEYS})
        if k >= 0:
            records, _ = read_log(out)
            st = load_checkpoint(ckpt_path(out, k), records[k]["ckpt_sha256"])
            restore_state(st, model, opt, sched, gens)
            global_step = int(st["meta"]["global_step"])
            start = k + 1
        rep.update({"resumed_at_epoch": start, "cache_verify": cache_rec, "env_key": ek,
                    "continuation_exact": continuation_exact, "git_dirty": git["git_dirty"]})
        atomic_write_json(rp, hist + [rep])
        print(f"resume: last committed epoch {k}, continuing at {start}; archived {len(rep['archived'])} file(s)")

    print(f"{a.kind} {a.model} seed {a.seed}: {len(train_slots)} train slots, {spe} steps/epoch, "
          f"{epochs} epochs, val {n_val} x {sorted(val_sets)}, {n_params(model):,} params", flush=True)

    # ---------------------------------------------------------------- epochs
    last, prev_log_commit = None, None
    for e in range(start, epochs):
        t_epoch = time.perf_counter()                  # whole epoch: train, val, CSV, checkpoint
        sampler.set_epoch(e)
        tr = train_epoch(model, train_loader, opt, sched, device, cfg, sync_timing=a.sync_timing,
                         progress_every=a.progress_every)
        global_step += tr["steps"]
        if global_step != (e + 1) * spe:
            raise AssertionError(f"global_step {global_step} != (epoch + 1) * steps_per_epoch {(e + 1) * spe}")
        _crash_hook("after_train", e)
        sync(device)
        t = time.perf_counter()
        rows, val = [], {}
        for cond, (vds, vl) in val_sets.items():
            r, s = run_val(model, vl, vds.slots, labels_cfg, device, cond)
            rows += r
            val[cond] = s
        sync(device)
        val_s = time.perf_counter() - t
        t = time.perf_counter()
        csv_sha = atomic_write_bytes(csv_path(out, e), rows_to_csv_bytes(rows))
        csv_s = time.perf_counter() - t
        _crash_hook("after_csv", e)
        t = time.perf_counter()
        meta = {"run_id": run_id, "epoch": e, "global_step": global_step, "csv_sha256": csv_sha,
                **{m: run[m] for m in META_KEYS}}
        ck_sha = save_checkpoint(ckpt_path(out, e), capture_state(model, opt, sched, gens, meta))
        ckpt_s = time.perf_counter() - t
        _crash_hook("after_ckpt", e)
        # everything up to the log append; the append itself (one fsync) is reported in the NEXT record
        wall = time.perf_counter() - t_epoch
        proj = (wall + (prev_log_commit or 0.0)) * epochs
        rec = {"run_id": run_id, "epoch": e, "global_step": global_step, "ckpt_sha256": ck_sha,
               "csv_sha256": csv_sha, "lr_end": tr["lr_end"], "loss": tr["loss"],
               "train_holes": {"hole_frac_mean": tr["hole_frac_mean"],
                               "covered_lie_frac_mean": tr["covered_lie_frac_mean"],
                               "per_severity": tr["per_severity"]},
               "val": val, "time": {"train_s": tr["train_s"], "loader_wait_s_approx": tr["loader_wait_s"],
                                    "val_s": val_s, "csv_s": csv_s, "ckpt_s": ckpt_s, "epoch_wall_s": wall,
                                    "prev_log_commit_s": prev_log_commit,
                                    "samples_per_s": tr["samples_per_s"]},
               "budget_valid": not a.sync_timing, "continuation_exact": continuation_exact,
               "projected_run_s": proj, "projection_includes_log_commit": prev_log_commit is not None}
        t = time.perf_counter()
        append_log(out, rec)
        prev_log_commit = time.perf_counter() - t
        _crash_hook("after_log", e)
        v0 = val.get("none", next(iter(val.values())))
        note = "" if not a.sync_timing else " (sync-timing: NOT a budget number)"
        print(f"epoch {e}: monitor loss (Q pooled over all pixels) {tr['loss']['total']:.5f} | val {v0['condition']} primary "
              f"{v0['top1_primary']:.4f} strict {v0['top1_strict']:.4f} invalid {v0['invalid_decodes']} | "
              f"train {tr['train_s']:.0f}s (wait {tr['loader_wait_s']:.0f}s) val {val_s:.0f}s "
              f"csv {csv_s:.1f}s ckpt {ckpt_s:.1f}s | epoch {wall:.0f}s, projected {proj / 60:.1f} min{note}",
              flush=True)
        if proj > BUDGET_S and not a.sync_timing:
            print(f"WARNING: projected {proj / 60:.0f} min exceeds the 90-min budget (SPEC_trajectory go/no-go)",
                  flush=True)
        last = rec
    if last is None:                                   # every epoch was already committed
        records, _ = read_log(out)
        if len(records) != epochs:
            raise AssertionError(f"{len(records)} committed epochs, expected {epochs}")
        last = records[-1]
        fp = out / "final.json"                        # keep the measurement from the run that did it
        if prev_log_commit is None and fp.exists():
            prev_log_commit = json.loads(fp.read_text()).get("last_log_commit_s")
    final = {"run_id": run_id, "completed_epochs": epochs, "selection": cfg["selection"],
             "final_checkpoint": str(ckpt_path(out, epochs - 1)), "final_val": last["val"],
             "last_log_commit_s": prev_log_commit, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_write_json(out / "final.json", final)
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
