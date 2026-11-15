#!/usr/bin/env python3
"""Validation pilot analysis: config loss vs SQ, and the training-budget check (proj_v4 section 5.13,
pre-registered before any pilot run). Reads finished runs only; trains nothing.

Inputs: three train.py pilot runs, A0 seed 1, batch 32, same commit / data / train slots:
  --config-arm  config loss (configs/train_v1.json, q_reduction "all"), E epochs
  --sq-arm      SQ (configs/train_v1_sq.json, q_reduction "split"), E epochs
  --sq-2e       SQ, 2E epochs (optional until it exists; the budget decision is then "pending")
Each run's FINAL epoch CSV (val/epoch_<E-1>.csv), clean condition "none", paired by slot.

Statistic: primary top-1 accuracy difference in percentage points, paired by scene. Uncertainty:
object-level bootstrap (SPEC_trajectory "Statistics": independence unit = object, the scenes of one object
resampled together, repeat-draw multiplicities kept), 10,000 resamples, percentile 95 % CI. ONE seed, so
one level: the interval omits seed-to-seed variation, which the spec says dominates. The decisions are built
so that this mainly risks a safe "stop for a manual decision" (see below).

Decisions (pre-registered):
  loss    d = config - SQ. CONFIG_WINS (-> manual decision) iff d > 0 and the CI excludes 0; otherwise SQ
          (ties included: the diagnosis showed the config loss does not fit Q).
  budget  d2 = SQ(2E) - SQ(E). BUDGET_TOO_SHORT (-> manual decision: longer runs vs fewer seeds) iff
          d2 >= 1 pp and the CI excludes 0; otherwise E. (1 pp is below the spec's ~3-point "practical"
          margin on purpose: it only triggers a human decision. Missing a real shortfall smaller than the
          seed noise is the one unsafe direction; recorded as a limitation.)
  narrow  among SQ's primary misses, the share whose picked cell has a target opening < 4 px at 224 (R6
          class). Void-cell picks have no target opening: counted as not narrow, reported separately.
          NARROW_DOMINATES (-> manual decision before production) iff the share > 0.5.
Reported, never deciding: the strict metric (same bootstrap), misses by opening bin (< 4, 4-8, >= 8 px at
224, void) x category, invalid decodes, the exact-target baseline (tt_match), per-arm accuracies.

Usage:
  python scripts/pilot_compare.py --config-arm RUNS/pilot_A0_cfg --sq-arm RUNS/pilot_A0_sq \
      [--sq-2e RUNS/pilot_A0_sq_2e] --cache-dir CACHE --out RUNS/pilot_compare
"""
import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.provenance import git_fields, refuse_existing  # noqa: E402
from training.checkpoint import atomic_write_json, csv_path, read_log  # noqa: E402
from training.runcfg import Q_REDUCTION_TEXT  # noqa: E402

METRICS = ("primary", "strict")
RESAMPLES = 10_000
BOOT_SEED = 20261114
BUDGET_PP = 1.0
NARROW_PX = 4.0
NARROW_SHARE = 0.5
BINS = ("<4", "4-8", ">=8", "void")
PAIRED = ("model", "seed", "batch_size", "n_train", "train_slots_sha256", "val_split", "git_commit",
          "manifest_sha256", "labels_sha256", "train_data_sha256", "holes", "flag_setting", "kind")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-arm", required=True)
    p.add_argument("--sq-arm", required=True)
    p.add_argument("--sq-2e", default=None)
    p.add_argument("--cache-dir", required=True, help="for <cache>/<val_split>/index.csv (slot -> object)")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


# ------------------------------------------------------------------ pure pieces (tested)
def _b(s):
    s = "" if s is None else str(s).strip()
    if s in ("True", "False"):
        return s == "True"
    if s == "":
        return None
    raise ValueError(f"not a boolean field: {s!r}")


def read_val_csv(path, condition="none"):
    """-> {slot: row} for one condition; refuses duplicate slots."""
    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["condition"] != condition:
                continue
            s = int(r["slot"])
            if s in rows:
                raise ValueError(f"{path}: duplicate slot {s}")
            rows[s] = r
    if not rows:
        raise ValueError(f"{path}: no rows for condition {condition!r}")
    return rows


def object_ids(index_csv, slots):
    """slot -> object key (part/object_id) from the cache index; every slot must be present."""
    m = {}
    with open(index_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            m[int(r["slot"])] = f"{r['part']}/{r['object_id']}"
    missing = sorted(set(slots) - set(m))
    if missing:
        raise ValueError(f"{len(missing)} slots missing from {index_csv}, e.g. {missing[:5]}")
    return {s: m[s] for s in slots}


def object_bootstrap(a, b, objects, resamples=RESAMPLES, seed=BOOT_SEED):
    """a, b: 0/1 arrays per scene (paired); objects: object key per scene. Difference of accuracies (a - b)
    in percentage points with an object-level percentile 95 % CI (multiplicities kept). -> dict."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(objects) != len(a) or not len(a):
        raise ValueError("a, b and objects must be equal-length, non-empty 1-D")
    keys, inv = np.unique(np.asarray(objects), return_inverse=True)
    n_obj = len(keys)
    sa, sb = np.bincount(inv, a, n_obj), np.bincount(inv, b, n_obj)
    n = np.bincount(inv, minlength=n_obj).astype(float)
    point = 100.0 * (sa.sum() - sb.sum()) / n.sum()
    rng = np.random.default_rng(seed)
    w = rng.multinomial(n_obj, np.full(n_obj, 1.0 / n_obj), size=resamples).astype(float)
    reps = 100.0 * (w @ (sa - sb)) / (w @ n)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"diff_pp": float(point), "ci95": [float(lo), float(hi)], "n_scenes": int(len(a)), "n_objects": int(n_obj),
            "acc_a": float(100.0 * sa.sum() / n.sum()), "acc_b": float(100.0 * sb.sum() / n.sum()),
            "resamples": int(resamples), "seed": int(seed)}


def decide_loss(boot):
    d, (lo, _) = boot["diff_pp"], boot["ci95"]
    return "CONFIG_WINS" if (d > 0 and lo > 0) else "SQ"


def decide_budget(boot):
    if boot is None:
        return "PENDING"
    d, (lo, _) = boot["diff_pp"], boot["ci95"]
    return "BUDGET_TOO_SHORT" if (d >= BUDGET_PP and lo > 0) else "E"


def opening_bin(row):
    if _b(row["target_void"]):
        return "void"
    o = float(row["target_opening_224"])
    return "<4" if o < NARROW_PX else ("4-8" if o < 8.0 else ">=8")


def miss_table(rows, metric):
    """{bin: {category: count}} over the misses of one metric, plus totals."""
    t = {b: {} for b in BINS}
    for r in rows.values():
        if _b(r[f"match_{metric}"]):
            continue
        c = r[f"cat_{metric}"] or "?"
        b = opening_bin(r)
        t[b][c] = t[b].get(c, 0) + 1
    return {"by_bin": t, "misses": sum(sum(v.values()) for v in t.values())}


def narrow_share(rows, metric="primary"):
    tab = miss_table(rows, metric)
    n = tab["misses"]
    narrow = sum(tab["by_bin"]["<4"].values())
    void = sum(tab["by_bin"]["void"].values())
    share = narrow / n if n else 0.0
    return {"misses": n, "narrow": narrow, "void_picks": void, "share": share,
            "decision": "NARROW_DOMINATES" if share > NARROW_SHARE else "OK"}


def check_runs(cfg_arm, sq_arm, sq_2e):
    """Each arg: (run_config dict, final dict). Refuses unpaired or mislabelled runs."""
    def qr(rc_):
        return Q_REDUCTION_TEXT[rc_["config"]["loss"]["q_reduction"]]
    (c, fc), (s, fs) = cfg_arm, sq_arm
    if qr(c) != "all" or qr(s) != "split":
        raise ValueError(f"arms mislabelled: config arm q_reduction {qr(c)!r}, SQ arm {qr(s)!r}")
    if c.get("kind") != "pilot":
        raise ValueError("arms must be pilot runs")
    for k in PAIRED:
        if c.get(k) != s.get(k):
            raise ValueError(f"config and SQ arms differ in {k}: {c.get(k)!r} vs {s.get(k)!r}")
    if c["epochs"] != s["epochs"]:
        raise ValueError("config and SQ arms need equal epochs (equal exposure)")
    cc, sc = copy.deepcopy(c["config"]), copy.deepcopy(s["config"])
    cc["loss"].pop("q_reduction", None)
    sc["loss"].pop("q_reduction", None)
    if cc != sc:
        raise ValueError("the arms' recorded configs differ beyond loss.q_reduction")
    for name, (r, f) in (("config", cfg_arm), ("SQ", sq_arm)) + ((("SQ-2E", sq_2e),) if sq_2e else ()):
        if f.get("completed_epochs") != r["epochs"]:
            raise ValueError(f"{name} arm not finished: {f.get('completed_epochs')} of {r['epochs']} epochs")
    if sq_2e:
        (t, _) = sq_2e
        if qr(t) != "split":
            raise ValueError("SQ-2E run is not SQ")
        for k in PAIRED:
            if t.get(k) != s.get(k):
                raise ValueError(f"SQ and SQ-2E differ in {k}")
        if t["epochs"] != 2 * s["epochs"]:
            raise ValueError(f"SQ-2E has {t['epochs']} epochs, expected {2 * s['epochs']}")
        if t["config"] != s["config"]:
            raise ValueError("SQ-2E's recorded config differs from the SQ arm's")


def check_csv_sha(path, expected):
    """The final CSV must be exactly the one the run log committed."""
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != expected:
        raise ValueError(f"{path}: sha256 {got[:12]} != run log csv_sha256 {str(expected)[:12]}")


# ------------------------------------------------------------------ main
def _load(d):
    d = Path(d).expanduser()
    rcfg = json.loads((d / "run_config.json").read_text())
    fin = json.loads((d / "final.json").read_text())
    return rcfg, fin, d


def main(argv=None):
    a = parse_args(argv)
    cfg_arm, sq_arm = _load(a.config_arm), _load(a.sq_arm)
    sq_2e = _load(a.sq_2e) if a.sq_2e else None
    try:
        check_runs(cfg_arm[:2], sq_arm[:2], sq_2e[:2] if sq_2e else None)
        for r in (cfg_arm, sq_arm) + ((sq_2e,) if sq_2e else ()):
            e_last = r[0]["epochs"] - 1
            records, _ = read_log(r[2])
            rec = next((x for x in records if x["epoch"] == e_last), None)
            if rec is None:
                raise ValueError(f"{r[2]}: no committed log record for epoch {e_last}")
            check_csv_sha(csv_path(r[2], e_last), rec["csv_sha256"])
    except ValueError as e:
        raise SystemExit(f"pilot_compare refused: {e}")
    rows = {n: read_val_csv(csv_path(r[2], r[0]["epochs"] - 1))
            for n, r in (("config", cfg_arm), ("sq", sq_arm)) + ((("sq2e", sq_2e),) if sq_2e else ())}
    slots = sorted(rows["sq"])
    if any(sorted(v) != slots for v in rows.values()):
        raise SystemExit("pilot_compare refused: the runs' validation slots differ")
    objs = object_ids(Path(a.cache_dir).expanduser() / sq_arm[0]["val_split"] / "index.csv", slots)
    ob = [objs[s] for s in slots]
    res = {"boot": {}, "accuracy": {}}
    for m in METRICS:
        hit = {n: [int(_b(v[s][f"match_{m}"])) for s in slots] for n, v in rows.items()}
        res["boot"][f"loss_{m}"] = object_bootstrap(hit["config"], hit["sq"], ob)
        if sq_2e:
            res["boot"][f"budget_{m}"] = object_bootstrap(hit["sq2e"], hit["sq"], ob)
        res["accuracy"][m] = {n: 100.0 * sum(h) / len(h) for n, h in hit.items()}
        res["accuracy"][f"tt_{m}"] = 100.0 * sum(bool(_b(rows["sq"][s][f"tt_match_{m}"])) for s in slots) / len(slots)
    res["decisions"] = {"loss": decide_loss(res["boot"]["loss_primary"]),
                        "budget": decide_budget(res["boot"].get("budget_primary")),
                        "narrow": narrow_share(rows["sq"], "primary")}
    res["miss_tables"] = {n: {m: miss_table(v, m) for m in METRICS} for n, v in rows.items()}
    res["invalid_decodes"] = {n: sum(not _b(r["valid"]) for r in v.values()) for n, v in rows.items()}
    res["limitations"] = ["one seed: CIs are object-level only and omit seed variation (SPEC: seeds dominate)"]
    res["runs"] = {n: str(r[2]) for n, r in (("config", cfg_arm), ("sq", sq_arm)) + ((("sq2e", sq_2e),) if sq_2e else ())}
    out = Path(a.out).expanduser()
    refuse_existing([out], overwrite=False)
    out.mkdir(parents=True)
    atomic_write_json(out / "pilot_compare.json", {**res, **git_fields()})
    lp, d = res["boot"]["loss_primary"], res["decisions"]
    print(f"primary: config {lp['acc_a']:.2f} %  SQ {lp['acc_b']:.2f} %  diff {lp['diff_pp']:+.2f} pp "
          f"[{lp['ci95'][0]:+.2f}, {lp['ci95'][1]:+.2f}]  ({lp['n_scenes']} scenes, {lp['n_objects']} objects)")
    if sq_2e:
        bp = res["boot"]["budget_primary"]
        print(f"budget:  SQ 2E - SQ E {bp['diff_pp']:+.2f} pp [{bp['ci95'][0]:+.2f}, {bp['ci95'][1]:+.2f}]")
    print(f"DECISIONS: loss {d['loss']} | budget {d['budget']} | narrow {d['narrow']['decision']} "
          f"(share {d['narrow']['share']:.2f} of {d['narrow']['misses']} SQ misses; void picks {d['narrow']['void_picks']})")
    print(f"strict (reported only): {res['boot']['loss_strict']['diff_pp']:+.2f} pp "
          f"{res['boot']['loss_strict']['ci95']}; invalid decodes {res['invalid_decodes']}")
    print(f"report: {out / 'pilot_compare.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
