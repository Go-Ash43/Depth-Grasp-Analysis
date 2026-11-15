#!/usr/bin/env python3
"""H1 harness on the validation evaluation (P-E1 rev 4; docs/proj_v7.md sections 13-15; SPEC_hypotheses "H1";
SPEC_trajectory "Statistics, adapted to Jacquard", "Exit gate G1"). CPU only, reads files, trains nothing.

Inputs, all bound by sha256 in h1.json: selected_runs.json (scripts/m1_runs.py), the val evaluation
(scripts/eval_m1.py: <eval-dir>/results with eval_record.json, nine runs x four conditions, CSV + sidecar),
the LOCAL metric decision (eval.metric_policy.load_decision, recheck on; it must be the record m1_runs.py
bound), <cache>/val/index.csv, and the current configs (frozen, hashes equal to the runs').

Checks: exactly nine (model, seed) ids and the four conditions; every CSV and sidecar covers exactly the
cache's eligible slots (index status ok), each once; per slot and condition the corruption identity
(n_synth_cells, m_synth_sha256, flag_sha256, severity, hole_frac, covered_lie_frac) is identical across all
nine runs, and the CSV's severity / hole_frac / covered_lie_frac equal the sidecar's. Objects =
(part, object_id) as scripts/pilot_compare.object_ids.

Primary (pre-registered): M - A0 top-1 on mid (injected holes, truthful flags), metric from the decision
record, two-level crossed bootstrap (eval.h1), 95 % and 90 % percentile intervals and the independent
reading flags. Everything else is reported with intervals and NO verdict:
  M - A0 on low, high, none (natural stereo / no injected holes, all eligible scenes) and on the named subset
  "natural holes on the object" (none, m_sensor_obj_224 > 0); R / A0 / M accuracy per condition; each seed
  alone (object bootstrap); invalid decodes; covered_lie_frac on mid and the lie strata (eval.h1.lie_strata).
G1: clean A0 (none) per seed and pooled. Below about 50 % fails G1; above 95 % is the promotion branch,
which only the registered metric policy decides (production A0 seed 1 final logged val), never this report.

--split val (default) or test_module. The index is <cache>/<split>/index.csv and must be byte-identical to
that file's entry in the manifest the evaluation verified. test_module: the one-look ledger must be
"completed" with identities and outputs equal to the results dir. G1 is gated on val only; on test_module
clean A0 is reported, never gated.

Usage:
  python scripts/h1_harness.py --selected $R/m1_selected/selected_runs.json --eval-dir $R/m1_eval_val \
      --metric-local $R/metric_decision_local/metric_decision.json --out $R/m1_h1_val
  python scripts/h1_harness.py --split test_module --eval-dir $R/m1_eval_test ... --out $R/m1_h1_test
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.paths import load_paths  # noqa: E402
from data.provenance import git_fields  # noqa: E402
from eval import h1  # noqa: E402
from eval import metric_policy as mp  # noqa: E402
from scripts.eval_m1 import CONDITIONS, SIDE_FIELDS, load_selected, parse_rows  # noqa: E402
from scripts.m1_runs import sha256  # noqa: E402
from scripts.pilot_compare import miss_table  # noqa: E402
from training.checkpoint import atomic_write_json, atomic_write_bytes  # noqa: E402

MODELS, SEEDS = ("R", "A0", "M"), (1, 2, 3)
PRIMARY_CONDITION = "mid"
IDENTITY = ("n_synth_cells", "m_synth_sha256", "flag_sha256", "severity", "hole_frac", "covered_lie_frac")


class HarnessError(RuntimeError):
    pass


def _need(ok, msg):
    if not ok:
        raise HarnessError(msg)


def verified_index(rec, cache_dir, split):
    """Review fix 4: <cache>/<split>/index.csv must be byte-identical to the entry of the manifest the
    evaluation verified the cache against (cache_verify), and that manifest must be the one recorded."""
    cv = rec["cache_verify"]
    man = Path(cv["manifest"])
    _need(man.is_file() and sha256(man) == cv["manifest_sha256"] == rec["identities"]["manifest_sha256"],
          f"{man}: manifest missing or not the one the evaluation verified")
    entries = {f["path"]: f["sha256"] for f in json.loads(man.read_text())["files"]}
    rel = f"{split}/index.csv"
    _need(rel in entries, f"{man} has no entry {rel}")
    index = Path(cache_dir).expanduser() / rel
    _need(sha256(index) == entries[rel], f"{index}: sha256 differs from the verified manifest entry")
    return index


def read_index(path):
    """-> sorted eligible slots, {slot: object key}, {slot: m_sensor_obj_224}."""
    objs, msens = {}, {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] == "ok":
                s = int(r["slot"])
                objs[s] = f"{r['part']}/{r['object_id']}"
                msens[s] = int(r["m_sensor_obj_224"])
    _need(objs, f"{path}: no eligible slots")
    return sorted(objs), objs, msens


def load_eval(res, files, slots):
    """-> csv[tag][cond] {slot: row}, side[tag][cond] {slot: row}; completeness checked."""
    want = sorted(f"{m}_s{s}/{c}.{k}" for m in MODELS for s in SEEDS for c in CONDITIONS for k in ("csv", "sidecar.csv"))
    have = sorted(k for k in files if k != "eval_record.json")
    _need(have == want, f"evaluation files differ from nine runs x four conditions: "
                        f"missing {sorted(set(want) - set(have))[:4]}, extra {sorted(set(have) - set(want))[:4]}")
    rows, side = {}, {}
    for m in MODELS:
        for s in SEEDS:
            tag = f"{m}_s{s}"
            for c in CONDITIONS:
                b = (res / tag / f"{c}.csv").read_bytes()
                raw = list(csv.DictReader(b.decode().splitlines()))
                _need(len(raw) == len(slots), f"{tag}/{c}: {len(raw)} rows, {len(slots)} eligible slots")
                r = parse_rows(b, c)
                _need(sorted(r) == slots, f"{tag}/{c}: slots differ from the eligible set")
                sd = list(csv.DictReader((res / tag / f"{c}.sidecar.csv").read_text().splitlines()))
                _need(list(sd[0].keys()) == list(SIDE_FIELDS), f"{tag}/{c}: sidecar header")
                sdm = {int(x["slot"]): x for x in sd}
                _need(len(sd) == len(slots) and sorted(sdm) == slots, f"{tag}/{c}: sidecar slots differ")
                for sl in slots:
                    for k in ("severity", "hole_frac", "covered_lie_frac"):
                        _need(r[sl][k] == sdm[sl][k], f"{tag}/{c} slot {sl}: CSV {k} != sidecar")
                rows.setdefault(tag, {})[c] = r
                side.setdefault(tag, {})[c] = sdm
    for c in CONDITIONS:                               # identical corruption instances across all nine
        ref = side["A0_s1"][c]
        for tag in side:
            for sl in slots:
                _need(tuple(side[tag][c][sl][k] for k in IDENTITY) == tuple(ref[sl][k] for k in IDENTITY),
                      f"{c} slot {sl}: corruption identity of {tag} differs from A0_s1")
    return rows, side


def hits(rows, model, cond, slots, metric):
    return np.array([[rows[f"{model}_s{s}"][cond][sl][f"match_{metric}"] == "True" for sl in slots] for s in SEEDS],
                    float)


def analyse(rows, side, slots, objects, msens, metric, resamples, boot_seed, gate_g1=True):
    obj = [objects[s] for s in slots]
    H = {m: {c: hits(rows, m, c, slots, metric) for c in CONDITIONS} for m in MODELS}
    out = {"metric": metric, "primary": {"contrast": "M - A0", "condition": PRIMARY_CONDITION,
                                         **h1.contrast(H["M"][PRIMARY_CONDITION], H["A0"][PRIMARY_CONDITION], obj,
                                                       resamples, boot_seed)}}
    sec = {}
    for c in CONDITIONS:
        if c != PRIMARY_CONDITION:
            sec[f"M-A0 {c}"] = h1.crossed_bootstrap(H["M"][c], H["A0"][c], obj, resamples, boot_seed)
    nat = np.array([msens[s] > 0 for s in slots])
    sec["M-A0 none, natural holes on the object (m_sensor_obj_224 > 0)"] = (
        h1.crossed_bootstrap(H["M"]["none"][:, nat], H["A0"]["none"][:, nat], list(np.array(obj, object)[nat]),
                             resamples, boot_seed) if nat.sum() and len(set(np.array(obj, object)[nat])) > 1
        else {"estimate": "NA", "n_scenes": int(nat.sum())})
    out["secondary_no_verdict"] = sec
    out["accuracy"] = {f"{m} {c}": h1.crossed_bootstrap(H[m][c], None, obj, resamples, boot_seed)
                       for m in MODELS for c in CONDITIONS}
    out["per_seed_M-A0"] = {f"seed {s} {c}": h1.crossed_bootstrap(H["M"][c][i:i + 1], H["A0"][c][i:i + 1], obj,
                                                                    resamples, boot_seed)
                            for i, s in enumerate(SEEDS) for c in CONDITIONS}
    a0 = {}
    for mk in ("primary", "strict"):
        hk = hits(rows, "A0", "none", slots, mk)
        a0[mk] = {"pooled_pct": float(100 * hk.mean()), **{f"seed {s}": float(100 * hk[i].mean()) for i, s in enumerate(SEEDS)}}
    pooled = a0["primary"]["pooled_pct"]
    note = ("FAIL: clean A0 below ~50 %" if pooled < 50 else
            ("above 95 %: promotion branch; the metric is decided only by the registered policy" if pooled > 95
             else "inside ~50-95 %"))
    out["G1_clean_A0"] = {**a0, "gated": bool(gate_g1),
                          "passed": bool(pooled >= 50) if gate_g1 else None,   # > 95 % is the promotion branch, not a fail
                          "band_note": note if gate_g1 else "reported only (G1 is gated on validation)"}
    # Descriptive only (review round 2, finding 3): misses by target opening at the picked cell
    # (<4 px = R6 class, 4-8, >=8, void); evidence context for the item-10 waiver (proj_v6 7.4).
    out["misses_by_target_opening_no_verdict"] = {
        f"{m}_s{s} {c}": {mk: miss_table(rows[f"{m}_s{s}"][c], mk) for mk in ("primary", "strict")}
        for m in MODELS for s in SEEDS for c in CONDITIONS}
    out["invalid_decodes"] = {f"{m}_s{s} {c}": sum(r["valid"] == "False" for r in rows[f"{m}_s{s}"][c].values())
                              for m in MODELS for s in SEEDS for c in CONDITIONS}
    sd = side["A0_s1"][PRIMARY_CONDITION]
    n_syn = np.array([int(sd[s]["n_synth_cells"]) for s in slots])
    cov = np.array([float(sd[s]["covered_lie_frac"]) for s in slots])
    lab, cuts = h1.lie_strata(n_syn, cov)
    out["covered_lie_mid"] = {
        "mean_all_scenes": float(cov.mean()), "mean_holed_scenes": float(cov[n_syn > 0].mean()) if (n_syn > 0).any() else None,
        "n_no_hole": int((n_syn == 0).sum()), "tertile_cuts": cuts,
        "strata_M-A0_no_verdict": h1.stratified_contrast(H["M"][PRIMARY_CONDITION], H["A0"][PRIMARY_CONDITION],
                                                         obj, lab, resamples, boot_seed)}
    return out


def tables(res):
    p = res["primary"]
    r = p["reading"]
    name = {"val": "validation", "test_module": "module test (one look)"}.get(res.get("split"), res.get("split"))
    lines = [f"# H1 ({name}), metric: {res['metric']}", "",
             "| contrast | point (pp) | 95 % CI | 90 % CI | reading |", "|---|---|---|---|---|",
             f"| **M - A0, {p['condition']} (primary)** | {p['point_pp']:+.2f} | [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] | "
             f"[{p['ci90'][0]:+.2f}, {p['ci90'][1]:+.2f}] | {r['label']} |"]
    for k, v in res["secondary_no_verdict"].items():
        lines.append(f"| {k} | " + ("NA | | | |" if v.get("estimate") == "NA" else
                     f"{v['point_pp']:+.2f} | [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] | "
                     f"[{v['ci90'][0]:+.2f}, {v['ci90'][1]:+.2f}] | (no verdict) |"))
    lines += ["", "| model condition | accuracy % | 95 % CI |", "|---|---|---|"]
    for k, v in res["accuracy"].items():
        lines.append(f"| {k} | {v['acc_a_pct']:.2f} | [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] |")
    lines += ["", f"G1 clean A0: {json.dumps(res['G1_clean_A0'])}", ""]
    return "\n".join(lines) + "\n"


def parse_args(argv=None, allowed=("val", "test_module")):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected", required=True)
    p.add_argument("--split", default="val", choices=allowed)
    p.add_argument("--eval-dir", required=True, help="eval_m1.py --out of the evaluation of --split")
    p.add_argument("--metric-local", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--cache-dir", default=None, help="default: load_paths()['cache_dir']")
    p.add_argument("--out", required=True)
    p.add_argument("--resamples", type=int, default=h1.RESAMPLES)
    p.add_argument("--boot-seed", type=int, default=h1.BOOT_SEED)
    p.add_argument("--seed", type=int, default=0, help="unused; the bootstrap uses --boot-seed")
    return p.parse_args(argv)


def main(argv=None, *, allowed=("val", "test_module"), ledger_splits=("test_module",), g1_splits=("val",),
         _ledger_dir=None):
    a = parse_args(argv, allowed)
    try:
        return _main(a, ledger_splits, g1_splits, _ledger_dir)
    except (HarnessError, mp.PolicyError) as e:
        raise SystemExit(f"h1_harness refused: {e}")


def _main(a, ledger_splits, g1_splits, _ledger_dir):
    from scripts import eval_m1 as em
    from training import runcfg as rc
    sel_sha = sha256(a.selected)
    sel, _ = load_selected(a.selected, a.config or str(rc.DEFAULT_CONFIG))
    res_dir = Path(a.eval_dir).expanduser() / "results"
    rec = json.loads((res_dir / "eval_record.json").read_text())
    _need(rec["split"] == a.split, f"evaluation split {rec['split']!r} != --split {a.split!r}")
    _need(rec["selected_sha256"] == sel_sha, "evaluation was made from a different selected_runs.json")
    for rel, h in rec["files"].items():
        _need(sha256(res_dir / rel) == h, f"{rel}: sha256 differs from eval_record.json")
    if a.split in ledger_splits:                     # test: only a COMPLETED one-look run is analysed
        current = rc.require_clean_tree(False)       # review round 2, finding 1: frozen analysis code only
        _need(current["git_commit"] == rec["identities"]["analysis_commit"],
              "test analysis must use the frozen analysis commit (no commit between test run and analysis)")
        ld =Path(_ledger_dir) if _ledger_dir is not None else em.ledger_dir()
        L = json.loads((ld / em.LEDGER_NAME).read_text())
        _need(L["status"] == "completed", f"test ledger status {L['status']!r}; not completed")
        _need(L["identities"] == rec["identities"], "test ledger identities differ from the evaluation record")
        _need(L["outputs"] == {**rec["files"], "eval_record.json": sha256(res_dir / "eval_record.json")},
              "test ledger outputs differ from the results dir")
    dec = mp.load_decision(a.metric_local, recheck=True)
    md = sel.get("metric_decision") or {}
    _need(md.get("local_sha256") == sha256(a.metric_local), "metric record is not the one m1_runs.py bound")
    _need(dec["metric"] == md.get("metric"), "metric differs from the selection record")
    cache_dir = Path(a.cache_dir or load_paths()["cache_dir"]).expanduser()
    index = verified_index(rec, cache_dir, a.split)
    slots, objects, msens = read_index(index)
    rows, side = load_eval(res_dir, rec["files"], slots)
    out_dir = Path(a.out).expanduser()
    _need(not out_dir.exists(), f"{out_dir} exists; use a fresh --out")
    res = analyse(rows, side, slots, objects, msens, dec["metric"], a.resamples, a.boot_seed,
                  gate_g1=a.split in g1_splits)
    res["split"] = a.split
    res["inputs"] = {"selected": a.selected, "selected_sha256": sel_sha,
                     "eval_record_sha256": sha256(res_dir / "eval_record.json"),
                     "metric_local": a.metric_local, "metric_local_sha256": sha256(a.metric_local),
                     "index": str(index), "index_sha256": sha256(index), "m3": sel["m3"],
                     "config_sha256": sel["shared"]["config_sha256"]}
    res["made"] = {**git_fields(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a)}
    out_dir.mkdir(parents=True)
    atomic_write_json(out_dir / "h1.json", res)
    atomic_write_bytes(out_dir / "h1_tables.md", tables(res).encode())
    print(tables(res))
    print(f"written: {out_dir}")
    if res["G1_clean_A0"]["passed"] is False:        # review round 2, finding 2: a G1 failure stops a chain
        print("G1 FAIL: clean validation A0 below ~50 %", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
