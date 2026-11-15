#!/usr/bin/env python3
"""Module 1 run integrity + D7-1 selection -> selected_runs.json (P-E1 rev 4; docs/proj_v7.md sections 6, 13-15).

Replaces check_hybrid for decision D7-1 (the helper file is not edited). READ-ONLY on every run directory
(never training.checkpoint.recover, which archives / rewrites files). CPU, about a minute (re-hashes every
checkpoint and val CSV of every run).

Per run (check_run): run_config kind production, logical (model, seed), epochs, git_dirty false,
budget_valid; final.json complete with the same run_id and the last epoch's checkpoint; log.jsonl epochs
0..E-1, one run_id, no torn tail, budget_valid / continuation_exact true, global_step (e+1) x steps_per_epoch;
EVERY checkpoint and val CSV re-hashed against its log line; final checkpoint meta (run_id, epoch E-1,
global_step = horizon, csv_sha256, model, seed, holes, horizon, config_sha256) equal to log and run_config.

Across the nine selected runs: exactly {R, A0, M} x {1, 2, 3}; one value of every SHARED field; git_commit
starts with --commit; the CURRENT config / labels / train_data / frozen files hash to the recorded values
(labels / train_data / frozen: or are bridged to them, data.provenance.config_matches);
the config is frozen and its epochs equal the runs'.
Venue groups (env_key): pod = the six seed-1/2 runs; laptop = A0_s3, R_s3. Each group has one env_key.

D7-1 rule (pre-registered, docs/proj_v7.md section 6): the pod M_s3 is used if it passes
check_run AND has the pod group's env_key; otherwise the laptop M_s3 (which must pass and have the laptop
group's env_key). Never chosen by accuracy. The unused candidate is recorded as unused / incomplete /
rejected (with the reason) and never blocks selection. The logical id is separate from the directory, so
the pod M_s3 may live at M_s3_pod.

Metric decision (rev 3 b / rev 2 point 2): the LOCAL record (scripts/metric_decision.py run on the copied
A0_s1) is validated with eval.metric_policy.load_decision (recheck on); the copied POD record is read as
plain JSON only (never load_decision / comparable: both use its absolute pod paths) and its portable fields
must equal the local record's. The local record's run_id must be the selected A0 seed 1.

Usage (laptop, after the copy-back):
  python scripts/m1_runs.py --runs-root $R --commit 2b85aa6 --m3-pod $R/M_s3_pod --m3-laptop $R/M_s3 \
      --metric-local $R/metric_decision_local/metric_decision.json \
      --metric-pod $R/metric_decision_pod/metric_decision.json --out $R/m1_selected
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import training.runcfg as rc  # noqa: E402
from data.provenance import config_matches, git_fields  # noqa: E402
from eval import metric_policy as mp  # noqa: E402
from training.checkpoint import atomic_write_json, ckpt_path, csv_path, load_checkpoint, read_log  # noqa: E402

MODELS = ("R", "A0", "M")
SEEDS = (1, 2, 3)
POD_IDS = tuple((m, s) for s in (1, 2) for m in MODELS)
LAPTOP_IDS = (("A0", 3), ("R", 3))
SHARED = ("kind", "epochs", "batch_size", "steps_per_epoch", "horizon", "n_train", "train_slots_sha256",
          "limit_train", "holes", "flag_setting", "val_split", "val_conditions", "n_val", "config_sha256",
          "labels_sha256", "train_data_sha256", "frozen_sha256", "manifest_sha256", "git_commit")
CURRENT = {"labels_sha256": REPO / "configs" / "labels.json",
           "train_data_sha256": REPO / "configs" / "train_data.json",
           "frozen_sha256": REPO / "configs" / "frozen_day1.json"}
PORTABLE = ("decision_version", "policy", "metric", "promoted", "threshold_pct", "comparison", "clean")
PORTABLE_RUN = ("run_id", "kind", "model", "seed", "epochs", "git_commit", "config_sha256", "continuation_exact")


class RunError(ValueError):
    """A run directory, the run set, or a metric record fails a Module 1 check."""


def _need(ok, msg):
    if not ok:
        raise RunError(msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json(p):
    p = Path(p)
    _need(p.is_file(), f"{p} not found")
    return json.loads(p.read_text())


def check_run(d, model, seed, epochs):
    """Read-only integrity of one run directory for logical id (model, seed). -> record dict."""
    d = Path(d).expanduser().resolve()
    rcfg = _json(d / "run_config.json")
    fin = _json(d / "final.json")
    for k in SHARED + ("run_id", "model", "seed", "env_key", "git_dirty", "budget_valid"):
        _need(k in rcfg, f"{d.name}: run_config lacks {k}")
    _need(rcfg["kind"] == "production", f"{d.name}: kind {rcfg['kind']!r}")
    _need((rcfg["model"], rcfg["seed"]) == (model, seed), f"{d.name}: is {rcfg['model']}/{rcfg['seed']}, expected {model}/{seed}")
    _need(rcfg["epochs"] == epochs, f"{d.name}: epochs {rcfg['epochs']} != {epochs}")
    _need(rcfg["git_dirty"] is False, f"{d.name}: git_dirty {rcfg['git_dirty']!r}")
    _need(rcfg["budget_valid"] is True, f"{d.name}: budget_valid {rcfg['budget_valid']!r}")
    run_id, spe = rcfg["run_id"], int(rcfg["steps_per_epoch"])
    _need(rcfg["horizon"] == epochs * spe, f"{d.name}: horizon {rcfg['horizon']} != {epochs} x {spe}")
    _need(fin.get("run_id") == run_id, f"{d.name}: final.json run_id differs")
    _need(fin.get("completed_epochs") == epochs, f"{d.name}: completed {fin.get('completed_epochs')} of {epochs}")
    last_ck = ckpt_path(d, epochs - 1)
    _need(Path(str(fin.get("final_checkpoint"))).parts[-2:] == last_ck.parts[-2:],
          f"{d.name}: final_checkpoint {fin.get('final_checkpoint')!r} is not epoch {epochs - 1}")
    records, torn = read_log(d)                      # read-only parse
    _need(torn is None, f"{d.name}: torn final log record")
    _need([r.get("epoch") for r in records] == list(range(epochs)), f"{d.name}: log epochs {[r.get('epoch') for r in records]}")
    for i, r in enumerate(records):
        _need(r.get("run_id") == run_id, f"{d.name}: log record {i} run_id differs")
        _need(r.get("budget_valid") is True and r.get("continuation_exact") is True,
              f"{d.name}: epoch {i} budget_valid / continuation_exact not true")
        _need(r.get("global_step") == (i + 1) * spe, f"{d.name}: epoch {i} global_step {r.get('global_step')}")
        _need(sha256(ckpt_path(d, i)) == r["ckpt_sha256"], f"{d.name}: checkpoint {i} sha256 differs from the log")
        _need(sha256(csv_path(d, i)) == r["csv_sha256"], f"{d.name}: val CSV {i} sha256 differs from the log")
    last = records[-1]
    _need(fin.get("final_val") == last.get("val"), f"{d.name}: final_val differs from the last log record")
    meta = load_checkpoint(last_ck, last["ckpt_sha256"])["meta"]
    want = {"run_id": run_id, "epoch": epochs - 1, "global_step": epochs * spe, "csv_sha256": last["csv_sha256"],
            "model": model, "seed": seed, "holes": rcfg["holes"], "horizon": rcfg["horizon"],
            "config_sha256": rcfg["config_sha256"]}
    for k, v in want.items():
        _need(meta.get(k) == v, f"{d.name}: final checkpoint meta {k}={meta.get(k)!r} != {v!r}")
    return {"model": model, "seed": seed, "path": str(d), "run_id": run_id,
            "final_ckpt": {"path": str(last_ck), "sha256": last["ckpt_sha256"]},
            "final_csv": {"path": str(csv_path(d, epochs - 1)), "sha256": last["csv_sha256"]},
            "env_key": rcfg["env_key"], "shared": {k: rcfg[k] for k in SHARED}}


def _env(rec):
    return json.dumps(rec["env_key"], sort_keys=True)


def _candidate(path, group_env, epochs, ref_shared):
    """-> (record or None, status string). Never raises for a candidate. Eligible = integrity, the venue
    group's env_key AND every SHARED field equal to the eight fixed runs' (review fix 3: an incompatible pod
    M_s3 falls back to the backup instead of aborting the selection)."""
    if path is None:
        return None, "absent (not given)"
    if not Path(path).expanduser().exists():
        return None, f"absent ({path})"
    try:
        r = check_run(path, "M", 3, epochs)
    except (RunError, OSError, KeyError, ValueError, RuntimeError) as e:
        return None, f"incomplete or rejected: {e}"
    if _env(r) != group_env:
        return None, "rejected: env_key differs from its venue group"
    bad = [k for k in SHARED if json.dumps(r["shared"][k], sort_keys=True) != json.dumps(ref_shared[k], sort_keys=True)]
    if bad:
        return None, f"rejected: provenance differs from the fixed runs in {bad}"
    return r, "passes"


def portable(rec):
    out = {k: rec.get(k) for k in PORTABLE}
    out["run"] = {k: rec.get("run", {}).get(k) for k in PORTABLE_RUN}
    out["final_checkpoint_sha256"] = rec.get("final_checkpoint", {}).get("sha256")
    out["final_csv_sha256"] = rec.get("final_csv", {}).get("sha256")
    return out


def check_metric(local_path, pod_path, a0_s1):
    """Local record via load_decision (recheck); pod record as plain JSON; portable fields equal."""
    try:
        local = mp.load_decision(local_path, recheck=True)
    except mp.PolicyError as e:
        raise RunError(f"local metric decision: {e}") from None
    pod = _json(pod_path)                            # plain JSON only: its paths are the pod's
    pl, pp = portable(local), portable(pod)
    diff = sorted(k for k in pl if pl[k] != pp[k])
    _need(not diff, f"pod and local metric records differ in {diff}")
    _need(local["run"]["run_id"] == a0_s1["run_id"], "metric decision is not from the selected A0 seed 1")
    _need(local["final_checkpoint"]["sha256"] == a0_s1["final_ckpt"]["sha256"], "metric decision checkpoint differs")
    return {"metric": local["metric"], "promoted": local["promoted"], "local": str(Path(local_path).resolve()),
            "local_sha256": sha256(local_path), "pod": str(Path(pod_path).resolve()), "pod_sha256": sha256(pod_path),
            "portable": pl}


def select(runs, m3_pod, m3_laptop, commit, config_path):
    """runs: {(model, seed): path} for the eight fixed ids. -> selection record (raises RunError)."""
    cfg, cfg_path = rc.load_train_cfg(config_path)
    _need(cfg["status"] == "frozen", f"{cfg_path}: status {cfg['status']!r}; selection needs the frozen config")
    epochs = int(cfg["epochs"])
    fixed = POD_IDS + LAPTOP_IDS
    _need(set(runs) == set(fixed), f"run ids {sorted(runs)} != {sorted(fixed)}")
    recs = {k: check_run(runs[k], k[0], k[1], epochs) for k in fixed}
    pod_env = {_env(recs[k]) for k in POD_IDS}
    lap_env = {_env(recs[k]) for k in LAPTOP_IDS}
    _need(len(pod_env) == 1, "pod group (seeds 1-2) has more than one env_key")
    _need(len(lap_env) == 1, "laptop group (A0_s3, R_s3) has more than one env_key")
    pod_env, lap_env = pod_env.pop(), lap_env.pop()
    for k in SHARED:                                 # the eight fixed runs agree first (candidates checked against them)
        vals = {json.dumps(recs[i]["shared"][k], sort_keys=True) for i in fixed}
        _need(len(vals) == 1, f"runs differ in {k}: {sorted(vals)}")
    ref = recs[fixed[0]]["shared"]
    cp, sp = _candidate(m3_pod, pod_env, epochs, ref)
    cl, sl = _candidate(m3_laptop, lap_env, epochs, ref)
    if cp is not None:
        chosen, why = "pod", "pod M_s3 passes (D7-1 rule)"
        sl = sl if cl is None else "passes; unused (pod candidate preferred by D7-1)"
    elif cl is not None:
        chosen, why = "laptop", f"pod M_s3 not usable ({sp}); laptop backup used (D7-1 rule)"
    else:
        raise RunError(f"no usable M_s3: pod {sp}; laptop {sl}")
    recs[("M", 3)] = cp if chosen == "pod" else cl
    ids = sorted(recs)
    _need(ids == sorted((m, s) for m in MODELS for s in SEEDS), f"selected ids {ids}")
    _need(len({r["path"] for r in recs.values()}) == 9, "two logical ids share a directory")
    for k in SHARED:
        vals = {json.dumps(r["shared"][k], sort_keys=True) for r in recs.values()}
        _need(len(vals) == 1, f"runs differ in {k}: {sorted(vals)}")
    shared = next(iter(recs.values()))["shared"]
    _need(len(commit) >= 7 and shared["git_commit"].startswith(commit), f"runs at {shared['git_commit']}, expected {commit}")
    _need(shared["epochs"] == epochs, f"runs have {shared['epochs']} epochs, config {epochs}")
    _need(rc.file_sha256(cfg_path) == shared["config_sha256"], f"{cfg_path} differs from the runs' config_sha256")
    for k, p in CURRENT.items():
        _need(config_matches(p, shared[k]), f"{p} differs from the runs' {k}")
    return {"runs": {f"{m}:{s}": {kk: v for kk, v in recs[(m, s)].items() if kk != "shared"} for m, s in ids},
            "shared": shared, "venues": {"pod": json.loads(pod_env), "laptop": json.loads(lap_env)},
            "m3": {"chosen": chosen, "reason": why, "pod_candidate": {"path": m3_pod, "status": sp},
                   "laptop_candidate": {"path": m3_laptop, "status": sl}},
            "config_path": str(cfg_path)}


def default_runs(root):
    root = Path(root).expanduser()
    return {(m, s): root / f"{m}_s{s}" for m, s in POD_IDS + LAPTOP_IDS}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True, help="directory holding {A0,R,M}_s{1,2} and {A0,R}_s3")
    p.add_argument("--commit", required=True, help="training commit (prefix, >= 7 chars)")
    p.add_argument("--m3-pod", default=None)
    p.add_argument("--m3-laptop", default=None)
    p.add_argument("--metric-local", default=None, help="local metric_decision.json (required unless --no-metric)")
    p.add_argument("--metric-pod", default=None)
    p.add_argument("--no-metric", action="store_true",
                   help="candidate check only (e.g. before stopping the laptop backup); writes no selection")
    p.add_argument("--config", default=str(rc.DEFAULT_CONFIG))
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0, help="unused (no randomness); recorded by convention")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    try:
        sel = select(default_runs(a.runs_root), a.m3_pod, a.m3_laptop, a.commit, a.config)
        if a.no_metric:
            print(f"runs OK; M_s3 -> {sel['m3']['chosen']} ({sel['m3']['reason']}); "
                  f"pod: {sel['m3']['pod_candidate']['status']}; laptop: {sel['m3']['laptop_candidate']['status']}")
            print("--no-metric: no selected_runs.json written")
            return 0
        if not (a.metric_local and a.metric_pod):
            raise RunError("--metric-local and --metric-pod are required (or --no-metric)")
        sel["metric_decision"] = check_metric(a.metric_local, a.metric_pod, sel["runs"]["A0:1"])
    except RunError as e:
        raise SystemExit(f"m1_runs refused: {e}")
    out = Path(a.out).expanduser()
    target = out / "selected_runs.json"
    if target.exists():
        raise SystemExit(f"{target} exists; selection is written once (use a fresh --out)")
    out.mkdir(parents=True, exist_ok=True)
    sel["made"] = {**git_fields(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a)}
    atomic_write_json(target, sel)
    print(f"selected_runs.json written: {target} (sha256 {sha256(target)[:12]}); M_s3 -> {sel['m3']['chosen']}; "
          f"metric {sel['metric_decision']['metric']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
