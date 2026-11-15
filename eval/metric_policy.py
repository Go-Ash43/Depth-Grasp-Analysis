"""Enforced metric promotion (proj_v6 7.5, item 7; SPEC_trajectory.md:77,264,268).

The single authoritative reference is production A0 seed 1's final checkpoint (config "selection":
final_epoch). If its clean-val primary top-1 (30 deg, IoU > 0.25) is strictly greater than 95 %, the strict
metric (15 deg, IoU > 0.40) becomes primary for every module; otherwise "primary" stays primary.

decide(run_dir, config_path)  checks the run end to end and returns the decision record (no side effects).
load_decision(path)           the ONLY way downstream H1 / G1 / analysis code obtains the metric key;
                              it re-hashes the files and (by default) re-derives the decision.

The threshold test uses integer counts (matches * 100 > 95 * n), so exactly 95.00 % stays "primary" and no
float rounding can flip it.
"""
import csv
import hashlib
import json
from pathlib import Path

from training import runcfg as rc
from training.checkpoint import ckpt_path, csv_path, read_log

# configs/train_v1.json "metric_promotion", byte-for-byte; any other wording is refused
POLICY_TEXT = "A0 seed 1 final checkpoint, clean val primary top-1 > 0.95 -> strict primary"
POLICY = {"kind": "production", "model": "A0", "seed": 1, "checkpoint": "final_epoch", "condition": "none",
          "metric": "primary", "threshold_pct": 95, "comparison": ">", "promote_to": "strict"}
METRIC_KEYS = ("primary", "strict")
DECISION_VERSION = 1
DECIDED_KEY = "decided"             # provenance of the decision call itself; excluded from comparisons


class PolicyError(ValueError):
    """The run, config or decision file does not satisfy the registered promotion policy."""


def parse_policy(text):
    if text != POLICY_TEXT:
        raise PolicyError(f"metric_promotion text {text!r} is not the registered policy {POLICY_TEXT!r}")
    return dict(POLICY)


def promote(matches, n, threshold_pct=POLICY["threshold_pct"]):
    """Strict becomes primary iff matches / n > threshold_pct / 100 (exact integer comparison)."""
    if not (isinstance(matches, int) and isinstance(n, int)) or isinstance(matches, bool) or n <= 0 \
            or not 0 <= matches <= n:
        raise PolicyError(f"bad counts: matches={matches!r} n={n!r}")
    return matches * 100 > threshold_pct * n


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    p = Path(path)
    if not p.is_file():
        raise PolicyError(f"{p} not found (run incomplete?)")
    return json.loads(p.read_text())


def _bool(s):
    if s not in ("True", "False"):
        raise PolicyError(f"not a boolean CSV field: {s!r}")
    return s == "True"


def _need(ok, msg):
    if not ok:
        raise PolicyError(msg)


def count_clean(path, condition="none"):
    """Recount from the per-slot CSV -> {"n", "matches_primary", "matches_strict"}; duplicate slots refused."""
    seen, mp, ms = set(), 0, 0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["condition"] != condition:
                continue
            s = int(r["slot"])
            _need(s not in seen, f"{path}: duplicate slot {s} in condition {condition!r}")
            seen.add(s)
            mp += _bool(r["match_primary"])
            ms += _bool(r["match_strict"])
    _need(seen, f"{path}: no rows for condition {condition!r}")
    return {"n": len(seen), "matches_primary": mp, "matches_strict": ms}


def decide(run_dir, config_path):
    """Check the reference run against the frozen config and return the decision record."""
    run_dir = Path(run_dir).expanduser().resolve()
    cfg, cfg_path = rc.load_train_cfg(Path(config_path).expanduser().resolve())
    pol = parse_policy(cfg.get("metric_promotion"))
    _need(cfg["status"] == "frozen", f"{cfg_path}: status {cfg['status']!r}, the decision needs the frozen config")
    epochs = cfg["epochs"]
    cfg_sha = rc.file_sha256(cfg_path)

    # --- the run is the registered reference (production A0 seed 1, frozen config, clean tree)
    rcfg = _json(run_dir / "run_config.json")
    for k in ("kind", "model", "seed"):
        _need(rcfg.get(k) == pol[k], f"run {k} = {rcfg.get(k)!r}; the reference is {pol[k]!r}")
    _need(rcfg.get("git_dirty") is False, f"run git_dirty = {rcfg.get('git_dirty')!r}; needs a clean tree")
    _need(rcfg.get("config_sha256") == cfg_sha, "run config_sha256 differs from the config file given")
    _need(rcfg.get("config") == cfg, "run's recorded config differs from the config file given")
    _need(rcfg.get("epochs") == epochs, f"run epochs {rcfg.get('epochs')} != config epochs {epochs}")
    _need(pol["condition"] in rcfg.get("val_conditions", ()), "run has no clean ('none') val condition")

    # --- complete: final.json + every epoch committed once, in order, same run_id, no torn tail
    fin = _json(run_dir / "final.json")
    run_id = rcfg["run_id"]
    _need(fin.get("run_id") == run_id, "final.json run_id differs from run_config")
    _need(fin.get("completed_epochs") == epochs, f"final.json completed_epochs {fin.get('completed_epochs')} != {epochs}")
    _need(fin.get("selection") == "final_epoch", f"final.json selection {fin.get('selection')!r}")
    ck = ckpt_path(run_dir, epochs - 1)
    _need(Path(str(fin.get("final_checkpoint"))).parts[-2:] == ck.parts[-2:],
          f"final.json final_checkpoint {fin.get('final_checkpoint')!r} is not the last epoch's")
    records, torn = read_log(run_dir)
    _need(torn is None, "run log has a torn final record")
    _need([r.get("epoch") for r in records] == list(range(epochs)),
          f"run log epochs {[r.get('epoch') for r in records]} != 0..{epochs - 1}")
    _need(all(r.get("run_id") == run_id for r in records), "run log mixes run_ids")
    last = records[-1]
    _need(last["val"] == fin["final_val"], "final.json final_val differs from the last log record")

    # --- files are the ones the log committed; the counts agree three ways (log, final.json, CSV)
    ck_sha = _sha(ck)
    _need(ck_sha == last["ckpt_sha256"], f"{ck}: sha256 differs from the run log")
    cp = csv_path(run_dir, epochs - 1)
    csv_sha = _sha(cp)
    _need(csv_sha == last["csv_sha256"], f"{cp}: sha256 differs from the run log")
    v = last["val"][pol["condition"]]
    c = count_clean(cp, pol["condition"])
    _need(c["n"] == v["n"] == rcfg.get("n_val"), f"clean n: CSV {c['n']}, log {v['n']}, run_config {rcfg.get('n_val')}")
    for m in METRIC_KEYS:
        _need(c[f"matches_{m}"] == v[f"matches_{m}"], f"clean {m} matches: CSV {c[f'matches_{m}']} != log {v[f'matches_{m}']}")
        _need(v[f"top1_{m}"] == v[f"matches_{m}"] / v["n"], f"log top1_{m} != matches / n")

    promoted = promote(c["matches_primary"], c["n"], pol["threshold_pct"])   # Sec. 7.5: > 95 % -> strict
    return {
        "decision_version": DECISION_VERSION,
        "policy": POLICY_TEXT,
        "metric": pol["promote_to"] if promoted else pol["metric"],
        "promoted": promoted,
        "threshold_pct": pol["threshold_pct"],
        "comparison": pol["comparison"],
        "clean": {"n": c["n"], "matches_primary": c["matches_primary"], "top1_primary": c["matches_primary"] / c["n"],
                  "matches_strict": c["matches_strict"], "top1_strict": c["matches_strict"] / c["n"]},
        "run": {"run_dir": str(run_dir), "run_id": run_id, "kind": rcfg["kind"], "model": rcfg["model"],
                "seed": rcfg["seed"], "epochs": epochs, "git_commit": rcfg.get("git_commit"),
                "config_sha256": cfg_sha, "config_path": str(cfg_path),
                "continuation_exact": all(r.get("continuation_exact") is True for r in records)},
        "final_checkpoint": {"path": str(ck), "sha256": ck_sha},
        "final_csv": {"path": str(cp), "sha256": csv_sha},
    }


def comparable(rec):
    """The decision without the provenance of the call (for overwrite / re-derivation checks)."""
    return json.loads(json.dumps({k: v for k, v in rec.items() if k != DECIDED_KEY}, default=str))


def load_decision(path, recheck=True):
    """-> the decision record; use rec["metric"] ("primary" or "strict") as the primary metric key.

    Always: version, policy text, internal consistency, and the checkpoint / CSV sha256 still match.
    recheck=True (default): re-derive the decision from the run and the config and require equality.
    """
    rec = _json(path)
    _need(rec.get("decision_version") == DECISION_VERSION, f"{path}: decision_version {rec.get('decision_version')!r}")
    _need(rec.get("policy") == POLICY_TEXT, f"{path}: policy text differs from the registered policy")
    _need(rec.get("metric") in METRIC_KEYS, f"{path}: metric {rec.get('metric')!r}")
    cl = rec["clean"]
    promoted = promote(cl["matches_primary"], cl["n"], POLICY["threshold_pct"])
    _need(rec.get("promoted") is promoted and rec["metric"] == ("strict" if promoted else "primary"),
          f"{path}: metric / promoted inconsistent with the recorded counts")
    for k in ("final_checkpoint", "final_csv"):
        _need(_sha(rec[k]["path"]) == rec[k]["sha256"], f"{path}: {rec[k]['path']} changed since the decision")
    if recheck:
        fresh = decide(rec["run"]["run_dir"], rec["run"]["config_path"])
        _need(comparable(fresh) == comparable(rec), f"{path}: re-derived decision differs from the recorded one")
    return rec
