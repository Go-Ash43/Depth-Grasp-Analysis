"""Export the recorded Module 1 results into the repo (artifacts/), so every result can be recreated from the repo.

Reads the production runs folder (--runs) and writes, with every hash checked against the recorded chain:
  artifacts/checkpoints/<model>_s<seed>.pt   the nine final checkpoints of the selected runs (R / A0 / M x seeds
                                             1-3), sha256 equal to m1_selection/selected_runs.json and to the
                                             last line of each run's log.jsonl
  artifacts/logs/<model>_s<seed>.jsonl       each selected run's epoch log (training curves); no machine paths
  artifacts/records/<split>/h1.json          the recorded H1 output (val, test_module) without its "inputs" and
                                             "made" fields (machine paths, time); h1_tables.md unchanged
  artifacts/records/<split>/eval_files.json  sha256 of every per-scene CSV and sidecar of the recorded evaluation
  artifacts/records/<split>/eval_summary.json  the recorded per-run, per-condition top-1 (primary, strict)
  artifacts/manifests/<name>.json            the two cache manifests' file lists (relative path, bytes, sha256 per
                                             file); the build's machine path, time and commit are left out, and
                                             artifacts.json keeps the original manifest's sha256
  artifacts/artifacts.json                   index: sha256 of every file above and the recorded identities
Binding checks: each recorded h1.json names the sha256 of the selection and of the evaluation record it was made
from (both must match); every evaluation file still has the hash its eval_record.json lists; the test_module
manifest has the pre-registered sha256. Refuses a dirty tree and a non-empty --out.

Usage:
  python scripts/export_artifacts.py --runs /path/to/depth-lies/runs
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.provenance import git_fields  # noqa: E402
from scripts.m1_runs import CURRENT  # noqa: E402

MODELS, SEEDS, SPLITS = ("R", "A0", "M"), (1, 2, 3), ("val", "test_module")
TEST_MANIFEST_SHA256 = "34c993195b9183d5fa0c3336226af49900631f0baf876da043ecd03b05b72ac8"   # pre-registered
# recorded folder names inside --runs
SELECTION = "m1_selection/selected_runs.json"
EVAL_DIRS = {"val": "m1_eval_val", "test_module": "m1_eval_test"}
H1_DIRS = {"val": "m1_h1_val", "test_module": "m1_h1_test"}
MANIFESTS = {"cache_manifest_v1": "cache_manifest_v1/manifest.json",
             "cache_manifest_test_module": "cache_manifest_test_module/manifest.json"}
STRIP = ("inputs", "made")
MANIFEST_KEEP = ("version", "splits", "n_files", "total_bytes", "files")   # drop machine path, time, commit


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def need(cond, msg):
    if not cond:
        raise SystemExit(f"refusing: {msg}")


def copy_checked(src, dst, sha=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    got = sha256_file(dst)
    need(got == sha256_file(src), f"copy of {src} differs from its source")
    need(sha is None or got == sha, f"{src}: sha256 {got} != recorded {sha}")
    return got


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    return sha256_file(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="production runs folder (selected runs, evaluations, H1, manifests)")
    p.add_argument("--out", default=str(REPO / "artifacts"), help="output folder (default: artifacts/ in the repo)")
    p.add_argument("--seed", type=int, default=0, help="unused (no randomness); recorded by convention")
    p.add_argument("--allow-dirty", action="store_true", help="tests only: allow a dirty git tree")
    return p.parse_args()


def main():
    a = parse_args()
    g = git_fields()
    need(a.allow_dirty or not g["git_dirty"], "dirty git tree: commit first (artifacts.json records the commit)")
    runs, out = Path(a.runs).expanduser(), Path(a.out).expanduser().resolve()
    need(not out.exists() or not any(out.iterdir()), f"{out} exists and is not empty")
    sel_path = runs / SELECTION
    sel, sel_sha = json.loads(sel_path.read_text()), sha256_file(sel_path)
    need(sorted(sel["runs"]) == sorted(f"{m}:{s}" for m in MODELS for s in SEEDS), "selection is not 3 models x 3 seeds")
    idx = {"format": 1, "metric": sel["metric_decision"]["metric"], "selected_runs_sha256": sel_sha,
           "config_sha256": sel["shared"]["config_sha256"],
           "data_config_hashes": {k: sel["shared"][k] for k in CURRENT}, "checkpoints": [], "logs": {}, "records": {},
           "manifests": {}, "git_commit": g["git_commit"]}

    # checkpoints + logs (the M seed-3 run is the one the selection chose; its folder name is recorded too)
    for m in MODELS:
        for s in SEEDS:
            r = sel["runs"][f"{m}:{s}"]
            run_dir, tag = Path(r["path"]), f"{m}_s{s}"
            log = run_dir / "log.jsonl"
            last = json.loads(log.read_text().strip().splitlines()[-1])
            need(last["ckpt_sha256"] == r["final_ckpt"]["sha256"], f"{tag}: last log line names another checkpoint")
            sha = copy_checked(Path(r["final_ckpt"]["path"]), out / "checkpoints" / f"{tag}.pt", r["final_ckpt"]["sha256"])
            idx["checkpoints"].append({"model": m, "seed": s, "file": f"checkpoints/{tag}.pt", "sha256": sha,
                                       "run": run_dir.name, "run_id": r["run_id"], "epochs": int(last["epoch"]) + 1})
            idx["logs"][tag] = {"file": f"logs/{tag}.jsonl", "sha256": copy_checked(log, out / "logs" / f"{tag}.jsonl")}
            print(f"{tag}: checkpoint and log exported (run {run_dir.name})")

    # recorded evaluations and H1
    for split in SPLITS:
        res = runs / EVAL_DIRS[split] / "results"
        rec_path = res / "eval_record.json"
        rec = json.loads(rec_path.read_text())
        files = rec["files"]
        need(rec["split"] == split and rec["selected_sha256"] == sel_sha, f"{split}: evaluation not from this selection")
        for rel, h in files.items():
            need(sha256_file(res / rel) == h, f"{split}/{rel}: changed since the evaluation recorded it")
        h1_dir = runs / H1_DIRS[split]
        h1 = json.loads((h1_dir / "h1.json").read_text())
        need(h1["split"] == split and h1["inputs"]["selected_sha256"] == sel_sha
             and h1["inputs"]["eval_record_sha256"] == sha256_file(rec_path), f"{split}: H1 not from this evaluation")
        d = out / "records" / split
        idx["records"][split] = {
            "h1.json": write_json(d / "h1.json", {k: v for k, v in h1.items() if k not in STRIP}),
            "h1_tables.md": copy_checked(h1_dir / "h1_tables.md", d / "h1_tables.md"),
            "eval_files.json": write_json(d / "eval_files.json", {"split": split, "files": files}),
            "eval_summary.json": copy_checked(runs / EVAL_DIRS[split] / "summary.json", d / "eval_summary.json"),
            "source_h1_sha256": sha256_file(h1_dir / "h1.json"), "source_eval_record_sha256": sha256_file(rec_path),
            "manifest_sha256": rec["identities"]["manifest_sha256"]}
        print(f"{split}: evaluation ({len(files)} files) and H1 records exported")

    for name, rel in MANIFESTS.items():               # file list only: no machine path, time or commit
        src = json.loads((runs / rel).read_text())
        sha = write_json(out / "manifests" / f"{name}.json", {k: src[k] for k in MANIFEST_KEEP})
        idx["manifests"][name] = {"file": f"manifests/{name}.json", "sha256": sha,
                                  "original_sha256": sha256_file(runs / rel)}   # the file the evaluation verified
    need(idx["manifests"]["cache_manifest_test_module"]["original_sha256"] == TEST_MANIFEST_SHA256,
         "test_module manifest is not the pre-registered one")
    for split, name in (("val", "cache_manifest_v1"), ("test_module", "cache_manifest_test_module")):
        need(idx["records"][split]["manifest_sha256"] == idx["manifests"][name]["original_sha256"],
             f"{split}: the evaluation verified its cache against another manifest")

    write_json(out / "artifacts.json", idx)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
