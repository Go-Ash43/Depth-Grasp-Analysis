"""Recreate the Module 1 evaluation and every H1 number from the repo's checkpoints and a 224-px cache.

For one split (val or test_module):
  1. cache check: every file of the split is hashed and compared with the shipped cache manifest
     (artifacts/manifests). summary.json records the build's machine, commit and time, so it differs for any
     rebuilt cache; every other file (the arrays, index.csv, grasps) should be identical. Differences are
     reported, not fatal: what counts is step 3.
  2. evaluation: the nine final checkpoints (artifacts/checkpoints, sha256 checked) x the four conditions
     (none, low, mid, high) with the evaluation's own function (scripts/eval_m1.py eval_one: same dataset,
     corruption seed, decoding, matching, batch size, and float32 precision settings on CUDA). Per-scene CSVs
     and sidecars are written to <out>/results/<model>_s<seed>/<condition>.{csv,sidecar.csv}, the layout
     scripts/eval_m1.py writes (scripts/make_fig3.py reads it), and compared with the recorded hashes;
     <out>/accuracy_table.md (the report's accuracy table: mean of the three seeds, primary / strict) is compared
     with the recorded values.
  3. H1: the harness's own statistics (scripts/h1_harness.py read_index, load_eval, analyse, tables: two-level
     crossed bootstrap, 10,000 resamples, fixed bootstrap seed) on these CSVs -> <out>/h1.json and
     <out>/h1_tables.md, compared with the recorded ones (artifacts/records/<split>; h1.json without its
     "inputs" / "made" fields). These two files hold every number of the report's H1 tables.
Different GPUs or library versions change the last bits of the predicted maps, so per-scene CSVs may differ
from the recorded bytes while every match outcome, and therefore every H1 number, is unchanged; the H1
comparison (with the accuracy table) is the pass criterion. Exit status 1 if either differs.

This is a re-computation for verification. It writes no test ledger (the one-look test record is the one made
by scripts/eval_m1.py); the recorded chain itself (metric_decision, m1_runs, eval_m1, h1_harness) needs every
epoch of every training run and is described in the README.

Usage (GPU recommended: ~30 min per split):
  python scripts/reproduce.py --split val --cache-dir /path/to/cache --out repro_val
  python scripts/reproduce.py --split test_module --cache-dir /path/to/cache --out repro_test
"""
import os

_CUBLAS = ":4096:8"                                  # must be set before torch initialises CUDA (as eval_m1.py)
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", _CUBLAS) != _CUBLAS:
    raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG is {os.environ['CUBLAS_WORKSPACE_CONFIG']!r}; unset it or use {_CUBLAS!r}")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import training.runcfg as rc  # noqa: E402
from data.labels import load_labels_cfg  # noqa: E402
from data.paths import load_paths  # noqa: E402
from data.provenance import config_matches, git_fields  # noqa: E402
from eval import h1 as h1stats  # noqa: E402
from scripts import cache_manifest as cm  # noqa: E402
from scripts.eval_m1 import CONDITIONS, eval_one, summary_from_csv  # noqa: E402
from scripts.h1_harness import analyse, load_eval, read_index, tables  # noqa: E402
from scripts.m1_runs import CURRENT  # noqa: E402

DEFAULT_ARTIFACTS = REPO / "artifacts"
MANIFEST_OF = {"val": "cache_manifest_v1", "test_module": "cache_manifest_test_module"}
PROVENANCE_FILES = ("summary.json",)                 # differ for any rebuilt cache (machine, commit, time)
STRIP = ("inputs", "made")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_bytes(path, b):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b)
    return hashlib.sha256(b).hexdigest()


def check_cache(cache_dir, manifest, split, workers):
    """-> {"identical": [...], "differ": [...], "provenance_differ": [...], "missing": [...]} for the split's files."""
    want = {f["path"]: f["sha256"] for f in manifest["files"] if f["path"].startswith(f"{split}/")}
    if not want:
        raise SystemExit(f"the manifest has no files for {split}")
    have = [r for r in sorted(want) if (cache_dir / r).is_file()]
    got, _ = cm.hash_all(cache_dir, have, workers)
    rep = {"identical": [], "differ": [], "provenance_differ": [], "missing": sorted(set(want) - set(have))}
    for r in have:
        if got[r][1] == want[r]:
            rep["identical"].append(r)
        elif Path(r).name in PROVENANCE_FILES:
            rep["provenance_differ"].append(r)
        else:
            rep["differ"].append(r)
    return rep


def accuracy_table(split, summ):
    """The report's accuracy table: top-1 (%) per model and condition, mean of the three seeds, primary / strict."""
    lines = [f"# Top-1 accuracy (%), mean of seeds 1-3, primary / strict ({split})", "",
             "| model | " + " | ".join(CONDITIONS) + " |", "|---" * (len(CONDITIONS) + 1) + "|"]
    for m in ("R", "A0", "M"):
        cells = []
        for c in CONDITIONS:
            v = [summ[f"{m}_s{s}/{c}"] for s in (1, 2, 3)]
            cells.append(f"{100 * sum(x['top1_primary'] for x in v) / 3:.2f} / "
                         f"{100 * sum(x['top1_strict'] for x in v) / 3:.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=("val", "test_module"))
    p.add_argument("--cache-dir", default=None, help="224-px cache folder holding <split>/ (default: cache_dir from "
                                                     "the paths config)")
    p.add_argument("--out", required=True, help="output folder (must not exist)")
    p.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS), help="default: artifacts/ in the repo")
    p.add_argument("--device", default="cuda", help="cuda (default) or cpu (slow: hours)")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--hash-workers", type=int, default=4, help="threads for the cache hash")
    p.add_argument("--seed", type=int, default=0, help="unused (all randomness is keyed); recorded by convention")
    return p.parse_args()


def main():
    a = parse_args()
    art = Path(a.artifacts).expanduser().resolve()
    idx = json.loads((art / "artifacts.json").read_text())
    cache_dir = Path(a.cache_dir or load_paths()["cache_dir"]).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} exists: pick a fresh --out")

    # configs: the ones the runs were trained with (a changed config would change preprocessing or labels)
    cfg, cfg_path = rc.load_train_cfg()
    if rc.file_sha256(cfg_path) != idx["config_sha256"]:
        raise SystemExit(f"{cfg_path} differs from the runs' training config")
    for k, p in CURRENT.items():
        if not config_matches(p, idx["data_config_hashes"][k]):
            raise SystemExit(f"{p} differs from the runs' {k}")
    for c in idx["checkpoints"]:
        if sha256_file(art / c["file"]) != c["sha256"]:
            raise SystemExit(f"{c['file']}: sha256 differs from artifacts.json")
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but no CUDA device (use --device cpu, slow)")
    precision = rc.setup_precision(cfg) if device.type == "cuda" else None
    out.mkdir(parents=True)
    t_start = time.time()

    # 1. cache check
    mname = MANIFEST_OF[a.split]
    manifest = json.loads((art / idx["manifests"][mname]["file"]).read_text())
    print(f"[1/3] hashing {cache_dir / a.split} against {mname} ...", flush=True)
    cache_rep = check_cache(cache_dir, manifest, a.split, a.hash_workers)
    if cache_rep["missing"]:
        raise SystemExit(f"cache files missing: {cache_rep['missing']}")
    index_ok = f"{a.split}/index.csv" in cache_rep["identical"]
    print(f"      {len(cache_rep['identical'])} files identical, {len(cache_rep['provenance_differ'])} provenance "
          f"file(s) differ (expected for a rebuilt cache), {len(cache_rep['differ'])} other file(s) differ"
          + (f": {cache_rep['differ']}" if cache_rep["differ"] else ""))

    # 2. evaluation: nine checkpoints x four conditions
    print("[2/3] evaluating 9 checkpoints x 4 conditions ...", flush=True)
    labels_cfg = load_labels_cfg()
    recorded_files = json.loads((art / "records" / a.split / "eval_files.json").read_text())["files"]
    res_dir = out / "results"
    files, summ = {}, {}
    for c in sorted(idx["checkpoints"], key=lambda c: (c["model"], c["seed"])):
        tag = f"{c['model']}_s{c['seed']}"
        rec = {"model": c["model"], "seed": c["seed"], "path": tag,
               "final_ckpt": {"path": str(art / c["file"]), "sha256": c["sha256"]}}
        for cond in CONDITIONS:
            t0 = time.time()
            b, sb = eval_one(rec, a.split, cond, cache_dir, cfg, labels_cfg, a.device, a.num_workers)
            files[f"{tag}/{cond}.csv"] = write_bytes(res_dir / tag / f"{cond}.csv", b)
            files[f"{tag}/{cond}.sidecar.csv"] = write_bytes(res_dir / tag / f"{cond}.sidecar.csv", sb)
            summ[f"{tag}/{cond}"] = summary_from_csv(b)
            print(f"      {tag} {cond}: primary {summ[f'{tag}/{cond}']['top1_primary']:.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    same = sorted(k for k in files if recorded_files.get(k) == files[k])
    print(f"      {len(same)} / {len(files)} per-scene files byte-identical to the recorded evaluation")
    rec_summ = json.loads((art / "records" / a.split / "eval_summary.json").read_text())["summary"]
    acc_ok = summ == rec_summ
    acc_tab = accuracy_table(a.split, summ)
    (out / "accuracy_table.md").write_text(acc_tab)
    print(acc_tab)

    # 3. H1 with the harness's own statistics
    print("[3/3] H1 statistics (10,000 bootstrap resamples) ...", flush=True)
    slots, objects, msens = read_index(cache_dir / a.split / "index.csv")
    rows, side = load_eval(res_dir, files, slots)
    res = analyse(rows, side, slots, objects, msens, idx["metric"], h1stats.RESAMPLES, h1stats.BOOT_SEED,
                  gate_g1=a.split == "val")
    res["split"] = a.split
    tab = tables(res)
    rec_h1 = json.loads((art / "records" / a.split / "h1.json").read_text())
    rec_tab = (art / "records" / a.split / "h1_tables.md").read_text()
    new_h1 = json.loads(json.dumps(res, default=str))  # same serialisation as the recorded file (atomic_write_json)
    diff = sorted(k for k in set(rec_h1) | set(new_h1) if k not in STRIP and rec_h1.get(k) != new_h1.get(k))
    h1_ok, tab_ok = not diff, tab == rec_tab
    res["inputs"] = {"artifacts_commit": idx.get("git_commit"), "cache_dir": str(cache_dir), "cache_check": cache_rep,
                     "index_identical_to_manifest": index_ok, "per_scene_files_identical": len(same),
                     "per_scene_files": len(files), "recorded_h1_equal": h1_ok, "recorded_tables_equal": tab_ok,
                     "recorded_accuracy_equal": acc_ok}
    res["made"] = {**git_fields(), "args": vars(a), "precision": precision, "torch": torch.__version__,
                   "python": platform.python_version(), "wall_s": round(time.time() - t_start, 1)}
    (out / "h1.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (out / "h1_tables.md").write_text(tab)
    (out / "summary.json").write_text(json.dumps({"split": a.split, "summary": summ}, indent=1) + "\n")
    print(tab)
    print(f"cache:      {len(cache_rep['differ'])} data file(s) differ from the manifest"
          + ("" if index_ok else " (index.csv differs: slot set not the recorded one)"))
    print(f"evaluation: {len(same)} / {len(files)} per-scene files byte-identical; accuracy table "
          f"{'IDENTICAL' if acc_ok else 'DIFFERS'} to the recorded one")
    print(f"H1:         h1.json {'IDENTICAL' if h1_ok else 'DIFFERS in ' + str(diff)}; "
          f"h1_tables.md {'IDENTICAL' if tab_ok else 'DIFFERS'} to the recorded {a.split} result")
    ok = h1_ok and tab_ok and acc_ok
    print(f"{'PASS' if ok else 'FAIL'}: written {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
