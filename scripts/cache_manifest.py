#!/usr/bin/env python3
"""sha256 manifest of a built cache: carry it to another machine (RunPod) and prove the copy is exact.

  write   hash every file of the named splits under --cache-dir -> <out>/manifest.json
  verify  re-hash the files listed in --manifest under --cache-dir; FAIL on any missing, extra,
          resized or changed file. Exit 0 = identical, 1 = differs, 3 = error. verify.json reads
          INCOMPLETE until the run finishes, so an interrupted rerun never leaves an old PASS.

Why: accept_cache.py cannot run on a pod (its rebuild check needs the raw dataset), so the pod's copy
is accepted by being byte-identical to the accepted workstation cache. Each split must carry its
COMPLETE marker. Read-only on the cache; outputs go to --out, which may not lie inside the cache or the
dataset root. No randomness, so no --seed.

Usage:
  workstation:  python scripts/cache_manifest.py write --splits train,val \
                    --out $RUNS/cache_manifest_v1
  pod:          python scripts/cache_manifest.py verify --cache-dir /root/cache --workers 8 \
                    --manifest /workspace/cache_manifest_v1/manifest.json --out /workspace/runs/manifest_verify
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.paths import load_paths, paths_source  # noqa: E402
from data.provenance import git_fields, refuse_existing, refuse_inside  # noqa: E402

KNOWN_SPLITS = ("train", "val", "test_module", "test_sealed")
CHUNK = 8 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def list_files(cache_dir, splits):
    """Sorted relative POSIX paths ("train/rgb.npy", ...) of every regular file of the splits."""
    root = Path(cache_dir).expanduser()
    out = []
    for sp in splits:
        d = root / sp
        if not (d / "COMPLETE").is_file():
            raise SystemExit(f"{d}: no COMPLETE marker (missing or unfinished split)")
        for p in d.rglob("*"):
            if p.is_symlink():
                raise SystemExit(f"{p}: symlink in the cache, refusing (hash the real file)")
            if p.is_file():
                out.append(p.relative_to(root).as_posix())
    return sorted(out)


def hash_all(root, rels, workers, log_every_gb=5.0):
    """{rel: (bytes, sha256)} in rels order; threads (hashlib releases the GIL on large reads)."""
    root = Path(root).expanduser()
    total = sum((root / r).stat().st_size for r in rels)
    t0, done, next_log = time.time(), 0, log_every_gb * 1e9

    def one(rel):
        p = root / rel
        return rel, p.stat().st_size, sha256_file(p)

    res = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for rel, n, h in ex.map(one, rels):
            res[rel] = (n, h)
            done += n
            if done >= next_log:
                print(f"  {done / 1e9:.1f} / {total / 1e9:.1f} GB ({time.time() - t0:.0f} s)", flush=True)
                next_log += log_every_gb * 1e9
    return res, total


def compare(manifest, got):
    """manifest: written dict; got: {rel: (bytes, sha256)} for the files present now."""
    want = {f["path"]: (f["bytes"], f["sha256"]) for f in manifest["files"]}
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    resized = sorted(r for r in set(want) & set(got) if want[r][0] != got[r][0])
    changed = sorted(r for r in set(want) & set(got) if want[r][0] == got[r][0] and want[r][1] != got[r][1])
    return {"missing": missing, "extra": extra, "resized": resized, "changed": changed,
            "identical": not (missing or extra or resized or changed)}


def parse_args(argv=None):
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["write", "verify"])
    ap.add_argument("--cache-dir", default=paths["cache_dir"])
    ap.add_argument("--splits", default="train,val", help="write only; verify takes them from the manifest")
    ap.add_argument("--manifest", help="verify: the manifest.json written on the workstation")
    ap.add_argument("--workers", type=int, default=1, help="hashing threads (1 on a USB HDD, 8 on NVMe)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    a.split_list = [s.strip() for s in a.splits.split(",") if s.strip()]
    if a.mode == "write" and (not a.split_list or len(set(a.split_list)) != len(a.split_list)
                              or any(s not in KNOWN_SPLITS for s in a.split_list)):
        ap.error(f"--splits must be a non-empty list without repeats from {KNOWN_SPLITS}")
    if a.mode == "verify" and not a.manifest:
        ap.error("verify needs --manifest")
    if a.workers < 1:
        ap.error("--workers >= 1")
    return a


def write_json_atomic(path, obj):
    """Write to <name>.tmp, then rename: a crash never leaves a half-written file under the real name."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def main(argv=None):
    """Output lifecycle. write: an old manifest.json is deleted FIRST and the new one is written
    atomically, so a crash leaves no manifest (never an old one beside a new config). verify:
    verify.json = INCOMPLETE is written FIRST; an exception rewrites it as ERROR with the traceback
    (exit 3); Ctrl-C or a killed process leaves INCOMPLETE. Only a finished run writes PASS / FAIL."""
    a = parse_args(argv)
    out = refuse_inside(a.out, load_paths()["data_root"], a.cache_dir)
    name = "manifest.json" if a.mode == "write" else "verify.json"
    refuse_existing([out / name, out / "config.json"], a.overwrite)
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    if a.mode == "write":
        (out / name).unlink(missing_ok=True)
    else:
        write_json_atomic(out / name, {"verdict": "INCOMPLETE", "started": started})
    config = {"script": "scripts/cache_manifest.py", "args": vars(a), **git_fields(), "paths": paths_source(),
              "python": platform.python_version(), "started": started}
    if a.mode == "write":
        (out / "config.json").write_text(json.dumps(config, indent=1, default=str))
        return run_write(a, out, name, started)
    try:
        (out / "config.json").write_text(json.dumps(config, indent=1, default=str))
        return run_verify(a, out, name)
    except (Exception, SystemExit) as e:          # SystemExit too: a split without COMPLETE
        write_json_atomic(out / name, {"verdict": "ERROR", "error": f"{type(e).__name__}: {e}",
                                       "traceback": traceback.format_exc(),
                                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"ERROR  {type(e).__name__}: {e}  (report: {out / name})")
        return 3


def run_write(a, out, name, started):
    t0 = time.time()
    rels = list_files(a.cache_dir, a.split_list)
    print(f"hashing {len(rels)} files of {a.split_list} under {a.cache_dir} ...", flush=True)
    got, total = hash_all(a.cache_dir, rels, a.workers)
    manifest = {"version": 1, "splits": a.split_list, "n_files": len(rels), "total_bytes": total,
                "files": [{"path": r, "bytes": got[r][0], "sha256": got[r][1]} for r in rels],
                "source_cache_dir": str(Path(a.cache_dir).expanduser().resolve()), **git_fields(),
                "created": started, "seconds": round(time.time() - t0, 1)}
    write_json_atomic(out / name, manifest)
    print(f"wrote {out / name}: {len(rels)} files, {total / 1e9:.2f} GB, {time.time() - t0:.0f} s")
    return 0


def run_verify(a, out, name):
    t0 = time.time()
    manifest = json.loads(Path(a.manifest).expanduser().read_text())
    rels = list_files(a.cache_dir, manifest["splits"])
    print(f"verifying {len(rels)} files of {manifest['splits']} under {a.cache_dir} ...", flush=True)
    got, total = hash_all(a.cache_dir, rels, a.workers)
    res = compare(manifest, got)
    report = {"verdict": "PASS" if res["identical"] else "FAIL", **res, "manifest": str(a.manifest),
              "n_files": len(rels), "total_bytes": total, "seconds": round(time.time() - t0, 1),
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json_atomic(out / name, report)
    print(f"{report['verdict']}: missing {len(res['missing'])}, extra {len(res['extra'])}, "
          f"resized {len(res['resized'])}, changed {len(res['changed'])}  (report: {out / name})")
    return 0 if res["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
