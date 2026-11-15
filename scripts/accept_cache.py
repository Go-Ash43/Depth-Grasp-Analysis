#!/usr/bin/env python3
"""One-time acceptance of the production 224-px cache (review findings 3/4; Phase 0).

Read-only on the cache and the dataset. Order: global checks, then A + B for every split, then the
heavy checks (D, E) only for splits whose metadata (B) passed. Checks, per split:
  A  open      open_cache with the identity checks (split name, membership against configs/splits.json
               whose sha256 must match the build, no duplicates, sorted, n_slots == n_scenes_in_split)
  B  metadata  per split, against the EXPECTED preprocessing (not against the other splits, so a
               separately scheduled build such as test_sealed may have another commit):
               summary.cfg == frozen_cfg(configs/frozen_day1.json) by CONTENT (the file gained
               normalisation constants after the build, so its hash differs by design); top-level
               size / taus / lie_rules == the same config; splits_sha256 == current splits.json;
               manifest_sha256 == splits.json's; clean build (git_dirty false, commit recorded).
               Global: manifest file hash == splits.json's; all splits in splits.json pairwise disjoint.
  C  coverage  slot list == scripts/build_cache.split_items(...) (the builder's own order), exactly;
               no cached object listed under another split (test_sealed never opened)
  D  payload   every slot. ok slots: masks in {0,1}; ds NaN <=> m_sensor; finite ds inside the clamp;
               ds_comp finite and == ds where m_sensor == 0; dstar finite and inside the row's
               [dstar_min, dstar_max] (+-0.05 mm: INTER_AREA averages); m_sensor not all 1 (R1);
               row counts recomputed from the arrays (obj_px_224, small_obj, m_sensor_obj_224,
               lie_eff_obj_224@t, lie_frac_obj_224@t): catches rows permuted against their records;
               lie masks nested across tau (t_hi subset of t_lo) and fraction subset of any;
               grasps finite, opening > 0, jaw > 0, centre inside the 1024 frame.
               Excluded slots: every array all zero (grasp slices are checked by open_cache).
  E  rederive  a seeded sample of ok slots rebuilt from the ORIGINAL scenes (try_load_scene +
               process_scene) and compared bit for bit: every array, every index field, the grasps.
               At least one slot per split (--n-rederive >= 1): acceptance without E is not offered.

Verdict: PASS (exit 0) / FAIL (exit 1) / VERSION-MISMATCH (exit 2) / ERROR (exit 3).
  VERSION-MISMATCH only when every problem is a NUMERIC rebuild difference (rederive_array,
  rederive_index_field) in a split whose own build numpy / cv2 differ from the current ones. A load
  skip, an exception, a status change, a different array set, changed grasps or changed annotation
  counts (n_grasps, n_grasps_dropped) are structural and always FAIL. E runs with
  cv2.setNumThreads(1), as the builder's workers did.
Report lifecycle: report.json = INCOMPLETE is written FIRST (before config.json is replaced); an
exception while writing the config or checking rewrites it as ERROR with the traceback; only a finished
run writes the verdict. So a stale PASS never sits beside a newer config. A KeyboardInterrupt (Ctrl-C)
or a killed process is not caught: it leaves INCOMPLETE, which never counts as acceptance. Refuses an --out inside the dataset root or the
cache, and existing outputs without --overwrite.

Usage:
  python scripts/accept_cache.py --out $RUNS/accept_cache_v1
"""
import argparse
import json
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.cache_layout import LIE_RULES, SPLITS_FILE, _sha256, lie_name, open_cache  # noqa: E402
from data.geometry import in_image  # noqa: E402
from data.jacquard import data_root  # noqa: E402
from data.lies import lie_eff_from_224  # noqa: E402
from data.paths import load_paths, resolve_manifest  # noqa: E402
from data.provenance import config_matches, git_fields, refuse_existing, refuse_inside  # noqa: E402
from scripts import build_cache as bc  # noqa: E402

MAX_EXAMPLES = 20
KNOWN_SPLITS = ("train", "val", "test_module", "test_sealed")
DSTAR_TOL_MM = 0.05
VERSION_SENSITIVE = {"rederive_array", "rederive_index_field"}   # numeric outputs of numpy / cv2
ANNOTATION_FIELDS = {"n_grasps", "n_grasps_dropped"}   # parsed from the grasp files: never version-sensitive
EXIT = {"PASS": 0, "FAIL": 1, "VERSION-MISMATCH": 2, "ERROR": 3}


class Problems:
    """Counts per check name, with at most MAX_EXAMPLES example messages each."""

    def __init__(self):
        self.counts = Counter()
        self.examples = defaultdict(list)

    def add(self, check, msg):
        self.counts[check] += 1
        if len(self.examples[check]) < MAX_EXAMPLES:
            self.examples[check].append(msg)

    def as_dict(self):
        return {"counts": dict(self.counts), "examples": dict(self.examples)}


# ------------------------------------------------------------------ B metadata
def check_metadata(sp, s, cfg_now, splits, splits_hash, prob):
    """One split's summary against the EXPECTED preprocessing. cfg_now: frozen_cfg(...) without 'root'.
    Returns True if no metadata problem was found (the heavy checks may run)."""
    n0 = sum(prob.counts.values())
    if s.get("cfg") != cfg_now:
        prob.add("cfg_differs_from_frozen", f"{sp}: summary cfg {s.get('cfg')} != frozen {cfg_now}")
    for k, want in (("size", cfg_now["size"]), ("taus", cfg_now["taus"]), ("lie_rules", list(LIE_RULES))):
        if s.get(k) != want:                     # the payload checks follow the TOP-LEVEL fields
            prob.add("summary_layout_differs", f"{sp}: summary {k}={s.get(k)!r}, expected {want!r}")
    if s.get("splits_sha256") != splits_hash and not config_matches(SPLITS_FILE, s.get("splits_sha256")):
        # current hash as before; else only a pre-cleanup hash bridged to the current file (data.provenance)
        prob.add("splits_hash", f"{sp}: splits_sha256 {s.get('splits_sha256')} != current {splits_hash}")
    if s.get("manifest_sha256") != splits.get("manifest_sha256"):
        prob.add("manifest_hash", f"{sp}: manifest_sha256 != splits.json's")
    if s.get("git_dirty") is not False or not s.get("git_commit"):
        prob.add("unclean_build", f"{sp}: git_commit={s.get('git_commit')!r}, git_dirty={s.get('git_dirty')!r}")
    if build_versions(s) is None:
        prob.add("build_versions_missing", f"{sp}: summary versions={s.get('versions')!r}")
    return sum(prob.counts.values()) == n0


def build_versions(summary):
    """{"numpy", "cv2"} recorded at build time, or None if missing / malformed."""
    v = summary.get("versions")
    if not isinstance(v, dict) or not all(isinstance(v.get(k), str) and v.get(k) for k in ("numpy", "cv2")):
        return None
    return {k: v[k] for k in ("numpy", "cv2")}


def versions_match(summary, now):
    """True / False for KNOWN recorded versions. Unknown versions count as matching, so they can never
    excuse a rebuild difference as VERSION-MISMATCH (check_metadata reports them as a failure)."""
    built = build_versions(summary)
    return True if built is None else all(built[k] == now[k] for k in ("numpy", "cv2"))


def check_splits_disjoint(splits, prob):
    names = list(splits["splits"])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = set(splits["splits"][a]) & set(splits["splits"][b])
            if both:
                prob.add("splits_overlap", f"{a} & {b}: {len(both)} objects, e.g. {sorted(both)[:3]}")


def check_manifest_file(splits, prob):
    m = resolve_manifest(splits["manifest"])
    h = _sha256(m)
    if h != splits["manifest_sha256"]:
        prob.add("manifest_hash", f"{m}: sha256 {h} != splits.json's {splits['manifest_sha256']}")
    return m


# ------------------------------------------------------------------ C coverage
def check_coverage(split, index, splits, manifest, prob):
    try:
        _, items = bc.split_items(splits, split, manifest)
    except SystemExit as e:
        prob.add("coverage", f"{split}: split_items failed: {e}")
        return
    got = [(f"{r['part']}/{r['object_id']}", int(r["view"])) for r in index]
    if got != items:
        a, b = set(got), set(items)
        prob.add("coverage", f"{split}: {len(got)} slots vs {len(items)} manifest scenes; "
                             f"{len(a - b)} extra, {len(b - a)} missing, e.g. extra {sorted(a - b)[:2]} "
                             f"missing {sorted(b - a)[:2]}"
                             + ("; same scenes, different order" if a == b and len(got) == len(items) else ""))
    others = {o for sp, objs in splits["splits"].items() if sp != split for o in objs}
    leak = sorted({o for o, _ in got} & others)
    if leak:
        prob.add("object_in_other_split", f"{split}: {len(leak)} objects, e.g. {leak[:3]}")


# ------------------------------------------------------------------ D payload
def _mask_names(taus):
    return ["m_sensor", "obj"] + [lie_name(r, t) for t in taus for r in LIE_RULES]


def check_slot(slot, r, a, g, cfg, taus, prob):
    """a: {name: array for this slot}; r: index row; g: (n, 5) grasps at 1024."""
    where = f"slot {slot} {r['part']}/{r['object_id']} v{r['view']}"
    if r["status"] != "ok":
        nz = [n for n, v in a.items() if np.any(v)]
        if nz:
            prob.add("excluded_slot_nonzero", f"{where} ({r['status']}): {nz}")
        return
    nonbinary = [n for n in _mask_names(taus) if a[n].max() > 1]
    for n in nonbinary:
        prob.add("mask_not_binary", f"{where}: {n} max {int(a[n].max())}")
    if nonbinary:                    # every later check reads these masks as 0/1: skip the slot, it already failed
        return
    ms, obj = a["m_sensor"] != 0, a["obj"] != 0
    ds, comp, dstar = a["ds"], a["ds_comp"], a["dstar"]
    lo, hi = cfg["clamp"]
    if not np.array_equal(np.isnan(ds), ms):
        prob.add("ds_nan_vs_m_sensor", f"{where}: {int((np.isnan(ds) != ms).sum())} cells")
    v = ds[~ms]
    if not np.isfinite(v).all():
        prob.add("ds_nonfinite_valid", f"{where}")
    elif v.size and (v.min() < lo or v.max() > hi):
        prob.add("ds_outside_clamp", f"{where}: [{v.min()}, {v.max()}] vs {cfg['clamp']}")
    if not np.isfinite(comp).all():
        prob.add("ds_comp_nonfinite", f"{where}: {int((~np.isfinite(comp)).sum())} cells")
    if not np.array_equal(comp[~ms], ds[~ms]):
        prob.add("ds_comp_ne_ds_outside_holes", f"{where}")
    if not np.isfinite(dstar).all():
        prob.add("dstar_nonfinite", f"{where}")
    else:
        dmin, dmax = float(r["dstar_min"]), float(r["dstar_max"])
        if dstar.min() < dmin - DSTAR_TOL_MM or dstar.max() > dmax + DSTAR_TOL_MM:
            prob.add("dstar_outside_row_range", f"{where}: [{dstar.min()}, {dstar.max()}] vs [{dmin}, {dmax}]")
    if ms.all():
        prob.add("r1_all_invalid", f"{where}")
    n_obj = int(obj.sum())
    want = {"obj_px_224": n_obj, "small_obj": int(n_obj < cfg["min_obj_px"]),
            "m_sensor_obj_224": int((ms & obj).sum())}
    if n_obj == 0:
        prob.add("r1_no_object", f"{where}")
    ms_u8 = a["m_sensor"]
    for t in taus:
        k = f"{float(t):g}"
        want[f"lie_eff_obj_224@{k}"] = int((lie_eff_from_224(a[lie_name("any", t)], ms_u8).astype(bool) & obj).sum())
        want[f"lie_frac_obj_224@{k}"] = int((lie_eff_from_224(a[lie_name("fraction", t)], ms_u8).astype(bool)
                                             & obj).sum())
        if np.any((a[lie_name("fraction", t)] != 0) & (a[lie_name("any", t)] == 0)):
            prob.add("lie_fraction_not_in_any", f"{where}: tau {k}")
    st = sorted(float(t) for t in taus)
    for rule in LIE_RULES:
        for t_lo, t_hi in zip(st, st[1:]):
            if np.any((a[lie_name(rule, t_hi)] != 0) & (a[lie_name(rule, t_lo)] == 0)):
                prob.add("lie_not_nested", f"{where}: {rule} t{t_hi:g} not inside t{t_lo:g}")
    for k, w in want.items():
        try:
            got = int(r[k])
        except (KeyError, TypeError, ValueError):
            prob.add(f"count_{k.split('@')[0]}", f"{where}: index {k}={r.get(k)!r}, arrays give {w}")
            continue
        if got != w:
            prob.add(f"count_{k.split('@')[0]}", f"{where}: index {k}={got}, arrays give {w}")
    if not np.isfinite(g).all() or (g[:, 3] <= 0).any() or (g[:, 4] <= 0).any():
        prob.add("grasp_values", f"{where}: non-finite or non-positive opening/jaw")
    elif not in_image(g[:, :2], (1024, 1024)).all():
        prob.add("grasp_centre_outside_frame", f"{where}: {int((~in_image(g[:, :2], (1024, 1024))).sum())} rows")


def check_payload(cache, cfg, prob, chunk=256, log=None):
    A, index, taus = cache["arrays"], cache["index"], cache["summary"]["taus"]
    n = len(index)
    t0, step = time.time(), max(n // 10, 1)
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        block = {k: np.asarray(v[s0:s1]) for k, v in A.items()}     # sequential read
        for j, slot in enumerate(range(s0, s1)):
            check_slot(slot, index[slot], {k: v[j] for k, v in block.items()}, cache["grasps"](slot),
                       cfg, taus, prob)
            if log and (slot + 1) % step == 0:
                log(f"  payload {slot + 1}/{n} ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ E rederive
def _same_field(stored, v):
    if isinstance(v, float):
        try:
            f = float(stored)
        except (TypeError, ValueError):
            return False
        return (np.isnan(f) and np.isnan(v)) or f == v
    return str(stored) == str(v)


def check_rederive(cache, cfg, slots, prob, log=None):
    """cfg must hold 'root'. Rebuilds each slot with the builder's own functions; bit-exact compare."""
    A, index = cache["arrays"], cache["index"]
    for i, slot in enumerate(slots):
        r = index[slot]
        where = f"slot {slot} {r['part']}/{r['object_id']} v{r['view']}"
        try:                                      # one bad slot is logged, the others still run
            scene, skip = bc.try_load_scene(f"{r['part']}/{r['object_id']}", int(r["view"]), root=cfg["root"])
            if skip is not None:
                prob.add("rederive_load_skip", f"{where}: {skip.get('message', '')[:200]}")
                continue
            row, arrs, g, _ = bc.process_scene(scene, cfg, abort_on_clamp=False)
        except Exception as e:
            prob.add("rederive_exception", f"{where}: {type(e).__name__}: {str(e)[:200]}")
            continue
        if row["status"] != "ok" or arrs is None:
            prob.add("rederive_status", f"{where}: rebuilt status {row['status']}")
            continue
        if set(arrs) != set(A):
            prob.add("rederive_array_set", f"{where}: rebuilt {sorted(set(arrs) ^ set(A))} differ from the cache's")
        for name in sorted(set(arrs) & set(A)):
            stored = np.asarray(A[name][slot])
            ref = np.asarray(arrs[name]).astype(stored.dtype, copy=False)
            if not np.array_equal(stored, ref, equal_nan=np.issubdtype(stored.dtype, np.floating)):
                prob.add("rederive_array", f"{where}: {name}")
        for k, v in row.items():
            if k in r and not _same_field(r[k], v):
                check = "rederive_annotation_field" if k in ANNOTATION_FIELDS else "rederive_index_field"
                prob.add(check, f"{where}: {k} stored {r[k]!r} rebuilt {v!r}")
        if not np.array_equal(np.asarray(g, np.float64), cache["grasps"](slot)):
            prob.add("rederive_grasps", f"{where}")
        if log and (i + 1) % 10 == 0:
            log(f"  rederive {i + 1}/{len(slots)}")


def pick_slots(index, n, seed, split):
    ok = [i for i, r in enumerate(index) if r["status"] == "ok"]
    rng = np.random.default_rng([seed, KNOWN_SPLITS.index(split)])
    return sorted(int(s) for s in rng.choice(ok, size=min(n, len(ok)), replace=False))


# ------------------------------------------------------------------ main
def verdict(per_split, glob, vmatch):
    """per_split: {split: {"counts": ...}}; glob: global problems; vmatch: {split: bool}.
    A numeric rebuild difference is excused (-> VERSION-MISMATCH) only in a split whose OWN build
    versions differ from the current ones; anything else is FAIL."""
    if not glob["counts"] and not any(p["counts"] for p in per_split.values()):
        return "PASS"
    if glob["counts"]:
        return "FAIL"
    for sp, p in per_split.items():
        for k in p["counts"]:
            if k not in VERSION_SENSITIVE or vmatch.get(sp, True):
                return "FAIL"
    return "VERSION-MISMATCH"


def parse_args(argv=None):
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=paths["cache_dir"])
    ap.add_argument("--splits", default="train,val,test_module",
                    help="comma list; test_sealed only for the final sealed-test run")
    ap.add_argument("--frozen", default=str(REPO / "configs" / "frozen_day1.json"))
    ap.add_argument("--n-rederive", type=int, default=50, help="ok slots rebuilt from originals per split")
    ap.add_argument("--seed", type=int, default=0, help="picks the rederive sample")
    ap.add_argument("--chunk", type=int, default=256, help="slots per sequential read")
    ap.add_argument("--out", required=True, help="directory for config.json and report.json")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    a.split_list = [s.strip() for s in a.splits.split(",") if s.strip()]
    if not a.split_list or len(set(a.split_list)) != len(a.split_list) or \
            any(s not in KNOWN_SPLITS for s in a.split_list):
        ap.error(f"--splits must be a non-empty list without repeats from {KNOWN_SPLITS}")
    if a.n_rederive < 1 or a.chunk <= 0 or a.seed < 0:
        ap.error("--n-rederive must be >= 1 (E is part of acceptance), --chunk > 0, --seed >= 0")
    return a


def main(argv=None):
    a = parse_args(argv)
    droot = Path(data_root()).expanduser().resolve()
    out = refuse_inside(a.out, droot)
    cache_root = Path(a.cache_dir).expanduser().resolve()
    if out == cache_root or cache_root in out.parents:
        raise SystemExit(f"refusing to write inside the cache directory {cache_root}: {out}")
    refuse_existing([out / "config.json", out / "report.json"], a.overwrite)
    out.mkdir(parents=True, exist_ok=True)
    rep_path = out / "report.json"
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    # invalidate any previous report FIRST: if this write fails, the old config and report still pair up
    rep_path.write_text(json.dumps({"verdict": "INCOMPLETE", "started": started}, indent=1))
    try:
        versions = {"python": platform.python_version(), "numpy": np.__version__, "cv2": cv2.__version__}
        config = {"script": "scripts/accept_cache.py", "args": {k: v for k, v in vars(a).items()},
                  **git_fields(), "versions": versions, "started": started}
        (out / "config.json").write_text(json.dumps(config, indent=1, default=str))
        report = run(a, droot, versions)
    except Exception as e:                        # never leave INCOMPLETE (or a stale verdict) behind
        report = {"verdict": "ERROR", "error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    rep_path.write_text(json.dumps(report, indent=1, default=str))
    v = report["verdict"]
    _log(f"\n{v}  (report: {rep_path})")
    if v == "ERROR":
        _log(f"  {report['error']}")
    else:
        for sp, p in report["per_split"].items():
            _log(f"  {sp}: {p['counts'] or 'ok'}{'' if report['splits'].get(sp, {}).get('heavy_checks_run') else '  (D/E skipped)'}")
        _log(f"  global: {report['global']['counts'] or 'ok'}")
    return EXIT[v]


def _log(msg):
    print(msg, flush=True)


def run(a, droot, versions):
    """Global checks -> A + B + C for every split -> D + E only where B passed. Returns the report."""
    cv2.setNumThreads(1)                          # same OpenCV threading as the builder's workers (E)
    cfg = bc.frozen_cfg(a.frozen)                 # without 'root', as stored in summary["cfg"]
    splits = json.loads(SPLITS_FILE.read_text())
    splits_hash = _sha256(SPLITS_FILE)
    glob = Problems()
    manifest = check_manifest_file(splits, glob)
    check_splits_disjoint(splits, glob)
    per, info, caches, vmatch = {}, {}, {}, {}
    for sp in a.split_list:                       # cheap stage: open, metadata, coverage
        prob = per[sp] = Problems()
        info[sp] = {"heavy_checks_run": False}
        _log(f"[{sp}] open + metadata")
        try:
            cache = open_cache(a.cache_dir, sp)
        except Exception as e:                    # A failed: report it, keep going with other splits
            prob.add("open_cache", f"{type(e).__name__}: {e}")
            continue
        s = cache["summary"]
        vmatch[sp] = versions_match(s, versions)
        info[sp].update(n_slots=len(cache["index"]), n_ok=s["n_ok"], git_commit=s.get("git_commit"),
                        build_versions=s.get("versions"), versions_match=vmatch[sp])
        meta_ok = check_metadata(sp, s, cfg, splits, splits_hash, prob)
        check_coverage(sp, cache["index"], splits, manifest, prob)
        if meta_ok:
            caches[sp] = cache
        else:
            _log(f"[{sp}] metadata problems {dict(prob.counts)}: payload and rederive skipped")
    for sp, cache in caches.items():              # heavy stage, only where the metadata passed
        prob = per[sp]
        _log(f"[{sp}] payload over {len(cache['index'])} slots")
        t0 = time.time()
        check_payload(cache, cfg, prob, a.chunk, _log)
        info[sp]["payload_seconds"] = round(time.time() - t0, 1)
        slots = pick_slots(cache["index"], a.n_rederive, a.seed, sp)
        info[sp]["rederive_slots"] = slots
        if not slots:                             # E is part of acceptance: a split with no ok scene fails
            prob.add("rederive_no_eligible_scene", f"{sp}: no ok slot to rebuild")
        _log(f"[{sp}] rederive {len(slots)} slots from the originals")
        check_rederive(cache, {**cfg, "root": str(droot)}, slots, prob, _log)
        info[sp]["heavy_checks_run"] = True
        _log(f"[{sp}] problems: {dict(prob.counts) or 'none'}")
    per_d = {sp: p.as_dict() for sp, p in per.items()}
    v = verdict(per_d, glob.as_dict(), vmatch)
    if v == "PASS" and not all(i["heavy_checks_run"] for i in info.values()):
        v = "FAIL"                                # defensive: a skipped split can never pass
    return {"verdict": v, "splits": info, "per_split": per_d, "global": glob.as_dict(),
            "versions_now": versions, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}


if __name__ == "__main__":
    sys.exit(main())
