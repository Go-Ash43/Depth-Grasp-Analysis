"""224-px preprocessing cache: the ONE definition of its layout, shared by scripts/build_cache.py (writer),
the overlays and the dataset class (readers). SPEC_trajectory Module 1 item 2; RULES R1-R4.

<cache_dir>/<split>/
  <name>.npy          one memmap per array below, first axis = scene slot (index.csv row order)
  grasps_1024.npy     (G, 5) float64 x, y, theta_deg, opening, jaw at 1024 px, as in the files.
                      Stored ONLY at 1024: users scale with data.geometry.scale_grasps, so a 224 copy
                      can never be scaled twice.
  grasp_offsets.npy   (N + 1,) int64: scene i's grasps are grasps_1024[off[i]:off[i + 1]]
  index.csv           one row per scene slot (part, object_id, view, status, sizes, counts)
  summary.json        config, provenance, R1/R3/R4 records, train-only normalisation + noise floor
  COMPLETE            written last; a split without it is invalid (open_cache refuses it)

Every scene of the split has a slot, in a fixed order (sorted objects, sorted views). Scenes excluded
by R1 (status != "ok") keep an all-zero slot and no grasps, so indices never depend on exclusions.

Masks are uint8 with 1 = invalid / lie, except `obj` (1 = object, the documented exception).
lie_<rule>_t<tau> is resize_<rule>(native M_lie at tau), BEFORE any flag is subtracted: the R2 mask
for a sample is data.lies.lie_eff_from_224(lie_..., deployed_224 incl. that sample's M_synth).
dstar is oracle / A* only: the dataset class must whitelist it explicitly (D* leak guard).
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from data.provenance import config_matches

REPO = Path(__file__).resolve().parents[1]
SPLITS_FILE = REPO / "configs" / "splits.json"
LAYOUT_VERSION = 1
LIE_RULES = ("any", "fraction")


def lie_name(rule, tau):
    if rule not in LIE_RULES:
        raise ValueError(f"rule must be one of {LIE_RULES}, got {rule!r}")
    return f"lie_{rule}_t{float(tau):g}"


def array_specs(size, taus):
    """{name: (dtype, per-scene shape)} for one split."""
    s = (size, size)
    specs = {
        "rgb": (np.uint8, s + (3,)),       # RGB order, INTER_AREA
        "ds": (np.float32, s),             # clamped D_s (R4) at 224, NaN where m_sensor = 1
        "ds_comp": (np.float32, s),        # clean A0 input: telea_fill(ds, m_sensor), finite everywhere
        "dstar": (np.float32, s),          # D* at 224, never clamped; ORACLE / A* ONLY
        "m_sensor": (np.uint8, s),         # resize_mask_any(M_sensor); 1 = invalid
        "obj": (np.uint8, s),              # resize_mask_majority(object); 1 = OBJECT
    }
    for t in taus:
        for r in LIE_RULES:
            specs[lie_name(r, t)] = (np.uint8, s)
    return specs


INDEX_FIELDS = ["slot", "part", "object_id", "view", "status", "detail",
                "obj_px_224", "small_obj", "n_grasps", "n_grasps_dropped",
                "m_sensor_obj_224", "dstar_min", "dstar_max", "ds_valid_max",
                "n_clamped", "n_clamped_truthful"]


def lie_index_fields(taus):
    """Per-tau count columns: M_lie at 1024 on the object, and the 224 cells lied about relative to
    the sensor flag only (no M_synth), any / fraction rule, on the 224 object (same as the audit)."""
    out = []
    for t in taus:
        k = f"{float(t):g}"
        out += [f"lie_obj_1024@{k}", f"lie_eff_obj_224@{k}", f"lie_frac_obj_224@{k}"]
    return out


def check_index(d, index, summary, g, off):
    """Integrity of the slot <-> record <-> grasp mapping (review 10/11). Status-aware: process_scene
    records n_grasps BEFORE the R1 checks, so no_object_224 / no_valid_depth_224 rows keep n_grasps > 0
    with an EMPTY grasp slice, and load skips have a blank count. Contract:
      status ok  -> slice length == n_grasps, a positive integer
      otherwise  -> slice length == 0 (n_grasps may be the original count or blank)
    Counts are compared with n_slots / n_processed / n_ok (the built slots: for a smoke cache these
    are the sampled scenes, never the full split's n_scenes_in_split)."""
    n = len(index)
    if g.ndim != 2 or g.shape[1] != 5 or g.dtype != np.float64:
        raise RuntimeError(f"{d}: grasps_1024 must be (G, 5) float64, got {g.dtype} {g.shape}")
    if off.ndim != 1 or not np.issubdtype(off.dtype, np.integer) or len(off) != n + 1:
        raise RuntimeError(f"{d}: grasp_offsets must be 1-D integers of length {n + 1}, got {off.dtype} {off.shape}")
    if off[0] != 0 or off[-1] != len(g) or (np.diff(off) < 0).any():
        raise RuntimeError(f"{d}: grasp_offsets must start at 0, never decrease and end at {len(g)}")
    for k in ("n_slots", "n_processed", "n_ok"):          # required: a deleted field must not bypass a check
        if k not in summary:
            raise RuntimeError(f"{d}: summary lacks required field {k!r}")
    for k in ("n_slots", "n_processed"):
        if summary[k] != n:
            raise RuntimeError(f"{d}: summary {k}={summary[k]} != {n} index rows")
    n_ok = 0
    for i, r in enumerate(index):
        if str(r.get("slot")) != str(i):
            raise RuntimeError(f"{d}: index row {i} has slot {r.get('slot')!r}")
        length = int(off[i + 1] - off[i])
        if r.get("status") == "ok":
            n_ok += 1
            try:
                ng = int(r.get("n_grasps"))
            except (TypeError, ValueError):
                raise RuntimeError(f"{d}: ok slot {i} has n_grasps {r.get('n_grasps')!r}") from None
            if ng <= 0 or length != ng:
                raise RuntimeError(f"{d}: ok slot {i}: {length} stored grasps != n_grasps {ng}")
        elif length != 0:
            raise RuntimeError(f"{d}: excluded slot {i} ({r.get('status')}) stores {length} grasps")
    if summary["n_ok"] != n_ok:
        raise RuntimeError(f"{d}: summary n_ok={summary['n_ok']} != {n_ok} ok rows")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_identity(d, split, index, summary, splits):
    """Which split and which scenes (review finding 3). The summary must name `split`;
    rows are unique (part, object_id, view) in the builder's sorted order; every object belongs to the
    frozen split; a production build holds every object of the split and n_slots == n_scenes_in_split
    (a smoke build holds a subset)."""
    if summary.get("split") != split:
        raise RuntimeError(f"{d}: summary belongs to split {summary.get('split')!r}, not {split!r}")
    if split not in splits.get("splits", {}):
        raise RuntimeError(f"{d}: split {split!r} is not defined in the splits file")
    try:
        keys = [(r["part"], r["object_id"], int(r["view"])) for r in index]
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"{d}: index row without a valid part/object_id/view ({e})") from None
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"{d}: duplicate (part, object_id, view) rows in index.csv")
    if keys != sorted(keys):                                 # builder contract: sorted objects, sorted views
        raise RuntimeError(f"{d}: index rows are not in the builder's sorted order")
    objs, want = {f"{p}/{o}" for p, o, _ in keys}, set(splits["splits"][split])
    extra = sorted(objs - want)
    if extra:
        raise RuntimeError(f"{d}: {len(extra)} objects not in split {split!r}, e.g. {extra[:3]}")
    if not summary.get("smoke"):
        missing = sorted(want - objs)
        if missing:
            raise RuntimeError(f"{d}: {len(missing)} objects of split {split!r} missing from the cache, "
                               f"e.g. {missing[:3]}")
        if summary.get("n_scenes_in_split") != len(index):
            raise RuntimeError(f"{d}: {len(index)} slots != n_scenes_in_split "
                               f"{summary.get('n_scenes_in_split')!r}")


def open_cache(cache_dir, split, allow_smoke=False, splits=None):
    """Read-only view of one built split: {"arrays": {name: memmap}, "index": [row dicts],
    "summary": dict, "grasps": fn(slot) -> (n, 5) at 1024}. Refuses incomplete or smoke caches.

    splits: None (production) -> configs/splits.json, which must be the file summary["splits_sha256"] names
            (same sha256, or bridged by data.provenance.config_matches);
            a dict ({"splits": {name: [objects]}}) -> used as given, no hash check (tests, synthetic).
    """
    d = Path(cache_dir).expanduser() / split
    marker = d / "COMPLETE"
    if not marker.is_file():
        raise RuntimeError(f"{d}: no COMPLETE marker (build unfinished or failed)")
    summary = json.loads((d / "summary.json").read_text())
    if summary.get("layout_version") != LAYOUT_VERSION:
        raise RuntimeError(f"{d}: layout_version {summary.get('layout_version')} != {LAYOUT_VERSION}")
    if summary.get("smoke") and not allow_smoke:
        raise RuntimeError(f"{d}: smoke cache (--max-scenes); pass allow_smoke=True only for checks")
    if summary.get("complete") is not True or summary.get("error") is not None:
        raise RuntimeError(f"{d}: summary says complete={summary.get('complete')!r}, "
                           f"error={summary.get('error')!r}")
    specs = array_specs(summary["size"], summary["taus"])
    arrays = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in specs}
    with open(d / "index.csv", newline="") as fh:
        index = list(csv.DictReader(fh))
    for n, a in arrays.items():
        if a.shape != (len(index),) + specs[n][1] or a.dtype != specs[n][0]:
            raise RuntimeError(f"{d}/{n}.npy: {a.dtype} {a.shape} does not match the layout")
    g = np.load(d / "grasps_1024.npy", mmap_mode="r")
    off = np.load(d / "grasp_offsets.npy")
    check_index(d, index, summary, g, off)
    if splits is None:
        if not summary.get("splits_sha256"):
            raise RuntimeError(f"{d}: summary lacks splits_sha256; pass splits= explicitly for a non-production cache")
        if not config_matches(SPLITS_FILE, summary["splits_sha256"]):
            raise RuntimeError(f"{d}: {SPLITS_FILE} sha256 {_sha256(SPLITS_FILE)} != the cache's splits_sha256 "
                               f"{summary['splits_sha256']} (splits changed since the build)")
        splits = json.loads(SPLITS_FILE.read_text())
    check_identity(d, split, index, summary, splits)

    def grasps(slot):
        return np.array(g[off[slot]:off[slot + 1]])

    return {"arrays": arrays, "index": index, "summary": summary, "grasps": grasps}
