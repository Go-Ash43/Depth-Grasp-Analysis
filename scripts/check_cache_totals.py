#!/usr/bin/env python3
"""Cache acceptance check across splits (docs/RULES_preprocessing.md R3, clarification).

R3 says "abort if more than 50 scenes (~0.1%) are skipped in total". The builder enforces 50 / 10-in-a-row
PER SPLIT while it runs; this script enforces the chosen clarification on the TOTAL:

  sum over the required production splits of skip_budget.n_skipped  <=  50

(loader skips, i.e. SkipBudget records. `no_grasps` is both a loader skip and an R1 exclusion;
`no_object_224` / `no_valid_depth_224` are R1 exclusions only and are not counted). It also fails if
any required split is missing, incomplete (no COMPLETE marker or complete != true), has an error, is
not explicitly a production build (smoke must be exactly false), belongs to another split (summary
"split" != folder), or has an invalid skip count (not a non-negative int, a bool, or != the number of
stored skip records). Only the current summary.json of each split is read, so superseded attempts
(replaced by --overwrite rebuilds) are never counted.

The sealed shard is NOT read unless listed explicitly (final sealed-test run only: --splits train,val,test_module,test_sealed).
Read-only: opens <cache>/<split>/summary.json and checks <cache>/<split>/COMPLETE; writes nothing.
Exit status 0 = PASS, 1 = FAIL.

Usage:
  python scripts/check_cache_totals.py                                   # train,val,test_module
  python scripts/check_cache_totals.py --splits train,val,test_module,test_sealed   # final sealed-test run only
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.paths import load_paths  # noqa: E402

MAX_TOTAL_SKIPS = 50                      # RULES R3
DEFAULT_SPLITS = ("train", "val", "test_module")
KNOWN_SPLITS = ("train", "val", "test_module", "test_sealed")


def _count(v):
    """A real non-negative int (bool is an int subclass in Python and is rejected)."""
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def check_totals(cache_dir, splits, max_total=MAX_TOTAL_SKIPS):
    """-> report dict with per-split facts, problems and verdict ("PASS" / "FAIL")."""
    root = Path(cache_dir).expanduser()
    per, problems, total = {}, [], 0
    bad = [s for s in splits if s not in KNOWN_SPLITS]
    if bad:
        raise ValueError(f"unknown split(s) {bad}; choose from {KNOWN_SPLITS}")
    for sp in splits:
        d = root / sp
        p = d / "summary.json"
        if not p.is_file():
            problems.append(f"{sp}: no summary.json")
            per[sp] = None
            continue
        s = json.loads(p.read_text())
        sb = s.get("skip_budget") or {}
        n_sk, recs = sb.get("n_skipped"), sb.get("skips")
        per[sp] = {"split": s.get("split"), "complete": s.get("complete"), "error": s.get("error"),
                   "smoke": s.get("smoke"), "marker": (d / "COMPLETE").is_file(), "n_skipped": n_sk,
                   "n_skip_records": len(recs) if isinstance(recs, list) else None, "n_ok": s.get("n_ok"),
                   "n_slots": s.get("n_slots"), "excluded_by_reason": s.get("excluded_by_reason"),
                   "git_commit": s.get("git_commit")}
        if s.get("split") != sp:
            problems.append(f"{sp}: summary belongs to split {s.get('split')!r}, not {sp!r}")
        if s.get("smoke") is not False:
            problems.append(f"{sp}: not an explicit production build (smoke={s.get('smoke')!r})")
        if s.get("complete") is not True or not (d / "COMPLETE").is_file():
            problems.append(f"{sp}: incomplete (complete={s.get('complete')}, COMPLETE marker="
                            f"{(d / 'COMPLETE').is_file()})")
        if s.get("error") is not None:
            problems.append(f"{sp}: error recorded: {str(s.get('error'))[:200]}")
        if not _count(n_sk):
            problems.append(f"{sp}: skip_budget.n_skipped missing or invalid ({n_sk!r})")
        elif not isinstance(recs, list) or len(recs) != n_sk:
            problems.append(f"{sp}: skip_budget.n_skipped {n_sk} != stored skip records "
                            f"({len(recs) if isinstance(recs, list) else recs!r})")
        else:
            total += n_sk
    if total > max_total:
        problems.append(f"total loader skips {total} > {max_total} (R3): cache invalid, examine before use")
    return {"splits": list(splits), "per_split": per, "total_skips": total, "max_total_skips": max_total,
            "problems": problems, "verdict": "PASS" if not problems else "FAIL"}


def main():
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=paths["cache_dir"])
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                    help="comma list; add test_sealed only for the final sealed-test run")
    ap.add_argument("--seed", type=int, default=0, help="unused (deterministic); logged for convention")
    a = ap.parse_args()
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    if not splits or len(set(splits)) != len(splits):
        ap.error("--splits must be a non-empty list without repeats")
    if any(s not in KNOWN_SPLITS for s in splits):
        ap.error(f"--splits must come from {','.join(KNOWN_SPLITS)}")
    rep = check_totals(a.cache_dir, splits)
    print(json.dumps(rep, indent=1, default=str))
    print(f"\n{rep['verdict']}: total loader skips {rep['total_skips']} / {rep['max_total_skips']}"
          + ("" if not rep["problems"] else "\n  " + "\n  ".join(rep["problems"])))
    sys.exit(0 if rep["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
