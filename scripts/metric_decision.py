#!/usr/bin/env python3
"""Record the metric-promotion decision once, from production A0 seed 1's final checkpoint (proj_v6 7.5).

Run once after the A0 seed-1 production run prints `done:`. CPU only, seconds (hashes one checkpoint and
one CSV). Writes <out>/metric_decision.json. Re-running with an identical decision is a no-op (exit 0);
a different decision for the same --out is refused. Downstream code reads it via
eval.metric_policy.load_decision, never by parsing the file itself.

    python scripts/metric_decision.py --a0-run $R/prod_A0_s1 --config configs/train_v1.json \
        --out $R/metric_decision
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.provenance import git_fields  # noqa: E402
from eval.metric_policy import DECIDED_KEY, PolicyError, comparable, decide, load_decision  # noqa: E402
from training.checkpoint import atomic_write_json  # noqa: E402

OUT_NAME = "metric_decision.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a0-run", required=True, help="production A0 seed-1 run directory")
    p.add_argument("--config", default=str(REPO / "configs" / "train_v1.json"), help="the frozen config")
    p.add_argument("--out", required=True, help="output directory (metric_decision.json)")
    p.add_argument("--seed", type=int, default=0, help="unused (no randomness); recorded by convention")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    try:
        rec = decide(a.a0_run, a.config)
    except PolicyError as e:
        raise SystemExit(f"metric_decision refused: {e}")
    out = Path(a.out).expanduser()
    target = out / OUT_NAME
    if target.exists():
        old = json.loads(target.read_text())
        if comparable(old) != comparable(rec):
            raise SystemExit(f"metric_decision refused: {target} holds a different decision "
                             f"(recorded metric {old.get('metric')!r}, now {rec['metric']!r}); not overwriting")
        print(f"already recorded (identical): {target} -> metric {rec['metric']!r}")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    rec[DECIDED_KEY] = {**git_fields(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a)}
    atomic_write_json(target, rec)
    load_decision(target)                                  # read back through the only sanctioned path
    c = rec["clean"]
    print(f"metric {rec['metric']!r}: clean primary {c['matches_primary']}/{c['n']} = {100 * c['top1_primary']:.2f} % "
          f"(promote iff > {rec['threshold_pct']} %) -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
