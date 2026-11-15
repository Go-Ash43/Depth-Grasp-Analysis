#!/usr/bin/env bash
# Dummy runs: every pipeline command on a tiny input (data pipeline, smoke training of all four models,
# Protocol A, metric decision, run selection, H1 on validation), plus comparisons against the recorded
# production outputs where the result must be identical. Reads the dataset, cache and production runs; writes
# ONLY inside <out_dir>. Each step: log in <out_dir>/logs/<step>.log, PASS / FAIL / SKIP line in
# <out_dir>/summary.txt; a failure does not stop later steps. Exit 0 iff no step failed.
#
# Usage (from anywhere; commit first: the post-training steps refuse a dirty tree):
#   DATA_ROOT=... CACHE=... MANIFEST_DIR=... RUNS=... MANIFEST=... bash scripts/dummy_runs.sh <out_dir>
# Required environment:
#   DATA_ROOT     Jacquard v2 "Extracted" folder (read-only)
#   CACHE         production 224-px cache dir (read-only)
#   MANIFEST_DIR  folder holding the dataset manifest.csv that configs/splits.json pins (read-only)
#   RUNS          production runs root (A0_s1 ... M_s3_pod, metric_decision_*, m1_selection, m1_h1_val, ...)
#   MANIFEST      cache manifest.json the production runs were verified against
# Optional: WORKERS (default 4), SEED (default 0, the subsample seed), TRAIN_COMMIT (default 2b85aa6),
#   SKIP_ASTAR=1, RUN_EVAL=1 (adds eval_m1 on val from the FRESH selection, ~30 min, and H1 on that new
#   evaluation with the fresh metric decision; both compared with the recorded ones), DEMO_PKG (demo package
#   folder, default demo_data/ in the repo; demo.py --check runs when it holds samples.json).
set -u
OUT=${1:?usage: bash scripts/dummy_runs.sh <out_dir>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
: "${DATA_ROOT:?}" "${CACHE:?}" "${MANIFEST_DIR:?}" "${RUNS:?}" "${MANIFEST:?}"
W=${WORKERS:-4}; SEED=${SEED:-0}; TRAIN_COMMIT=${TRAIN_COMMIT:-2b85aa6}

OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
case "$OUT/" in "$REPO"/*) echo "refusing: $OUT is inside the repo"; exit 2;; esac
[ -z "$(ls -A "$OUT")" ] || { echo "refusing: $OUT is not empty (use a fresh directory)"; exit 2; }
mkdir -p "$OUT/logs" "$OUT/manifests" "$OUT/runs"
SUMMARY="$OUT/summary.txt"

# Paths: the real dataset / cache / manifest folder, every default output folder inside $OUT.
PATHS_REAL="$OUT/paths_real.json"; PATHS_SCRATCH="$OUT/paths_scratch.json"
printf '{"data_root": "%s", "cache_dir": "%s", "runs_dir": "%s", "manifest_dir": "%s"}\n' \
    "$DATA_ROOT" "$CACHE" "$OUT/runs" "$MANIFEST_DIR" > "$PATHS_REAL"
printf '{"data_root": "%s", "cache_dir": "%s", "runs_dir": "%s", "manifest_dir": "%s"}\n' \
    "$DATA_ROOT" "$CACHE" "$OUT/runs" "$OUT/manifests" > "$PATHS_SCRATCH"
export DEPTH_LIES_PATHS="$PATHS_REAL"

printf '{"script": "scripts/dummy_runs.sh", "out": "%s", "git_commit": "%s", "git_dirty": %s, "workers": %s, "seed": %s, "train_commit": "%s", "data_root": "%s", "cache": "%s", "manifest_dir": "%s", "runs": "%s", "manifest": "%s", "skip_astar": "%s", "run_eval": "%s"}\n' \
    "$OUT" "$(git rev-parse HEAD)" "$([ -z "$(git status --porcelain)" ] && echo false || echo true)" "$W" "$SEED" \
    "$TRAIN_COMMIT" "$DATA_ROOT" "$CACHE" "$MANIFEST_DIR" "$RUNS" "$MANIFEST" "${SKIP_ASTAR:-}" "${RUN_EVAL:-}" \
    > "$OUT/dummy_config.json"

NFAIL=0
step() {   # step <name> <command...>
    local name=$1; shift
    local t0; t0=$(date +%s)
    echo "--- [$name] $*" >> "$OUT/commands.log"
    printf '%-26s running...\r' "$name"
    "$@" > "$OUT/logs/$name.log" 2>&1
    local rc=$? st=PASS
    [ $rc -eq 0 ] || { st=FAIL; NFAIL=$((NFAIL + 1)); }
    printf '%-26s %-4s exit %-3s %5ss\n' "$name" "$st" "$rc" "$(( $(date +%s) - t0 ))" | tee -a "$SUMMARY"
    return $rc
}
skip() { printf '%-26s SKIP (%s)\n' "$1" "$2" | tee -a "$SUMMARY"; }

json_same() {   # json_same <new> <recorded> <ignored key...>: equal except the ignored keys (at any depth)
    python - "$@" <<'EOF'
import json, sys
ign = set(sys.argv[3:])
def drop(o):
    if isinstance(o, dict):
        return {k: drop(v) for k, v in o.items() if k not in ign}
    if isinstance(o, list):
        return [drop(v) for v in o]
    return o
a, b = (drop(json.load(open(p))) for p in sys.argv[1:3])
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print("compared", sys.argv[1], "with", sys.argv[2], "ignoring (any depth)", sorted(ign))
print("differing top-level keys:", diff or "none")
sys.exit(1 if diff else 0)
EOF
}
eval_files_same() {   # the per-file sha256 map of two eval_record.json files
    python - "$@" <<'EOF'
import json, sys
a, b = (json.load(open(p))["files"] for p in sys.argv[1:3])
bad = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print(len(a), "files;", "differing:", bad or "none")
sys.exit(1 if bad else 0)
EOF
}

echo "dummy runs -> $OUT   (commit $(git rev-parse --short HEAD), workers $W)" | tee "$SUMMARY"

# ---------------------------------------------------------------- A. data pipeline (tiny inputs)
step manifest_limit200    python scripts/make_manifest.py --config "$PATHS_SCRATCH" --limit 200 --workers "$W"
step make_split           python scripts/make_split.py --config "$PATHS_REAL" --manifest "$MANIFEST_DIR/manifest.csv" \
                              --seed 0 --out "$OUT/splits_rebuilt.json" \
  && step cmp_splits      json_same "$OUT/splits_rebuilt.json" configs/splits.json manifest created git_commit git_dirty
step audit_train_200      python scripts/audit_train.py --max-scenes 200 --seed "$SEED" --workers "$W" --out "$OUT/audit_train" \
  && step audit_report    python scripts/audit_report.py --audit-dir "$OUT/audit_train"
step angle_convention     python scripts/check_angle_convention.py --config "$PATHS_REAL" \
                              --manifest "$MANIFEST_DIR/manifest.csv" --n 3 --seed "$SEED" --out "$OUT/angle_check"
step build_cache_val50    python scripts/build_cache.py --splits val --max-scenes 50 --seed "$SEED" --workers "$W" \
                              --out "$OUT/cache_smoke"
step check_cache_totals   python scripts/check_cache_totals.py --cache-dir "$CACHE"
step accept_cache_val     python scripts/accept_cache.py --cache-dir "$CACHE" --splits val --n-rederive 5 --seed "$SEED" \
                              --out "$OUT/accept_cache"
step overlays             python scripts/overlays.py --cache-dir "$CACHE" --n-scenes 3 --n-stats 50 --seed "$SEED" \
                              --out "$OUT/overlays"
step zone_diagnostics     python scripts/zone_diagnostics.py --cache-dir "$CACHE" --n-scenes 50 --n-overlays 2 \
                              --seed "$SEED" --out "$OUT/zone_diagnostics"
step cache_manifest_write python scripts/cache_manifest.py write --cache-dir "$CACHE" --splits val --workers 8 \
                              --out "$OUT/cache_manifest_val" \
  && step cache_manifest_verify python scripts/cache_manifest.py verify --cache-dir "$CACHE" \
                              --manifest "$OUT/cache_manifest_val/manifest.json" --workers 8 --out "$OUT/cache_manifest_verify"

# ---------------------------------------------------------------- B. smoke training + Protocol A
MODELS="R A0 M"; [ -n "${SKIP_ASTAR:-}" ] || MODELS="$MODELS Astar"
for m in $MODELS; do
    step "train_smoke_$m"  python scripts/train.py --kind smoke --model "$m" --seed 1 --epochs 1 --limit-train 64 \
                              --batch-size 8 --num-workers 2 --hash-workers 8 --progress-every 4 --allow-dirty \
                              --manifest "$MANIFEST" --device cuda --out "$OUT/train_smoke_$m"
done
CK="$OUT/train_smoke_%s/ckpt/epoch_000.pt"
if [ -f "$(printf "$CK" R)" ] && [ -f "$(printf "$CK" A0)" ] && [ -f "$(printf "$CK" M)" ]; then
    step protocol_a       python scripts/gate_protocol_a.py --ckpt-R "$(printf "$CK" R)" --ckpt-A0 "$(printf "$CK" A0)" \
                              --ckpt-M "$(printf "$CK" M)" --n 64 --batch-size 8 --device cuda --out "$OUT/protocol_a"
else
    skip protocol_a "a smoke checkpoint is missing"
fi

# ---------------------------------------------------------------- C. post-training chain on the real runs
step metric_decision      python scripts/metric_decision.py --a0-run "$RUNS/A0_s1" --out "$OUT/metric_decision" \
  && step cmp_metric      json_same "$OUT/metric_decision/metric_decision.json" \
                              "$RUNS/metric_decision_local/metric_decision.json" decided config_path
if [ -f "$OUT/metric_decision/metric_decision.json" ]; then
    step m1_runs          python scripts/m1_runs.py --runs-root "$RUNS" --commit "$TRAIN_COMMIT" \
                              --m3-pod "$RUNS/M_s3_pod" --m3-laptop "$RUNS/M_s3" \
                              --metric-local "$OUT/metric_decision/metric_decision.json" \
                              --metric-pod "$RUNS/metric_decision_pod/metric_decision.json" --out "$OUT/m1_selection" \
      && step cmp_selection json_same "$OUT/m1_selection/selected_runs.json" "$RUNS/m1_selection/selected_runs.json" \
                              metric_decision made config_path
else
    skip m1_runs "no metric decision"
fi
# H1 on the RECORDED val evaluation and selection (the evaluation is bound to that selection's sha256).
# The recorded metric decision stores the absolute config path of the machine that made it: on any other
# machine this step is skipped (by design); RUN_EVAL=1 below re-runs the chain on the fresh records instead.
REC_CFG=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"]["config_path"])' \
              "$RUNS/metric_decision_local/metric_decision.json" 2>/dev/null)
if [ -n "$REC_CFG" ] && [ -f "$REC_CFG" ]; then
    step h1_val           python scripts/h1_harness.py --selected "$RUNS/m1_selection/selected_runs.json" --split val \
                              --eval-dir "$RUNS/m1_eval_val" --config configs/train_v1.json \
                              --metric-local "$RUNS/metric_decision_local/metric_decision.json" --out "$OUT/h1_val" \
      && step cmp_h1_val  json_same "$OUT/h1_val/h1.json" "$RUNS/m1_h1_val/h1.json" inputs made \
      && step cmp_h1_tables cmp "$OUT/h1_val/h1_tables.md" "$RUNS/m1_h1_val/h1_tables.md"
else
    skip h1_val "recorded config path not on this machine"
fi

# ---------------------------------------------------------------- D. optional: val evaluation + H1 on the FRESH records
if [ -z "${RUN_EVAL:-}" ]; then
    skip eval_m1_val "RUN_EVAL not set"
elif [ ! -f "$OUT/m1_selection/selected_runs.json" ]; then
    skip eval_m1_val "no fresh selection"
else
    step eval_m1_val      python scripts/eval_m1.py --selected "$OUT/m1_selection/selected_runs.json" --split val \
                              --manifest "$MANIFEST" --device cuda --num-workers 2 --hash-workers 8 --out "$OUT/eval_val" \
      && step cmp_eval_val eval_files_same "$OUT/eval_val/results/eval_record.json" \
                              "$RUNS/m1_eval_val/results/eval_record.json" \
      && step h1_val_new_eval python scripts/h1_harness.py --selected "$OUT/m1_selection/selected_runs.json" --split val \
                              --eval-dir "$OUT/eval_val" --config configs/train_v1.json \
                              --metric-local "$OUT/metric_decision/metric_decision.json" --out "$OUT/h1_val_new_eval" \
      && step cmp_h1_new_eval json_same "$OUT/h1_val_new_eval/h1.json" "$RUNS/m1_h1_val/h1.json" inputs made \
      && step cmp_h1_new_tables cmp "$OUT/h1_val_new_eval/h1_tables.md" "$RUNS/m1_h1_val/h1_tables.md"
fi

# ---------------------------------------------------------------- E. demo on the shipped package, vs the recorded evaluation
DEMO_PKG=${DEMO_PKG:-$REPO/demo_data}
if [ -f "$DEMO_PKG/samples.json" ]; then
    step demo_check       python demo.py --package "$DEMO_PKG" --check --device cuda --out "$OUT/demo"
else
    skip demo_check "no demo package at $DEMO_PKG"
fi

echo "failed steps: $NFAIL   (summary: $SUMMARY, logs: $OUT/logs/)" | tee -a "$SUMMARY"
[ "$NFAIL" -eq 0 ]
