#!/usr/bin/env bash
# Unattended end-to-end run of the physics-first model, committing results after every step.
#
#   1. shoplite LP for every case (resumes from checkpoints; waits if already running)
#   2. walk-forward of LP + LightGBM, stride 3          -> report/runs/walkforward_physics_s3.csv
#   3. submission trained on all 2,922 labelled cases    -> submission_physics.csv
#   4. walk-forward of the submitted XGBoost model, stride 3 (fold-by-fold baseline)
#                                                        -> report/runs/walkforward_baseline_s3.csv
#
# Usage (from sintef_project/src):  nohup bash run_physics_pipeline.sh > pipeline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
RUNS=../report/runs
mkdir -p "$RUNS"
BRANCH=$(git rev-parse --abbrev-ref HEAD)

push() {   # commit the given paths and push, retrying on network errors
    local msg=$1; shift
    git add "$@" && git commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L7eDZui1LadL3GHQ9XHVUB" || return 0
    for wait in 2 4 8 16 0; do
        git push -q -u origin "$BRANCH" && return 0
        [ "$wait" -gt 0 ] && sleep "$wait"
    done
    echo "push failed after retries"
}

step() { echo "=== $(date -u +%H:%M:%S) $*"; }

step "1. LP solutions"
while [ ! -f ../data/processed/shoplite_lp_unit.parquet ]; do
    if ! pgrep -f "shoplite_batch.py lp" > /dev/null; then
        python shoplite_batch.py lp
    else
        sleep 30
    fi
done

step "2. walk-forward, LP + LightGBM"
python physics_model.py walkforward 3 "$RUNS/walkforward_physics_s3.csv" \
    | tee "$RUNS/walkforward_physics_s3.txt"
push "Walk-forward results: shoplite LP + LightGBM (stride 3)" "$RUNS"

step "3. submission"
python physics_model.py submit ../submission_physics.csv | tee "$RUNS/submit_physics.txt"
push "Submission from shoplite LP + LightGBM trained on all labelled cases" \
    ../submission_physics.csv "$RUNS"

step "4. walk-forward, submitted XGBoost baseline"
if ! python baseline_walkforward.py 3 "$RUNS/walkforward_baseline_s3.csv" \
        | tee "$RUNS/walkforward_baseline_s3.txt"; then
    step "baseline at stride 3 failed (memory?), retrying at stride 5"
    python baseline_walkforward.py 5 "$RUNS/walkforward_baseline_s5.csv" \
        | tee "$RUNS/walkforward_baseline_s5.txt"
fi
push "Walk-forward results: submitted XGBoost model, same folds" "$RUNS"
step "done"
