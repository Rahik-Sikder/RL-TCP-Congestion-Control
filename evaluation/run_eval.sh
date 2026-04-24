#!/usr/bin/env bash
# run_eval.sh — orchestrate all scenario × algorithm evaluation runs.
#
# Usage:
#   ./run_eval.sh [PPO_MODEL_DIR]
#
# PPO_MODEL_DIR defaults to the most recent training/outputs/ppo_* directory.
# Results land in evaluation/results/{scenario}_{algo}_{timestamp}/
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$EVAL_DIR/src"
SCENARIOS_DIR="$EVAL_DIR/kathara/scenarios"
RESULTS_DIR="$EVAL_DIR/results"

# Use first arg as PPO model dir, or auto-detect most recent training output
PPO_MODEL="${1:-}"
if [ -z "$PPO_MODEL" ]; then
    PPO_MODEL="$(ls -td "$EVAL_DIR/../training/outputs"/ppo_* 2>/dev/null | head -1 || true)"
fi

ALGORITHMS="newreno cubic ppo"
SCENARIOS="simple_p2p bottleneck variable_latency high_loss"
RECEIVER_IP="10.0.0.2"
RECEIVER_PORT="5000"
TRANSFER_BYTES="$((10 * 1024 * 1024))"   # 10 MB

log() { echo "[run_eval $(date '+%H:%M:%S')] $*"; }

# ── Build ──────────────────────────────────────────────────────────────────
log "Building evaluation binaries..."
make -C "$SRC_DIR" all
log "Build complete."

mkdir -p "$RESULTS_DIR"

# ── Run matrix ────────────────────────────────────────────────────────────
for scenario in $SCENARIOS; do
    for algo in $ALGORITHMS; do

        if [ "$algo" = "ppo" ] && [ -z "$PPO_MODEL" ]; then
            log "WARNING: No PPO model found — skipping ppo × $scenario"
            continue
        fi

        ts="$(date +%Y%m%d_%H%M%S)"
        run_id="${scenario}_${algo}_${ts}"
        run_dir="$RESULTS_DIR/$run_id"
        mkdir -p "$run_dir"

        log "===== $scenario × $algo ====="
        log "Results → $run_dir"

        lab_dir="$SCENARIOS_DIR/$scenario"

        # Start Kathara lab
        log "Starting Kathara lab..."
        (cd "$lab_dir" && kathara start) || { log "kathara start failed, skipping"; continue; }
        sleep 2   # let containers initialize networking

        # Launch receiver in background inside its container
        log "Starting receiver in container..."
        kathara exec receiver -- /eval/receiver --port "$RECEIVER_PORT" &
        RECV_PID=$!
        sleep 1

        # Build model flag for ppo
        MODEL_ARG=""
        [ "$algo" = "ppo" ] && MODEL_ARG="--model $PPO_MODEL"

        # Run sender inside its container
        log "Starting sender (algo=$algo)..."
        kathara exec sender -- /eval/sender \
            --cc      "$algo" \
            $MODEL_ARG \
            --dest-ip "$RECEIVER_IP" \
            --port    "$RECEIVER_PORT" \
            --bytes   "$TRANSFER_BYTES" \
            --out-dir /results \
            --scenario "$scenario" \
        || log "sender exited with error (non-fatal)"

        # Copy results out of container
        log "Collecting results..."
        kathara exec sender -- cat /results/summary.json    > "$run_dir/summary.json"    2>/dev/null || true
        kathara exec sender -- cat /results/timeseries.csv  > "$run_dir/timeseries.csv"  2>/dev/null || true

        # Tear down
        kill "$RECV_PID" 2>/dev/null || true
        log "Stopping Kathara lab..."
        (cd "$lab_dir" && kathara wipe -F) 2>/dev/null || log "kathara wipe failed (non-fatal)"

        log "Run $run_id complete."
        echo ""
    done
done

# ── Summary ───────────────────────────────────────────────────────────────
log "All runs complete. Results in $RESULTS_DIR"
log "Throughput summary:"
for d in "$RESULTS_DIR"/*/; do
    if [ -f "$d/summary.json" ]; then
        tp=$(grep -o '"avg_throughput_mbps": [0-9.]*' "$d/summary.json" || echo "?")
        printf "  %-45s  %s\n" "$(basename "$d")" "$tp"
    fi
done
