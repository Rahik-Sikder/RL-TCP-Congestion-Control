#!/usr/bin/env bash
# run_eval.sh — orchestrate all scenario × algorithm evaluation runs.
#
# Usage:
#   ./run_eval.sh [MODEL_DIR ...]
#
# Each MODEL_DIR should be a path whose basename starts with the algorithm name
# followed by an underscore (e.g. ppo_2026-05-03_23-09-09, ddpg_..., dqn_...).
# The algorithm name is extracted automatically from the directory prefix.
#
# If no MODEL_DIRs are given, the script defaults to the most recent
# training/outputs/ppo_* directory (preserving backwards-compatibility).
#
# Baseline algorithms (newreno, cubic) always run regardless of model args.
# All results for one run land in evaluation/results/run_TIMESTAMP/.
# When Kathara is not available, falls back to local loopback mode.
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$EVAL_DIR/src"
SCENARIOS_DIR="$EVAL_DIR/kathara/scenarios"
RESULTS_DIR="$EVAL_DIR/results"

# Build parallel arrays: MODEL_ALGOS and MODEL_DIRS
MODEL_ALGOS=()
MODEL_DIRS=()

if [ "$#" -gt 0 ]; then
    for arg in "$@"; do
        base="$(basename "$arg")"
        algo="${base%%_*}"   # everything before the first underscore
        MODEL_ALGOS+=("$algo")
        MODEL_DIRS+=("$(cd "$arg" && pwd)")
    done
else
    # Backwards-compatible default: latest ppo_* directory
    default_ppo="$(ls -td "$EVAL_DIR/../training/outputs"/ppo_* 2>/dev/null | head -1 || true)"
    if [ -n "$default_ppo" ]; then
        MODEL_ALGOS+=("ppo")
        MODEL_DIRS+=("$(cd "$default_ppo" && pwd)")
    fi
fi

BASELINE_ALGORITHMS="newreno cubic"
SCENARIOS="simple_p2p bottleneck variable_latency high_loss"
RECEIVER_PORT="5000"
TRANSFER_BYTES="$((10 * 1024 * 1024))"   # 10 MB

# Receiver IP differs per scenario (bottleneck routes through a separate subnet)
receiver_ip_for() {
    case "$1" in
        bottleneck) echo "10.0.1.2" ;;
        *)          echo "10.0.0.2" ;;
    esac
}

log() { echo "[run_eval $(date '+%H:%M:%S')] $*"; }

# ── Build ──────────────────────────────────────────────────────────────────
log "Building evaluation binaries..."
make -C "$SRC_DIR" all
log "Build complete."

# ── Single run directory for all tests in this invocation ─────────────────
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RESULTS_DIR/run_$RUN_TS"
mkdir -p "$RUN_DIR"
log "Run directory: $RUN_DIR"

# ── Detect Kathara ────────────────────────────────────────────────────────
USE_KATHARA=0
if command -v kathara &>/dev/null; then
    USE_KATHARA=1
    log "Kathara detected — using container mode."
else
    log "Kathara not found — using local loopback mode (127.0.0.1)."
fi


# ── Run matrix ────────────────────────────────────────────────────────────
run_one() {
    local scenario="$1" algo="$2" model_dir="${3:-}"
    local test_dir="$RUN_DIR/${scenario}_${algo}"
    log "===== $scenario × $algo ====="

    MODEL_ARG=""
    [ -n "$model_dir" ] && MODEL_ARG="--model $model_dir"

    if [ "$USE_KATHARA" = "1" ]; then
        # ── Kathara container mode ────────────────────────────────────
        local lab_dir="$SCENARIOS_DIR/$scenario"
        local recv_ip
        recv_ip="$(receiver_ip_for "$scenario")"

        # Copy model files into the scenario shared dir so containers can access them
        if [ -n "$model_dir" ]; then
            rm -rf "$lab_dir/shared/${algo}"
            cp -r "$model_dir" "$lab_dir/shared/${algo}"
            MODEL_ARG="--model /shared/${algo}"
        fi

        # Wipe any leftover containers from previous runs or failed teardowns
        kathara wipe -f 2>/dev/null || true

        log "Starting Kathara lab..."
        if ! (cd "$lab_dir" && kathara lstart --noterminals); then
            log "kathara lstart failed — skipping $scenario × $algo"
            kathara wipe -f 2>/dev/null || true
            return
        fi
        sleep 2

        log "Starting receiver in container..."
        (cd "$lab_dir" && kathara exec receiver -- /shared/run_receiver.sh --port "$RECEIVER_PORT") &
        local RECV_PID=$!
        sleep 1

        log "Starting sender (algo=$algo)..."
        # shellcheck disable=SC2086
        (cd "$lab_dir" && kathara exec sender -- /shared/run_sender.sh \
            --cc "$algo" $MODEL_ARG \
            --dest-ip "$recv_ip" \
            --port "$RECEIVER_PORT" \
            --bytes "$TRANSFER_BYTES" \
            --out-dir /tmp/results \
            --scenario "$scenario") \
        || log "sender exited with error (non-fatal)"

        # Copy results out of container
        mkdir -p "$test_dir"
        (cd "$lab_dir" && kathara exec sender -- cat /tmp/results/summary.json)   > "$test_dir/summary.json"   2>/dev/null || true
        (cd "$lab_dir" && kathara exec sender -- cat /tmp/results/timeseries.csv) > "$test_dir/timeseries.csv" 2>/dev/null || true

        kill "$RECV_PID" 2>/dev/null || true
        log "Stopping Kathara lab..."
        (cd "$lab_dir" && kathara lclean) 2>/dev/null || log "kathara lclean failed (non-fatal)"

        # Clean up copied model directory from shared dir
        [ -n "$model_dir" ] && rm -rf "$lab_dir/shared/${algo}"

    else
        # ── Local loopback mode ───────────────────────────────────────
        mkdir -p "$test_dir"

        log "Starting local receiver on port $RECEIVER_PORT..."
        "$SRC_DIR/receiver" --port "$RECEIVER_PORT" &
        local RECV_PID=$!
        sleep 0.3

        log "Starting local sender (algo=$algo)..."
        # shellcheck disable=SC2086
        "$SRC_DIR/sender" \
            --cc      "$algo" \
            $MODEL_ARG \
            --dest-ip "127.0.0.1" \
            --port    "$RECEIVER_PORT" \
            --bytes   "$TRANSFER_BYTES" \
            --out-dir "$test_dir" \
            --scenario "$scenario" \
        || log "sender exited with error (non-fatal)"

        kill "$RECV_PID" 2>/dev/null || true
        wait "$RECV_PID" 2>/dev/null || true
    fi

    log "Test ${scenario}_${algo} complete → $test_dir"
    echo ""
}

for scenario in $SCENARIOS; do
    # Baselines
    for algo in $BASELINE_ALGORITHMS; do
        run_one "$scenario" "$algo"
    done

    # Model-based algorithms
    for i in "${!MODEL_ALGOS[@]}"; do
        run_one "$scenario" "${MODEL_ALGOS[$i]}" "${MODEL_DIRS[$i]}"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────
log "All runs complete. Results in $RUN_DIR"
log "Throughput summary:"
for d in "$RUN_DIR"/*/; do
    if [ -f "$d/summary.json" ]; then
        tp=$(grep -o '"avg_throughput_mbps": [0-9.]*' "$d/summary.json" || echo "?")
        printf "  %-45s  %s\n" "$(basename "$d")" "$tp"
    fi
done
