#!/usr/bin/env bash
# run_eval.sh — orchestrate all scenario × algorithm evaluation runs.
#
# Usage:
#   ./run_eval.sh [PPO_MODEL_DIR]
#
# PPO_MODEL_DIR defaults to the most recent training/outputs/ppo_* directory.
# All results for one run land in evaluation/results/run_TIMESTAMP/.
# When Kathara is not available, falls back to local loopback mode.
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$EVAL_DIR/src"
SCENARIOS_DIR="$EVAL_DIR/kathara/scenarios"
RESULTS_DIR="$EVAL_DIR/results"

PPO_MODEL="${1:-}"
if [ -z "$PPO_MODEL" ]; then
    PPO_MODEL="$(ls -td "$EVAL_DIR/../training/outputs"/ppo_* 2>/dev/null | head -1 || true)"
fi

ALGORITHMS="newreno cubic ppo"
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
for scenario in $SCENARIOS; do
    for algo in $ALGORITHMS; do

        if [ "$algo" = "ppo" ] && [ -z "$PPO_MODEL" ]; then
            log "WARNING: No PPO model found — skipping ppo × $scenario"
            continue
        fi

        test_dir="$RUN_DIR/${scenario}_${algo}"
        log "===== $scenario × $algo ====="

        MODEL_ARG=""
        [ "$algo" = "ppo" ] && MODEL_ARG="--model $PPO_MODEL"

        if [ "$USE_KATHARA" = "1" ]; then
            # ── Kathara container mode ────────────────────────────────────
            lab_dir="$SCENARIOS_DIR/$scenario"
            recv_ip="$(receiver_ip_for "$scenario")"

            # Copy PPO model files into the scenario shared dir so containers can access them
            if [ "$algo" = "ppo" ] && [ -n "$PPO_MODEL" ]; then
                cp -r "$PPO_MODEL" "$lab_dir/shared/ppo.onnx"
                info_json="${PPO_MODEL%.onnx}.info.json"
                [ -f "$info_json" ] && cp "$info_json" "$lab_dir/shared/ppo.info.json"
                MODEL_ARG="--model /shared/ppo.onnx"
            fi

            # Wipe any leftover containers from previous runs or failed teardowns
            kathara wipe -f 2>/dev/null || true

            log "Starting Kathara lab..."
            if ! (cd "$lab_dir" && kathara lstart --noterminals); then
                log "kathara lstart failed — skipping $scenario × $algo"
                kathara wipe -f 2>/dev/null || true
                continue
            fi
            sleep 2

            log "Starting receiver in container..."
            (cd "$lab_dir" && kathara exec receiver -- /shared/run_receiver.sh --port "$RECEIVER_PORT") &
            RECV_PID=$!
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

        else
            # ── Local loopback mode ───────────────────────────────────────
            mkdir -p "$test_dir"

            log "Starting local receiver on port $RECEIVER_PORT..."
            "$SRC_DIR/receiver" --port "$RECEIVER_PORT" &
            RECV_PID=$!
            sleep 0.3

            log "Starting local sender (algo=$algo)..."
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
