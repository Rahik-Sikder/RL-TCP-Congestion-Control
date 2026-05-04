# Evaluation Suite

Network testing suite that runs PPO, NewReno, and Cubic congestion control algorithms across four emulated network scenarios using Kathara containers.

## How It Works

The suite uses a **UDP-based custom reliable transport** written in C. Because standard Linux does not expose a socket option to set TCP's kernel-managed congestion window from userspace, we implement our own reliable transport on top of UDP. This gives us full ownership of `cwnd`, RTT measurement, duplicate-ACK tracking, and timeout detection — the same signals the model was trained on.

Kathara emulates each network scenario by shaping traffic inside Linux containers via `tc netem`. The sender binary picks a CC algorithm at startup; everything else is identical across algorithms, making results directly comparable.

## Prerequisites

**On the host (build machine):**
```bash
# macOS
brew install onnxruntime

# Linux (Debian/Ubuntu)
apt-get install libonnxruntime-dev
# or download from https://github.com/microsoft/onnxruntime/releases
```

**Kathara** must be installed and the `kathara` CLI must be on `$PATH`. See https://github.com/KatharaFramework/Kathara.

## Quick Start

### 1. Build

```bash
cd evaluation/src
make all          # release build
make debug        # same + -DDEBUG -g (verbose stderr output)
make clean
```

The Makefile auto-detects the ONNX Runtime prefix via `brew --prefix onnxruntime` on macOS, or falls back to `/usr` on Linux.

### 2. Local smoke test (no Kathara)

Verify the binaries work before running any Kathara scenarios:

```bash
# Terminal 1
./evaluation/src/receiver --port 5000

# Terminal 2
mkdir -p /tmp/eval_test
./evaluation/src/sender \
  --cc newreno \
  --dest-ip 127.0.0.1 --port 5000 \
  --bytes 10485760 \
  --out-dir /tmp/eval_test \
  --scenario loopback_test
```

Check output:
```bash
cat /tmp/eval_test/summary.json
wc -l /tmp/eval_test/timeseries.csv
```

To test PPO, add `--cc ppo --model path/to/ppo_output_dir`. The directory must contain both `ppo.onnx` and `ppo.info.json` (produced automatically by `training/train_agent.py`).

### 3. Run all scenarios

NewReno and Cubic always run as baselines. Pass one or more model directories to also run RL-based algorithms — the algorithm name is taken from the directory prefix (everything before the first `_`):

```bash
# Baselines only (newreno + cubic)
./evaluation/run_eval.sh

# Baselines + PPO (auto-detects most recent training/outputs/ppo_* directory)
./evaluation/run_eval.sh training/outputs/ppo_2026-04-15_21-49-25

# Baselines + multiple RL models in one run
./evaluation/run_eval.sh \
  training/outputs/ppo_2026-05-01_10-00-00 \
  training/outputs/dqn_2026-05-02_14-30-00 \
  training/outputs/ddpg_2026-05-03_09-15-00
```

Each model directory must contain `<algo>.onnx` and `<algo>.info.json` (produced by `training/train_agent.py`). The directory name prefix determines which `--cc` value is passed to the sender (e.g. `ppo_...` → `--cc ppo`).

This iterates over all 4 scenarios × (2 baselines + N models). Each run:
1. Starts the Kathara lab for the scenario
2. Launches the receiver inside its container
3. Runs the sender with the chosen CC algorithm
4. Copies `summary.json` and `timeseries.csv` out of the container
5. Tears the lab down

Results land in `evaluation/results/run_TIMESTAMP/{scenario}_{algo}/`.

### 4. Run a single scenario manually

```bash
cd evaluation/kathara/scenarios/simple_p2p
kathara lstart

# Inside sender container:
kathara exec sender -- /eval/sender \
  --cc cubic \
  --dest-ip 10.0.0.2 --port 5000 \
  --bytes 10485760 \
  --out-dir /results \
  --scenario simple_p2p

kathara wipe -F
```

## Sender CLI Reference

```
./sender --cc ppo|newreno|cubic
         [--model DIR]       # required when --cc ppo; dir with ppo.onnx + ppo.info.json
         --dest-ip IP
         --port PORT
         [--bytes N]         # total bytes to send (default: 10 MB)
         [--out-dir DIR]     # where to write summary.json + timeseries.csv (default: results/)
         [--scenario NAME]   # label written into summary.json
```

## Output Files

Each run produces two files in `--out-dir`:

**`summary.json`**
```json
{
  "scenario": "high_loss",
  "algorithm": "ppo",
  "model_path": "training/outputs/ppo_2026-04-15_21-49-25",
  "avg_throughput_mbps": 4.21,
  "avg_rtt_ms": 23.1,
  "total_loss_rate": 0.031,
  "total_packets_sent": 7500,
  "total_packets_acked": 7267,
  "total_timeouts": 14
}
```

**`timeseries.csv`** — one row per ACK event:
```
timestamp_ms,cwnd_bytes,rtt_ms,dup_ack_count,timeout_count
0.000,14000,0.312,0,0
0.412,15400,0.298,0,0
...
```

Use these for comparing cwnd evolution, RTT response, and loss behavior across algorithms and scenarios.

## Scenarios

| Name | Topology | Network conditions |
|---|---|---|
| `simple_p2p` | sender ↔ receiver | 20 ms delay, 10 Mbit/s |
| `bottleneck` | sender → router → receiver | 5 ms each hop, 1 Mbit/s bottleneck on router egress |
| `variable_latency` | sender ↔ receiver | 10 Mbit/s, delay cycles 10 ms → 30 ms → 50 ms → 20 ms every 5 s |
| `high_loss` | sender ↔ receiver | 20 ms delay, 5% random packet loss, 10 Mbit/s |

## Algorithms

| Name | `--cc` value | Notes |
|---|---|---|
| TCP NewReno | `newreno` | Baseline — slow start, AIMD, fast retransmit after 3 dupACKs |
| TCP Cubic | `cubic` | Baseline — W_cubic window function (C=0.4, β=0.7) |
| RL agent (any) | e.g. `ppo`, `dqn`, `ddpg` | Loads `<algo>.onnx`; reads `k` from `<algo>.info.json` in the model dir |

## Debug Output

Build with `make debug` (or `-DDEBUG`) to get verbose per-packet logs on stderr:

```
[transport] SEND seq=0 chunk=1400 bytes_sent=1400
[transport] ACK seq=0 rtt_ms=0.312 cwnd=14000 dup=0
[new_reno] on_ack: cwnd=15400 ssthresh=65536
[ppo] on_ack: rtt=0.31ms dup=0 timeout=0 cwnd=14000
[ppo] queue not full yet (1/10), skipping inference
...
[ppo] action_mean=0.1234 old_cwnd=14000 new_cwnd=15374
```

Redirect to a file per run: `./sender ... 2>debug.log`.
