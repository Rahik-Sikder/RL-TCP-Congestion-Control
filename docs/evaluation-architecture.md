# Evaluation Suite Architecture

## Overview

The evaluation suite tests TCP congestion control algorithms (PPO, NewReno, Cubic) inside Kathara network emulation containers. Because Linux does not expose a standard socket option for setting TCP's kernel-managed congestion window from userspace, the suite implements its own **UDP-based reliable transport** in C. This lets every algorithm control `cwnd`, RTT measurement, duplicate-ACK tracking, and timeout detection directly — without kernel modifications or eBPF.

---

## Directory Layout

```
evaluation/
├── README.md
├── run_eval.sh                          # full 4-scenario × 3-algorithm orchestration
├── results/                             # output written here per run
│   └── {scenario}_{algo}_{timestamp}/
│       ├── summary.json
│       └── timeseries.csv
├── kathara/
│   └── scenarios/
│       ├── simple_p2p/                  # 2-node, 20 ms, 10 Mbit/s
│       ├── bottleneck/                  # 3-node, 1 Mbit/s bottleneck
│       ├── variable_latency/            # 2-node, cycling delay every 5 s
│       └── high_loss/                   # 2-node, 5% random loss
└── src/
    ├── Makefile
    ├── sender.c                         # main sender binary
    ├── receiver.c                       # main receiver binary
    ├── transport/
    │   ├── transport.h                  # shared types + transport API
    │   ├── transport.c                  # UDP event loop implementation
    │   ├── queue.h                      # drop-head ring buffer API
    │   └── queue.c                      # ring buffer implementation
    └── cc/
        ├── cc.h                         # cc_ops_t interface + constructor declarations
        ├── new_reno.h / new_reno.c      # NewReno algorithm
        ├── cubic.h   / cubic.c          # Cubic algorithm
        └── ppo.h     / ppo.c            # PPO via ONNX Runtime
```

---

## Component Descriptions

### `transport/transport.h` — Shared Types

The single source of truth for all types used across the transport and CC layers.

**`tcp_info_t`** — per-packet telemetry, our userspace equivalent of the kernel's `tcp_info` struct:
```c
typedef struct {
    float rtt_ms;   // round-trip time for this packet
    int   dup_ack;  // 1 = this ACK was a duplicate
    int   timeout;  // 1 = this packet timed out before being ACKed
    float cwnd;     // cwnd in bytes at the time this packet was sent
} tcp_info_t;
```

**`cc_ops_t`** — the congestion control plugin interface. Every algorithm implements all four function pointers:
```c
typedef struct {
    void  (*on_ack)(void *ctx, tcp_info_t *info);
    void  (*on_timeout)(void *ctx, tcp_info_t *info);
    float (*get_cwnd)(void *ctx);   // returns current cwnd in bytes
    void  (*destroy)(void *ctx);
} cc_ops_t;
```

**Wire formats:**
```c
// Packet sent from sender to receiver
typedef struct __attribute__((packed)) {
    uint32_t seq;
    uint16_t data_len;
    uint64_t send_time_us;   // monotonic timestamp at send time
} pkt_hdr_t;

// ACK sent from receiver back to sender
typedef struct __attribute__((packed)) {
    uint32_t seq;            // echoes the packet's seq number
} ack_hdr_t;
```

**Constants:**
| Name | Value | Meaning |
|---|---|---|
| `MSS` | 1400 bytes | Max segment size |
| `MAX_WINDOW` | 1024 | Max in-flight packets tracked simultaneously |
| `TIMEOUT_US` | 200 000 µs | Per-packet timeout (200 ms) |
| `INIT_CWND_SEGS` | 10 | Initial cwnd in segments |

---

### `transport/transport.c` — UDP Transport

Implements a sliding-window reliable transport on top of UDP.

**Send path (`transport_run`):**
1. Compute `max_inflight = max(1, cwnd / MSS)` from the current CC module
2. Send new packets (header + payload via `sendmsg`) up to `max_inflight`
3. Record `window[seq % MAX_WINDOW] = {send_time_us, seq, acked=0}`

**ACK path (`drain_acks`):**
1. Non-blocking `select(1 ms)` poll
2. `recv` each `ack_hdr_t`
3. Validate `window[seq % MAX_WINDOW].seq == ack.seq` to guard against slot aliasing after timeouts
4. Compute RTT, detect dupACKs (same `seq` as previous ACK → `dup_ack_count++`)
5. Call `cc->on_ack(ctx, &tcp_info_t{...})`
6. Append a row to `timeseries.csv`

**Timeout path (`check_timeouts`):**
1. Scan all `MAX_WINDOW` slots
2. Retire any slot where `age > TIMEOUT_US`
3. Call `cc->on_timeout(ctx, &tcp_info_t{timeout=1, ...})`

**All debug output** is gated on `#ifdef DEBUG` and prefixed `[transport]`.

---

### `transport/queue.h` / `transport/queue.c` — Drop-Head Ring Buffer

Used exclusively by the PPO module to maintain a sliding window of `k` recent telemetry entries.

**Semantics:**
- `queue_push`: if full, **drop the head** (oldest entry) before inserting at tail — keeps the most recent `k` entries
- `queue_read_all`: copies all entries in FIFO order into a caller-provided buffer **without draining** the queue — inference reads without consuming
- `queue_size`: returns current count

**Future improvement** (noted in `queue.c`): instead of dropping head entries, split the buffer into `k` equal buckets and replace the oldest entry per bucket. `queue_read_all` then samples one entry per bucket in order, giving uniform temporal coverage across the full window regardless of arrival rate.

---

### `cc/cc.h` — CC Interface

Declares the `cc_ops_t` type (imported from `transport.h`) and the three constructor functions:

```c
cc_ops_t *new_reno_create(void **ctx);
cc_ops_t *cubic_create(void **ctx);
cc_ops_t *ppo_create(void **ctx, const char *model_dir);
```

Each constructor allocates the algorithm's private state, sets it in `*ctx`, and returns a pointer to a static `cc_ops_t` table.

---

### `cc/new_reno.c` — NewReno

| Event | Behavior |
|---|---|
| ACK, `cwnd < ssthresh` | Slow start: `cwnd += MSS` |
| ACK, `cwnd >= ssthresh` | Congestion avoidance: `cwnd += MSS²/cwnd` |
| 3rd dupACK | Fast retransmit: `ssthresh = cwnd/2`, `cwnd = ssthresh + 3·MSS` |
| Timeout | `ssthresh = max(cwnd/2, 2·MSS)`, `cwnd = MSS` (slow-start restart) |

Initial state: `cwnd = 10·MSS`, `ssthresh = 65536`.

---

### `cc/cubic.c` — Cubic

Implements the Cubic window function:

```
W_cubic(t) = C · (t − K)³ + W_max
```

where `C = 0.4`, `β = 0.7`, `K = cbrt(W_max · (1−β) / C)`, and `t` is seconds since the last congestion event.

| Event | Behavior |
|---|---|
| ACK | If `W_cubic(t) > cwnd`, set `cwnd = W_cubic(t)` (clamped to [MSS, 1000·MSS]) |
| Timeout | `W_max = cwnd`, `cwnd = cwnd · β`, recompute `K`, reset epoch |

---

### `cc/ppo.c` — PPO via ONNX Runtime

Loads a trained PPO model and runs inference after every `k` ACKs.

**Startup (`ppo_create`):**
1. Initialize ONNX Runtime via `OrtGetApiBase()->GetApi(ORT_API_VERSION)`
2. Read `k` from `{model_dir}/ppo.info.json`
3. Create ONNX session from `{model_dir}/ppo.onnx`
4. Allocate drop-head queue of capacity `k`

**On each ACK (`ppo_on_ack`):**
1. Push `tcp_info_t` to the queue
2. If `inference_busy` flag is set: skip (packet telemetry is dropped for this inference cycle)
3. If `queue_size < k`: skip (warming up)
4. Run inference

**Inference (`ppo_run_inference`):**
1. Read all `k` entries from queue (FIFO, no drain)
2. Build state vector of length `k·3 + 1`:
   - `state[0..k-1]` = RTT history (ms)
   - `state[k..2k-1]` = dupACK flags (0 or 1)
   - `state[2k..3k-1]` = timeout flags (0 or 1)
   - `state[3k]` = `cwnd / MSS` (cwnd in segments)
3. Run ONNX session: input `"state"` → outputs `"action_mean"`, `"action_std"`, `"value"`
4. Apply cwnd update: `cwnd = cwnd · 2^action_mean` (same formula used during NS-3 training)
5. Clamp result to `[1·MSS, 1000·MSS]`

The state vector composition matches exactly what the NS-3 training environment sent to the model, ensuring inference is consistent with training.

---

### `sender.c` — Main Sender

CLI entry point. Selects the CC module, creates a UDP socket connected to the receiver, opens the timeseries CSV, runs `transport_run`, then writes `summary.json`.

**Algorithm selection:**
```c
if      (strcmp(cc_name, "newreno") == 0) cc = new_reno_create(&ctx);
else if (strcmp(cc_name, "cubic")   == 0) cc = cubic_create(&ctx);
else if (strcmp(cc_name, "ppo")     == 0) cc = ppo_create(&ctx, model_dir);
```

After `transport_run` returns, it reads stats directly from the `transport_t` struct (`total_sent`, `total_acked`, `total_timeouts`, `rtt_sum_ms`, `rtt_count`) to compute summary metrics.

---

### `receiver.c` — Main Receiver

Binds a UDP socket, loops on `recvfrom`, and sends an `ack_hdr_t` back to the sender for every received packet. Prints bytes/sec throughput to stdout every second. Exits cleanly on `SIGINT`.

---

### Kathara Scenarios

Each scenario is a self-contained Kathara lab directory. To modify a scenario, edit its `lab.conf` and `*.startup` files:

| File | Controls |
|---|---|
| `lab.conf` | Container names, link assignments, memory limits |
| `sender.startup` | IP address, `tc netem` shaping on eth0, optional background delay-cycling script |
| `receiver.startup` | IP address setup |
| `router.startup` | IP forwarding, `tc tbf` for bandwidth limiting (bottleneck only) |

**Shaping commands by scenario:**

| Scenario | Command |
|---|---|
| `simple_p2p` | `tc qdisc add dev eth0 root netem delay 20ms rate 10mbit` |
| `bottleneck` | router: `tc qdisc add dev eth1 root tbf rate 1mbit burst 10kb latency 50ms` |
| `variable_latency` | starts at 10 ms, background loop changes every 5 s: `tc qdisc change dev eth0 root netem delay Xms` |
| `high_loss` | `tc qdisc add dev eth0 root netem delay 20ms loss 5% rate 10mbit` |

---

### `run_eval.sh` — Orchestration

Iterates over all `{scenario} × {algorithm}` combinations in order. For each run:

```
build → kathara start → launch receiver → run sender → copy results → kathara wipe
```

PPO runs are skipped automatically if no model directory is found. Pass the model directory as `$1` to override auto-detection.

---

## Data Flow

```
sender.c
  │
  ├── new_reno_create / cubic_create / ppo_create
  │      └── returns cc_ops_t* + ctx*
  │
  └── transport_create(sock_fd, cc, ctx, ts_file)
         │
         └── transport_run(t, total_bytes)
                │
                ├── [send packet] → UDP → receiver.c → [send ACK] → UDP
                │
                ├── drain_acks()
                │     └── cc->on_ack(ctx, &tcp_info_t)
                │           ├── new_reno: update cwnd via AIMD
                │           ├── cubic:   update cwnd via W_cubic(t)
                │           └── ppo:     push to queue → maybe run inference → update cwnd
                │
                ├── check_timeouts()
                │     └── cc->on_timeout(ctx, &tcp_info_t)
                │
                └── write row to timeseries.csv
```

---

## Build System

The `Makefile` in `evaluation/src/` auto-detects the ONNX Runtime location:

```makefile
ONNX_PREFIX ?= $(shell brew --prefix onnxruntime 2>/dev/null || echo /usr)
INCS        = -I. -Itransport -Icc -I$(ONNX_PREFIX)/include
SENDER_LIBS = -lm -L$(ONNX_PREFIX)/lib -lonnxruntime
```

On macOS with Homebrew this resolves to `/opt/homebrew/opt/onnxruntime`. On Linux it falls back to `/usr`, which is correct for `apt-get install libonnxruntime-dev`.

The `sender` binary links against `libonnxruntime`; the `receiver` binary does not.
