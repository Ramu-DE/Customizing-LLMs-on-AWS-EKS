# Day 20 — Dynamic Batching

> **Hook:** Trading latency vs throughput intelligently.

---

## The Post

Every inference system faces a fundamental tension:

**Throughput** wants bigger batches. Process more requests per GPU second.  
**Latency** wants smaller batches. Respond to each user faster.

Dynamic batching is how you navigate this trade-off — not by picking one side, but by adapting to real-time conditions.

---

## The Core Trade-off

```
  THE BATCHING DILEMMA
  ═════════════════════

  SMALL BATCH (batch size = 1)
  ┌─────────────────────────────────────────────────────────────┐
  │  Pros: Minimum TTFT (no wait for other requests)           │
  │  Cons: GPU massively under-utilized                        │
  │        Each request pays full GPU overhead                 │
  │        Cost per request: HIGH                              │
  │                                                             │
  │  GPU:  [Req A]     [Req B]     [Req C]     [Req D]        │
  │        ████░░░░    ████░░░░    ████░░░░    ████░░░░        │
  │        25% util    25% util    25% util    25% util        │
  └─────────────────────────────────────────────────────────────┘

  LARGE BATCH (batch size = 32)
  ┌─────────────────────────────────────────────────────────────┐
  │  Pros: GPU at 90%+ utilization                             │
  │        Much lower cost per request                         │
  │  Cons: TTFT = wait until batch is full                     │
  │        If requests trickle in: users wait for nothing      │
  │        Tail latency explodes                               │
  │                                                             │
  │  GPU:  ░░░░░░░░░░░░░░░░░░[WAIT COLLECTING][A+B+C+...+Z]   │
  │        waiting...                          ███████████████  │
  └─────────────────────────────────────────────────────────────┘

  DYNAMIC BATCHING: adapt batch size to real-time traffic
```

---

## Dynamic Batching Strategies

```
  THREE DYNAMIC BATCHING APPROACHES
  ════════════════════════════════════

  ① TIMEOUT-BASED DYNAMIC BATCHING
  ──────────────────────────────────
  Wait up to T milliseconds OR until batch reaches max size:

  arrival: ─A─────B──C────────────────D──E──F──G──H──
  time:     0ms   5ms 8ms                50ms 52ms 55ms 60ms 63ms
                      │                  │
                      ▼                  ▼
  batch:         [A, B, C]         [D, E, F, G, H]
                  timeout=10ms      timeout=10ms, maxsize=5

  Config: timeout=10ms, max_batch_size=32
  Sweet spot: 5-20ms timeout for interactive workloads

  ② TOKEN-BUDGET BATCHING
  ──────────────────────────
  Fill batch until total token count hits budget:

  Req A: 500 tokens total  ▓▓▓▓▓▓▓▓▓▓
  Req B: 300 tokens total  ▓▓▓▓▓▓
  Req C: 800 tokens total  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  Req D: 200 tokens total  ▓▓▓▓
  Req E: 900 tokens total  would overflow 2048 budget

  Batch 1: [A + B + C + D] = 1800 tokens → send (under 2048)
  Batch 2: [E + ...]

  vLLM: --max-num-batched-tokens 8192

  ③ CONTINUOUS BATCHING (per-step iteration)
  ────────────────────────────────────────────
  No fixed batch window — reassemble batch at every decode step:

  Step N:   running=[A,B,C]    waiting=[D,E,F]
  C finishes→ Step N+1: running=[A,B,D]   waiting=[E,F]
  B finishes→ Step N+2: running=[A,D,E]   waiting=[F]

  This is vLLM's default. Most powerful. (See Day 15)
```

---

## The Batching Decision Matrix

```
  WHICH STRATEGY FOR WHICH WORKLOAD?
  ════════════════════════════════════

  ┌──────────────────────┬──────────────┬────────────────────────┐
  │ Workload Type        │ Strategy     │ Key Config             │
  ├──────────────────────┼──────────────┼────────────────────────┤
  │ Interactive chat     │ Continuous   │ max-num-seqs=256       │
  │ (low latency req'd)  │ (vLLM)      │ no added wait          │
  │                      │             │                        │
  │ Batch API / offline  │ Max size     │ max-batch-size=256     │
  │ (throughput matters) │ timeout=5s  │ long timeout OK        │
  │                      │             │                        │
  │ Mixed traffic        │ Token budget │ max-tokens=8192        │
  │ (variable lengths)   │ + timeout   │ timeout=50ms           │
  │                      │             │                        │
  │ Real-time streaming  │ Continuous  │ chunked prefill        │
  │ (TPOT critical)      │ + prefetch  │ streaming=true         │
  │                      │             │                        │
  │ Agent workloads      │ Priority    │ priority queue         │
  │ (multi-step)         │ scheduling  │ short req first        │
  └──────────────────────┴──────────────┴────────────────────────┘
```

---

## The Latency-Throughput Curve in Practice

```
  MEASURING YOUR OPTIMAL OPERATING POINT
  ════════════════════════════════════════

  Load generator (inference-perf, Module 300) sweeps concurrency:

  Concurrency:  1    2    4    8   16   32   64  128
  ──────────────────────────────────────────────────────────
  Throughput:   2    4    8   14   22   30   33   34  req/s
  TTFT p50:    50   50   55   70  120  280  650  2000 ms
  TTFT p99:    80   90  110  160  350  800 2000  8000 ms
  GPU Util:   20%  35%  55%  70%  80%  88%  92%   93%

  ┌──────────────────────────────────────────────────────────┐
  │                    TTFT p99 (ms)                        │
  │ 8000│                                              ●    │
  │ 2000│                                         ●        │
  │  800│                                    ●             │
  │  350│                               ●                  │
  │  160│                          ●                       │
  │  110│               ●                                  │
  │   90│          ●                                       │
  │   80│     ●                                            │
  │      ──────────────────────────────────────────────    │
  │       1    2    4    8   16   32   64   128             │
  │                  Concurrency                            │
  │                                                        │
  │  SLA: TTFT p99 < 500ms → MAX concurrency = 32         │
  │  At concurrency 32: 30 req/s throughput, 88% GPU util │
  │  This is your operating point.                        │
  └────────────────────────────────────────────────────────┘
```

---

## Priority Scheduling — Not All Requests Are Equal

```
  PRIORITY-AWARE BATCHING
  ════════════════════════

  Real production systems have mixed request types:

  ┌─────────────────────┬──────────┬──────────────────────────┐
  │ Request Type        │ Priority │ SLA                      │
  ├─────────────────────┼──────────┼──────────────────────────┤
  │ Real-time chat      │ HIGH     │ TTFT < 200ms             │
  │ Agent reasoning     │ MEDIUM   │ TTFT < 500ms             │
  │ Document processing │ LOW      │ TTFT < 5s                │
  │ Batch fine-tune eval│ LOWEST   │ Best-effort              │
  └─────────────────────┴──────────┴──────────────────────────┘

  Priority queue in inference server:

  Incoming:  [Chat:HIGH][Doc:LOW][Agent:MED][Batch:LOWEST][Chat:HIGH]
                                                                │
                                                   Sorted by priority
                                                                │
  GPU batch: [Chat:HIGH][Chat:HIGH][Agent:MED][Doc:LOW][Batch:LOWEST]

  High-priority requests never wait behind slow batch jobs.
  Preemption: running low-priority request can be paused
              when high-priority request arrives.
  vLLM supports priority-based scheduling with --scheduling-policy priority
```

---

## Prefill/Decode Disaggregation — Advanced Dynamic Batching

```
  THE PREFILL STALL PROBLEM
  ══════════════════════════

  In mixed continuous batching, a large prefill step STALLS decode:

  Step N:   [Decode A: 1ms][Decode B: 1ms][Decode C: 1ms] = 3ms total
  Step N+1: [Prefill D: 800ms][Decode A: 1ms]             = 801ms total
                              ↑
             Decode A's TPOT jumps from 3ms to 801ms!
             D's 800-token prompt disrupted everyone else.

  Solution: Separate Prefill and Decode workers (P/D Disaggregation)
  ┌──────────────────────────────────────────────────────────────┐
  │                                                               │
  │  PREFILL NODES (compute-optimized)                          │
  │  ┌─────────────────────────────┐                            │
  │  │  Process all new prompts    │  KV blocks transferred     │
  │  │  Compute-heavy, parallel    │ ──────────────────────────▶│
  │  └─────────────────────────────┘                            │
  │                                                               │
  │  DECODE NODES (memory-BW-optimized)                         │
  │  ┌─────────────────────────────┐                            │
  │  │  Only run decode steps      │                            │
  │  │  Steady, predictable TPOT  │                            │
  │  └─────────────────────────────┘                            │
  │                                                               │
  │  Result: Decode TPOT never spikes regardless of prefill load│
  └──────────────────────────────────────────────────────────────┘
```

---

## Workshop Connection

```
  DYNAMIC BATCHING SETTINGS IN OUR WORKSHOP
  ═══════════════════════════════════════════

  Module 100 (vLLM baseline):
  --max-num-seqs 256              # continuous batch size cap
  --max-num-batched-tokens 8192   # token budget per step

  Module 300 (Benchmarking):
  # inference-perf sweeps concurrency automatically
  # Saturation test reveals batch behavior under load
  # Grafana panel: "Running/Waiting Requests" shows queue depth

  Module 800 (Ray Serve):
  # Ray Serve adds another layer of dynamic batching
  # Across multiple vLLM replicas
  # Request routing: round-robin or least-loaded
  # Auto-scales replica count based on queue depth
  #   scale up when queue > 10 requests
  #   scale down when GPU util < 30%
```

---

## Key Takeaway

> Dynamic batching is how inference servers serve thousands of users from one GPU.  
> The goal: maximize GPU utilization without violating your TTFT SLA.  
> Find the "knee" in your latency-throughput curve — that's your optimal operating point.  
> For production: continuous batching + token budget + priority scheduling + P/D disaggregation.

---

*30-Day Series: LLM Inference Is Everything | Day 20 of 30*
*← [Day 19](./day-19-prefix-caching.md) | Next → [Day 21](./day-21-inference-throughput.md)*
