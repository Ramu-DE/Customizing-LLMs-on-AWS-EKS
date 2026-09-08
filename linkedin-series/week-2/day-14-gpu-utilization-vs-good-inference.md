# Day 14 — GPU Utilization ≠ Good Inference

> **Hook:** You can have 90% GPU utilization and still have terrible economics.

---

## The Post

"GPU utilization is at 90%. We're in great shape."

I've heard this from engineers who were simultaneously burning money and delivering slow responses.

GPU utilization is one metric. It's not the metric. Here's why — and what to track instead.

---

## The Dangerous Vanity Metric

```
  THE PROBLEM WITH CHASING GPU UTILIZATION
  ══════════════════════════════════════════

  Scenario A: 90% GPU utilization — GOOD
  ─────────────────────────────────────────

  GPU:  [Req A compute][Req B compute][Req C compute][Req D compute]
        ████████████████████████████████████████████████████████████
        90% utilization — doing real work, batching well

  Throughput: 500 req/min
  Cost:       $1.00/hr
  Cost/req:   $0.003

  ─────────────────────────────────────────────────────────────────

  Scenario B: 90% GPU utilization — BAD
  ─────────────────────────────────────────

  GPU:  [Req A — 128K context compute]
        ████████████████████████████████████████████████████████████
        90% utilization — but doing 1 request at a time, huge context

  Throughput: 3 req/min
  Cost:       $1.00/hr
  Cost/req:   $5.56

  ─────────────────────────────────────────────────────────────────

  Same GPU utilization number. 1,850× different cost per request.
  Utilization tells you the GPU is busy. Not that it's efficient.
```

---

## The Metrics That Actually Matter

```
  THE INFERENCE ECONOMICS DASHBOARD
  ════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │                                                              │
  │  LATENCY METRICS (user experience)                          │
  │  ─────────────────────────────────                         │
  │  TTFT p50/p95/p99    → Is the system responsive?           │
  │  TPOT p50/p95/p99    → Does generation feel smooth?        │
  │  End-to-end latency  → Full request duration               │
  │                                                              │
  │  THROUGHPUT METRICS (capacity)                              │
  │  ─────────────────────────────                             │
  │  Requests/second     → How many users served               │
  │  Tokens/second       → Raw generation capacity             │
  │  Input tokens/second → Prefill throughput                  │
  │  Output tokens/second→ Decode throughput                   │
  │                                                              │
  │  EFFICIENCY METRICS (economics)                             │
  │  ─────────────────────────────                             │
  │  Cost per 1M tokens  → Dollar efficiency                   │
  │  GPU memory utilized → Are we packing enough requests?     │
  │  Batch size average  → Are we batching effectively?        │
  │  Queue depth         → Are we bottlenecked?                │
  │  Cache hit rate      → Is prefix caching working?          │
  │                                                              │
  │  GPU HARDWARE METRICS (hardware health)                     │
  │  ─────────────────────────────────────                     │
  │  SM Utilization      → Streaming Multiprocessor activity   │
  │  Memory BW utilization → Is memory the bottleneck?        │
  │  VRAM used/available → KV cache headroom                  │
  │  GPU temperature     → Thermal throttling risk             │
  │                                                              │
  └─────────────────────────────────────────────────────────────┘
```

---

## What High GPU Utilization Can Actually Mean

```
  5 SCENARIOS WHERE 90% UTILIZATION IS MISLEADING
  ═════════════════════════════════════════════════

  ❶ You're processing one huge request
  ─────────────────────────────────────
  GPU:  [128K context, 1 user] ████████████████████████████████
  Util: 90%  |  Concurrent users: 1  |  Cost/user: $$$

  ❷ Your batch size is too small (underutilized batch dimension)
  ─────────────────────────────────────────────────────────────
  GPU:  [small batch] █ [idle] ░░░ [small batch] █ [idle] ░░░
  Utilization averages to 90% but each batch underutilizes FLOPS
  Fix: Increase max_batch_size in vLLM config

  ❸ Memory bandwidth is the actual bottleneck, not compute
  ──────────────────────────────────────────────────────────
  Compute units: 90% "busy" (waiting for data)
  Memory bandwidth: 99% saturated ← THE REAL BOTTLENECK
  Adding more compute cores would do nothing
  Fix: Quantize to reduce weight sizes → less data to load

  ❹ The model is too large (offloading to CPU RAM)
  ────────────────────────────────────────────────
  GPU compute: 90% busy loading data from CPU RAM
  Actual throughput: 10× below potential
  Fix: Quantize or use a smaller model

  ❺ You have a single slow request holding up fast ones
  ──────────────────────────────────────────────────────
  Request A: 128K context  [GPU busy: 5 seconds]
  Request B: 50 token chat [waiting in queue: 5 seconds]
  Request C: 50 token chat [waiting in queue: 5 seconds]
  GPU utilization: 90%  |  P99 TTFT: 5+ seconds  ← disaster
  Fix: Preemption + priority scheduling (vLLM supports this)
```

---

## The Real Efficiency Metric: MFU

```
  MODEL FLOPS UTILIZATION (MFU)
  ══════════════════════════════

  MFU = Actual FLOPS achieved / Peak FLOPS possible

  This tells you what fraction of the GPU's raw capability
  you're actually using for useful model computation.

  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  Peak FLOPS (L40S, FP16):  183 TFLOPS                    │
  │                                                            │
  │  Scenario A (good batching):                              │
  │  Actual FLOPS: ~90 TFLOPS   MFU = 90/183 = 49%           │
  │  This is excellent for LLM inference (decode is hard)    │
  │                                                            │
  │  Scenario B (1 request at a time):                       │
  │  Actual FLOPS: ~5 TFLOPS    MFU = 5/183 = 2.7%           │
  │  GPU utilization: "90%" (memory BW bound)                │
  │  But effective compute utilization: terrible             │
  │                                                            │
  └────────────────────────────────────────────────────────────┘

  High SM utilization + low MFU = memory-bound decode
  (Very common in LLM inference. Not a bug — it's physics.)
```

---

## The Utilization vs Saturation vs Error Framework

```
  USE (Utilization, Saturation, Errors) FOR GPU INFERENCE
  ═════════════════════════════════════════════════════════

  UTILIZATION: Is the resource being used?
  ──────────────────────────────────────────
  GPU SM Utilization     → % time doing compute
  GPU Memory Utilization → % VRAM used
  Target: High SM util during prefill, moderate during decode

  SATURATION: Is the resource overloaded?
  ─────────────────────────────────────────
  Request queue depth    → Are requests waiting?
  TTFT degradation       → Is prefill taking longer under load?
  KV cache at 100%       → New requests being rejected/queued
  Target: Queue depth < 10, TTFT < 2× baseline under load

  ERRORS: Is anything failing?
  ──────────────────────────────
  OOM (out of memory) errors  → VRAM exceeded
  Request timeouts           → Decode taking too long
  Dropped requests           → Server overloaded
  Target: Zero OOM, < 0.1% timeouts

  All three must be healthy simultaneously.
  High utilization + high saturation = bad (you're overloaded).
  High utilization + low saturation + zero errors = good.
```

---

## The Throughput-Latency Trade-off Curve

```
  FINDING YOUR OPERATING POINT
  ═════════════════════════════

  Latency
  (TTFT ms)
  │
  800│                                              ●
    │                                         ●
  600│                                    ●
    │                               ●
  400│                          ●
    │                      ●
  200│               ●  ●
    │          ●  ●
  100│    ●  ●
    │  ●
   50│ ●
    └────────────────────────────────────────────▶
      0    5   10   15   20   25   30   35   40
                 Requests/second (RPS)

  ← UNDER-UTILIZED →    ← OPTIMAL ZONE →    ← OVERLOADED →
  Low RPS               Good latency         TTFT explodes
  GPU idle              Good throughput      Requests queue
  Wasted money          Sweet spot           SLA violations

  Goal: Find the "knee" in this curve.
  That's your target operating point.
  Everything left of it wastes GPU.
  Everything right of it breaks SLAs.
```

---

## What to Actually Watch in Grafana

```
  MODULE 300 — THE DASHBOARDS THAT MATTER
  ═════════════════════════════════════════

  Our Grafana setup (Module 300) tracked:

  Panel 1: Request Throughput
  ┌────────────────────────────────────────┐
  │  req/s over time                       │
  │  Is it stable? Growing? Dropping?     │
  └────────────────────────────────────────┘

  Panel 2: TTFT Distribution
  ┌────────────────────────────────────────┐
  │  Histogram: p50 / p95 / p99           │
  │  p99 spike = real user pain           │
  └────────────────────────────────────────┘

  Panel 3: TPOT over time
  ┌────────────────────────────────────────┐
  │  Token generation speed               │
  │  Degradation = memory pressure        │
  └────────────────────────────────────────┘

  Panel 4: GPU VRAM Used (DCGM: DCGM_FI_DEV_FB_USED)
  ┌────────────────────────────────────────┐
  │  At 95%+ → KV cache pressure incoming │
  │  Add LMCache (Module 400) at this pt  │
  └────────────────────────────────────────┘

  Panel 5: GPU SM Utilization (DCGM: DCGM_FI_DEV_GPU_UTIL)
  ┌────────────────────────────────────────┐
  │  Low util + high TTFT = queue problem  │
  │  High util + high TTFT = overloaded   │
  └────────────────────────────────────────┘

  Panel 6: Batch size average
  ┌────────────────────────────────────────┐
  │  < 4 avg batch → batching not working │
  │  > 32 avg batch → great utilization   │
  └────────────────────────────────────────┘
```

---

## Week 2 Complete 🎉

```
  WHAT YOU NOW UNDERSTAND — WEEK 2 RECAP
  ════════════════════════════════════════

  Day 8:  GPUs win because LLMs are parallel matrix math problems
  Day 9:  Memory bandwidth, not FLOPS, bottlenecks decode
  Day 10: KV cache converts O(n²) generation to O(1) per step
  Day 11: Long context is quadratic in cost — use RAG first
  Day 12: Quantization halves memory with < 2% quality loss
  Day 13: Fine-tuned small models beat generic large models
  Day 14: GPU utilization tells you the GPU is busy, not efficient

  Next week: How inference SERVERS actually work
  (vLLM internals, scheduling, batching strategies, speculative decoding)
```

---

## Key Takeaway

> GPU utilization is a health signal, not a success metric.  
> Track cost per token, TTFT p99, throughput at SLA, and cache hit rate.  
> Those four numbers tell you whether your inference system is working.  
> High utilization with bad economics means you need to rethink the system, not celebrate it.

---

*30-Day Series: LLM Inference Is Everything | Day 14 of 30 — Week 2 Complete*
*← [Day 13](./day-13-model-size-vs-inference-cost.md) | [Back to Index](../README.md)*
