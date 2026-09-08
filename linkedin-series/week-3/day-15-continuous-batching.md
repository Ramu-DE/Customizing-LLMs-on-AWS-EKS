# Day 15 — Continuous Batching

> **Hook:** How inference servers keep GPUs busy.

---

## The Post

A GPU sitting idle is money on fire.

But naive inference servers do exactly that — they wait for one request to fully finish before starting the next one.

**Continuous batching** is the technique that changed everything. It's why vLLM can serve 10-100× more requests per GPU than a basic server.

Here's exactly how it works.

---

## The Full Inference Server Architecture

> Inspired by reference: *Inference Server + Engine diagram*

```
  INFERENCE SERVER — HOW REQUESTS FLOW
  ══════════════════════════════════════════════════════════════════

  END USERS                         SERVER
  ───────────                       ──────────────────────────────────────────────────────
                                    ┌──────────────────────────────────────────────────┐
  ┌─────────────┐  Requests         │  ENGINE                Memory & Model Optimisation│
  │ Application │ ─────────────────▶│                                                   │
  └─────────────┘  HTTP / gRPC      │  ┌─────────────┐   ┌────────────────────────┐   │
                                    │  │ Query Queue  │──▶│  Dynamic Batching      │   │
  ┌─────────────┐                   │  │ Scheduler   │   │  (Continuous/Inflight) │   │
  │ Application │ ─────────────────▶│  └─────────────┘   └────────────┬───────────┘   │
  └─────────────┘                   │                                  │               │
                                    │                                  ▼               │
  ┌─────────────┐                   │                    ┌────────────────────────┐   │
  │ Application │ ─────────────────▶│                    │  Model (PyTorch/etc)   │   │
  └─────────────┘                   │                    │  + KV Cache System     │   │
                                    │                    └────────────┬───────────┘   │
  ┌─────────────┐                   │                                  │               │
  │ Application │ ─────────────────▶│  ┌─────────────────────────┐   │               │
  └─────────────┘  Query Response   │  │     Query Response       │◀──┘               │
       ▲           HTTP / gRPC      │  └─────────────────────────┘                   │
       └─────────────────────────────│                                                │
    Multiple Requests                │  Metrics (Throughput, Latency, GPU Util...)   │
                                    └──────────────────────────────────────────────────┘
                                                          │
                                            ┌─────────────▼────────────┐
                                            │  Hardware (GPU / CPU)    │
                                            │  NVIDIA L40S / H100      │
                                            └──────────────────────────┘
```

---

## Static Batching — The Old Way

```
  STATIC (NAIVE) BATCHING
  ════════════════════════

  The server waits for a full batch, then processes it start-to-finish.

  Time ──────────────────────────────────────────────────────────▶

  GPU:  ░░░░░░░[WAIT FOR BATCH]░░░░░░░[PREFILL A+B+C][DECODE→→→→→→→][WAIT][PREFILL...]
              ↑                                        ↑            ↑
         collecting                              generating      waiting for
         requests                              all together      next batch

  Problems:
  ┌────────────────────────────────────────────────────────────────┐
  │  1. Short requests wait for slow requests to finish           │
  │  2. GPU idles between batches                                 │
  │  3. Batch size must be fixed upfront                          │
  │  4. Memory reserved for max_tokens even if not used          │
  └────────────────────────────────────────────────────────────────┘

  Example horror scenario:
  Batch = [Req A: 10 tokens out] [Req B: 10 tokens out] [Req C: 500 tokens out]
  → A and B finish quickly, but GPU must keep running for C
  → A and B slots sit empty until C finishes = wasted GPU capacity
```

---

## Continuous Batching — The vLLM Way

```
  CONTINUOUS BATCHING (also called In-flight Batching)
  ══════════════════════════════════════════════════════

  Core idea: As soon as one request finishes a decode step,
             insert a new request WITHOUT waiting.

  Time ──────────────────────────────────────────────────────────▶

  Slot 1: [Req A prefill][A tok1][A tok2][A tok3][DONE][Req E prefill][E tok1]...
  Slot 2: [Req B prefill][B tok1][B tok2][DONE][Req D prefill][D tok1][D tok2]...
  Slot 3: [Req C prefill][C tok1][C tok2][C tok3][C tok4][C tok5]...

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  Every decode step = a mini-batch that may include:             │
  │  - Decode steps for in-progress requests                        │
  │  - Prefill for newly arrived requests                           │
  │  - Nothing for completed requests (slot freed immediately)      │
  │                                                                  │
  │  GPU never waits. New requests fill freed slots instantly.      │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step: One Iteration Cycle

```
  WHAT HAPPENS IN EACH vLLM SCHEDULER ITERATION
  ════════════════════════════════════════════════

  Iteration N:
  ┌──────────────────────────────────────────────────────────────┐
  │                                                               │
  │  Running requests: [A: decode step 4]                        │
  │                    [B: decode step 2]                        │
  │                    [C: decode step 8]  ← about to finish    │
  │                                                               │
  │  Waiting queue:    [D: new arrival]                         │
  │                    [E: new arrival]                         │
  │                                                               │
  │  GPU executes: batch decode step for A, B, C simultaneously  │
  └──────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ C generates <EOS> — done!
  Iteration N+1:
  ┌──────────────────────────────────────────────────────────────┐
  │                                                               │
  │  C is EVICTED. KV cache pages freed immediately.            │
  │  D is ADMITTED. D's prefill runs this iteration.            │
  │                                                               │
  │  Running: [A: decode step 5]                                 │
  │           [B: decode step 3]                                 │
  │           [D: prefill  ← new!]                              │
  │                                                               │
  │  GPU never stops. Slot never wasted.                        │
  └──────────────────────────────────────────────────────────────┘
```

---

## Throughput Impact

```
  STATIC vs CONTINUOUS BATCHING — REAL NUMBERS
  ══════════════════════════════════════════════

  Workload: Mix of short (50 tok) and long (500 tok) requests
  Hardware: NVIDIA L40S, Ministral-3-8B

  ┌─────────────────────────┬──────────────┬──────────────────────┐
  │ Batching Strategy       │ Throughput   │ GPU Utilization      │
  ├─────────────────────────┼──────────────┼──────────────────────┤
  │ Static (naive)          │  ~2 req/s    │ ~25% average         │
  │ Dynamic (fixed window)  │  ~6 req/s    │ ~55% average         │
  │ Continuous (vLLM)       │ ~20 req/s    │ ~85% average         │
  └─────────────────────────┴──────────────┴──────────────────────┘

  Continuous batching = 10× throughput improvement over naive
  Same GPU. Same model. Different scheduler.
```

---

## P/D Disaggregation — The Next Frontier

```
  PREFILL / DECODE DISAGGREGATION
  ═════════════════════════════════

  Problem with mixed batching:
  Prefill (compute-heavy) and Decode (memory-heavy) compete for the same GPU.
  A large prefill step "stalls" all decode steps for that iteration.
  → Decode TPOT spikes whenever a long prompt enters the batch.

  P/D Disaggregation (emerging in production systems):
  ┌──────────────────────────────────────────────────────────────┐
  │                                                               │
  │  PREFILL WORKERS          DECODE WORKERS                    │
  │  ┌────────────────┐       ┌────────────────────────────┐   │
  │  │  GPU optimized │       │  GPU optimized for memory  │   │
  │  │  for compute   │──KV──▶│  bandwidth (decode)        │   │
  │  │  (prefill)     │ cache │                            │   │
  │  └────────────────┘ xfer  └────────────────────────────┘   │
  │                                                               │
  │  Prompts processed separately from generation               │
  │  No decode stall during long prefills                       │
  │  Decode TPOT stays stable under heavy prefill load         │
  │                                                               │
  └──────────────────────────────────────────────────────────────┘

  This is labeled "KEY FEATURE" in production inference engines.
  vLLM v0.6+ supports experimental P/D disaggregation.
```

---

## vLLM Scheduler Config (Our Workshop)

```
  MODULE 100 — vLLM CONTINUOUS BATCHING SETTINGS
  ════════════════════════════════════════════════

  # vllm-deployment.yml (Module 100)
  args:
    - --max-num-seqs 256          # max concurrent sequences in scheduler
    - --max-num-batched-tokens 8192  # max tokens per batch iteration
    - --scheduler-delay-factor 0.0   # no artificial delay between iterations

  What this means:
  ┌──────────────────────────────────────────────────────────────┐
  │  max-num-seqs 256: Up to 256 requests in flight at once     │
  │  max-num-batched-tokens 8192: GPU processes up to 8192      │
  │    tokens per scheduler step (prefill + decode combined)    │
  │  scheduler-delay-factor 0.0: Run iterations as fast as      │
  │    hardware allows — never artificially wait               │
  └──────────────────────────────────────────────────────────────┘

  The continuous batching scheduler runs in a tight loop:
  while True:
      batch = scheduler.schedule()   # pick running + new requests
      outputs = model.step(batch)    # run one GPU forward pass
      scheduler.update(outputs)      # free finished, admit waiting
```

---

## Key Takeaway

> Continuous batching is the single most impactful optimization in modern inference servers.  
> It keeps the GPU busy by mixing prefill and decode across requests in every iteration.  
> vLLM's scheduler does this automatically — 10× throughput over naive serving.  
> Everything else (PagedAttention, prefix caching) amplifies this foundation.

---

*30-Day Series: LLM Inference Is Everything | Day 15 of 30 — Week 3 Begins*
*← [Day 14](../week-2/day-14-gpu-utilization-vs-good-inference.md) | Next → [Day 16](./day-16-paged-attention.md)*
