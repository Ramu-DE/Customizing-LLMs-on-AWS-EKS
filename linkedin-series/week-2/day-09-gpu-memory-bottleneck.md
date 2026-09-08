# Day 9 — GPU Memory: The Real Bottleneck

> **Hook:** Why VRAM often matters more than raw FLOPS.

---

## The Post

Everyone spec-hunts for FLOPS.

"Does it do 312 TFLOPS? 989 TFLOPS?"

Meanwhile, the actual bottleneck in LLM inference is almost always **memory** — not compute.

Here's why, and what to do about it.

---

## The Roofline Model — Compute vs Memory Bound

```
  THE ROOFLINE MODEL FOR LLM INFERENCE
  ══════════════════════════════════════

  Performance
  (FLOPS/s)
       │
       │              ╱ Compute ceiling (peak FLOPS)
       │             ╱──────────────────────────────
       │            ╱
       │           ╱  ← Compute-bound region
       │          ╱     (prefill with large batches)
       │         ╱
       │        ╱
       │       ╱  ← Memory-bound region
       │      ╱     (decode — almost always here)
       │     ╱
       └────┴───────────────────────────────────────▶
            Arithmetic Intensity (FLOPs / byte loaded)

  PREFILL: High arithmetic intensity → compute-bound
           Many tokens processed in parallel
           GPU cores are the bottleneck

  DECODE:  Low arithmetic intensity → memory-bound
           One token at a time, weights loaded repeatedly
           Memory bandwidth is the bottleneck
```

---

## What Lives in GPU VRAM

```
  GPU VRAM BUDGET — MINISTRAL-3-8B ON L40S (48 GB)
  ══════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │  VRAM ALLOCATION                             SIZE       │
  │  ─────────────────────────────────────────  ────────   │
  │                                                         │
  │  ① Model Weights (BF16)                      ~7 GB     │
  │     All 3.8B parameters × 2 bytes each                 │
  │     Fixed, loaded once at startup                      │
  │                                                         │
  │  ② KV Cache (dynamic)                       ~30 GB     │
  │     Keys and values per layer per token                │
  │     Grows with: layers × heads × seq_len × precision  │
  │     vLLM pre-allocates this pool at launch            │
  │                                                         │
  │  ③ Activation buffers (temporary)            ~3 GB     │
  │     Intermediate values during forward pass           │
  │     Freed after each decode step                       │
  │                                                         │
  │  ④ CUDA kernels + overhead                   ~2 GB     │
  │     Runtime libraries, cuDNN, cuBLAS                   │
  │                                                         │
  │  TOTAL USED: ~42 GB  /  48 GB available               │
  │  HEADROOM:    ~6 GB  (safety margin)                   │
  └─────────────────────────────────────────────────────────┘
```

---

## The Three Memory Pressure Scenarios

```
  SCENARIO A — COMFORTABLE FIT ✅
  ════════════════════════════════

  ┌────────────────────────────────────────────────────────┐
  │  [Weights 7GB] [KV Cache 25GB] [Buffers 3GB] [Free 13GB]│
  │  ████████████  ░░░░░░░░░░░░░░  ▒▒▒▒▒▒▒▒▒▒▒▒           │
  └────────────────────────────────────────────────────────┘

  Result: High concurrency, fast decode, stable TPOT


  SCENARIO B — KV CACHE PRESSURE ⚠️
  ════════════════════════════════════

  Many concurrent long-context requests fill the KV cache:
  ┌────────────────────────────────────────────────────────┐
  │  [Weights 7GB] [KV Cache 40GB] [Buffers 3GB] [Free 0 ]│
  │  ████████████  ████████████████████████████  ▒▒▒▒▒▒▒▒ │
  └────────────────────────────────────────────────────────┘

  Result: New requests QUEUE (waiting for KV cache space)
          TTFT spikes → user experience degrades


  SCENARIO C — MODEL OVERFLOW ❌
  ═══════════════════════════════

  Model too large for VRAM:
  ┌────────────────────────────────────────────────────────┐
  │  GPU VRAM (48 GB): [Weights partial 48GB]              │
  │  ████████████████████████████████████████████████████  │
  │                                                        │
  │  Remaining weights → CPU RAM (50 GB/s bandwidth)      │
  └────────────────────────────────────────────────────────┘

  Result: 10-40× slower decode (CPU RAM bandwidth vs HBM)
          This is the "runs but terribly" scenario
```

---

## Memory Bandwidth Is the Decode Bottleneck

```
  DECODE STEP: WHAT THE GPU ACTUALLY DOES
  ════════════════════════════════════════

  For every single output token:

  1. Load attention weights from VRAM        → 7 GB transfer
  2. Compute Q/K/V projections               → math
  3. Load KV cache for current sequence      → N GB transfer
  4. Compute attention scores                → math
  5. Load FFN weights                        → ~4 GB transfer
  6. Compute FFN output                      → math
  7. Sample next token                       → tiny

  Data movement per decode step: ~11-15 GB
  At L40S bandwidth (864 GB/s):
  → ~13ms minimum per decode step = ~77 tok/s (ceiling)

  At H100 bandwidth (3,350 GB/s):
  → ~3.5ms minimum per decode step = ~285 tok/s (ceiling)

  ┌──────────────────────────────────────────────────────┐
  │  This is why H100 is 3-4× faster for decode         │
  │  than L40S — despite similar FLOPS at some tasks.   │
  │  It moves data 3.9× faster.                         │
  └──────────────────────────────────────────────────────┘
```

---

## Memory Hierarchy in the Full System

```
  MEMORY PYRAMID FOR LLM INFERENCE
  ══════════════════════════════════

                    ┌─────────────┐
                    │  GPU Cores  │  Registers + L1 cache
                    │  ~few MB    │  ~20 TB/s (on-chip)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  L2 Cache   │  On-chip unified cache
                    │  ~50-100 MB │  ~10 TB/s
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │      GPU VRAM (HBM)     │  High Bandwidth Memory
              │      48 GB (L40S)       │  864 GB/s ← PRIMARY STORE
              │      KV cache lives here│
              └────────────┬────────────┘
                           │  KV cache overflow (LMCache)
              ┌────────────▼────────────┐
              │      CPU RAM (DDR5)     │  System memory
              │      ~512 GB typical    │  ~50 GB/s
              │      L2 cache (LMCache) │
              └────────────┬────────────┘
                           │  Remote cache (LMCache)
              ┌────────────▼────────────┐
              │   Valkey / Redis        │  Network memory
              │   (ElastiCache in EKS)  │  ~10 GB/s effective
              │   L3 cache (LMCache)    │  (but huge capacity)
              └─────────────────────────┘

  Trade-off: Speed ↓ as you go down, Capacity ↑ as you go down
  LMCache (Module 400) intelligently manages all three tiers
```

---

## VRAM Sizing Guide

```
  HOW MUCH VRAM DO YOU NEED?
  ═══════════════════════════

  Model          Params   BF16 size   Min VRAM   Recommended
  ─────────────  ──────   ─────────   ────────   ───────────
  Ministral-3B   3.8B     7.6 GB      16 GB      24 GB+
  Llama-3-8B     8B       16 GB       24 GB      40 GB+
  Llama-3-70B    70B      140 GB      4× 40GB    4× 80GB
  Llama-3-405B   405B     810 GB      16× 80GB   32× 80GB

  "Minimum" = model barely fits, no room for KV cache
  "Recommended" = model + healthy KV cache for concurrency

  Rule of thumb:
  ┌──────────────────────────────────────────────────────┐
  │  VRAM needed = model_params × 2 (bytes, BF16)       │
  │              + KV cache budget (2-4× model size)    │
  │              + 3-5 GB overhead                      │
  │                                                      │
  │  For good throughput: KV cache should be ≥ 2× model │
  └──────────────────────────────────────────────────────┘
```

---

## Workshop Connection

```
  HOW WE MANAGED GPU MEMORY IN THE WORKSHOP
  ══════════════════════════════════════════

  Module 100 (vLLM):
  - gpu_memory_utilization = 0.90  (use 90% of VRAM for KV cache)
  - PagedAttention: dynamic KV cache allocation, no fragmentation
  - --max-model-len 8192  (cap context to control KV cache size)

  Module 400 (LMCache):
  - When GPU VRAM KV cache fills → spill to CPU RAM automatically
  - When CPU RAM fills → spill to Valkey (ElastiCache)
  - No requests dropped, just slightly slower cache hits

  Module 300 (Benchmarking):
  - DCGM Exporter metric: DCGM_FI_DEV_FB_USED (VRAM used)
  - Grafana dashboard tracked VRAM pressure in real time
  - We could see KV cache fill up under load
```

---

## Key Takeaway

> FLOPS tell you how fast a GPU can compute.  
> Memory bandwidth tells you how fast it can feed those computations.  
> For LLM decode — the phase users wait for — memory bandwidth wins.  
> More VRAM = more concurrent requests. More bandwidth = lower TPOT.

---

*30-Day Series: LLM Inference Is Everything | Day 9 of 30*
*← [Day 8](./day-08-why-gpus-are-critical.md) | Next → [Day 10](./day-10-kv-cache-explained.md)*
