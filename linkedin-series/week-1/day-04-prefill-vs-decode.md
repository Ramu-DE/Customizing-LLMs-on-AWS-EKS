# Day 4 — Prefill vs Decode

> **Hook:** The two phases that determine your LLM's performance.

---

## The Post

LLM inference has two distinct phases. Most people don't know either exists.

Understanding them changes how you debug latency, choose hardware, and design inference systems.

---

## The Two Phases — Side by Side

```
  LLM INFERENCE: TWO PHASES
  ══════════════════════════════════════════════════════════════════

  YOUR REQUEST: "Explain Kubernetes in simple terms." (6 tokens in)
  EXPECTED OUTPUT: ~150 tokens

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PHASE 1: PREFILL                    PHASE 2: DECODE
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Input:  [tok1][tok2][tok3]          Output generated token-by-token
          [tok4][tok5][tok6]
                 │                    tok1 → tok2 → tok3 → ... → tok150
                 ▼                       (sequential, one at a time)

  All 6 tokens processed              Each decode step:
  IN PARALLEL simultaneously          ┌─────────────────────────┐
                                      │  Take last token        │
  Duration: ~50-200ms                 │  + KV cache context     │
  (depends on prompt length)          │  Run through model      │
                                      │  Sample next token      │
                                      └─────────────────────────┘
                                      Repeat 150 times...
                                      Duration: ~3-5 seconds
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## The Fundamental Difference

```
  PREFILL                              DECODE
  ═══════════════════════════          ═══════════════════════════

  ┌─────────────────────────┐          ┌─────────────────────────┐
  │  COMPUTE BOUND          │          │  MEMORY BANDWIDTH BOUND  │
  │                         │          │                          │
  │  GPU is doing heavy     │          │  GPU is mostly waiting   │
  │  parallel matrix math   │          │  for data from VRAM      │
  │                         │          │                          │
  │  More compute = faster  │          │  More memory BW = faster │
  │  H100 >> A100 >> A10    │          │  HBM3 >> HBM2e >> GDDR6  │
  │                         │          │                          │
  │  Utilization: HIGH      │          │  Utilization: LOW-MED    │
  │  (GPU is busy)          │          │  (GPU often idle)        │
  └─────────────────────────┘          └─────────────────────────┘
```

---

## Bottleneck Comparison Table

```
  ┌─────────────────────┬──────────────────────┬──────────────────────┐
  │ Property            │ PREFILL              │ DECODE               │
  ├─────────────────────┼──────────────────────┼──────────────────────┤
  │ Parallelism         │ ✅ Fully parallel     │ ❌ Sequential         │
  │ Bottleneck          │ Compute (FLOPs)      │ Memory bandwidth     │
  │ Scales with         │ GPU FLOPS            │ Memory bandwidth     │
  │ Input dependency    │ Prompt length        │ Output length        │
  │ KV cache writes     │ Heavy (all at once)  │ Light (1 token/step) │
  │ Batching benefit    │ Huge                 │ Moderate             │
  │ Affects metric      │ TTFT                 │ TPOT                 │
  └─────────────────────┴──────────────────────┴──────────────────────┘
```

---

## What Happens in GPU Memory

```
  GPU VRAM DURING INFERENCE
  ══════════════════════════

  AT PREFILL START:
  ┌─────────────────────────────────────────────────────────┐
  │  Model Weights  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (fixed)      │
  │  KV Cache       ░ (empty)                               │
  └─────────────────────────────────────────────────────────┘

  AFTER PREFILL (all input tokens computed):
  ┌─────────────────────────────────────────────────────────┐
  │  Model Weights  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (fixed)      │
  │  KV Cache       ▓▓▓ (input tokens cached)              │
  └─────────────────────────────────────────────────────────┘

  DURING DECODE (each new token extends cache):
  ┌─────────────────────────────────────────────────────────┐
  │  Model Weights  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (fixed)      │
  │  KV Cache       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (grows each token)   │
  └─────────────────────────────────────────────────────────┘

  More output tokens → larger KV cache → less room for other requests
```

---

## Why vLLM's PagedAttention Is Clever

```
  TRADITIONAL KV CACHE (wasteful):

  Request A needs 1000 tokens  → Reserve 1000 token slots upfront
  Request B needs 200 tokens   → Reserve 200 token slots upfront
  
  Problem: Reserved slots often go unused → memory waste


  vLLM PagedAttention (efficient):

  Memory divided into fixed-size pages (like OS virtual memory)
  
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ A1 │ A2 │ B1 │ A3 │ B2 │ A4 │ C1 │ A5 │  Pages assigned dynamically
  └────┴────┴────┴────┴────┴────┴────┴────┘
  
  No upfront reservation → up to 24x more efficient memory use
  → More concurrent requests → Higher throughput
```

---

## Workshop Connection — Module 100 (vLLM) + Module 400 (LMCache)

```
  Module 100: vLLM handles both phases automatically

  Prefill optimization:
  - Continuous batching groups prefill across requests
  - PagedAttention manages KV cache efficiently
  
  Decode optimization:
  - Token streaming returns tokens as they're generated (no waiting)
  - Speculative decoding (draft + verify) for faster generation

  Module 400: LMCache extends this
  
  If your system prompt is the same across 1000 requests:
  ┌──────────────────────────────────────────────────────────┐
  │  Without LMCache: Prefill runs for EVERY request         │
  │  With LMCache:    Prefill runs ONCE, KV reused 999 times │
  │                                                          │
  │  TTFT improvement: 30-70% faster on repeated prefixes   │
  └──────────────────────────────────────────────────────────┘
```

---

## Key Takeaway

> Prefill is parallel and compute-bound. Decode is sequential and memory-bound.  
> They need different optimizations, different hardware profiles, different batching strategies.  
> Treat them as separate problems — because they are.

---

*30-Day Series: LLM Inference Is Everything | Day 4 of 30*  
*← [Day 3](./day-03-what-happens-during-inference.md) | Next → [Day 5](./day-05-why-tokens-matter.md)*
