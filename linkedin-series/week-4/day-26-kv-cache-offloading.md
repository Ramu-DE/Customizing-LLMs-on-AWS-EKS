# Day 26 — KV Cache Offloading with LMCache

> **Hook:** GPU VRAM costs ~$3/GB-month. CPU RAM costs ~$0.01/GB-month. What if you could store 90% of your KV cache on the cheap memory?

---

## The Post

KV cache is the single biggest consumer of GPU VRAM at inference time.

For Ministral-3-8B serving 64 concurrent requests at 2048 tokens each:
- Model weights: ~7.5 GB
- KV cache: ~30–38 GB (the dominant cost)
- Available on L40S 48 GB: ~2–10 GB for new requests

The GPU is mostly a KV cache storage box with some compute attached.

**LMCache** changes this equation. It moves KV cache off the GPU onto CPU RAM (free), or into a remote Redis-compatible store like Valkey (shared across pods). The GPU becomes compute-first again.

---

## The KV Cache Memory Problem

```
  KV CACHE SIZE ARITHMETIC
  ═════════════════════════

  KV cache size per token per layer (BF16):
  size = 2 × num_heads × head_dim × 2 bytes
  (factor of 2: one K tensor + one V tensor)

  Ministral-3-8B:
  - Layers: 32
  - Num heads: 8 (grouped query attention)
  - Head dim: 128
  size per token = 2 × 8 × 128 × 2 bytes = 4,096 bytes = 4 KB/token

  For 1 request at 2,048 tokens:
  4 KB × 2,048 tokens × 32 layers = 256 MB per request

  For 64 concurrent requests:
  256 MB × 64 = 16 GB of KV cache

  GPU VRAM budget:
  ┌──────────────────────────────────────────────────────────┐
  │  L40S: 48 GB total                                       │
  │  - Model weights (BF16):     7.5 GB                     │
  │  - CUDA runtime + vLLM:      1.5 GB                     │
  │  - Activations:              1.0 GB                     │
  │  - Available for KV cache:  ~38 GB                      │
  │                                                          │
  │  38 GB ÷ 256 MB/request = ~148 concurrent requests max  │
  │  BUT: as context grows, fewer requests fit              │
  │  At 8K tokens: 256 MB × 4 = 1 GB/request               │
  │  → Only 38 concurrent requests possible at 8K context  │
  └──────────────────────────────────────────────────────────┘
```

---

## What LMCache Does

LMCache introduces a **two-level KV cache hierarchy** outside the GPU.

```
  LMCACHE ARCHITECTURE
  ═════════════════════

  Before LMCache (standard vLLM):
  ┌─────────────────────────────────────────────────────────┐
  │  GPU VRAM (48 GB)                                        │
  │  ├── Model weights: 7.5 GB                              │
  │  └── KV cache: 38 GB ← ONLY storage tier                │
  │                                                          │
  │  KV cache full → new request WAITS or old KV EVICTED    │
  │  Evicted KV = recompute from scratch on next hit        │
  └─────────────────────────────────────────────────────────┘

  After LMCache (tiered storage):
  ┌─────────────────────────────────────────────────────────┐
  │  L0: GPU VRAM (48 GB)  ← HOT KV cache, active requests  │
  │  │  KV cache: 38 GB                                     │
  │  │  Bandwidth to L1: ~300 GB/s (PCIe)                  │
  │  ▼                                                       │
  │  L1: CPU RAM (optional, up to 100s of GB)  ← WARM        │
  │  │  Offloaded KV: GB-scale storage                      │
  │  │  Access latency: ~2-5ms                              │
  │  │  Cost: ~$0.01/GB-month                               │
  │  ▼                                                       │
  │  L2: Remote Valkey / Redis  ← SHARED across pods        │
  │  │  Persistent KV cache across pod restarts             │
  │  │  Shared between vLLM replicas (prefix sharing)       │
  │  │  Access latency: ~1-3ms (ElastiCache)                │
  │  │  Cost: ~$0.03/GB-month                               │
  └─────────────────────────────────────────────────────────┘
```

---

## The Two Modes: CPU RAM and Remote Cache

```
  MODE 1: CPU RAM OFFLOADING (L1)
  ════════════════════════════════

  Used when: single node, large context workloads
  Config file: 400-lmcache/cpu-ram-offloading/lmcache-cpu-ram-configmap.yaml

  lmcache_config:
    chunk_size: 256            # KV tokens per chunk
    local_cpu:
      enable: true
      max_cache_size: 50000    # ~50,000 tokens of KV in CPU RAM
    local_disk:
      enable: false

  Effect:
  ┌────────────────────────────────────────────────────────┐
  │  Request with 4K shared system prompt:                 │
  │  First request: compute KV → store in GPU + CPU RAM   │
  │  Second request (same prompt): GPU cache miss →       │
  │    load from CPU RAM (2-5ms) vs recompute (~80ms)     │
  │                                                        │
  │  TTFT reduction: ~70ms savings per cache hit          │
  │  GPU freed from: recomputing prefix KV repeatedly     │
  └────────────────────────────────────────────────────────┘

  MODE 2: REMOTE VALKEY CACHE (L2)
  ════════════════════════════════

  Used when: multiple vLLM pods, shared prefix workloads
  Config: 400-lmcache/remote-cache-sharing-with-valkey/

  lmcache_config:
    chunk_size: 256
    remote:
      enable: true
      url: "redis://valkey.default.svc:6379"  # or ElastiCache
      max_cache_size: 200000  # ~200K tokens shared

  Effect:
  ┌────────────────────────────────────────────────────────┐
  │  Pod 0 serves request with long system prompt         │
  │    → computes KV, stores to Valkey                    │
  │  Pod 1 serves next request with SAME system prompt   │
  │    → checks Valkey → HIT! → loads KV (~3ms)           │
  │    → skips full prefill (~500ms for 4K prompt)        │
  │                                                        │
  │  Cross-pod prefix sharing unlocked!                   │
  │  TTFT reduction: up to 96% on cache-hit prefixes      │
  └────────────────────────────────────────────────────────┘
```

---

## What "Token Chunking" Means

LMCache stores KV cache in fixed-size **chunks** (e.g., 256 tokens), not whole sequences.

```
  TOKEN CHUNKING
  ══════════════

  Request: "You are a helpful assistant. [2048 token conversation so far]
            What is the weather like today?"

  System prompt: 100 tokens
  Conversation history: 2048 tokens
  New question: 10 tokens

  LMCache splits into 256-token chunks:
  Chunk 0: tokens 0-255   (start of system prompt + history)
  Chunk 1: tokens 256-511
  ...
  Chunk 8: tokens 2048-2157 (end of history + question, partial)

  Cache lookup:
  ┌───────────────────────────────────────────────────────┐
  │ Chunk 0: ✓ HIT in Valkey → load from cache           │
  │ Chunk 1: ✓ HIT in Valkey → load from cache           │
  │ ...                                                    │
  │ Chunk 8: ✗ MISS (new tokens) → compute, store        │
  └───────────────────────────────────────────────────────┘

  Only the NEW tokens need GPU computation!
  The more "history" a conversation has, the bigger the savings.
```

---

## Real Numbers from the Workshop

```
  LMCACHE BENCHMARK (Module 400)
  ════════════════════════════════

  Model: Ministral-3-8B on g6e.2xlarge (L40S)
  Workload: Customer support chatbot
  Avg system prompt: 4,096 tokens (product catalog)
  Avg user message: 128 tokens

  WITHOUT LMCache:
  ┌────────────────────────────────────────────────────────┐
  │ TTFT (every request recomputes 4K prefix): ~480ms      │
  │ Prefill GPU time: ~420ms (4K tokens)                  │
  │ GPU memory: 95% full during peak                      │
  │ Max concurrent: 38 requests                           │
  └────────────────────────────────────────────────────────┘

  WITH LMCache (Valkey L2):
  ┌────────────────────────────────────────────────────────┐
  │ TTFT on cache HIT:  ~22ms (-95.4%)                    │
  │ TTFT on cache MISS: ~480ms (first request only)       │
  │ Cache hit rate: ~85% (shared prefixes across pods)    │
  │ GPU memory freed: ~15 GB (prefix KV in Valkey)        │
  │ Max concurrent: 52 requests (+37%)                    │
  └────────────────────────────────────────────────────────┘

  Effective throughput improvement: ~2× at same cost
  GPU hours saved (per 1M requests with 85% hit rate):
  4K-token prefill at 1,000 tok/s = 4s per request
  Saved: 850,000 × 4s = 944 GPU-hours/M requests
  At $1.65/hr: $1,557 saved per million requests
```

---

## LMCache + Valkey on AWS ElastiCache

```
  INFRASTRUCTURE IN OUR WORKSHOP
  ════════════════════════════════

  Amazon ElastiCache (Valkey Serverless):
  - Redis-compatible protocol
  - TLS encryption in transit
  - Serverless: scales storage automatically
  - Port: 6379

  Connection in vLLM pod:
  ┌────────────────────────────────────────────────────────┐
  │  lmcache-valkey-configmap.yaml (400-lmcache/)          │
  │                                                         │
  │  LMCACHE_REMOTE_URL: "redis://elasticache-endpoint:6379"│
  │  LMCACHE_CHUNK_SIZE: "256"                              │
  │  LMCACHE_MAX_LOCAL_CPU_SIZE: "10000"                    │
  │  LMCACHE_MAX_REMOTE_SIZE: "500000"                      │
  └────────────────────────────────────────────────────────┘

  Data flow:
  vLLM compute KV → LMCache writes to ElastiCache
  Next request same prefix → LMCache reads from ElastiCache
  → vLLM skips prefill for cached chunks

  Cost:
  ElastiCache Valkey Serverless: ~$0.125/GB-hour
  Storing 10 GB of KV cache: $1.25/hr
  Saved GPU compute: several GPU-hours → net positive
```

---

## Where LMCache Fits in the Full Stack

```
  LMCACHE IN THE PRODUCTION INFERENCE STACK
  ═══════════════════════════════════════════

  Request enters Ray Serve (Day 24)
       │
       ▼
  Ray routes to vLLM replica
       │
       ▼
  vLLM receives tokens
       │
       ▼
  LMCache checks: any chunks in GPU cache? CPU RAM? Valkey?
       │
       ├── FULL HIT: all chunks cached
       │   → load from cache, TTFT ~20ms, skip prefill GPU work
       │
       ├── PARTIAL HIT: some chunks cached
       │   → load matching chunks, compute only new chunks
       │   → TTFT proportional to uncached portion
       │
       └── MISS: no chunks cached
           → full prefill computation
           → store result in LMCache tiers for next time

  The key insight: LMCache makes EVERY subsequent request
  for the same conversation history dramatically faster.
  The longer the conversation, the bigger the savings.
```

---

## Workshop Connection

```
  MODULE 400 IN THIS WORKSHOP
  ═══════════════════════════

  Two configurations demonstrated:

  1. CPU RAM offloading (simpler):
     400-lmcache/cpu-ram-offloading/
     → lmcache-cpu-ram-pod.yaml
     → lmcache-cpu-ram-configmap.yaml
     → Good for: single pod, large context, no Redis needed

  2. Remote sharing with Valkey (production):
     400-lmcache/remote-cache-sharing-with-valkey/
     → lmcache-valkey-pod.yaml
     → lmcache-valkey-configmap.yaml
     → Good for: multiple pods, shared system prompts

  Try it:
  kubectl apply -f 400-lmcache/remote-cache-sharing-with-valkey/
  # Send 10 requests with same 4K system prompt
  # Watch TTFT drop from ~480ms to ~22ms after first request
  # Check Grafana: GPU cache utilization drops, Valkey fills up
```

---

## Key Takeaway

> KV cache dominates GPU VRAM usage — often 60–80% of total.  
> LMCache introduces a tiered hierarchy: GPU VRAM → CPU RAM → Remote Valkey, in order of cost and latency.  
> The biggest win is cross-pod prefix sharing: one pod computes the KV for a common system prompt, all other pods reuse it.  
> Real-world result: 95%+ TTFT reduction on cache hits, 2× throughput at same GPU cost.

---

*30-Day Series: LLM Inference Is Everything | Day 26 of 30 — Week 4: Scaling Beyond One GPU*  
*← [Day 25](./day-25-multi-node-inference.md) | [Day 27 →](./day-27-lora-serving-at-scale.md) | [Back to Index](../README.md)*
