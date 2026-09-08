# Day 19 — Prefix Caching

> **Hook:** Why repeated prompts shouldn't be recomputed.

---

## The Post

Every request to your LLM starts with a system prompt.

If you're running 10,000 requests a day with a 500-token system prompt, you're paying for 5 million tokens of prefill **every single day** — recomputing the same thing 10,000 times.

Prefix caching eliminates that waste. It's one of the highest-ROI optimizations in production LLM serving.

---

## The Repeated Prefix Problem

```
  THE WASTED COMPUTE PATTERN
  ═══════════════════════════

  Your inference server receives:

  Request 1:
  ┌─────────────────────────────────────────────────────────────┐
  │  [SYSTEM: You are a customer support agent for Acme Corp.  │
  │   Always be polite. Respond in JSON format. You have       │
  │   access to order history. Never share private data...     │
  │   (500 tokens)]                                            │
  │  [USER: What is my order status?  (8 tokens)]              │
  └─────────────────────────────────────────────────────────────┘

  Request 2:    [SAME 500-token system prompt] + [USER: Where is my refund?]
  Request 3:    [SAME 500-token system prompt] + [USER: Cancel my order]
  ...
  Request 10000:[SAME 500-token system prompt] + [USER: Track my package]

  WITHOUT prefix caching:
  ┌─────────────────────────────────────────────────────────────┐
  │  TTFT per request = prefill(500 system) + prefill(user)    │
  │  = ~200ms + ~5ms = ~205ms                                  │
  │                                                             │
  │  Compute wasted: 500 tokens × 10,000 requests = 5M tokens  │
  │  Cost: 5M tokens × $0.002/1k = $10/day on system prompts  │
  │  That's $3,650/year for computing the SAME thing daily    │
  └─────────────────────────────────────────────────────────────┘
```

---

## How Prefix Caching Works

```
  RADIX TREE STRUCTURE — THE KEY DATA STRUCTURE
  ══════════════════════════════════════════════

  Prefix caching stores KV blocks indexed by a hash of the token sequence.

  Token sequence → SHA hash → KV block pointer

  Example token sequences sharing prefixes:

  Request A: [sys_tok_1][sys_tok_2]...[sys_tok_500][user_A_1][user_A_2]
  Request B: [sys_tok_1][sys_tok_2]...[sys_tok_500][user_B_1][user_B_2]
  Request C: [sys_tok_1][sys_tok_2]...[sys_tok_500][user_C_1]

                         Radix tree:
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
  [sys 1-16 KV]          [sys 1-16]       [sys 1-16]
       │                     ...               ...
  [sys 17-32 KV]         ← SHARED ──────────────┘
       ...
  [sys 481-496 KV]
  [sys 497-500 KV]  ← ALL SHARED by A, B, C
       │
  ┌────┼────┐
  [A unique] [B unique] [C unique]

  When Request B arrives:
  → Hash(sys_tok_1..500) = 0xAB34... → CACHE HIT
  → Skip 500-token prefill entirely
  → Only prefill user_B tokens (2 tokens)
  → TTFT: ~205ms → ~5ms (97.5% reduction)
```

---

## Caching Strategies Compared

```
  THREE LEVELS OF PREFIX CACHING
  ═══════════════════════════════

  LEVEL 1: vLLM Automatic Prefix Caching (in-process)
  ┌──────────────────────────────────────────────────────────────┐
  │  Cache lives in: GPU VRAM (KV block pool)                   │
  │  Scope: single vLLM process                                 │
  │  Hit when: same prefix, same process, blocks not evicted    │
  │  Config: --enable-prefix-caching (vLLM default ON)         │
  │  Capacity: limited to VRAM KV pool (~30 GB on L40S)        │
  └──────────────────────────────────────────────────────────────┘

  LEVEL 2: LMCache CPU Tier (cross-request, same node)
  ┌──────────────────────────────────────────────────────────────┐
  │  Cache lives in: CPU RAM (hundreds of GB)                   │
  │  Scope: single node, persists across GPU evictions          │
  │  Hit when: prefix seen on this node, any time              │
  │  Transfer cost: CPU→GPU copy (~20ms for 500 tokens)        │
  │  Still 10× faster than full prefill recompute              │
  └──────────────────────────────────────────────────────────────┘

  LEVEL 3: LMCache Valkey/Redis (cross-node, shared)
  ┌──────────────────────────────────────────────────────────────┐
  │  Cache lives in: ElastiCache Valkey (remote)                │
  │  Scope: ALL vLLM replicas (Ray Serve pods in Module 800)   │
  │  Hit when: any replica has seen this prefix                │
  │  Transfer cost: network (~50ms) — still better than 200ms  │
  │  Capacity: effectively unlimited                           │
  └──────────────────────────────────────────────────────────────┘
```

---

## The LLM Gateway Pattern — Multi-Model Caching

> Inspired by reference: *LLM Gateway flow diagram*

```
  CACHE-AWARE LLM GATEWAY
  ═════════════════════════

  ┌─────────────┐
  │ Application │
  └──────┬──────┘
         │ Single API Call
         ▼
  ┌─────────────────────────────────────────────────┐
  │              LLM GATEWAY                        │
  │         (Route & Cache layer)                   │
  └──────────────────┬──────────────────────────────┘
                     │
              Route & Cache Decision
                     │
         ┌───────────▼────────────┐
         │     DECISION LAYER     │
         └───┬───────┬───────┬───┘
             │       │       │
     Cache Hit?  Route to  Route to
         │       Provider  Provider
         ▼           │         │
  ┌──────────────┐  ▼         ▼
  │Return Cached │ ┌────┐  ┌──────────┐
  │Response ~50ms│ │vLLM│  │AWS Bedrock│
  └──────────────┘ │ on │  │  Backup  │
                   │ EKS│  └──────────┘
                   └──┬─┘
                      │
               ┌──────▼──────┐
               │Store & Return│
               └─────────────┘

  Hit rates in production:
  ┌──────────────────────────────────────────────────────────┐
  │  System prompt cache: 95-99% hit rate (same every req)  │
  │  Few-shot examples:   80-95% hit rate                   │
  │  RAG context chunks:  40-70% hit rate (popular docs)    │
  │  User conversation:   20-40% hit rate (multi-turn)      │
  └──────────────────────────────────────────────────────────┘
```

---

## Real Production Impact

```
  TTFT REDUCTION WITH PREFIX CACHING
  ════════════════════════════════════

  Setup: Ministral-3-8B, 500-token system prompt, L40S

  ┌──────────────────────────┬──────────┬──────────┬──────────┐
  │ Cache State              │ Hit Tier │ TTFT     │ Savings  │
  ├──────────────────────────┼──────────┼──────────┼──────────┤
  │ No cache (cold)          │ MISS     │ ~205ms   │ 0%       │
  │ GPU VRAM hit             │ L1       │  ~8ms    │ 96%      │
  │ CPU RAM hit (LMCache)    │ L2       │  ~25ms   │ 88%      │
  │ Valkey hit (LMCache)     │ L3       │  ~55ms   │ 73%      │
  └──────────────────────────┴──────────┴──────────┴──────────┘

  At 10,000 req/day with 95% L1 hit rate:
  ┌──────────────────────────────────────────────────────────┐
  │  Without caching: 10,000 × 205ms prefill = 34 min GPU  │
  │  With caching:    500 × 205ms + 9500 × 8ms = 3 min GPU │
  │                                                          │
  │  GPU savings: 31 min/day = 189 hours/month              │
  │  At $1/hr: $189/month saved — just from system prompts  │
  └──────────────────────────────────────────────────────────┘
```

---

## Designing for Cacheability

```
  HOW TO MAXIMIZE YOUR CACHE HIT RATE
  ═════════════════════════════════════

  ✅ DO: Put stable content FIRST
  ┌──────────────────────────────────────────────────────────┐
  │  [System prompt — STATIC, 500 tokens]                   │
  │  [Few-shot examples — STATIC, 300 tokens]               │
  │  [Retrieved context — SEMI-STATIC, 200 tokens]          │
  │  [User message — DYNAMIC, 50 tokens]  ← always last    │
  └──────────────────────────────────────────────────────────┘
  Cache can match on the first 1000 tokens consistently.

  ❌ DON'T: Inject dynamic content into the prefix
  ┌──────────────────────────────────────────────────────────┐
  │  [System prompt with timestamp: "Today is Sept 8 2026"] │
  │  ← cache miss every second!                             │
  │                                                          │
  │  [User ID embedded in system prompt]                    │
  │  ← different hash per user, no sharing possible         │
  └──────────────────────────────────────────────────────────┘

  ✅ DO: Use consistent token encoding
  ┌──────────────────────────────────────────────────────────┐
  │  Same whitespace, same special tokens, same formatting   │
  │  Tiny changes break the hash → cache miss              │
  └──────────────────────────────────────────────────────────┘
```

---

## Workshop Connection

```
  PREFIX CACHING ACROSS OUR MODULES
  ══════════════════════════════════

  Module 200 (Strands Agent):
  - System prompt + tool definitions = ~800 tokens
  - Same across all agent requests → near 100% cache hit
  - Config: --enable-prefix-caching in vllm-deployment-agents.yaml

  Module 400 (LMCache):
  - Extends vLLM prefix caching to CPU RAM + Valkey
  - KV blocks evicted from GPU reappear from CPU RAM on next hit
  - Shared across all Ray Serve pods (Module 800)

  Module 700 (RAG):
  - Popular document chunks cached after first retrieval
  - Cache key = hash(retrieved_chunks_text)
  - 2nd request for same documents: 0 prefill cost
```

---

## Key Takeaway

> Prefix caching is the highest-ROI optimization for most production deployments.  
> Your system prompt is computed 10,000 times/day without it.  
> Design prompts with stable content first and dynamic content last.  
> GPU hit rate: 96% TTFT reduction. Even a Valkey hit beats full recompute by 70%.

---

*30-Day Series: LLM Inference Is Everything | Day 19 of 30*
*← [Day 18](./day-18-flash-attention.md) | Next → [Day 20](./day-20-dynamic-batching.md)*
