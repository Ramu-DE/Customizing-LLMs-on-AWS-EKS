# Day 10 — KV Cache Explained

> **Hook:** The hidden memory system powering fast generation.

---

## The Post

There's a data structure inside every running LLM that most people have never heard of.

It's called the **KV Cache** — and without it, every token you generate would be 10-100× slower.

Here's how it works, why it matters, and how the workshop optimized it.

---

## The Problem KV Cache Solves

```
  WITHOUT KV CACHE — THE NAIVE APPROACH
  ══════════════════════════════════════

  Prompt: "The capital of France is"
  Model generates: "Paris"

  Step 1: Generate token "Paris"
  Input to model: ["The", "capital", "of", "France", "is"]
  → Run all 5 tokens through all 32 transformer layers
  → Get next token: "Paris"

  Step 2: Generate token "."
  Input to model: ["The", "capital", "of", "France", "is", "Paris"]
  → Run all 6 tokens through all 32 transformer layers AGAIN
  → Get next token: "."

  Step 3: Generate token "<EOS>"
  Input to model: ["The", "capital", "of", "France", "is", "Paris", "."]
  → Run all 7 tokens through all 32 transformer layers AGAIN
  → ...

  PROBLEM: Every new token reprocesses ALL previous tokens from scratch.
  If output is 500 tokens with 100-token prompt:
  → Total work = (100+1) + (100+2) + ... + (100+500) compute steps
  → This is O(n²) in total generation length
  → Completely unacceptable for production
```

---

## How KV Cache Works

```
  WITH KV CACHE — THE SMART APPROACH
  ════════════════════════════════════

  In each transformer layer, attention computes 3 matrices:
    Q = Query   (what am I looking for?)
    K = Key     (what do I contain?)
    V = Value   (what should I return?)

  KEY INSIGHT: For previous tokens, K and V never change.
  Once computed, we can CACHE them and reuse forever.

  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  PREFILL PHASE: Process entire prompt at once            │
  │                                                          │
  │  ["The", "capital", "of", "France", "is"]               │
  │       │        │       │       │      │                  │
  │       ▼        ▼       ▼       ▼      ▼                  │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │           COMPUTE K,V for all tokens           │    │
  │  │           STORE in KV Cache                    │    │
  │  └─────────────────────────────────────────────────┘    │
  │                                                          │
  │  KV Cache now holds: K[The], V[The],                    │
  │                      K[capital], V[capital],            │
  │                      K[of], V[of], ...                  │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  DECODE STEP 1: Generate "Paris"                         │
  │                                                          │
  │  New token "is" (last token) → compute Q only           │
  │  REUSE K,V from cache → compute attention               │
  │  Output: "Paris" → append K[Paris], V[Paris] to cache   │
  │                                                          │
  │  DECODE STEP 2: Generate "."                             │
  │                                                          │
  │  New token "Paris" → compute Q only                     │
  │  REUSE K,V from cache → compute attention               │
  │  Output: "." → append K[.], V[.] to cache               │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

  Result: Each decode step is O(1), not O(n)
  The cache grows by 1 entry per token generated.
```

---

## KV Cache Memory Cost

```
  HOW MUCH MEMORY DOES KV CACHE USE?
  ════════════════════════════════════

  KV cache size formula:

  size = 2 × num_layers × num_heads × head_dim × seq_len × precision

  For Ministral-3-8B:
  ┌──────────────────────────────────────────────────────────┐
  │  num_layers: 32                                          │
  │  num_kv_heads: 8 (uses Grouped Query Attention)         │
  │  head_dim: 128                                           │
  │  precision: BF16 = 2 bytes                              │
  │                                                          │
  │  Per token: 2 × 32 × 8 × 128 × 2 = 131,072 bytes       │
  │           = 128 KB per token                            │
  │                                                          │
  │  For 1 request × 8,192 token context:                   │
  │  128 KB × 8,192 = ~1 GB per request (full context)      │
  │                                                          │
  │  For 30 concurrent requests × 8,192 tokens each:        │
  │  ~30 GB just for KV cache                               │
  └──────────────────────────────────────────────────────────┘

  Implication: KV cache is why long context = fewer concurrent requests
```

---

## PagedAttention — vLLM's KV Cache Innovation

```
  THE MEMORY FRAGMENTATION PROBLEM
  ══════════════════════════════════

  Traditional approach: Pre-allocate max sequence length upfront

  Request A asks for up to 2048 tokens:
  ┌────────────────────────────────────────────────────────┐
  │ RESERVED for A: ████ (uses 200) ░░░░░░░░░░░░░░░░░░░░  │
  │                 used             WASTED (1848 tokens)  │
  └────────────────────────────────────────────────────────┘

  Memory fragmentation wastes 50-80% of KV cache capacity!


  PagedAttention (vLLM's solution — inspired by OS virtual memory):
  ══════════════════════════════════════════════════════════════════

  Physical memory divided into fixed-size PAGES (e.g., 16 tokens each)

  Page table maps logical sequence positions to physical pages:

  Request A sequence: [tok0..15] → Page 3
                      [tok16..31] → Page 7
                      [tok32..47] → Page 1
                      ...

  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │ A │ B │ A │ C │ A │ B │ D │ A │ C │ B │ D │ C │ D │ D │  Physical pages
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
    3   1   7   2   1   4   3   5   8   2   7   6   1   9    Page IDs

  Benefits:
  ✓ No upfront reservation — pages allocated on demand
  ✓ No internal fragmentation
  ✓ Pages can be SHARED across requests (prefix caching)
  ✓ 24× more efficient memory use reported by vLLM team
```

---

## Prefix Caching — The Next Level

```
  SHARED PREFIX ACROSS REQUESTS
  ═══════════════════════════════

  If 1,000 requests all start with the same system prompt:

  Request 1:  [System Prompt: 500 tokens] [User msg: 50 tokens]
  Request 2:  [System Prompt: 500 tokens] [User msg: 45 tokens]
  Request 3:  [System Prompt: 500 tokens] [User msg: 60 tokens]
  ...
  Request 1000: [System Prompt: 500 tokens] [User msg: 52 tokens]

  WITHOUT prefix caching:
  Each request computes KV for the 500-token system prompt from scratch
  → 1000 × 500 tokens = 500,000 redundant KV computations

  WITH prefix caching (LMCache, Module 400):
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  Request 1:  Compute KV for system prompt → STORE       │
  │  Request 2:  HIT cached KV → skip 500 tokens of prefill │
  │  Request 3:  HIT cached KV → skip 500 tokens of prefill │
  │  ...                                                     │
  │  Request 1000: HIT → skip                               │
  │                                                          │
  │  TTFT reduction: 30-70% for requests with shared prefix │
  │  Memory: Only 1 copy of system prompt KV stored         │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

---

## KV Cache Tiering (LMCache Architecture)

```
  THREE-TIER KV CACHE (Module 400 in our workshop)
  ══════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │                                                              │
  │  TIER 1: GPU VRAM                                           │
  │  ┌──────────────────────────────────────┐                  │
  │  │  Hot KV cache (active requests)       │  864 GB/s       │
  │  │  ~30 GB available on L40S            │  ~5ms access     │
  │  └──────────────────┬───────────────────┘                  │
  │                     │ overflow when full                   │
  │  TIER 2: CPU RAM                                            │
  │  ┌──────────────────▼───────────────────┐                  │
  │  │  Warm KV cache (recent requests)      │  ~50 GB/s       │
  │  │  Hundreds of GB available             │  ~20ms access   │
  │  └──────────────────┬───────────────────┘                  │
  │                     │ overflow when full                   │
  │  TIER 3: Valkey/Redis (ElastiCache)                        │
  │  ┌──────────────────▼───────────────────┐                  │
  │  │  Cold KV cache (shared across pods)   │  ~10 GB/s net  │
  │  │  Effectively unlimited capacity       │  ~50ms access  │
  │  │  Shared across vLLM replicas          │               │
  │  └──────────────────────────────────────┘                  │
  │                                                              │
  └─────────────────────────────────────────────────────────────┘

  When a request arrives, LMCache checks:
  1. GPU cache HIT → prefill skipped entirely (fastest)
  2. CPU cache HIT → transfer to GPU, partial prefill skipped
  3. Valkey HIT    → fetch over network, partial prefill skipped
  4. MISS          → full prefill from scratch (baseline)
```

---

## Key Takeaway

> The KV cache is what makes autoregressive generation feasible.  
> Without it, every token would require reprocessing the entire context.  
> With it, each decode step is O(1). The trade-off is VRAM.  
> Everything in inference optimization — batching, PagedAttention, prefix caching — ultimately serves the KV cache.

---

*30-Day Series: LLM Inference Is Everything | Day 10 of 30*
*← [Day 9](./day-09-gpu-memory-bottleneck.md) | Next → [Day 11](./day-11-why-long-context-is-expensive.md)*
