# Day 3 — What Actually Happens During LLM Inference?

> **Hook:** From prompt → tokens → GPU → response. Here's every step.

---

## The Post

You type a prompt. A response appears. Simple, right?

Not even close. Here's what's actually happening under the hood — step by step.

---

## The Full Inference Pipeline

```
  LLM INFERENCE — COMPLETE FLOW
  ══════════════════════════════════════════════════════════════════════

  YOUR PROMPT
  "Summarize this article in 3 bullet points: [article text]"
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  STEP 1: TOKENIZATION                                            │
  │                                                                  │
  │  "Summarize" → [6800]                                           │
  │  " this"     → [428]                                            │
  │  " article"  → [4356]     Text split into integer token IDs     │
  │  " in"       → [287]      Vocabulary size: ~50k-131k entries    │
  │  " 3"        → [513]      Our model (Ministral): 131k vocab     │
  │  ...         → [...]                                            │
  │                                                                  │
  │  Input: "Summarize this article..." (23 words)                  │
  │  Output: [6800, 428, 4356, 287, 513, ...] (~28 tokens)         │
  └──────────────────────┬───────────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  STEP 2: EMBEDDING LOOKUP                                        │
  │                                                                  │
  │  Token 6800 → [0.23, -1.4, 0.89, ..., 0.11]  (4096 numbers)   │
  │  Token 428  → [1.02, 0.33, -0.77, ..., 0.45]  (4096 numbers)   │
  │  Token 4356 → [-0.5, 0.91, 1.23, ..., -0.3]   (4096 numbers)   │
  │  ...                                                             │
  │                                                                  │
  │  Each token becomes a dense vector in high-dimensional space    │
  │  This is how meaning is encoded numerically                     │
  └──────────────────────┬───────────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  STEP 3: TRANSFORMER LAYERS (32-96 layers depending on model)   │
  │                                                                  │
  │  Layer 1 ──▶ Layer 2 ──▶ Layer 3 ──▶ ... ──▶ Layer 32         │
  │                                                                  │
  │  Each layer runs:                                               │
  │  ┌────────────────────────────────────────────────────────┐    │
  │  │  a) Multi-Head Attention  (tokens talk to each other)  │    │
  │  │  b) Feed-Forward Network  (per-token transformation)   │    │
  │  │  c) Layer Normalization   (stabilize values)           │    │
  │  └────────────────────────────────────────────────────────┘    │
  │                                                                  │
  │  Each layer = billions of matrix multiplications on GPU         │
  │  This is the compute-intensive core of inference               │
  └──────────────────────┬───────────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  STEP 4: ATTENTION MECHANISM (Inside each layer)                │
  │                                                                  │
  │  Every token "looks at" every other token                      │
  │                                                                  │
  │  Token:   "Summarize" "this" "article" "in" "3" "bullets"      │
  │               │          │       │       │   │      │           │
  │               ▼          ▼       ▼       ▼   ▼      ▼           │
  │  Attention weights:  Who matters most for predicting next token? │
  │                                                                  │
  │  Complexity: O(n²) in sequence length                           │
  │  → 100 tokens:  10,000 attention operations                     │
  │  → 1000 tokens: 1,000,000 attention operations  ← expensive!   │
  │  → 8000 tokens: 64,000,000 attention operations ← very costly  │
  └──────────────────────┬───────────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  STEP 5: TOKEN SAMPLING                                          │
  │                                                                  │
  │  Model outputs probability over entire vocabulary:              │
  │                                                                  │
  │  "The"    → 0.31  ████████████████████████████████             │
  │  "Here"   → 0.18  ██████████████████                           │
  │  "First"  → 0.12  ████████████                                 │
  │  "Below"  → 0.09  █████████                                    │
  │  ...      → ...                                                 │
  │                                                                  │
  │  Sampling strategies:                                           │
  │  - Temperature 0.0: Always pick highest probability (greedy)   │
  │  - Temperature 1.0: Sample proportionally                      │
  │  - Top-p 0.9: Sample from top tokens covering 90% probability  │
  └──────────────────────┬───────────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  STEP 6: REPEAT UNTIL DONE                                       │
  │                                                                  │
  │  Chosen token appended → Steps 3-5 run AGAIN for next token    │
  │                                                                  │
  │  Token 1: "The"    → run forward pass                          │
  │  Token 2: "first"  → run forward pass                          │
  │  Token 3: "bullet" → run forward pass                          │
  │  ...                                                             │
  │  Token 200: <EOS>  → stop                                       │
  │                                                                  │
  │  200 output tokens = 200 sequential GPU forward passes          │
  │  THIS is why long outputs are slow                              │
  └──────────────────────────────────────────────────────────────────┘
```

---

## GPU Memory Layout During Inference

```
  GPU VRAM (48 GB on L40S in our workshop)
  ══════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │                                                          │
  │   Model Weights (Static, loaded once)                   │
  │   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ~7 GB (BF16)      │
  │                                                          │
  │   KV Cache (Dynamic, grows per request)                 │
  │   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~30 GB available   │
  │   (Stores attention keys/values per token per layer)    │
  │                                                          │
  │   Runtime buffers, activations                          │
  │   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ~5 GB                               │
  │                                                          │
  └─────────────────────────────────────────────────────────┘

  KV Cache is why longer contexts = more memory = fewer parallel requests
```

---

## Workshop Connection

In Module 100 (vLLM on EKS), vLLM handles all of this automatically:

```
  Your HTTP request
        │
        ▼
  vLLM Inference Server (running on g6e.2xlarge, NVIDIA L40S)
        │
        ├── Tokenizes your prompt (Tekken tokenizer, 131k vocab)
        ├── Runs prefill (all input tokens through transformer)
        ├── Manages KV cache in GPU VRAM (PagedAttention)
        ├── Runs decode loop (one token at a time)
        └── Streams tokens back via SSE or returns full response
```

Module 400 (LMCache) specifically optimizes **Step 4** — it caches the computed KV values so identical prompt prefixes don't need to recompute attention.

---

## Key Takeaway

> LLM inference isn't one operation — it's hundreds of chained GPU computations.  
> Every token costs compute. Every layer costs memory. Every optimization matters.

---

*30-Day Series: LLM Inference Is Everything | Day 3 of 30*  
*← [Day 2](./day-02-training-vs-inference.md) | Next → [Day 4](./day-04-prefill-vs-decode.md)*
