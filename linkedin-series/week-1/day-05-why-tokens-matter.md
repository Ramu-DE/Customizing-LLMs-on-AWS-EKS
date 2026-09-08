# Day 5 — Why Tokens Matter More Than You Think

> **Hook:** More tokens = more latency + more GPU work + more cost. Every single time.

---

## The Post

Tokens are the unit of currency in LLM inference.

Every cost, every latency number, every GPU hour — it all traces back to tokens.

But most people treat them like characters or words. They're neither.

---

## What Is a Token?

```
  TOKENIZATION EXAMPLES
  ══════════════════════════════════════════════════════════════════

  Input Text                     Tokens                  Count
  ──────────────────────────────────────────────────────────────
  "Hello world"              →   [Hello] [world]           2
  "Hello, world!"            →   [Hello] [,] [world] [!]   4
  "unbelievable"             →   [un] [believ] [able]       3
  "ChatGPT"                  →   [Chat] [G] [PT]            3
  "AWS EKS"                  →   [AWS] [EKS]                2
  "192.168.1.1"              →   [192] [.] [168] [.] [1]   5+
                                 [.] [1]
  "def calculate_sum(a, b):" →   [def] [calculate] [_]     8
                                 [sum] [(] [a] [,] [b] [:]
  A typical paragraph        →   ~100 tokens
  This full LinkedIn post    →   ~400 tokens
  GPT-4 max context          →   128,000 tokens
  Our Ministral-3-8B         →   8,192 tokens (context limit)
  ──────────────────────────────────────────────────────────────

  Rule of thumb: 1 token ≈ 0.75 words (English)
                 1 token ≈ 4 characters (code/English mix)
```

---

## Why Token Count Hits You From Every Angle

```
  THE TOKEN COST MULTIPLIER
  ══════════════════════════════════════════════════════════════════

                         TOKENS
                            │
              ┌─────────────┼─────────────┬─────────────┐
              ▼             ▼             ▼             ▼
          LATENCY         COST         MEMORY       THROUGHPUT
              │             │             │             │
  ┌───────────┴──┐ ┌────────┴───┐ ┌──────┴─────┐ ┌────┴──────────┐
  │ Each output  │ │ APIs bill  │ │ KV cache   │ │ More tokens = │
  │ token =      │ │ per input  │ │ grows with │ │ longer GPU    │
  │ 1 forward    │ │ + output   │ │ context    │ │ hold time     │
  │ pass through │ │ token      │ │ length     │ │ → fewer       │
  │ the model    │ │            │ │            │ │ req/second    │
  │              │ │ Linear     │ │ Linear     │ │               │
  │ Sequential   │ │ pricing    │ │ memory use │ │ Inverse       │
  │ → no speedup │ │            │ │            │ │ throughput    │
  └──────────────┘ └────────────┘ └────────────┘ └───────────────┘
```

---

## The Real Cost: A Worked Example

```
  COST COMPARISON: CHATBOT SYSTEM PROMPT
  ═══════════════════════════════════════

  Scenario: Customer support bot, 10,000 requests/day

  VERSION A — Verbose System Prompt
  ┌──────────────────────────────────────────────────────────┐
  │ System prompt: 800 tokens (detailed instructions)        │
  │ Average user message: 50 tokens                          │
  │ Average response: 200 tokens                             │
  │                                                          │
  │ Total per request: 800 + 50 + 200 = 1,050 tokens        │
  │ Daily total: 10,000 × 1,050 = 10,500,000 tokens         │
  │ Monthly: 315,000,000 tokens                              │
  │ Cost @ $0.002/1k tokens: $630/month                      │
  │ TTFT: ~250ms (800 token prefill)                         │
  └──────────────────────────────────────────────────────────┘

  VERSION B — Tight System Prompt
  ┌──────────────────────────────────────────────────────────┐
  │ System prompt: 150 tokens (essential instructions only)  │
  │ Average user message: 50 tokens                          │
  │ Average response: 200 tokens                             │
  │                                                          │
  │ Total per request: 150 + 50 + 200 = 400 tokens          │
  │ Daily total: 10,000 × 400 = 4,000,000 tokens            │
  │ Monthly: 120,000,000 tokens                              │
  │ Cost @ $0.002/1k tokens: $240/month                      │
  │ TTFT: ~80ms (150 token prefill)                          │
  └──────────────────────────────────────────────────────────┘

  Savings: $390/month  (+62% cheaper, 3x faster TTFT)
  Just by trimming the system prompt.
```

---

## Token Count vs Context Length

```
  CONTEXT WINDOW IMPACT ON PERFORMANCE
  ═════════════════════════════════════

  Context     Attention        KV Cache      Relative
  Length      Operations       Memory        Cost
  ──────────  ───────────────  ────────────  ────────
  512 tokens  262,144          ~200 MB       1×
  1K tokens   1,048,576        ~400 MB       4×
  2K tokens   4,194,304        ~800 MB       16×
  4K tokens   16,777,216       ~1.6 GB       64×
  8K tokens   67,108,864       ~3.2 GB       256×
  32K tokens  1,073,741,824    ~13 GB        4,096×
  128K tokens 17,179,869,184   ~52 GB        65,536×

  Attention is O(n²) — doubling context = 4× the attention work
  This is why long contexts are expensive
```

---

## Practical Token Optimization Strategies

```
  BEFORE (wasteful)                   AFTER (optimized)
  ═══════════════════                 ══════════════════════

  ┌────────────────────────┐          ┌────────────────────────┐
  │ System prompt: 800 tok │          │ System prompt: 150 tok │
  │ "You are a helpful     │          │ "Customer support.     │
  │  AI assistant. Your    │          │  Be brief. JSON only." │
  │  goal is to provide    │──────▶   │                        │
  │  excellent customer    │ trim     │ Prompt caching:        │
  │  service responses...  │          │ Reuse KV for this 150  │
  │  [500 more tokens]"    │          │ across all requests    │
  └────────────────────────┘          └────────────────────────┘

  ┌────────────────────────┐          ┌────────────────────────┐
  │ Output: "Sure! I'd be  │          │ Output constrained:    │
  │ happy to help you with │──────▶   │ max_tokens=150         │
  │ that. Let me explain   │ limit    │ Response JSON schema   │
  │ in detail..."          │          │ enforced               │
  │ [unbounded output]     │          └────────────────────────┘
  └────────────────────────┘
```

---

## Workshop Connection — Tokenization in Ministral-3-8B

```
  OUR WORKSHOP MODEL: Ministral-3-8B-Instruct-2512

  Tokenizer: Tekken (Mistral's tokenizer)
  Vocabulary: 131,072 tokens (131k)
  Context limit: 8,192 tokens

  Why 131k vocabulary matters:
  ┌────────────────────────────────────────────────────────┐
  │ Larger vocabulary = fewer tokens per word              │
  │                                                        │
  │ GPT-2 (50k vocab): "unbelievable" → 5 tokens           │
  │ Tekken (131k vocab): "unbelievable" → 2-3 tokens       │
  │                                                        │
  │ Fewer tokens per word = faster inference               │
  │                       = lower cost                    │
  │                       = more fits in context window   │
  └────────────────────────────────────────────────────────┘

  In benchmarking (Module 300), we measured:
  - Input tokens per request (prompt length)
  - Output tokens per request (generation length)
  - Tokens per second (throughput)
  - These are the core axes of all LLM performance analysis
```

---

## Key Takeaway

> Tokens aren't just text units — they're compute units, cost units, and memory units.  
> Optimizing your prompts isn't just about clarity. It's engineering with real dollar values.  
> Count your tokens. They're not free.

---

*30-Day Series: LLM Inference Is Everything | Day 5 of 30*  
*← [Day 4](./day-04-prefill-vs-decode.md) | Next → [Day 6](./day-06-ttft-vs-tpot.md)*
