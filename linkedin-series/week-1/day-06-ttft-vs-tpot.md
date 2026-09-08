# Day 6 — TTFT vs TPOT

> **Hook:** The two inference metrics every AI architect should understand.

---

## The Post

If you're serious about LLM inference, you need exactly two metrics in your vocabulary.

Miss either one and you'll ship a product that feels broken — even if the model is excellent.

---

## The Two Metrics

```
  USER EXPERIENCE TIMELINE
  ════════════════════════════════════════════════════════════════

  User sends prompt
       │
       │◀──────── TTFT ──────────▶│◀──────── TPOT × N tokens ──────▶│
       │                          │                                   │
  [Request sent]          [First token arrives]           [Full response done]
       │                          │                                   │
       ▼                          ▼                                   ▼
  ─────●──────────────────────────●──●──●──●──●──●──●──●──●──●──●──●─
                                     T  h  i  s     i  s     t  h  e
  
  TTFT = Time to First Token
         "How long before I see anything?"
         
  TPOT = Time Per Output Token  
         "How fast does the text stream?"
```

---

## TTFT — Time to First Token

```
  WHAT DRIVES TTFT
  ══════════════════════════════════════════════════════════════════

  User Request → [Network] → [Tokenize] → [PREFILL PHASE] → First Token

  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  TTFT = Network latency                                      │
  │       + Tokenization time       (< 1ms usually)             │
  │       + Prefill compute time    (THE main factor)           │
  │       + Queue wait time         (if server is busy)         │
  │                                                              │
  │  Prefill time scales with: INPUT TOKEN COUNT                │
  │                                                              │
  │  100 token prompt   → ~50ms prefill                        │
  │  500 token prompt   → ~200ms prefill                       │
  │  2000 token prompt  → ~800ms prefill                       │
  │  8000 token prompt  → ~3000ms prefill  ← NOTICEABLE        │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  TTFT targets by use case:
  ┌──────────────────────┬────────────────┬──────────────────────┐
  │ Use Case             │ Target TTFT    │ Why                  │
  ├──────────────────────┼────────────────┼──────────────────────┤
  │ Chat UI              │ < 500ms        │ Feels responsive     │
  │ Voice assistant      │ < 200ms        │ No awkward silence   │
  │ Code completion      │ < 100ms        │ Matches typing speed │
  │ Search/RAG           │ < 300ms        │ Search-speed UX      │
  │ Batch processing     │ < 30 seconds   │ Throughput > latency │
  └──────────────────────┴────────────────┴──────────────────────┘
```

---

## TPOT — Time Per Output Token

```
  WHAT DRIVES TPOT
  ══════════════════════════════════════════════════════════════════

  [Decode Step 1] → [Decode Step 2] → ... → [Decode Step N]
      TPOT              TPOT                    TPOT

  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  TPOT = Decode step time per token                          │
  │  = Model forward pass (sequential) + sampling              │
  │                                                              │
  │  TPOT is driven by: GPU MEMORY BANDWIDTH                   │
  │  (not compute — this is why decode is memory-bound)        │
  │                                                              │
  │  At 20ms TPOT → 50 tokens/second  (very fast, smooth)     │
  │  At 50ms TPOT → 20 tokens/second  (good for most UI)      │
  │  At 100ms TPOT → 10 tokens/second (noticeable slowness)   │
  │  At 200ms TPOT → 5 tokens/second  (frustrating)           │
  │                                                              │
  │  Human reading speed: ~250 words/min ≈ ~5 words/sec        │
  │  = ~7 tokens/second                                        │
  │  Anything above this feels "fast" to readers               │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

---

## The Interaction — You Need Both

```
  FOUR SCENARIOS
  ══════════════════════════════════════════════════════════════════

  Scenario 1: Great TTFT, Great TPOT  ✅ THE GOAL
  ─────────────────────────────────────────────────
  ─────●─────────────────────────────────────────●
       │ fast first token                         │
              ████████████████████████████████
              fast, smooth streaming
  
  User experience: "This AI is incredibly responsive"


  Scenario 2: Poor TTFT, Great TPOT  ⚠️ FRUSTRATING START
  ─────────────────────────────────────────────────────────
  ─────────────────────────────────────────────●──────────●
                    long wait                  │          │
                                        ████████████████
                                        then fast stream
  
  User experience: "Is this thing working? Oh, it works now. Huh."


  Scenario 3: Great TTFT, Poor TPOT  ⚠️ READS LIKE MOLASSES
  ─────────────────────────────────────────────────────────
  ─────●──────────────────────────────────────────────────●
       │                                                   │
       █─────────────█──────────────█──────────────────────
       fast first token but... each... word... takes... forever
  
  User experience: "It started fast but why is it so slow now?"


  Scenario 4: Poor TTFT, Poor TPOT  ❌ BROKEN FEELING
  ─────────────────────────────────────────────────────────
  ───────────────────────────────────●─────────────────────●
                  long wait               slow stream
  
  User experience: "This is unusable"
```

---

## How to Measure Both

```
  MEASURING TTFT AND TPOT
  ═══════════════════════

  With vLLM + inference-perf (Module 300 in our workshop):

  inference-perf run --model ministral-3b \
    --concurrency 10 \
    --output-file results.json

  Output metrics:
  ┌────────────────────────────────────────────────┐
  │  ttft_mean:   187ms                            │
  │  ttft_p50:    165ms                            │
  │  ttft_p99:    412ms   ← outliers matter!      │
  │                                                │
  │  tpot_mean:   42ms                             │
  │  tpot_p50:    38ms                             │
  │  tpot_p99:    89ms                             │
  │                                                │
  │  tokens_per_second: 23.8                       │
  │  requests_per_second: 4.2                      │
  └────────────────────────────────────────────────┘

  Both p50 AND p99 matter.
  A great p50 with a terrible p99 means 1 in 100 users has a bad time.
  At scale, that's thousands of people.
```

---

## Workshop Connection — Module 300 Benchmarking

```
  OUR BENCHMARK SETUP (AWS EKS + inference-perf + Grafana)

  Load generator ──▶ vLLM on g6e.2xlarge (NVIDIA L40S)
                          │
                          ▼
  Grafana dashboard showed:
  ┌──────────────────────────────────────────────────────────┐
  │  TTFT panel: Real-time P50/P95/P99 histogram            │
  │  TPOT panel: Per-token latency over time                │
  │  Throughput: Tokens/sec vs concurrent requests          │
  │  GPU util:   How hard the GPU is working                │
  └──────────────────────────────────────────────────────────┘

  Key finding from our tests:
  - TTFT degraded with concurrency (prefill queuing)
  - TPOT stayed stable until GPU memory pressure hit
  - The "knee" in the curve is your optimal batch size
```

---

## Key Takeaway

> TTFT tells you if your system feels responsive.  
> TPOT tells you if the experience feels smooth.  
> Both matter. Neither alone is enough. Measure p99, not just p50.

---

*30-Day Series: LLM Inference Is Everything | Day 6 of 30*  
*← [Day 5](./day-05-why-tokens-matter.md) | Next → [Day 7](./day-07-why-llm-feels-slow.md)*
