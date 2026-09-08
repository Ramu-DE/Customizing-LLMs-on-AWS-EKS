# Day 7 — Why Your LLM Feels Slow

> **Hook:** Latency isn't just "model speed." Here are the 6 real culprits.

---

## The Post

"The model is slow."

I hear this constantly. And it's almost never the model itself.

LLM latency is a systems engineering problem. Here are the 6 real causes — and what to do about each one.

---

## The Full Latency Stack

```
  WHERE YOUR LATENCY ACTUALLY LIVES
  ══════════════════════════════════════════════════════════════════

  User clicks "Send"
       │
       │  ① Network round-trip         (5-50ms typical)
       ▼
  Load Balancer / API Gateway
       │
       │  ② Request queue wait         (0ms - ∞ms)
       ▼
  Inference Server (vLLM)
       │
       │  ③ Tokenization               (~1ms)
       ▼
  [PREFILL PHASE]
       │
       │  ④ KV cache miss / recompute  (50ms - 3000ms)
       ▼
  [DECODE PHASE]
       │
       │  ⑤ Decode per token           (20ms - 200ms × N tokens)
       ▼
  Response
       │
       │  ⑥ Serialization / streaming  (1-10ms overhead)
       ▼
  User sees response

  ──────────────────────────────────────────────────────────────
  Total time user perceives = sum of ALL of the above
  "The model" is only ④ and ⑤
```

---

## Root Cause 1: No Request Batching

```
  WITHOUT BATCHING (default naive setup)
  ═══════════════════════════════════════

  Time →
  GPU: [Request A────────────][Request B────────────][Request C──────]
       1 request at a time, full GPU overhead each time

  WITH CONTINUOUS BATCHING (vLLM default)
  ═══════════════════════════════════════

  Time →
  GPU: [A+B+C+D prefill][A decode + E prefill][B decode + F prefill]
       Multiple requests share GPU resources simultaneously

  ┌────────────────────────────────────────────────────────────┐
  │ Impact: 3-10× throughput improvement                       │
  │ Latency for each request: slightly higher                  │
  │ Latency per dollar: dramatically better                    │
  └────────────────────────────────────────────────────────────┘
```

---

## Root Cause 2: Long Prompts With No Caching

```
  THE REPEATED SYSTEM PROMPT PROBLEM
  ════════════════════════════════════

  Request 1:  [System Prompt: 800 tokens][User: 50 tokens]
  Request 2:  [System Prompt: 800 tokens][User: 60 tokens]
  Request 3:  [System Prompt: 800 tokens][User: 45 tokens]
  ...
  Request 1000: [System Prompt: 800 tokens][User: 52 tokens]

  WITHOUT prefix caching:
  Each request prefills the same 800 tokens from scratch
  800 tokens × 1000 requests = 800,000 tokens of wasted compute
  TTFT impact: +200-400ms per request

  WITH prefix caching (LMCache, Module 400):
  ┌──────────────────────────────────────────────────────────┐
  │  Request 1:  Compute KV for 800-token system prompt      │
  │  Request 2:  REUSE cached KV ← 0 compute for 800 tokens │
  │  Request 3:  REUSE cached KV                            │
  │  ...                                                     │
  │  TTFT improvement: 30-70%                               │
  └──────────────────────────────────────────────────────────┘
```

---

## Root Cause 3: Wrong Hardware Profile

```
  HARDWARE MISMATCH DIAGNOSIS
  ════════════════════════════

  Symptom: Slow despite "running on GPU"
  
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  Scenario A: Model doesn't fit in VRAM                     │
  │                                                             │
  │  GPU VRAM: 16 GB                                           │
  │  Model size: 20 GB                                         │
  │  Result: Model layers offloaded to CPU RAM                 │
  │          CPU RAM bandwidth: ~50 GB/s                       │
  │          GPU HBM bandwidth: ~2,000 GB/s                    │
  │          Performance: 40× slower than expected             │
  │                                                             │
  │  Scenario B: Right GPU, wrong precision                    │
  │                                                             │
  │  Running FP32 instead of BF16 or FP8                       │
  │  FP32 model = 2× memory of BF16                            │
  │  = half the batch size = half the throughput               │
  │                                                             │
  │  Our workshop: Ministral-3-8B in BF16 on 48GB L40S        │
  │  = Comfortably fits, fast KV cache, good batch sizes       │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

---

## Root Cause 4: KV Cache Thrashing

```
  KV CACHE UNDER PRESSURE
  ════════════════════════

  GPU VRAM: 48 GB (our L40S)
  Model weights: 7 GB
  Available for KV cache: ~35 GB

  Normal operation:
  ┌────────────────────────────────────────────────────────┐
  │ Weights ▓▓▓▓▓▓▓  KV Cache ░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │         7 GB                       35 GB available     │
  └────────────────────────────────────────────────────────┘

  Under heavy load (many long-context requests):
  ┌────────────────────────────────────────────────────────┐
  │ Weights ▓▓▓▓▓▓▓  KV Cache ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
  │                            FULL! → requests queued    │
  └────────────────────────────────────────────────────────┘

  LMCache fix (Module 400):
  ┌────────────────────────────────────────────────────────┐
  │  GPU KV Cache (hot)   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓                 │
  │  CPU RAM Cache (warm) ░░░░░░░░░░░░░░░░░░░░░░░░░░       │
  │  Valkey / Redis (cold)░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
  │                                                        │
  │  Overflow to CPU RAM instead of blocking requests     │
  └────────────────────────────────────────────────────────┘
```

---

## Root Cause 5: Unbounded Output Length

```
  OUTPUT LENGTH CONTROL
  ══════════════════════

  Common mistake:
  
  prompt = "Explain machine learning"
  response = model.generate(prompt)  # no max_tokens set!
  
  Model decides to write 2,000 tokens.
  At 40ms/token = 80 seconds total.
  
  User wanted a paragraph. Got a textbook chapter. Waited 80 seconds.

  Fix:
  
  response = model.generate(
      prompt,
      max_tokens=300,          # hard cap
      stop=["\n\n\n"],         # stop on triple newline
  )

  ┌──────────────────────────────────────────────────────┐
  │  Constraining output is one of the highest-ROI       │
  │  latency optimizations available.                    │
  │                                                      │
  │  300 tokens @ 40ms = 12 seconds total response     │
  │  2000 tokens @ 40ms = 80 seconds total response    │
  │                                                      │
  │  Same model. Same GPU. 6.7× latency difference.    │
  └──────────────────────────────────────────────────────┘
```

---

## Root Cause 6: The Network Tax

```
  WHERE TIME DISAPPEARS BETWEEN GPU AND USER
  ═══════════════════════════════════════════

  GPU finishes token generation: T=0ms
       │
       │ Serialize to JSON: +2ms
       ▼
  vLLM HTTP response buffer
       │
       │ Wait for full response (if not streaming): +varies
       ▼
  Load Balancer (ALB in our workshop)
       │
       │ SSL/TLS termination: +1-3ms
       │ Routing: +1ms
       ▼
  Client network
       │
       │ Internet latency: +5-100ms depending on region
       ▼
  User browser / app

  SOLUTION: Enable streaming (SSE / Server-Sent Events)
  
  Without streaming: User waits for ALL tokens then sees full response
  With streaming:    User sees tokens as they're generated
  
  ┌──────────────────────────────────────────────────────────┐
  │  Streaming doesn't reduce total time.                    │
  │  But perceived TTFT drops from "total latency"          │
  │  to "first token latency."                              │
  │  That's the difference between "fast" and "broken."    │
  └──────────────────────────────────────────────────────────┘
```

---

## The Diagnostic Checklist

```
  YOUR LLM IS SLOW — WHERE TO LOOK FIRST
  ════════════════════════════════════════

  □ Check GPU utilization
    → Low utilization = batching problem or memory bandwidth bottleneck
    → High utilization + slow = compute bound, consider tensor parallelism

  □ Check TTFT vs TPOT separately
    → High TTFT only = prompt too long or queue wait
    → High TPOT only = memory bandwidth bottleneck
    → Both high = hardware mismatch or VRAM full

  □ Check whether model fits in VRAM
    → nvidia-smi (or DCGM Exporter in Grafana, Module 300)
    → Any CPU offloading = problem

  □ Check if streaming is enabled
    → If not, enable it immediately

  □ Check system prompt length
    → > 500 tokens and not cached = low-hanging fruit

  □ Check max_tokens setting
    → If unset = open-ended output = unbounded latency

  □ Check concurrency
    → If 1 request at a time = batching not working
```

---

## Workshop Connection — Everything Comes Together

```
  MODULE   PROBLEM IT SOLVES               IMPACT
  ────────────────────────────────────────────────────────────
  100-vllm  Batching, PagedAttention       3-10× throughput
  300-bench Measure TTFT/TPOT/GPU util    Diagnose any issue
  400-lmcache Prefix caching, KV overflow 30-70% TTFT drop
  800-ray   Horizontal scaling            Handle traffic spikes

  Every module in this workshop is a specific answer
  to a specific category of LLM slowness.
```

---

## Key Takeaway

> When your LLM feels slow, the problem is almost never "the model."  
> Audit the full stack: network, queue, prompt length, KV cache, output bounds, hardware fit.  
> Each layer has its own fix. Most fixes take hours, not weeks.

---

## Week 1 Complete 🎉

You now understand:
- ✅ Why inference matters more than training
- ✅ What happens token-by-token inside an LLM
- ✅ The two phases: prefill (parallel) and decode (sequential)
- ✅ Why tokens are the unit of cost, latency, and memory
- ✅ The two metrics that define user experience: TTFT and TPOT
- ✅ The 6 real causes of slow LLMs — and how to fix them

Next week: **How inference servers actually work** (vLLM internals, batching strategies, scheduling).

---

*30-Day Series: LLM Inference Is Everything | Day 7 of 30 — Week 1 Complete*  
*← [Day 6](./day-06-ttft-vs-tpot.md) | [Back to Index](../README.md)*
