# Day 21 — Inference Throughput

> **Hook:** Tokens/sec is becoming the new infrastructure metric.

---

## The Post

Cloud infrastructure used to be measured in:
- Requests per second (web servers)
- Transactions per second (databases)
- GB/s (storage)

AI infrastructure has a new unit: **tokens per second.**

It's the single number that captures GPU efficiency, cost, and user experience simultaneously. Here's how to think about it — and how to measure it properly.

---

## What Tokens/Second Actually Means

```
  THE THROUGHPUT METRIC DECOMPOSED
  ══════════════════════════════════

  Tokens/second has two dimensions that are OFTEN CONFLATED:

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │  1. GENERATION THROUGHPUT (Output tokens/sec)                   │
  │     ─────────────────────────────────────────                   │
  │     How many tokens the GPU generates per second                │
  │     Across ALL concurrent requests                              │
  │     This is the GPU's "production rate"                         │
  │                                                                   │
  │     Formula: total_output_tokens / total_time                   │
  │     Example: 10 requests × 200 output tokens / 20 seconds      │
  │            = 100 tokens/sec                                     │
  │                                                                   │
  │  2. PER-REQUEST TOKEN RATE (Output tokens/sec per request)      │
  │     ───────────────────────────────────────────────────         │
  │     How fast a SINGLE user sees tokens stream                   │
  │     = 1 / TPOT                                                  │
  │     Example: TPOT=40ms → 25 tokens/sec per user               │
  │                                                                   │
  │  WHY THIS MATTERS:                                              │
  │  100 tok/s system throughput with 10 concurrent users          │
  │  = 10 tok/s per user TPOT                                       │
  │  Human reading speed: ~7 tok/s                                  │
  │  10 tok/s per user: fast enough! ✓                             │
  │                                                                   │
  └─────────────────────────────────────────────────────────────────┘
```

---

## The Full Infrastructure Stack Throughput View

> Inspired by reference: *AI Infrastructure for Agents diagram*

```
  END-TO-END THROUGHPUT CHAIN
  ════════════════════════════

  ┌────────────────────────────────────────────────────────────────┐
  │  USER: Prompt                                                   │
  │       │                                                         │
  │       ▼                                                         │
  │  ┌────────────────┐  Monitoring  ┌────────────────────────┐   │
  │  │  API Gateway   │◀────────────▶│  Prometheus / Grafana  │   │
  │  │  (Kong / ALB)  │              │  tokens/s per endpoint │   │
  │  └───────┬────────┘              └────────────────────────┘   │
  │          │ HTTP/gRPC                                            │
  │          ▼                                                      │
  │  ┌────────────────────────────────────────┐                    │
  │  │  Model Serving & Orchestration         │                    │
  │  │  ┌─────────────┐  ┌────────────────┐  │                    │
  │  │  │ Model Router│  │ Inference vLLM │  │                    │
  │  │  │ (Ray Serve) │◀▶│ (continuous    │  │                    │
  │  │  └─────────────┘  │  batching)     │  │                    │
  │  │                   └────────────────┘  │                    │
  │  └────────────────────────┬───────────────┘                    │
  │                           │                                    │
  │  ┌────────────────────────▼───────────────────────────────┐   │
  │  │  Cache Check → Vector DB (RAG) → KV Cache              │   │
  │  └────────────────────────┬───────────────────────────────┘   │
  │                           │                                    │
  │  ┌────────────────────────▼───────────────────────────────┐   │
  │  │  Kubernetes Pod → NVIDIA L40S GPU                      │   │
  │  │  THE THROUGHPUT BOTTLENECK IS ALWAYS HERE              │   │
  │  └────────────────────────────────────────────────────────┘   │
  └────────────────────────────────────────────────────────────────┘

  Every layer above the GPU adds latency overhead.
  Throughput ceiling = GPU tokens/sec ÷ overhead multiplier.
```

---

## Throughput Benchmarks — Real Numbers

```
  THROUGHPUT ACROSS MODEL/HARDWARE COMBINATIONS
  ═══════════════════════════════════════════════

  Hardware: g6e.2xlarge (1× NVIDIA L40S, 48 GB VRAM)
  Framework: vLLM 0.21.0, BF16, continuous batching
  Workload: 512 input tokens, 256 output tokens, concurrency=32

  ┌──────────────────────┬──────────────┬──────────────┬─────────┐
  │ Model                │ Output tok/s │ TPOT (ms)    │ TTFT ms │
  ├──────────────────────┼──────────────┼──────────────┼─────────┤
  │ Ministral-3-8B BF16  │  ~2,400      │  ~13ms       │ ~120ms  │
  │ Ministral-3-8B FP8   │  ~3,800      │  ~8ms        │ ~80ms   │
  │ Llama-3-8B BF16      │  ~1,800      │  ~18ms       │ ~150ms  │
  │ Llama-3-70B INT4     │   ~400       │  ~80ms       │ ~600ms  │
  └──────────────────────┴──────────────┴──────────────┴─────────┘

  Cost efficiency (at $1/hr for g6e.2xlarge):
  ┌──────────────────────┬──────────────────────────────────────┐
  │ Ministral-3-8B BF16  │  $0.42 / 1M output tokens           │
  │ Ministral-3-8B FP8   │  $0.26 / 1M output tokens           │
  │ Llama-3-8B BF16      │  $0.56 / 1M output tokens           │
  │ Llama-3-70B INT4     │  $2.50 / 1M output tokens           │
  └──────────────────────┴──────────────────────────────────────┘
```

---

## The Three Throughput Regimes

```
  THROUGHPUT SCALING BEHAVIOR
  ════════════════════════════

  REGIME 1: Under-loaded (concurrency too low)
  ────────────────────────────────────────────
  Concurrency: 1-4 requests

  GPU utilization: 15-40%
  Output tokens/s: scales linearly with concurrency
  TTFT: minimal (no queue)
  Cost efficiency: POOR (GPU mostly idle)
  Bottleneck: Not enough requests to fill GPU batch

  REGIME 2: Optimal Zone
  ──────────────────────
  Concurrency: 8-32 requests (depends on model/hardware)

  GPU utilization: 70-85%
  Output tokens/s: near-peak for the hardware
  TTFT: acceptable (moderate queue wait)
  Cost efficiency: EXCELLENT
  Bottleneck: Memory bandwidth (decode) / compute (prefill)

  REGIME 3: Saturated (over-loaded)
  ──────────────────────────────────
  Concurrency: 64+ requests

  GPU utilization: 90%+
  Output tokens/s: PLATEAUS or slightly decreases
  TTFT: explodes (deep queue backlog)
  Cost efficiency: GOOD throughput but SLA violations
  Bottleneck: Request queue depth, KV cache pressure

  ┌────────────────────────────────────────────────────────────┐
  │  Output     ●─────●─────●─────●─────●─────●               │
  │  Tokens/s                                   plateau        │
  │                                                            │
  │  TTFT                                            ●──────  │
  │                                         ●─────            │
  │                              ●─────                       │
  │             ●─────●─────                                  │
  │  ────────────────────────────────────────────────────────  │
  │   0    8   16   24   32   40   48   56   64  concurrency  │
  │        ↑              ↑                   ↑               │
  │      under          optimal            saturated          │
  └────────────────────────────────────────────────────────────┘
```

---

## Tokens/Sec as a Business Metric

```
  TRANSLATING THROUGHPUT TO BUSINESS VALUE
  ═════════════════════════════════════════

  Ministral-3-8B on g6e.2xlarge:
  Peak throughput: 2,400 output tokens/sec

  ┌──────────────────────────────────────────────────────────────┐
  │  Working backwards from user capacity:                       │
  │                                                              │
  │  Average response: 200 output tokens                        │
  │  Average response time: 10 seconds (TTFT + decode)         │
  │  Concurrent users at any moment: 32                         │
  │                                                              │
  │  User capacity: 32 concurrent × 60s/10s = 192 req/min      │
  │               = 11,520 req/hour                             │
  │               = 276,480 req/day per GPU                     │
  │                                                              │
  │  At $1/hr instance cost:                                    │
  │  Cost per request: $1 / 11,520 = $0.000087                 │
  │  Cost per 1M requests: $87                                  │
  │  Compared to GPT-4 API: ~$15-30 per 1M tokens → 100-350×  │
  └──────────────────────────────────────────────────────────────┘
```

---

## The Throughput Optimization Stack

```
  ALL WEEK 3 TECHNIQUES — COMPOUNDING EFFECT
  ════════════════════════════════════════════

  Baseline (naïve server, no optimization):
  Throughput: ~200 tokens/sec, 1 request at a time

                          Apply each technique ↓

  ① Continuous Batching (Day 15)
  ┌──────────────────────────────────────────────────────────┐
  │  Throughput: 200 → ~1,200 tokens/sec (+6×)              │
  │  Keep GPU busy between requests                         │
  └──────────────────────────────────────────────────────────┘

  ② PagedAttention (Day 16)
  ┌──────────────────────────────────────────────────────────┐
  │  Throughput: 1,200 → ~2,000 tokens/sec (+1.7×)          │
  │  More concurrent requests fit in VRAM                  │
  └──────────────────────────────────────────────────────────┘

  ③ FlashAttention (Day 18)
  ┌──────────────────────────────────────────────────────────┐
  │  Throughput: 2,000 → ~2,400 tokens/sec (+1.2×)          │
  │  Faster prefill, lower attention memory pressure        │
  └──────────────────────────────────────────────────────────┘

  ④ Prefix Caching (Day 19)
  ┌──────────────────────────────────────────────────────────┐
  │  Effective throughput: +30-50% for cached prefixes       │
  │  TTFT drops → faster perceived throughput               │
  └──────────────────────────────────────────────────────────┘

  ⑤ Quantization FP8 (Day 12)
  ┌──────────────────────────────────────────────────────────┐
  │  Throughput: 2,400 → ~3,800 tokens/sec (+1.6×)          │
  │  Weights smaller → faster memory bandwidth              │
  └──────────────────────────────────────────────────────────┘

  ⑥ Speculative Decoding (Day 17)
  ┌──────────────────────────────────────────────────────────┐
  │  Throughput: 3,800 → ~7,000+ tokens/sec (+1.8× at α=0.8)│
  │  Multiple tokens per GPU step when drafts accepted      │
  └──────────────────────────────────────────────────────────┘

  FINAL COMPOUNDED RESULT:
  ~200 tokens/sec → ~7,000 tokens/sec = 35× improvement
  Same GPU. Same model. All software optimizations.
```

---

## Tokens/Sec in the Wild — Industry Numbers

```
  REAL-WORLD THROUGHPUT BENCHMARKS (2024-2025)
  ═════════════════════════════════════════════

  ┌───────────────────────┬───────────────────┬──────────────────┐
  │ System                │ Output tok/s      │ Notes            │
  ├───────────────────────┼───────────────────┼──────────────────┤
  │ GPT-4 API             │ ~40-60 tok/s/user │ per user stream  │
  │ Claude 3.5 Sonnet     │ ~50-80 tok/s/user │ per user stream  │
  │ Llama-3-8B on A100   │ ~5,000 tok/s      │ system, batch=32 │
  │ Llama-3-70B on H100×8│ ~3,000 tok/s      │ tensor parallel  │
  │ Groq (LPU hardware)  │ ~800 tok/s/user   │ single stream!   │
  │ Our L40S + Ministral │ ~2,400 tok/s      │ system throughput│
  └───────────────────────┴───────────────────┴──────────────────┘

  The infrastructure wars:
  ┌──────────────────────────────────────────────────────────┐
  │  Whoever achieves highest tokens/sec/dollar wins        │
  │  Not the company with the biggest model                 │
  │  Not the company with the most GPUs                     │
  │  The one who squeezes the most tokens from each GPU     │
  └──────────────────────────────────────────────────────────┘
```

---

## Workshop Connection — Module 300 Benchmarking

```
  MEASURING THROUGHPUT IN OUR WORKSHOP
  ══════════════════════════════════════

  inference-perf tool (Module 300):

  inference-perf run \
    --model ministral-3b \
    --backend vllm \
    --concurrency 32 \
    --num-requests 1000 \
    --input-length 512 \
    --output-length 256 \
    --output-file week3-throughput.json

  Key output metrics:
  ┌──────────────────────────────────────────────────────────┐
  │  output_token_throughput: 2,387 tokens/sec              │
  │  input_token_throughput:  4,890 tokens/sec              │
  │  request_throughput:      9.3 requests/sec              │
  │  ttft_mean:               127ms                         │
  │  ttft_p99:                412ms                         │
  │  tpot_mean:               13.4ms                        │
  │  tpot_p99:                28.1ms                        │
  │  gpu_utilization_mean:    82%                           │
  │  gpu_memory_utilization:  87%                           │
  └──────────────────────────────────────────────────────────┘

  Grafana dashboard tracks all these live during benchmark runs.
  The vllm-benchmarking-dashboard.json (Module 300) renders them
  as real-time panels visible to the whole team.
```

---

## Week 3 Complete 🎉

```
  WEEK 3 RECAP — MAKING INFERENCE FAST
  ══════════════════════════════════════

  Day 15: Continuous Batching
    → Keeps GPU busy every decode step. 10× naive throughput.

  Day 16: PagedAttention
    → Virtual memory for KV cache. 24× more concurrent requests.

  Day 17: Speculative Decoding
    → Small model drafts, big model verifies. 2-4× decode speedup.

  Day 18: FlashAttention
    → IO-aware attention. Never writes n×n matrix to VRAM. 2-4× faster.

  Day 19: Prefix Caching
    → Cache KV for repeated prefixes. 96% TTFT reduction on cache hits.

  Day 20: Dynamic Batching
    → Adapt batch size to traffic. Find the throughput-latency knee.

  Day 21: Inference Throughput
    → Tokens/sec is the unifying metric. 35× improvement compounded.

  Combined effect: A naïve server doing 200 tok/s becomes
                   an optimized server doing 7,000+ tok/s.
                   Same GPU. Same model. All software.

  Next week: Scaling inference beyond a single GPU
  (Tensor parallelism, pipeline parallelism, Ray Serve, multi-node)
```

---

## Key Takeaway

> Tokens per second is the infrastructure metric of the AI era.  
> It links GPU utilization, cost efficiency, and user experience in one number.  
> Every optimization in Week 3 stacks. Combined: 35× improvement on the same hardware.  
> Measure it. Track it. Optimize for it. It's the number that determines whether you can scale.

---

*30-Day Series: LLM Inference Is Everything | Day 21 of 30 — Week 3 Complete*
*← [Day 20](./day-20-dynamic-batching.md) | [Back to Index](../README.md)*
