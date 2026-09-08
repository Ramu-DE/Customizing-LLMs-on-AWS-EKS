# Day 29 — The Complete Production LLM Stack

> **Hook:** 29 days of concepts. One diagram to rule them all. Here's what a production LLM inference system actually looks like end-to-end.

---

## The Post

Every day this month, we've zoomed in on one piece of the puzzle.

Today we zoom out.

This is the complete architecture — every layer, every component, every optimization we've covered — assembled into one production-grade system. The kind of system that serves millions of requests per day, handles traffic spikes automatically, and costs a fraction of managed API alternatives.

Let's walk through it top to bottom.

---

## The Full Stack Diagram

```
  COMPLETE PRODUCTION LLM INFERENCE ARCHITECTURE
  ════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────┐
  │                         USERS / CLIENTS                              │
  │           Web App │ Mobile │ API Consumers │ Internal Tools          │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │ HTTPS
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    AWS APPLICATION LOAD BALANCER                     │
  │                    SSL termination, WAF, rate limiting               │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         AMAZON EKS CLUSTER                           │
  │                                                                       │
  │  ┌──────────────────────────────────────────────────────────────┐   │
  │  │                    RAY SERVE HEAD (Day 24)                    │   │
  │  │  • Request routing & load balancing                          │   │
  │  │  • Autoscaling controller (target_ongoing_requests)          │   │
  │  │  • Health monitoring & replica management                    │   │
  │  │  • Zero-downtime rolling updates                             │   │
  │  └───────────────────────────┬──────────────────────────────────┘   │
  │                              │ dispatches to replicas                │
  │         ┌────────────────────┼────────────────────┐                 │
  │         ▼                    ▼                    ▼                 │
  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
  │  │  vLLM       │    │  vLLM       │    │  vLLM       │            │
  │  │  Replica 1  │    │  Replica 2  │    │  Replica N  │            │
  │  │             │    │             │    │             │            │
  │  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │            │
  │  │ │Cont.    │ │    │ │Cont.    │ │    │ │Cont.    │ │            │
  │  │ │Batching │ │    │ │Batching │ │    │ │Batching │ │            │
  │  │ │(Day 15) │ │    │ │(Day 15) │ │    │ │(Day 15) │ │            │
  │  │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │            │
  │  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │            │
  │  │ │Paged    │ │    │ │Paged    │ │    │ │Paged    │ │            │
  │  │ │Attn     │ │    │ │Attn     │ │    │ │Attn     │ │            │
  │  │ │(Day 16) │ │    │ │(Day 16) │ │    │ │(Day 16) │ │            │
  │  │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │            │
  │  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │            │
  │  │ │Flash    │ │    │ │Flash    │ │    │ │Flash    │ │            │
  │  │ │Attn     │ │    │ │Attn     │ │    │ │Attn     │ │            │
  │  │ │(Day 18) │ │    │ │(Day 18) │ │    │ │(Day 18) │ │            │
  │  │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │            │
  │  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │            │
  │  │ │Prefix   │ │    │ │Prefix   │ │    │ │Prefix   │ │            │
  │  │ │Cache    │ │    │ │Cache    │ │    │ │Cache    │ │            │
  │  │ │(Day 19) │ │    │ │(Day 19) │ │    │ │(Day 19) │ │            │
  │  │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │            │
  │  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │            │
  │  │ │LoRA     │ │    │ │LoRA     │ │    │ │LoRA     │ │            │
  │  │ │Adapters │ │    │ │Adapters │ │    │ │Adapters │ │            │
  │  │ │(Day 27) │ │    │ │(Day 27) │ │    │ │(Day 27) │ │            │
  │  │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │            │
  │  │             │    │             │    │             │            │
  │  │ NVIDIA L40S │    │ NVIDIA L40S │    │ NVIDIA L40S │            │
  │  │ 48 GB VRAM  │    │ 48 GB VRAM  │    │ 48 GB VRAM  │            │
  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘            │
  │         │                  │                  │                    │
  │  ┌──────▼──────────────────▼──────────────────▼──────────────┐   │
  │  │                  LMCache (Day 26)                          │   │
  │  │  L1: CPU RAM offload (100s GB, ~5ms)                      │   │
  │  └──────────────────────────┬──────────────────────────────── ┘   │
  │                             │                                      │
  │  ┌──────────────────────────▼───────────────────────────────┐     │
  │  │           Amazon ElastiCache — Valkey (Day 26)            │     │
  │  │  L2: Remote KV cache (shared across all replicas)        │     │
  │  │  TLS encrypted, serverless auto-scaling                  │     │
  │  └───────────────────────────────────────────────────────────┘     │
  │                                                                       │
  │  ┌───────────────────────────────────────────────────────────┐      │
  │  │              RAG PIPELINE (Day 28 / Module 700)            │      │
  │  │  Query → Embedding → S3 Vectors search → Top-K docs       │      │
  │  │  → Inject into prompt context before vLLM call            │      │
  │  └───────────────────────────────────────────────────────────┘      │
  │                                                                       │
  │  ┌───────────────────────────────────────────────────────────┐      │
  │  │              KARPENTER (GPU/ module)                       │      │
  │  │  Watches pod pending state → provisions GPU node          │      │
  │  │  g6e.2xlarge on demand, terminates when idle              │      │
  │  └───────────────────────────────────────────────────────────┘      │
  │                                                                       │
  │  ┌───────────────────────────────────────────────────────────┐      │
  │  │         MONITORING STACK (Module 300)                      │      │
  │  │  DCGM Exporter → Prometheus → Grafana dashboards          │      │
  │  │  Metrics: tokens/s, TTFT, TPOT, GPU util, KV cache fill  │      │
  │  └───────────────────────────────────────────────────────────┘      │
  └─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │ Amazon   │ │ Amazon   │ │ Amazon   │
             │    S3    │ │   S3     │ │   ECR    │
             │  Model   │ │  LoRA    │ │Container │
             │ Weights  │ │Adapters  │ │ Images   │
             └──────────┘ └──────────┘ └──────────┘
```

---

## How a Request Flows Through the Entire Stack

```
  REQUEST LIFECYCLE: END TO END
  ══════════════════════════════

  User: "What are the specs for the Sony WH-1000XM5 headphones?
         Please respond as our support specialist."

  ① ALB receives HTTPS request (TLS termination)
     Latency added: ~1ms

  ② Ray Serve head receives request
     Checks: which replica has lowest queue depth?
     Routes to vLLM Replica 2 (least busy)
     Latency added: ~0.5ms

  ③ RAG pipeline runs (if enabled)
     Embed user query → vector search S3 Vectors
     Retrieve top-3 product spec docs
     Inject into prompt: 500 token context added
     Latency added: ~80ms

  ④ vLLM Replica 2 receives full prompt
     Check LMCache L0 (GPU): partial hit on system prompt chunks
     Check LMCache L1 (CPU RAM): hit on 3 more chunks
     Check LMCache L2 (Valkey): miss (new product query)
     Latency saved: ~120ms (cached chunks skipped)

  ⑤ vLLM runs prefill on uncached tokens
     FlashAttention (Day 18): tiled IO-aware computation
     Prefix caching (Day 19): system prompt chunks reused
     TTFT: ~85ms for uncached portion

  ⑥ vLLM runs decode loop
     Speculative decoding (Day 17): draft model proposes tokens
     Continuous batching (Day 15): interleaved with other requests
     PagedAttention (Day 16): KV pages allocated dynamically
     TPOT: ~13ms per token

  ⑦ LoRA adapter applied at each layer (Day 27)
     "support-specialist" adapter modifies tone/behavior
     Overhead vs base model: ~12%

  ⑧ Response streams back
     Ray Serve → ALB → User (Server-Sent Events)
     First token: ~170ms total (TTFT)
     Stream: 13ms/token (~77 tokens/sec to user)

  ⑨ Metrics emitted
     vLLM → Prometheus → Grafana updates in real-time
     DCGM → GPU utilization, memory, temperature logged

  ⑩ KV cache written
     New prompt chunks → stored in Valkey for next request
     Next user asking same product question: ~20ms TTFT
```

---

## The Numbers: Everything Compounded

```
  OPTIMIZATIONS STACKED (this full production system)
  ═════════════════════════════════════════════════════

  Baseline: naïve single-GPU server, no optimizations
  Throughput: ~200 tokens/sec | TTFT: ~800ms | Cost: $1.65/hr

  ┌──────────────────────────────────────────────────────────────┐
  │ Optimization          │ Throughput gain │ TTFT impact        │
  ├──────────────────────────────────────────────────────────────┤
  │ Continuous Batching   │ +600%           │ neutral            │
  │ PagedAttention        │ +70%            │ -20% (more conc.)  │
  │ FlashAttention        │ +20%            │ -15%               │
  │ Prefix Caching        │ +30-50% (eff.)  │ -60% (cache hit)   │
  │ Quantization FP8      │ +58%            │ -30%               │
  │ Speculative Decoding  │ +85%            │ neutral            │
  │ LMCache (cross-pod)   │ +40% (eff.)     │ -96% (cache hit)   │
  │ Ray autoscaling       │ N× replicas     │ -queue wait        │
  └──────────────────────────────────────────────────────────────┘

  COMBINED RESULT (single GPU, all optimizations):
  Throughput: ~7,000+ tokens/sec (35× baseline)
  TTFT (cache hit): ~20ms (-97.5% vs baseline)
  TTFT (cache miss): ~85ms (-89% vs baseline)

  COMBINED RESULT (4-replica autoscaled fleet):
  Peak throughput: ~28,000 tokens/sec
  Cost at peak: 4 × $1.65/hr = $6.60/hr
  vs GPT-4 API equivalent: ~$84/hr for same throughput
  SAVINGS: 92% vs managed API at peak load
```

---

## Infrastructure Cost Model

```
  MONTHLY COST BREAKDOWN (production system, moderate traffic)
  ═════════════════════════════════════════════════════════════

  Traffic: 500,000 requests/month
  Avg response: 300 output tokens
  Peak: 50 req/min (9am-6pm weekdays)

  COMPUTE (Karpenter-managed, autoscaled):
  ┌─────────────────────────────────────────────────────────┐
  │ Peak hours (200 hrs/month): 3× g6e.2xlarge             │
  │   3 × $1.65 × 200 = $990                               │
  │ Off-peak (520 hrs/month): 1× g6e.2xlarge               │
  │   1 × $1.65 × 520 = $858                               │
  │ Compute total: $1,848/month                             │
  └─────────────────────────────────────────────────────────┘

  STORAGE & CACHE:
  ┌─────────────────────────────────────────────────────────┐
  │ S3 (model + adapters + RAG docs): ~15 GB → ~$0.35/month │
  │ ElastiCache Valkey Serverless: ~$45/month               │
  │ EKS cluster (system nodes): ~$150/month                 │
  └─────────────────────────────────────────────────────────┘

  NETWORKING:
  ┌─────────────────────────────────────────────────────────┐
  │ ALB: $16/month + data transfer                          │
  │ Data out (500K × ~1KB avg): ~$22/month                  │
  └─────────────────────────────────────────────────────────┘

  TOTAL: ~$2,081/month
  GPT-4 API equivalent (500K × 300 out tokens):
  150M tokens × $0.030/1K = $4,500/month

  Self-hosted savings: $2,419/month (54% cheaper)
  Break-even point: ~100K requests/month
```

---

## The Monitoring Dashboard

```
  WHAT TO WATCH IN PRODUCTION
  ════════════════════════════

  PRIMARY METRICS (alert on these):
  ┌──────────────────────────────────────────────────────────┐
  │ TTFT P95 > 500ms        → queue building, scale up       │
  │ TPOT P95 > 50ms         → GPU memory pressure           │
  │ GPU cache util > 95%    → KV cache nearly full           │
  │ Request error rate > 1% → replica health issue           │
  │ Ray replica count = max → need more GPU nodes            │
  └──────────────────────────────────────────────────────────┘

  EFFICIENCY METRICS (optimize for these):
  ┌──────────────────────────────────────────────────────────┐
  │ GPU utilization 70-85%  → healthy throughput zone        │
  │ LMCache hit rate > 70%  → prefix sharing working         │
  │ Tokens/sec/GPU          → infrastructure efficiency      │
  │ Cost per 1M tokens      → business metric                │
  └──────────────────────────────────────────────────────────┘

  CAPACITY METRICS (plan ahead):
  ┌──────────────────────────────────────────────────────────┐
  │ Karpenter node provision time                            │
  │ Max replicas headroom                                    │
  │ S3 model load time on cold start                        │
  └──────────────────────────────────────────────────────────┘
```

---

## The Concept → Module Map (Complete)

```
  29 DAYS → 8 WORKSHOP MODULES
  ══════════════════════════════

  Day 1-7   (Inference basics)    ──▶  100-vllm/
  Day 8-14  (GPU & memory)        ──▶  100-vllm/ + GPU/
  Day 15-21 (Making it fast)      ──▶  100-vllm/ + 300-benchmarking/
  Day 22-25 (Scaling multi-GPU)   ──▶  800-ray/
  Day 26    (KV offloading)       ──▶  400-lmcache/
  Day 27    (LoRA at scale)       ──▶  600-finetuning/
  Day 28    (RAG vs FT)           ──▶  600-finetuning/ + 700-rag/
  Day 29    (Full stack)          ──▶  All modules combined

  Every concept in this series maps to running code
  in this workshop repository. Not theory — production patterns.
```

---

## Key Takeaway

> A production LLM system isn't one piece of technology — it's 8+ layers working together.  
> Each layer we covered this month solves a specific, real problem: memory waste, slow prefill, idle GPUs, stale knowledge, behavior gaps.  
> The compounded result: 35× throughput improvement, 97% TTFT reduction, 92% cost savings vs managed APIs — on the same hardware.  
> You now have the mental model for every layer. Tomorrow: where this is all heading.

---

*30-Day Series: LLM Inference Is Everything | Day 29 of 30*  
*← [Day 28](./day-28-rag-vs-finetuning.md) | [Day 30 →](./day-30-whats-next.md) | [Back to Index](../README.md)*
