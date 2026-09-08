# 📢 LinkedIn 30-Day Series: "LLM Inference Is Everything"

> A plain-English, diagram-first series breaking down how Large Language Models actually run in production — written for engineers, architects, and technical leaders.

---

## Series Philosophy

Most content online covers **how to train** LLMs.  
This series covers **how to run** them — fast, cheaply, and at scale.

Every post includes:
- A clear hook
- ASCII flow/architecture diagrams
- Actionable takeaways
- Real numbers from production systems

---

## Week 1 — Understand Inference

| Day | Title | Core Concept |
|-----|-------|-------------|
| [Day 1](./week-1/day-01-inference-is-everything.md) | LLM Inference Is Everything | Training is a one-time event. Inference runs forever. |
| [Day 2](./week-1/day-02-training-vs-inference.md) | Training vs Inference | Why inference pays the bills. |
| [Day 3](./week-1/day-03-what-happens-during-inference.md) | What Actually Happens During LLM Inference | Prompt → Tokens → GPU → Response |
| [Day 4](./week-1/day-04-prefill-vs-decode.md) | Prefill vs Decode | The two phases that define LLM performance. |
| [Day 5](./week-1/day-05-why-tokens-matter.md) | Why Tokens Matter More Than You Think | Tokens = latency + cost + memory. |
| [Day 6](./week-1/day-06-ttft-vs-tpot.md) | TTFT vs TPOT | The two inference metrics every AI architect must know. |
| [Day 7](./week-1/day-07-why-llm-feels-slow.md) | Why Your LLM Feels Slow | Latency is a systems problem, not a model problem. |

---

## Week 2 — GPU & Memory

| Day | Title | Core Concept |
|-----|-------|-------------|
| [Day 8](./week-2/day-08-why-gpus-are-critical.md) | Why GPUs Are Critical for LLM Inference | LLMs are parallel matrix math — GPUs are built exactly for that. |
| [Day 9](./week-2/day-09-gpu-memory-bottleneck.md) | GPU Memory: The Real Bottleneck | Memory bandwidth matters more than FLOPS for decode. |
| [Day 10](./week-2/day-10-kv-cache-explained.md) | KV Cache Explained | The hidden structure that makes O(n²) generation O(1). |
| [Day 11](./week-2/day-11-why-long-context-is-expensive.md) | Why Long Context Is Expensive | 128K context = 62,500× more attention work than 512 tokens. |
| [Day 12](./week-2/day-12-quantization.md) | Quantization | FP32 → FP16 → INT8 → INT4: half the memory, same model. |
| [Day 13](./week-2/day-13-model-size-vs-inference-cost.md) | Model Size vs Inference Cost | A fine-tuned 7B model often beats a generic 70B at 10% of the cost. |
| [Day 14](./week-2/day-14-gpu-utilization-vs-good-inference.md) | GPU Utilization ≠ Good Inference | 90% GPU util with bad economics is not a success. |

---

## Connection to This Workshop

Everything in this series maps directly to what we built:

```
LinkedIn Series Concept          Workshop Module
─────────────────────────────────────────────────────────
Inference serving basics    ──▶  100-vllm/
Throughput benchmarking     ──▶  300-benchmarking/
KV cache optimization       ──▶  400-lmcache/
Token-level efficiency      ──▶  300-benchmarking/ + 400-lmcache/
TTFT / TPOT metrics         ──▶  300-benchmarking/ (Grafana dashboards)
Latency root causes         ──▶  All modules
```

---

*Series by Ramu-DE | Workshop: Customizing LLMs on AWS EKS*
