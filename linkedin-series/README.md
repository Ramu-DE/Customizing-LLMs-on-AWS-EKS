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
