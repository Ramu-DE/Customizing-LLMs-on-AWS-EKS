# Day 28 — RAG vs Fine-tuning: The Production Decision Guide

> **Hook:** "Should we RAG it or fine-tune it?" is the most common question I hear from teams shipping AI features. After a month of this series, here's the definitive framework.

---

## The Post

You've spent a month understanding how LLM inference works. Now the question every team faces:

Your product needs an LLM that knows about *your* data. Your product catalog, your knowledge base, your company's tone and policies.

Two paths:

1. **RAG** — Give the model fresh documents at query time via retrieval
2. **Fine-tuning** — Teach the model new knowledge or behavior through additional training

These are NOT interchangeable. Picking the wrong one costs months and tens of thousands of dollars.

Here's the complete decision framework.

---

## What Each Approach Actually Does

```
  RAG vs FINE-TUNING: THE CORE DIFFERENCE
  ════════════════════════════════════════

  RAG (Retrieval-Augmented Generation):
  ┌────────────────────────────────────────────────────────────┐
  │  User query                                                 │
  │     │                                                       │
  │     ▼                                                       │
  │  Embedding model converts query to vector                  │
  │     │                                                       │
  │     ▼                                                       │
  │  Vector search in document store                           │
  │  → Finds top-K relevant document chunks                    │
  │     │                                                       │
  │     ▼                                                       │
  │  Inject documents into prompt:                             │
  │  "Context: [retrieved docs]\n\nQuestion: [user query]"     │
  │     │                                                       │
  │     ▼                                                       │
  │  LLM generates answer grounded in retrieved docs           │
  └────────────────────────────────────────────────────────────┘

  FINE-TUNING (LoRA):
  ┌────────────────────────────────────────────────────────────┐
  │  Training phase (one-time):                                │
  │  Dataset of examples → train LoRA adapter → save to S3    │
  │                                                            │
  │  Serving phase (every request):                            │
  │  User query                                                │
  │     │                                                       │
  │     ▼                                                       │
  │  LLM with adapter applied                                  │
  │  → Model "knows" the domain from training                  │
  │  → No retrieval step, no context injection                 │
  │     │                                                       │
  │     ▼                                                       │
  │  Response reflects trained behavior/knowledge              │
  └────────────────────────────────────────────────────────────┘
```

---

## The Decision Framework

```
  WHEN TO USE RAG
  ════════════════

  ✓ Your data changes frequently (daily/weekly updates)
    → Product catalog, pricing, news, support tickets
    → Fine-tuning would need retraining every update

  ✓ Your data volume is massive (millions of documents)
    → Can't fit in a context window
    → Can't train on all of it efficiently
    → RAG retrieves the relevant 3-5 docs at query time

  ✓ You need source citations / grounding
    → RAG shows which documents were used
    → Hallucination is auditable
    → Compliance, legal, medical scenarios

  ✓ You want to avoid retraining costs
    → Add new documents to vector store = instant
    → No GPU training job, no model version management

  ✓ Your query types are heterogeneous
    → Different questions → different relevant docs
    → Dynamic retrieval handles this naturally

  ┌────────────────────────────────────────────────────────────┐
  │  Real example in this workshop:                            │
  │  Module 700 (RAG): electronics product Q&A                 │
  │  700-rag/electronics.jsonl → indexed in S3 Vectors         │
  │  User asks "Is the Sony WH-1000XM5 compatible with iOS?"  │
  │  → Retrieve Sony product docs                              │
  │  → Answer grounded in retrieved content                    │
  └────────────────────────────────────────────────────────────┘
```

```
  WHEN TO USE FINE-TUNING
  ════════════════════════

  ✓ You need to change the model's BEHAVIOR, not its knowledge
    → Always respond as a VC advisor
    → Enforce a specific output format (JSON with schema)
    → Adopt a custom persona or communication style

  ✓ Your domain has specialized vocabulary the base model struggles with
    → Medical ICD codes, legal clause types, financial instruments
    → RAG can retrieve the text, but model can't reason about it
    → Fine-tuning teaches the model to understand the domain

  ✓ Latency is critical and context windows are expensive
    → RAG adds retrieval latency (~50-200ms) and token cost
    → Long context = higher TTFT (more prefill tokens)
    → Fine-tuned model: no retrieval, shorter prompts

  ✓ Your dataset is small and curated (100–10,000 examples)
    → Just enough to teach a specific task
    → LoRA on 3.8B model: 1-4 hours on 1× GPU

  ┌────────────────────────────────────────────────────────────┐
  │  Real example in this workshop:                            │
  │  Module 600 (Fine-tuning): AnyVC startup advisor          │
  │  600-finetuning/anyvc-startup-dataset.jsonl                │
  │  → Trains model to respond as a VC advisor                 │
  │  → Tone, framework, and reasoning style learned           │
  │  → Not about retrieving specific startup data             │
  └────────────────────────────────────────────────────────────┘
```

---

## Side-by-Side Comparison

```
  RAG vs FINE-TUNING PRODUCTION COMPARISON
  ═════════════════════════════════════════

  ┌───────────────────┬─────────────────────┬─────────────────────┐
  │ Dimension         │ RAG                 │ Fine-tuning (LoRA)  │
  ├───────────────────┼─────────────────────┼─────────────────────┤
  │ Knowledge type    │ Factual, retrieval  │ Behavioral, style   │
  │ Data freshness    │ Real-time (update   │ Snapshot (retrain   │
  │                   │ vector store)       │ for new data)       │
  │ Data volume       │ Millions of docs    │ 100s–10,000s of     │
  │                   │ (only top-K used)   │ training examples   │
  │ Latency           │ +50-200ms retrieval │ Same as base model  │
  │ Token cost        │ Higher (long ctx)   │ Lower (no context)  │
  │ First deployment  │ Hours-days          │ Days-weeks          │
  │ Update cadence    │ Instant (add docs)  │ Hours per retrain   │
  │ Hallucination     │ Grounded (citable)  │ Possible (encoded)  │
  │ Infrastructure    │ Vector DB + embed   │ Training GPU + S3   │
  │ Transparency      │ Show sources        │ Opaque knowledge    │
  │ Data security     │ Docs in context     │ Encoded in weights  │
  └───────────────────┴─────────────────────┴─────────────────────┘
```

---

## The Third Path: RAG + Fine-tuning Together

Most production systems use BOTH. The key is separating concerns.

```
  COMBINED ARCHITECTURE
  ══════════════════════

  Fine-tuned model handles:                RAG handles:
  ┌──────────────────────────┐            ┌──────────────────────────┐
  │ • Response format        │            │ • Current product specs  │
  │ • Domain reasoning style │            │ • Company policies       │
  │ • Specialized vocabulary │            │ • Recent knowledge       │
  │ • Persona and tone       │            │ • Source citations       │
  └──────────────────────────┘            └──────────────────────────┘
              │                                       │
              └─────────────────┬─────────────────────┘
                                │
                          Combined LLM call:
                          Base model + LoRA adapter (behavior)
                          + Retrieved context (knowledge)
                          = Best of both worlds

  Example: Customer support bot
  - Fine-tuned on: company tone, escalation patterns (behavior)
  - RAG from: product docs, order history, FAQ (knowledge)
  - Result: responds in correct style with current accurate info
```

---

## Cost & Latency Breakdown (Real Numbers)

```
  COST MODEL: RAG vs FINE-TUNING
  ════════════════════════════════

  Scenario: 1M queries/month, mix of knowledge + behavior needs

  RAG COSTS:
  ┌──────────────────────────────────────────────────────────┐
  │ Embedding model:  $0 (using vLLM for embeddings)         │
  │ S3 Vectors:       $0.05/GB/month, 100k docs = ~1 GB → $5│
  │ Extra LLM tokens: +500 tokens context/request            │
  │   At 2,400 tok/s: +0.2s latency (prefill)               │
  │   Cost: 500M extra input tokens                          │
  │   At $0.42/1M (our L40S, Day 21): +$210/month           │
  │ Total RAG overhead: ~$215/month                          │
  └──────────────────────────────────────────────────────────┘

  FINE-TUNING COSTS:
  ┌──────────────────────────────────────────────────────────┐
  │ One-time training: 4 hrs × $1.65/hr = $6.60             │
  │ S3 adapter storage: 100 MB adapter → negligible         │
  │ Per-request overhead: ~0 (adapter already loaded)       │
  │ Retraining cadence: monthly → $6.60/month               │
  │ Total FT overhead: ~$6.60/month                         │
  └──────────────────────────────────────────────────────────┘

  WHICH IS CHEAPER?
  ┌──────────────────────────────────────────────────────────┐
  │ For BEHAVIOR (style, format, domain reasoning):          │
  │ Fine-tuning wins: $6.60/month vs $215/month (32×)        │
  │                                                          │
  │ For KNOWLEDGE (facts, docs, current info):               │
  │ RAG wins: instant updates, no retraining cost           │
  └──────────────────────────────────────────────────────────┘
```

---

## The Decision Tree

```
  QUICK DECISION TREE
  ════════════════════

  START HERE: What does your LLM need?
  │
  ├─▶ Does it need to know current/changing facts?
  │    YES → RAG (vector store of your documents)
  │    NO  ↓
  │
  ├─▶ Does it need to sound/behave a certain way?
  │    YES → Fine-tuning (LoRA adapter)
  │    NO  ↓
  │
  ├─▶ Does it need BOTH?
  │    YES → RAG + Fine-tuning combined
  │    NO  ↓
  │
  └─▶ Does the base model already do what you need?
       YES → Neither! Just prompt engineering.
       NO  → Re-evaluate: is this a knowledge or behavior gap?

  SPECIAL CASES:
  ┌─────────────────────────────────────────────────────────┐
  │ "We have 1M internal docs and need instant answers"     │
  │ → RAG (can't fine-tune 1M docs, can't fit in context)  │
  │                                                         │
  │ "We need the model to always output valid JSON"         │
  │ → Fine-tuning (teach format) or constrained decoding   │
  │                                                         │
  │ "We need the model to know last week's news"           │
  │ → RAG (fine-tuning is always a knowledge snapshot)     │
  │                                                         │
  │ "We need model to reason like our domain expert"       │
  │ → Fine-tuning (reasoning patterns, not just facts)     │
  └─────────────────────────────────────────────────────────┘
```

---

## Workshop Connection

```
  HOW THIS WORKSHOP DEMONSTRATES BOTH
  ═════════════════════════════════════

  RAG Pipeline — Module 700:
  ┌─────────────────────────────────────────────────────────────┐
  │  700-rag/electronics.jsonl → document corpus                 │
  │  rag-processor.yml → embeds docs, stores in S3 Vectors       │
  │  rag-serve.yml → retrieval + generation service              │
  │  rag-gradio-app.yml → UI for testing                        │
  │                                                              │
  │  Try: "What are the specs of the Sony WH-1000XM5?"          │
  │  → Retrieval finds product doc                               │
  │  → Model answers with cited source                          │
  └─────────────────────────────────────────────────────────────┘

  Fine-tuning Pipeline — Module 600:
  ┌─────────────────────────────────────────────────────────────┐
  │  600-finetuning/anyvc-startup-dataset.jsonl → training data  │
  │  train_lora.py → trains LoRA adapter (4hrs, 1× L40S)        │
  │  lora-training-job.yaml → Kubernetes Job                     │
  │  vllm-with-lora.yaml → serves trained adapter               │
  │                                                              │
  │  Try: "We have 50 paying users. What should we focus on?"   │
  │  → Model responds as VC advisor (trained behavior)          │
  │  → No retrieval, fast response, distinct tone               │
  └─────────────────────────────────────────────────────────────┘
```

---

## Week 4 Complete 🎉

```
  WEEK 4 RECAP — SCALING BEYOND ONE GPU
  ═══════════════════════════════════════

  Day 22: Tensor Parallelism
    → Split weight matrices across GPUs. Enables 70B+ on multi-GPU nodes.

  Day 23: Pipeline Parallelism
    → Assign layers to GPUs in sequence. Crosses node boundaries.

  Day 24: Ray Serve for LLM Autoscaling
    → Orchestrates vLLM fleet. Autoscales on demand. Zero-downtime updates.

  Day 25: Multi-node Inference
    → TP within node (NVLink) + PP across nodes (InfiniBand/EFA).

  Day 26: KV Cache Offloading with LMCache
    → GPU VRAM → CPU RAM → Valkey. 95% TTFT reduction on cache hits.

  Day 27: Serving LoRA Adapters at Scale
    → One GPU, many adapters. 88% cost reduction for multi-tenant serving.

  Day 28: RAG vs Fine-tuning
    → RAG for knowledge. Fine-tuning for behavior. Both for production.

  The series closes tomorrow (Day 29-30) with putting it all together:
  the complete production LLM inference architecture.
```

---

## Key Takeaway

> RAG and fine-tuning solve different problems — don't choose between them, understand when each applies.  
> RAG for dynamic knowledge: current facts, large document corpora, source citations.  
> Fine-tuning for permanent behavior: domain reasoning, output format, persona and tone.  
> Most production systems use both. Fine-tune the behavior, RAG the knowledge.  
> The wrong choice costs months and significant GPU budget. This decision tree should be on every AI team's wall.

---

*30-Day Series: LLM Inference Is Everything | Day 28 of 30 — Week 4 Complete*  
*← [Day 27](./day-27-lora-serving-at-scale.md) | [Back to Index](../README.md)*
