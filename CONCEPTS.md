# AI Inference – Beginner's Primer

> Read this first if you are new to AI inference. Every concept in this workshop traces back to something explained on this page. Based on the Red Hat guide: https://www.redhat.com/en/topics/ai/what-is-ai-inference

---

## What is AI Inference?

AI inference is **when an AI model gives you an answer**.

Think of it like a student who studied hard (training) and is now sitting an exam (inference). The studying is done – now the knowledge is being applied to a new question the student has never seen before.

```
┌─────────────────────────────────────────────────────────────┐
│                  The Two Phases of AI                        │
│                                                              │
│  PHASE 1: TRAINING                PHASE 2: INFERENCE        │
│  ┌─────────────────────┐         ┌─────────────────────┐   │
│  │  Feed the model      │         │  Ask the model       │   │
│  │  millions of         │  ────▶  │  a question.         │   │
│  │  examples.           │         │  It answers from     │   │
│  │  It learns patterns. │         │  what it learned.    │   │
│  └─────────────────────┘         └─────────────────────┘   │
│   Happens ONCE (expensive)         Happens EVERY REQUEST    │
│   Weeks / months on clusters       Milliseconds on a GPU    │
└─────────────────────────────────────────────────────────────┘
```

**Real-life analogy:** Your brain learned what a dog looks like from thousands of pictures. When you see a new dog you have never seen before, you instantly recognise it. That recognition is *inference*. The learning was *training*.

In this workshop, Ministral-3-8B has already been trained by Mistral AI. We only do inference (and fine-tuning, which is like extra studying on a specific topic).

---

## Why Does Inference Need a GPU?

An AI model doing inference performs billions of floating-point maths operations per second. A modern CPU has ~10–100 GFLOPS. A modern GPU (like the NVIDIA L40S used here) has **~733 TOPS** for AI workloads – roughly 100–1000× faster.

```
CPU  ──── 10–100 GFLOPS    ────  fast at sequential tasks
GPU  ──── 733 TOPS (INT8)  ────  fast at parallel matrix math
                                   ↑ this is what AI needs
```

The **NVIDIA L40S** in this workshop:
- 48 GB of VRAM (GPU memory)
- 733 TOPS for INT8 operations
- 18,176 CUDA cores running in parallel

---

## Types of AI Inference

The Red Hat article identifies three types. Each module in this workshop demonstrates one or more.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    3 Types of AI Inference                            │
├──────────────────┬──────────────────────────┬────────────────────────┤
│  ONLINE          │  BATCH                   │  STREAMING             │
│  (Dynamic)       │  (Offline/Static)        │                        │
├──────────────────┼──────────────────────────┼────────────────────────┤
│  Responds in     │  Processes large groups  │  Receives a constant   │
│  real time.      │  of requests at once.    │  flow of data and      │
│  User is waiting │  User is NOT waiting.    │  makes predictions     │
│  for the answer. │  Results delivered later.│  continuously.         │
│                  │                          │                        │
│  Example:        │  Example:                │  Example:              │
│  ChatGPT,        │  Overnight batch of      │  Fraud detection,      │
│  this workshop's │  1M customer emails      │  sensor monitoring,    │
│  Open WebUI chat │  run through a spam      │  live video analysis   │
│                  │  classifier              │                        │
│  Module: 100,    │  Module: 300             │  Module: 300           │
│  200, 800        │  (benchmarking)          │  (saturation test)     │
├──────────────────┴──────────────────────────┴────────────────────────┤
│  This workshop focuses primarily on ONLINE inference                  │
│  (real-time chat + agents), with BATCH used in benchmarking.         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What is an Inference Server?

An **inference server** is the software layer that:
1. Receives requests (HTTP API calls)
2. Schedules them onto the model
3. Returns responses

Without an inference server you would have to write all of that yourself. The inference server in this workshop is **vLLM**.

### Single-Model vs Multi-Model Servers

```
Single-model server          Multi-model (Multimodal) server
─────────────────────        ────────────────────────────────
One model loaded.            Several models loaded at once.
Extremely fast for           Can handle text, images, code,
that specific model.         audio on the same GPU.
Used in: Module 100,         Used in: production AI platforms
200, 300, 400                that serve many different tasks.
```

---

## What is vLLM? (The Inference Engine in This Workshop)

vLLM is an open-source inference server built specifically for Large Language Models (LLMs). It solves the biggest efficiency problems that make running LLMs expensive.

### The core problem vLLM solves: GPU Memory waste

When a language model generates text, it stores intermediate calculations called **Key-Value (KV) tensors** for every token it has processed. Without vLLM, these are stored statically – memory is allocated upfront and wasted if not fully used.

**vLLM's solution: PagedAttention**

```
WITHOUT PagedAttention (naive):
┌────────────────────────────────────────────┐
│  GPU Memory                                 │
│  Request A: [████████████░░░░░░░░░░░░░░░░] │ ← 50% wasted
│  Request B: [██░░░░░░░░░░░░░░░░░░░░░░░░░░] │ ← 75% wasted
│  Request C: REJECTED – not enough memory   │
└────────────────────────────────────────────┘

WITH PagedAttention (vLLM):
┌────────────────────────────────────────────┐
│  GPU Memory (split into small pages)       │
│  Page 1: [A][A][A][B][C][C][A][B][B][C]   │ ← no waste
│  Page 2: [A][B][C][C][A][A][B][C][A][B]   │ ← pages shared
│  All 3 requests fit!                       │
└────────────────────────────────────────────┘
```

**Continuous batching** is vLLM's scheduler that fills the GPU with work from multiple users simultaneously, instead of waiting for one request to finish before starting the next.

---

## What is the KV Cache?

Every token the model reads or generates gets turned into two sets of numbers: a **Key** and a **Value**. These are cached so the model does not recalculate them when it reads the same text again.

```
Without KV Cache:
  Request: "What is the capital of France?"
  Model recalculates math for every token every time. Slow.

With KV Cache:
  First request: "What is the capital of France?" → compute KV, store it
  Second request (same prompt): → reuse stored KV, skip computation, fast!
```

Module 400 (LMCache) extends the KV cache beyond the GPU into CPU RAM and a remote Redis-compatible store (Valkey), so cached computations survive across requests and even across pods.

---

## The 3 Big Challenges of AI Inference

The Red Hat article identifies these three challenges. This workshop addresses all three.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Challenge 1: COMPLEXITY                                              │
│  Simple task: "Write a haiku" → small model, fast                    │
│  Complex task: "Detect financial fraud in 1M transactions" →         │
│    needs massive model, huge data, specialised hardware               │
│                                                                       │
│  This workshop's solution: fine-tuning (Module 600) to teach the     │
│  base model a specific domain without training from scratch           │
├──────────────────────────────────────────────────────────────────────┤
│  Challenge 2: RESOURCES                                               │
│  GPU VRAM is expensive and finite.                                    │
│  Ministral-3-8B weights alone use ~7.5 GB of the 48 GB L40S VRAM.   │
│  KV cache for 256 concurrent users uses the remaining ~40 GB.        │
│                                                                       │
│  This workshop's solutions:                                           │
│  - vLLM PagedAttention (Module 100): pack GPU memory tightly         │
│  - LMCache (Module 400): spill KV cache to CPU RAM + Valkey          │
│  - Ray Serve (Module 800): spread load across multiple GPUs          │
├──────────────────────────────────────────────────────────────────────┤
│  Challenge 3: COST                                                    │
│  GPU instances are expensive.                                         │
│  g6e.2xlarge (1× L40S) costs $1.65–$3.50/hr on AWS.                 │
│  Poor utilisation = money wasted.                                     │
│                                                                       │
│  This workshop's solutions:                                           │
│  - Benchmarking (Module 300): find the saturation point              │
│  - Karpenter (GPU folder): provision GPU only when needed            │
│  - Autoscaling (Module 800): scale down to 0 when idle               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What is Fine-Tuning?

A pre-trained model like Ministral-3-8B knows a lot about the world (from its training data). But it may not know your company's products, your internal policies, or how to respond in your specific domain.

**Fine-tuning** is additional training on a small, focused dataset to teach the model a specific skill without starting from scratch.

```
Base Model (general knowledge)
        │
        │  Fine-tune on: AnyVC startup advisor conversations
        │  (Module 600 dataset: anyvc-startup-dataset.jsonl)
        ▼
Domain-Expert Model (startup advisor + general knowledge)
```

### LoRA – Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates all 3.8 billion weights → requires enormous compute and storage.

**LoRA (Low-Rank Adaptation)** is a smarter approach:
- Freeze all original weights (do not change them)
- Add two tiny matrices (A and B) next to each attention layer
- Only train those tiny matrices
- Result: ~4.7 million trainable parameters instead of 3.8 billion (0.12%)

```
Normal fine-tuning:  Train 3,800,000,000 parameters  → 15+ GB optimizer state
LoRA fine-tuning:    Train         4,700,000 parameters  →  0.1 GB optimizer state
                                        ↑
                              99.88% reduction in compute cost
```

The LoRA "adapter" is a tiny file (~50–100 MB) that plugs into vLLM at serving time. vLLM can load multiple LoRA adapters for one base model simultaneously.

---

## What is RAG (Retrieval-Augmented Generation)?

A model's knowledge is frozen at its training cutoff date. RAG lets the model answer questions using **up-to-date information** from your own documents.

```
WITHOUT RAG:
  User:  "What is the current price of Product X?"
  Model: "I don't have information about current prices." ← stale knowledge

WITH RAG (Module 700):
  User:  "What is the current price of Product X?"
  Step 1: Search product catalog for "Product X" → find matching documents
  Step 2: Add those documents to the prompt as context
  Step 3: Model answers using the retrieved context
  Model: "Product X is currently $49.99, available in 3 colours." ← grounded
```

RAG vs Fine-tuning – when to use which:

```
┌─────────────────────┬─────────────────────────────────────────────┐
│  Use RAG when...    │  Use Fine-tuning when...                     │
├─────────────────────┼─────────────────────────────────────────────┤
│  Data changes often │  Behaviour/style needs to change             │
│  (prices, news,     │  (always respond as a VC advisor)            │
│   product catalog)  │                                              │
│  Data is large      │  Domain vocabulary is specialised            │
│  (millions of docs) │  (medical, legal, financial terms)           │
│  No retraining      │  Latency is critical (no retrieval step)     │
│  budget/time        │                                              │
└─────────────────────┴─────────────────────────────────────────────┘
```

---

## What are AI Agents?

An AI agent is a model that can **take actions**, not just answer questions. It uses tools (functions it can call) to interact with the real world.

```
Simple LLM:      Question → Model → Text answer
                 "What time is it in Tokyo?" → "I don't have real-time data."

AI Agent:        Question → Model → decides to call a tool
(Module 200)     "What time is it in Tokyo?"
                   → calls current_time(location="Tokyo")
                   → tool fetches real time via API
                   → Model synthesises: "It is 3:14 AM in Tokyo."
```

Agents work because modern LLMs (including Ministral) support **tool calling** – a structured way for the model to say "I need to call function X with argument Y" instead of just producing text. vLLM's `--enable-auto-tool-choice` and `--tool-call-parser=mistral` flags enable this.

---

## What is Mixture of Experts (MoE)?

A standard LLM activates ALL its parameters for every token it processes. MoE models activate only a **subset of "expert" sub-networks** for each token, routing the computation to whichever experts are best suited.

```
Standard LLM (dense):
  Input token → ENTIRE network processes it → Output
  100% of parameters active per token

MoE LLM (sparse):
  Input token → Router → Expert 2 + Expert 7 handle it → Output
  Only 10–20% of parameters active per token
  ↑ Same quality, much faster inference, lower memory pressure
```

MoE is not directly demonstrated in this workshop but is the architecture behind many frontier models (e.g., Mixtral 8x7B, GPT-4 is believed to use MoE). When you scale to multiple GPUs (Module 800 / Ray Serve), the same concept of splitting work applies.

---

## What is Distributed Inference?

A single GPU has finite memory. Ministral-3-8B fits on one L40S (48 GB). But a model like Llama-3-70B (140 GB of weights) does not.

**Distributed inference** splits the model and/or the work across multiple GPUs.

```
Single GPU:
  GPU 0: [Full model – 7.5 GB weights + KV cache]
  ✓ Works for 3-8B models

Tensor Parallelism (multi-GPU):
  GPU 0: [Layer 0–15 of the model]  ──┐
  GPU 1: [Layer 16–31 of the model] ──┤── All GPUs process each request
  GPU 2: [Layer 32–47 of the model] ──┤   together in parallel
  GPU 3: [Layer 48–63 of the model] ──┘
  ✓ Required for 70B+ models

  vLLM flag: --tensor-parallel-size=4
  Module 800 (Ray Serve) orchestrates this at cluster level
```

Distributed inference also means **horizontal scaling** – running multiple copies of the same model on different GPUs, each handling a subset of users:

```
User A ──▶ GPU 0 (vLLM replica 1) ──▶ answer
User B ──▶ GPU 1 (vLLM replica 2) ──▶ answer   } Ray Serve autoscales
User C ──▶ GPU 0 (vLLM replica 1) ──▶ answer     replicas 1→N
```

---

## How All Modules Connect

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Red Hat AI Inference Concepts → Workshop Modules            │
├─────────────────────────────────────────┬───────────────────────────────┤
│  Concept                                │  Module                        │
├─────────────────────────────────────────┼───────────────────────────────┤
│  GPU as inference hardware              │  GPU/ folder                   │
│  Online inference server                │  100-vllm                     │
│  vLLM (PagedAttention, batching)        │  100-vllm                     │
│  Agentic AI / tool calling              │  200-strands-agent            │
│  Measuring inference performance        │  300-benchmarking             │
│  Resource challenge (KV cache)          │  400-lmcache                  │
│  Fine-tuning (LoRA/PEFT)               │  600-finetuning               │
│  RAG (alternative to fine-tuning)       │  700-rag                      │
│  Distributed inference + autoscaling    │  800-ray                      │
└─────────────────────────────────────────┴───────────────────────────────┘
```

---

## Glossary

| Term | Plain English Meaning |
|------|-----------------------|
| **Inference** | The model giving you an answer |
| **Token** | A word chunk (roughly ¾ of a word on average). "Hello world" = 2 tokens |
| **LLM** | Large Language Model – the kind of AI model used in this workshop |
| **GPU** | Graphics Processing Unit – hardware that runs AI fast due to parallel computing |
| **VRAM** | Video RAM – memory on the GPU where model weights and KV cache live |
| **KV Cache** | Stored intermediate math results, reused to avoid recalculating same tokens |
| **PagedAttention** | vLLM technique that eliminates wasted GPU memory by paging KV cache |
| **Continuous Batching** | Processing multiple users' requests simultaneously in one GPU pass |
| **TTFT** | Time To First Token – how long you wait before seeing the first word of a response |
| **Throughput** | How many tokens per second the server can produce across all users |
| **Latency** | How long a single request takes end-to-end |
| **LoRA** | Low-Rank Adaptation – efficient fine-tuning that trains only 0.1% of weights |
| **PEFT** | Parameter-Efficient Fine-Tuning – umbrella term for techniques like LoRA |
| **RAG** | Retrieval-Augmented Generation – give the model fresh documents at query time |
| **Agent** | An AI that can call external tools/APIs, not just generate text |
| **Tool Calling** | Structured way for an LLM to invoke a function with specific arguments |
| **MoE** | Mixture of Experts – model architecture that activates only part of the network per token |
| **Tensor Parallelism** | Splitting one model across multiple GPUs |
| **Distributed Inference** | Running inference across multiple machines/GPUs |
| **Autoscaling** | Automatically adding or removing GPU replicas based on demand |
| **S3 Vectors** | Amazon's native vector database built into S3 (used for RAG search) |
| **Embedding** | A list of numbers that represents the meaning of a text chunk (used in RAG) |
| **Similarity Search** | Finding documents whose embeddings are mathematically closest to the query |
