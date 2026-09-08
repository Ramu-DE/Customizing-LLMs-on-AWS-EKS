# Day 2 — Training vs Inference

> **Hook:** Why training gets the attention — but inference pays the bills.

---

## The Post

Training gets the headlines. Inference pays the bills.

Here's why the attention is backwards:

**Training is a science project.**  
You do it once (or a few times). It's expensive, yes — but it's a capital expense.  
You plan for it, budget for it, and it ends.

**Inference is a business.**  
It runs continuously. Every user request. Every API call. Every second your product is live.

The math is brutal:
- A large model might cost **$5M to train**
- At scale, inference can cost **$5M per month**

And unlike training, you can't just throw more GPUs at inference and call it done.  
You have to be smart about batching, memory, and latency/throughput tradeoffs.

---

## ASCII Diagram — The Cost Curve

```
  COST OVER TIME
  ═══════════════

  $$$
   │
   │  Training
   │  ████ (one-time spike)
   │  ████
   │  ████
   │  ████─────────────────────────────────────────────────────
   │          Inference ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
   │          (continuous, growing with users)
   │
   └──────────────────────────────────────────────────────────▶ Time
             │                    │
           Launch            At Scale
           
   Training cost: fixed, bounded, one-time
   Inference cost: variable, unbounded, forever
```

---

## Architecture Comparison

```
  TRAINING ARCHITECTURE              INFERENCE ARCHITECTURE
  ══════════════════════             ══════════════════════════

  ┌─────────────────────┐           ┌─────────────────────────────┐
  │   Training Cluster  │           │     Production Cluster      │
  │                     │           │                             │
  │  Node 1 ──┐         │           │  ┌──────────────────────┐  │
  │  Node 2 ──┤         │           │  │   Load Balancer      │  │
  │  Node 3 ──┼── All   │           │  └──────────┬───────────┘  │
  │  Node 4 ──┤  reduce │           │             │               │
  │  Node 5 ──┘ comms   │           │    ┌────────┼────────┐     │
  │                     │           │    │        │        │     │
  │  Duration: Weeks    │           │  vLLM    vLLM    vLLM     │
  │  Goal: Minimize     │           │  Pod 1   Pod 2   Pod 3    │
  │    training loss    │           │    │        │        │     │
  │                     │           │    └────────┼────────┘     │
  │  Optimize for:      │           │             │               │
  │  ✓ Throughput       │           │  Optimize for:             │
  │  ✓ Gradient sync    │           │  ✓ Latency (TTFT, TPOT)   │
  │  ✗ Latency          │           │  ✓ Cost per token          │
  └─────────────────────┘           │  ✓ Requests/second         │
                                     └─────────────────────────────┘
```

---

## The Staffing Gap

```
  TYPICAL ML TEAM STAFFING
  ═════════════════════════

  Research Engineers     ████████████████████  (heavy investment)
  Training Infra         ████████████          (good investment)
  MLOps / Fine-tuning    ████████              (moderate)
  Inference Engineers    ████                  (understaffed!)
  Inference Optimization ██                    (severely understaffed!)

  Reality check: Inference is where your product lives.
                 Inference is where most AI teams are weakest.
```

---

## Workshop Connection

Our workshop on AWS EKS was entirely on the **inference side**:

```
  We used Ministral-3-8B (pre-trained by Mistral AI)
                    │
                    ▼
  Module 100: Serve it with vLLM on GPU nodes
  Module 300: Measure TTFT, TPOT, throughput under load
  Module 400: Cut latency with LMCache KV offloading
  Module 600: Fine-tune (a small training job) → then back to inference
  Module 800: Scale inference with Ray Serve
                    │
                    ▼
  Everything comes back to: How fast? How cheap? How many users?
```

---

## Key Takeaway

> The gap between "we have a model" and "we have a product"  
> is almost entirely an inference engineering problem.

---

*30-Day Series: LLM Inference Is Everything | Day 2 of 30*  
*← [Day 1](./day-01-inference-is-everything.md) | Next → [Day 3](./day-03-what-happens-during-inference.md)*
