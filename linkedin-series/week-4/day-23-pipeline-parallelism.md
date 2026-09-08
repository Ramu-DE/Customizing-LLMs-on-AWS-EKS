# Day 23 — Pipeline Parallelism

> **Hook:** Tensor Parallelism splits each layer across GPUs. Pipeline Parallelism does something different — it gives each GPU its own set of layers.

---

## The Post

Yesterday we covered Tensor Parallelism: every GPU holds a slice of every layer, all GPUs work on every token together.

Today: **Pipeline Parallelism**. A completely different strategy.

Instead of splitting *within* each layer, you split *between* layers. GPU 0 handles layers 1–20. GPU 1 handles layers 21–40. Tokens flow through the pipeline like an assembly line.

Same goal (fit huge models). Different tradeoffs. Understanding both is what separates a GPU cluster architect from someone who just runs vLLM commands.

---

## The Core Idea: Layers as a Pipeline

```
  PIPELINE PARALLELISM VISUAL
  ════════════════════════════

  Llama-3-70B has 80 transformer layers.
  Split across 4 GPUs (PP=4, 20 layers each):

  ┌───────────────────────────────────────────────────────────┐
  │                    FORWARD PASS                            │
  │                                                            │
  │  Token embeddings                                          │
  │       │                                                    │
  │       ▼                                                    │
  │  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  │  GPU 0   │────▶│  GPU 1   │────▶│  GPU 2   │────▶│  GPU 3   │
  │  │Layer 1-20│     │Layer21-40│     │Layer41-60│     │Layer61-80│
  │  │  ~35 GB  │     │  ~35 GB  │     │  ~35 GB  │     │  ~35 GB  │
  │  └──────────┘     └──────────┘     └──────────┘     └──────────┘
  │       │               │               │               │
  │  Each GPU only communicates          Logits (output)  │
  │  the ACTIVATIONS to the next         returned to GPU 0│
  │  stage (~hidden_size × batch)                         │
  └───────────────────────────────────────────────────────────┘

  Communication cost: activation tensor between each stage
  Size: batch_size × seq_len × hidden_dim
  For Llama-3-70B (hidden=8192): 1 × 512 × 8192 × 2 bytes = 8 MB
  Compare to TP AllReduce: per layer, per token
```

---

## TP vs PP: The Key Difference

```
  TENSOR PARALLELISM vs PIPELINE PARALLELISM
  ═══════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │ TENSOR PARALLELISM (TP)                                      │
  │                                                              │
  │  ALL GPUs process EVERY layer of EVERY token TOGETHER        │
  │                                                              │
  │  Token ──▶ [GPU0+GPU1+GPU2+GPU3 do Layer 1 jointly]         │
  │         ──▶ [GPU0+GPU1+GPU2+GPU3 do Layer 2 jointly]         │
  │         ──▶ ... × 80 layers                                  │
  │                                                              │
  │  Communication: AllReduce after every layer (high frequency) │
  │  Requirement: NVLink (fast, low latency)                     │
  │  Idle GPUs: None — all GPUs always busy                      │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │ PIPELINE PARALLELISM (PP)                                    │
  │                                                              │
  │  EACH GPU processes its OWN LAYERS, in sequence              │
  │                                                              │
  │  Token ──▶ [GPU0 does Layers 1-20]                          │
  │         ──▶ [GPU1 does Layers 21-40]                         │
  │         ──▶ [GPU2 does Layers 41-60]                         │
  │         ──▶ [GPU3 does Layers 61-80]                         │
  │                                                              │
  │  Communication: Activation tensor between each stage (lower) │
  │  Requirement: Works over InfiniBand or even Ethernet         │
  │  Idle GPUs: The "pipeline bubble" problem                    │
  └─────────────────────────────────────────────────────────────┘
```

---

## The Pipeline Bubble Problem

Pipeline Parallelism has a fundamental inefficiency: while GPU 0 processes request #2, GPU 3 is still working on request #1. There's always a "bubble" of idle time.

```
  THE PIPELINE BUBBLE (naïve pipeline)
  ══════════════════════════════════════

  Time →
  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┐
  │GPU0│ R1 │ R2 │ R3 │ R4 │    │    │    │    │
  ├────┼────┼────┼────┼────┼────┼────┼────┼────┤
  │GPU1│idle│ R1 │ R2 │ R3 │ R4 │    │    │    │
  ├────┼────┼────┼────┼────┼────┼────┼────┼────┤
  │GPU2│idle│idle│ R1 │ R2 │ R3 │ R4 │    │    │
  ├────┼────┼────┼────┼────┼────┼────┼────┼────┤
  │GPU3│idle│idle│idle│ R1 │ R2 │ R3 │ R4 │    │
  └────┴────┴────┴────┴────┴────┴────┴────┴────┘
         ↑
  Bubble = (PP-1) steps of idle at startup = 3 idle GPU-steps

  Bubble fraction = (PP-1) / (PP-1 + batch_size)
  With PP=4, batch=1: bubble = 75% idle!
  With PP=4, batch=16: bubble = 16% idle — much better

  KEY INSIGHT: Pipeline Parallelism needs LARGE BATCHES
  to amortize the bubble overhead.
  With continuous batching in vLLM, this is manageable.
```

---

## Micro-batching: The Fix for Pipeline Bubbles

```
  MICRO-BATCHING TO FILL THE PIPELINE
  ═════════════════════════════════════

  Split each batch into M micro-batches:

  Time →
  ┌────┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  │GPU0│m1│m2│m3│m4│  │  │  │  │  │  │  │  │  │  │  │
  ├────┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
  │GPU1│  │m1│m2│m3│m4│  │  │  │  │  │  │  │  │  │  │
  ├────┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
  │GPU2│  │  │m1│m2│m3│m4│  │  │  │  │  │  │  │  │  │
  ├────┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
  │GPU3│  │  │  │m1│m2│m3│m4│  │  │  │  │  │  │  │  │
  └────┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

  With M=4 micro-batches, GPUs are mostly busy.
  Bubble fraction: (PP-1) / (PP-1+M) = 3/7 = 43% → much better
  With M=16: bubble = 3/19 = 16%

  This is how Megatron-LM, DeepSpeed, and production
  LLM training pipelines achieve near-linear scaling.
```

---

## Real Numbers: PP Across Nodes

```
  CROSS-NODE PIPELINE PARALLELISM
  ════════════════════════════════

  Setup: Llama-3-405B across 8 nodes × 8× H100 (80 GB each)
  Total VRAM: 64 × 80 GB = 5,120 GB
  Model weights: ~810 GB (BF16)

  Strategy: TP=8 within each node, PP=8 across nodes

  Node 0 (GPU 0-7, TP=8):   Layers 1-50    ~101 GB weights
  Node 1 (GPU 8-15, TP=8):  Layers 51-100  ~101 GB weights
  ...
  Node 7 (GPU 56-63, TP=8): Layers 351-400 ~101 GB weights

  Communication:
  ┌────────────────────────────────────────────────────────────┐
  │ Within node: NVLink (900 GB/s) — handles TP AllReduce      │
  │ Between nodes: InfiniBand 400G (50 GB/s) — handles PP      │
  │                                                            │
  │ Activation transfer between PP stages:                     │
  │ batch=8 × seq=2048 × hidden=16384 × 2 bytes = 537 MB      │
  │ At 50 GB/s IB: ~11ms per stage                            │
  │ 8 stages × 11ms = 88ms pipeline overhead per batch        │
  └────────────────────────────────────────────────────────────┘

  Achieved throughput: ~1,200 tokens/sec for 405B
  Cost: 64× H100 SXM ($32/hr each) = $2,048/hr
  Cost per 1M tokens: ~$0.47
```

---

## TP + PP Combined: The Production Pattern

For truly massive models, teams use TP and PP together:

```
  HYBRID PARALLELISM: 3D PARALLELISM
  ════════════════════════════════════

  TP within node (NVLink) + PP across nodes (InfiniBand)

  Node 0:  [GPU0 GPU1 GPU2 GPU3] ← TP=4, Layers 1-20
     ↕ InfiniBand
  Node 1:  [GPU4 GPU5 GPU6 GPU7] ← TP=4, Layers 21-40
     ↕ InfiniBand
  Node 2:  [GPU8 GPU9 GPU10 GPU11] ← TP=4, Layers 41-60
     ↕ InfiniBand
  Node 3:  [GPU12 GPU13 GPU14 GPU15] ← TP=4, Layers 61-80

  This is how GPT-4, Llama-3-405B, and Gemini Ultra are served.

  The third dimension (Data Parallelism) runs MULTIPLE COPIES
  of the entire TP+PP setup to scale request throughput.

  ┌────────────────────────────────────────────────────────┐
  │ Data Parallel = multiple independent model replicas   │
  │ Tensor Parallel = splits one layer across GPUs        │
  │ Pipeline Parallel = splits layers across GPU groups   │
  │                                                        │
  │ Production: DP × TP × PP configured per model/budget  │
  └────────────────────────────────────────────────────────┘
```

---

## vLLM Pipeline Parallelism Flag

```bash
# 8-GPU node: TP=4 within socket, PP=2 across sockets
vllm serve meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --dtype bfloat16

# For cross-node (Ray Serve + vLLM, Module 800):
# PP is configured via Ray's placement groups
# Each pipeline stage = one Ray actor on one node
```

---

## Choosing TP vs PP

```
  DECISION MATRIX
  ════════════════

  ┌─────────────────────────────┬──────────────┬───────────────┐
  │ Scenario                    │ Use          │ Why           │
  ├─────────────────────────────┼──────────────┼───────────────┤
  │ Model fits in 1 GPU         │ Neither      │ Run replicas  │
  │ Model needs 2-8 GPUs,       │ TP           │ NVLink fast,  │
  │   all on same node          │              │ no bubble     │
  │ Model needs 2+ nodes        │ PP (or TP+PP)│ IB handles PP │
  │ Low batch workload          │ TP           │ PP bubble bad │
  │   (<8 concurrent requests)  │              │ at low batch  │
  │ High batch workload         │ TP+PP        │ Bubble small  │
  │   (continuous batching)     │              │ at high batch │
  └─────────────────────────────┴──────────────┴───────────────┘

  Our workshop (g6e.12xlarge, 4× L40S):
  → Ministral-3-8B: TP=1, run 4 replicas (best throughput)
  → Hypothetical Llama-70B: TP=4, single replica
  → Hypothetical Llama-405B: Need TP=4 + PP=2+ across nodes
```

---

## Workshop Connection

```
  PIPELINE PARALLELISM IN OUR STACK
  ═══════════════════════════════════

  Module 800 (Ray Serve) is where PP becomes relevant.

  The RayService definition in 800-ray/ray-vllm-service.yaml
  uses Ray's actor placement to control which GPU group
  each vLLM instance runs on.

  For multi-node inference (Day 25), Ray handles the
  cross-node orchestration that makes PP across nodes work:

  RayCluster head node
       │
       ├── Worker node 0 (4× L40S) → vLLM replica 1
       ├── Worker node 1 (4× L40S) → vLLM replica 2
       └── Worker node 2 (4× L40S) → vLLM replica 3

  Each vLLM replica uses TP=4 within its node.
  Ray load-balances requests across replicas (data parallel).
  For a 70B model: would need TP=4+PP=2 across 2 nodes.
```

---

## Key Takeaway

> Pipeline Parallelism assigns different transformer layers to different GPUs — tokens flow through stages like an assembly line.  
> Unlike Tensor Parallelism, it works across nodes connected by InfiniBand.  
> The downside is pipeline bubbles — idle GPU time between micro-batches. Large batches minimize this.  
> Production at scale uses TP within a node + PP across nodes + data parallelism for replicas. That's 3D parallelism.

---

*30-Day Series: LLM Inference Is Everything | Day 23 of 30 — Week 4: Scaling Beyond One GPU*  
*← [Day 22](./day-22-tensor-parallelism.md) | [Day 24 →](./day-24-ray-serve-autoscaling.md) | [Back to Index](../README.md)*
