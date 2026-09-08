# Day 25 — Multi-node Inference

> **Hook:** One node maxes out at ~8 GPUs. What do you do when you need 32? You need distributed inference across multiple machines — and it's harder than it sounds.

---

## The Post

Single-node GPU servers have a ceiling.

The largest AWS EC2 instances give you 8× H100 or 8× L40S. That's ~640 GB VRAM on H100s. Impressive — but still not enough for frontier-scale models like Llama-3-405B (~810 GB), Falcon-180B, or future 1T+ parameter models.

Multi-node inference crosses the machine boundary. Now your model spans physical servers connected by a network. The challenges multiply: distributed coordination, fault tolerance, network latency, and scheduling complexity.

---

## Why Multi-node Is Fundamentally Different

```
  SINGLE-NODE vs MULTI-NODE INFERENCE
  ════════════════════════════════════

  SINGLE NODE:
  ┌──────────────────────────────────────┐
  │  Server (g6e.12xlarge)               │
  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
  │  │GPU 0  │ │GPU 1  │ │GPU 2  │ │GPU 3  │
  │  │L40S   │ │L40S   │ │L40S   │ │L40S   │
  │  │48 GB  │ │48 GB  │ │48 GB  │ │48 GB  │
  │  └───────┘ └───────┘ └───────┘ └───────┘
  │  Connected by: NVLink (600 GB/s)         │
  │  OS: shared memory, same process space   │
  └──────────────────────────────────────────┘

  MULTI-NODE:
  ┌────────────────────┐     NETWORK      ┌────────────────────┐
  │  Node 0            │   InfiniBand     │  Node 1            │
  │  ┌────┐ ┌────┐    │  ←─────────────▶ │  ┌────┐ ┌────┐    │
  │  │GPU0│ │GPU1│    │   50-400 GB/s    │  │GPU4│ │GPU5│    │
  │  └────┘ └────┘    │   (vs 600 NVLink)│  └────┘ └────┘    │
  │  ┌────┐ ┌────┐    │   10-100× slower │  ┌────┐ ┌────┐    │
  │  │GPU2│ │GPU3│    │                   │  │GPU6│ │GPU7│    │
  │  └────┘ └────┘    │                   │  └────┘ └────┘    │
  └────────────────────┘                   └────────────────────┘

  The gap: NVLink = 600 GB/s | InfiniBand 400G = 50 GB/s
  12× bandwidth drop when crossing a node boundary.
  This changes everything about how you partition the model.
```

---

## The Communication Stack for Multi-node

```
  WHAT TRAVELS BETWEEN NODES
  ════════════════════════════

  In Pipeline Parallelism (PP), nodes exchange activation tensors:
  ┌─────────────────────────────────────────────────────────┐
  │  Activation size: batch × seq_len × hidden_dim          │
  │  Llama-3-70B (hidden=8192), batch=8, seq=1024:          │
  │  8 × 1024 × 8192 × 2 bytes = 128 MB per stage boundary │
  │  At 50 GB/s: 2.56 ms per transfer — tolerable           │
  └─────────────────────────────────────────────────────────┘

  In Tensor Parallelism (TP) across nodes, every layer needs AllReduce:
  ┌─────────────────────────────────────────────────────────┐
  │  AllReduce size: batch × hidden_dim × 2 bytes            │
  │  Llama-3-70B, batch=8: 8 × 8192 × 2 = 128 KB           │
  │  At 50 GB/s: 0.0025 ms — very fast                      │
  │  BUT: 80 layers × 2 AllReduce each = 160 per token      │
  │  At 0.0025ms each: 0.4ms communication per token total  │
  │                                                          │
  │  Verdict: TP across nodes is viable with fast IB!        │
  └─────────────────────────────────────────────────────────┘

  RULE OF THUMB:
  ┌──────────────────────────────────────────────────────────┐
  │ Use PP to cross node boundaries (fewer, larger transfers) │
  │ Use TP within a node (many, smaller transfers over NVLink)│
  │ Combine: TP=8 per node, PP=N across N nodes              │
  └──────────────────────────────────────────────────────────┘
```

---

## AWS Infrastructure for Multi-node LLM Inference

```
  EKS MULTI-NODE INFERENCE ARCHITECTURE
  ══════════════════════════════════════

  VPC
  └── EKS Cluster
       │
       ├── Ray Head Pod (m5.xlarge)
       │   - Cluster coordinator
       │   - Autoscaler decisions
       │   - Serves Ray Dashboard
       │
       ├── GPU Worker Node 0 (p4d.24xlarge)
       │   - 8× A100 80 GB = 640 GB VRAM
       │   - EFA (Elastic Fabric Adapter) enabled
       │   - vLLM instance: TP=8, layers 1-40
       │
       ├── GPU Worker Node 1 (p4d.24xlarge)
       │   - 8× A100 80 GB = 640 GB VRAM
       │   - EFA enabled
       │   - vLLM instance: TP=8, layers 41-80
       │
       └── ...
       │
  ┌─────────────────────────────────────────────────────────┐
  │ EFA (Elastic Fabric Adapter):                            │
  │ → AWS-native high-speed fabric, ~400 Gbps                │
  │ → Latency: ~2 μs (comparable to datacenter InfiniBand)  │
  │ → Available on: p3dn, p4d, p4de, p5, Trn instances      │
  │ → Required for production multi-node LLM inference       │
  └─────────────────────────────────────────────────────────┘
```

---

## Instance Types for Multi-node on AWS

```
  AWS GPU INSTANCES FOR MULTI-NODE
  ════════════════════════════════

  ┌───────────────┬──────────────────┬─────────┬──────────────┐
  │ Instance      │ GPUs             │ VRAM    │ Interconnect │
  ├───────────────┼──────────────────┼─────────┼──────────────┤
  │ g6e.12xlarge  │ 4× L40S          │ 192 GB  │ PCIe         │
  │ g6e.48xlarge  │ 8× L40S          │ 384 GB  │ NVLink       │
  │ p4d.24xlarge  │ 8× A100 (40 GB)  │ 320 GB  │ NVLink + EFA │
  │ p4de.24xlarge │ 8× A100 (80 GB)  │ 640 GB  │ NVLink + EFA │
  │ p5.48xlarge   │ 8× H100 (80 GB)  │ 640 GB  │ NVLink + EFA │
  │ p5e.48xlarge  │ 8× H200 (141 GB) │ 1,128 GB│ NVLink + EFA │
  └───────────────┴──────────────────┴─────────┴──────────────┘

  For multi-node LLM inference:
  → p4d/p4de/p5 are the production choice (EFA + NVLink)
  → g6e clusters work but PCIe limits cross-node bandwidth
  → p5 cluster: 4 nodes = 32× H100 = 2,560 GB VRAM total
```

---

## Distributed Coordination: What Goes Wrong

```
  FAILURE MODES IN MULTI-NODE INFERENCE
  ══════════════════════════════════════

  1. NODE FAILURE DURING INFERENCE
  ────────────────────────────────
  PP: If any pipeline stage fails, the entire request fails.
  Recovery: Ray detects failure, recreates the actor, restores state.
  Downtime: seconds to minutes depending on checkpoint strategy.

  2. NETWORK PARTITION
  ────────────────────
  NCCL collective operations hang if a GPU can't communicate.
  Default NCCL timeout: 30 minutes(!) — silent hang is dangerous.

  Best practice:
  - Set NCCL_TIMEOUT=60  (fail fast)
  - Enable NCCL heartbeats
  - Use Ray's fault tolerance: actor restart on failure

  3. STRAGGLERS
  ─────────────
  In TP, the slowest GPU in each AllReduce determines speed.
  If one GPU is throttled (thermal, memory pressure) → all wait.
  Fix: NVIDIA MIG, node health checks via DCGM exporter

  4. KV CACHE STATE
  ─────────────────
  Each node only holds KV cache for ITS layers.
  Long conversations need coordinated eviction across all nodes.
  vLLM handles this within a job, but eviction logic is complex.
```

---

## Real Multi-node Benchmark: Llama-3-405B

```
  LLAMA-3-405B MULTI-NODE INFERENCE
  ═══════════════════════════════════

  Hardware: 4× p5.48xlarge (4 nodes × 8× H100 = 32 GPUs)
  Configuration: TP=8 per node, PP=4 across nodes
  Model precision: BF16

  ┌──────────────────────────────────────────────────────────┐
  │ Metric              │ Value                              │
  ├──────────────────────────────────────────────────────────┤
  │ Input: 512 tokens, Output: 256 tokens                    │
  │ Throughput (batch=8)│ ~380 tokens/sec output             │
  │ TTFT (P50)          │ ~2.1 seconds                       │
  │ TTFT (P95)          │ ~4.8 seconds                       │
  │ TPOT (P50)          │ ~85ms/token                        │
  │ GPU utilization     │ ~78% average                       │
  │ Interconnect util.  │ ~35% (EFA 400 Gbps)                │
  ├──────────────────────────────────────────────────────────┤
  │ Infrastructure cost │ 4 × $98.32/hr = $393/hr            │
  │ Cost per 1M tokens  │ ~$288 (output tokens only)         │
  └──────────────────────────────────────────────────────────┘

  Vs. GPT-4 API pricing (~$30/1M output tokens):
  Self-hosted 405B: more expensive at this scale.
  But: data privacy, no rate limits, no per-call cost.
```

---

## Multi-node with Ray Serve + vLLM (Our Workshop Stack)

```
  MULTI-NODE SETUP IN MODULE 800
  ═══════════════════════════════

  For Ministral-3-8B (fits in 1 GPU), multi-node = replicas:

  Ray Head (m5.xlarge) + 3 Worker Nodes (g6e.2xlarge each):

  ┌──────────────────────────────────────────────────────────┐
  │  Request                                                  │
  │     │                                                     │
  │     ▼                                                     │
  │  Ray Serve Head (load balancer)                          │
  │     │              │              │                       │
  │     ▼              ▼              ▼                       │
  │  Worker 0       Worker 1       Worker 2                  │
  │  vLLM+GPU       vLLM+GPU       vLLM+GPU                  │
  │  Ministral      Ministral      Ministral                  │
  │  (independent)  (independent)  (independent)             │
  └──────────────────────────────────────────────────────────┘

  This is DATA PARALLELISM: 3 independent replicas.
  NO shared state between workers.
  Each worker serves requests independently.
  3× throughput, 3× cost.

  For a 70B model that needs 4× L40S (TP=4):
  → Each worker = 1 g6e.12xlarge (4 GPUs, TP=4)
  → Multi-node means running multiple such workers
  → Ray handles routing; NCCL handles intra-worker TP
```

---

## When to Go Multi-node

```
  DECISION GUIDE
  ══════════════

  Model fits in 1 GPU?
  → Stay single node, run multiple replicas
  → Cheapest, simplest, most reliable

  Model needs multiple GPUs but fits in 1 node?
  → Use TP within the node (g6e.12xlarge, p4d, p5)
  → NVLink handles communication efficiently

  Model DOESN'T fit in 1 node's VRAM?
  → Multi-node is required
  → Use TP within node + PP across nodes
  → Need EFA/InfiniBand for reasonable performance

  Need high throughput AND a large model?
  → Multiple multi-node clusters (data parallel replicas)
  → Ray Serve load-balances across entire clusters

  Frontier/largest models (Llama-405B, future 1T+):
  → 8+ nodes, 64+ GPUs
  → Dedicated GPU cluster, not spot instances
  → Operations team needed for reliability
```

---

## Key Takeaway

> Multi-node inference is the only option when model weights exceed single-node VRAM.  
> The fundamental challenge is network bandwidth: crossing a machine boundary costs 10-100× more than NVLink.  
> The standard pattern: Tensor Parallelism within each node (NVLink), Pipeline Parallelism across nodes (InfiniBand/EFA).  
> For small models, "multi-node" just means independent replicas — simpler, cheaper, more reliable.

---

*30-Day Series: LLM Inference Is Everything | Day 25 of 30 — Week 4: Scaling Beyond One GPU*  
*← [Day 24](./day-24-ray-serve-autoscaling.md) | [Day 26 →](./day-26-kv-cache-offloading.md) | [Back to Index](../README.md)*
