# Day 22 — Tensor Parallelism

> **Hook:** A 70B model needs 140 GB of VRAM. The biggest single GPU has 80 GB. So how does anyone actually run it?

---

## The Post

You've seen the headlines:
- "We deployed Llama-3-70B in production"
- "Running Mixtral-8×7B at 2,000 tokens/sec"

But wait — a 70B model in BF16 is ~140 GB. Even an A100 80 GB can't fit that. So how?

**Tensor Parallelism.** The model doesn't live on one GPU. It's sliced across all of them.

---

## Why One GPU Isn't Enough for Large Models

```
  THE VRAM ARITHMETIC PROBLEM
  ════════════════════════════

  Model weights memory (BF16, 2 bytes/param):
  ┌────────────────────────────┬──────────────┬───────────────────┐
  │ Model                      │ Params       │ VRAM for weights  │
  ├────────────────────────────┼──────────────┼───────────────────┤
  │ Ministral-3-8B             │ 3.8B         │ ~7.5 GB           │
  │ Llama-3-8B                 │ 8B           │ ~16 GB            │
  │ Llama-3-70B                │ 70B          │ ~140 GB           │
  │ Llama-3-405B               │ 405B         │ ~810 GB           │
  │ GPT-4 (estimated)          │ ~1.76T (MoE) │ ~3,500 GB         │
  └────────────────────────────┴──────────────┴───────────────────┘

  GPU VRAM limits:
  ┌────────────────────────────┬──────────────┐
  │ NVIDIA L40S                │ 48 GB        │
  │ NVIDIA A100                │ 80 GB        │
  │ NVIDIA H100                │ 80 GB (SXM)  │
  │ NVIDIA H200                │ 141 GB       │
  └────────────────────────────┴──────────────┘

  Ministral-3-8B → fits comfortably on 1× L40S ✓
  Llama-3-70B    → does NOT fit on 1× L40S or A100 ✗
  Llama-3-405B   → needs 8+ H200s ✗
```

---

## What Tensor Parallelism Actually Does

Tensor Parallelism (TP) **splits the model's weight matrices** across multiple GPUs.

Every transformer layer has large linear operations (QKV projections, MLP layers). These matrices are sliced column-wise or row-wise across GPUs, so each GPU holds and computes only its portion.

```
  TENSOR PARALLELISM: SPLITTING WEIGHT MATRICES
  ═══════════════════════════════════════════════

  Without Tensor Parallelism (1 GPU):
  ┌─────────────────────────────────────────────────────────┐
  │ GPU 0                                                    │
  │  Layer 1: Attention [Q: 4096×128, K: 4096×128,         │
  │                      V: 4096×128, O: 4096×512]         │
  │  Layer 1: MLP [Gate: 4096×14336, Up: 4096×14336,       │
  │                Down: 14336×4096]                        │
  │  ... × 32 layers = 140 GB total                        │
  │  → DOESN'T FIT on a single 80 GB GPU                  │
  └─────────────────────────────────────────────────────────┘

  With Tensor Parallelism (4 GPUs, TP=4):
  ┌─────────────────────┐  ┌─────────────────────┐
  │ GPU 0               │  │ GPU 1               │
  │ Q[:, 0:32],         │  │ Q[:, 32:64],        │
  │ K[:, 0:32],         │  │ K[:, 32:64],        │
  │ MLP[0:3584, :]      │  │ MLP[3584:7168, :]   │
  │ ~35 GB              │  │ ~35 GB              │
  └─────────────────────┘  └─────────────────────┘
  ┌─────────────────────┐  ┌─────────────────────┐
  │ GPU 2               │  │ GPU 3               │
  │ Q[:, 64:96],        │  │ Q[:, 96:128],       │
  │ K[:, 64:96],        │  │ K[:, 96:128],       │
  │ MLP[7168:10752, :]  │  │ MLP[10752:14336, :] │
  │ ~35 GB              │  │ ~35 GB              │
  └─────────────────────┘  └─────────────────────┘

  Total: 4 × 35 GB = 140 GB — problem solved!
  Each GPU computes its portion, then results are AllReduced.
```

---

## The Communication Cost: AllReduce

After each tensor-parallel layer, the GPUs must **combine their partial results**. This is an AllReduce operation — every GPU sends its output to all others and sums them.

```
  ALLREDUCE COMMUNICATION PATTERN
  ════════════════════════════════

  After each attention/MLP layer:

  GPU 0: partial output [0.3, 0.7, ...]
  GPU 1: partial output [0.1, 0.2, ...]
  GPU 2: partial output [0.4, 0.5, ...]
  GPU 3: partial output [0.2, 0.1, ...]
          │
          │  NVLink (600 GB/s) or PCIe (64 GB/s)
          ▼
  All GPUs get: [1.0, 1.5, ...] (sum of all partials)
          │
          ▼
  Continue to next layer with full result

  KEY INSIGHT: AllReduce happens EVERY LAYER
  Llama-3-70B has 80 layers → 80 AllReduce ops per token
  This is why NVLink bandwidth matters enormously for TP.
```

---

## NVLink vs PCIe — Why Interconnect Decides Throughput

```
  INTERCONNECT BANDWIDTH COMPARISON
  ═══════════════════════════════════

  ┌─────────────────┬───────────────┬──────────────────────────┐
  │ Connection      │ Bandwidth     │ Latency                  │
  ├─────────────────┼───────────────┼──────────────────────────┤
  │ NVLink 4.0      │ 900 GB/s      │ ~1 μs                    │
  │ NVLink 3.0      │ 600 GB/s      │ ~1 μs                    │
  │ PCIe 5.0        │ 128 GB/s      │ ~10 μs                   │
  │ PCIe 4.0        │ 64 GB/s       │ ~10 μs                   │
  │ Infiniband 400G │ 50 GB/s       │ ~2 μs                    │
  └─────────────────┴───────────────┴──────────────────────────┘

  IMPACT ON THROUGHPUT (Llama-3-70B, TP=4, 512 tok prefill):

  With NVLink:
    AllReduce time per layer: ~0.5ms
    Total communication overhead: 80 × 0.5ms = 40ms
    Total prefill time: ~120ms → 40ms comms (25% overhead)

  With PCIe:
    AllReduce time per layer: ~4ms
    Total communication overhead: 80 × 4ms = 320ms
    Total prefill time: ~400ms → 320ms comms (80% overhead!)

  Rule of thumb: Tensor Parallelism only works efficiently
  on GPUs WITHIN THE SAME MACHINE with NVLink.
  Cross-machine TP = communication bottleneck.
```

---

## Tensor Parallelism in vLLM — It's One Flag

```bash
# Run Llama-3-70B across 4 GPUs on one g6e.12xlarge node
# g6e.12xlarge = 4× NVIDIA L40S (4× 48 GB = 192 GB total)

vllm serve meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

```
  WHAT --tensor-parallel-size 4 DOES
  ═════════════════════════════════════

  vLLM startup sequence:
  1. Detects 4× L40S GPUs
  2. Loads model weights from S3 / local cache
  3. Shards weight matrices across 4 GPUs automatically
  4. Initializes NCCL communication group for AllReduce
  5. Starts serving — all 4 GPUs process EVERY request together

  Memory usage per GPU (Llama-3-70B, TP=4):
  ┌──────────────────────────────────────────────────┐
  │ Model weights:   140 GB ÷ 4 = 35 GB per GPU     │
  │ KV cache:        ~8 GB per GPU                   │
  │ Activations:     ~2 GB per GPU                   │
  │ Total:           ~45 GB → fits in 48 GB L40S!   │
  └──────────────────────────────────────────────────┘
```

---

## TP Scaling Efficiency — Not Linear

```
  THROUGHPUT vs GPU COUNT (Llama-3-70B, 512→256 tokens)
  ═══════════════════════════════════════════════════════

  ┌───────────┬──────────────┬──────────────┬──────────────┐
  │ TP Size   │ GPUs         │ Throughput   │ Scaling Eff. │
  ├───────────┼──────────────┼──────────────┼──────────────┤
  │ TP=1      │ 1× H100      │ ~400 tok/s   │ 100% (base)  │
  │ TP=2      │ 2× H100      │ ~720 tok/s   │ 90%          │
  │ TP=4      │ 4× H100      │ ~1,300 tok/s │ 81%          │
  │ TP=8      │ 8× H100      │ ~2,100 tok/s │ 66%          │
  └───────────┴──────────────┴──────────────┴──────────────┘

  Efficiency drops because:
  - AllReduce overhead grows with each GPU added
  - Memory bandwidth advantage saturates

  But: without TP, the 70B model CAN'T RUN AT ALL.
  TP isn't primarily a speed optimization — it's an enabler.
```

---

## Workshop Connection — Module 800 Ray Serve

```
  HOW THIS WORKSHOP USES TENSOR PARALLELISM
  ═══════════════════════════════════════════

  Our g6e.12xlarge node: 4× NVIDIA L40S (4 × 48 GB)

  In Module 800 (Ray Serve), the RayService CRD configures
  vLLM with tensor parallelism via environment variables:

  ray-vllm-service.yaml (800-ray/):
  ─────────────────────────────────
  env:
  - name: TENSOR_PARALLEL_SIZE
    value: "4"

  This means a single RayService request:
  → Received by Ray Serve head
  → Dispatched to vLLM worker
  → That worker uses all 4 L40S GPUs via TP
  → AllReduce happens over NVLink inside the g6e.12xlarge
  → Single response returned

  For Ministral-3-8B (our 3.8B model), TP=1 is sufficient.
  The 4-GPU g6e.12xlarge gives us REPLICA scaling instead:
  4 independent vLLM replicas × 1 GPU each = 4× throughput.
```

---

## When to Use Tensor Parallelism

```
  DECISION GUIDE
  ══════════════

  Model fits in 1 GPU VRAM?
  ┌───────────────────────────────────────────┐
  │ YES → Use single-GPU. Scale replicas.     │
  │       Tensor parallelism adds overhead    │
  │       with no benefit for small models.   │
  │                                           │
  │ NO  → Use Tensor Parallelism.            │
  │       TP=2 for 2× VRAM budget             │
  │       TP=4 for 4× VRAM budget             │
  │       Must stay within 1 machine (NVLink) │
  └───────────────────────────────────────────┘

  Model STILL doesn't fit in 1 node even with TP?
  → Pipeline Parallelism (Day 23 tomorrow)
  → Or buy an H200 cluster

  Rule: Use the minimum TP needed to fit the model.
  Less TP = less communication = more efficient.

  Our workshop:
  Ministral-3-8B @ TP=1 → replicas = right choice
  Llama-3-70B @ TP=4    → required for 4× L40S node
```

---

## Key Takeaway

> Tensor Parallelism solves the "model too big for one GPU" problem by sharding weight matrices across GPUs.  
> Every GPU processes every request simultaneously, then combines results via AllReduce.  
> It only works efficiently when GPUs are connected by NVLink — PCIe kills the benefit.  
> Use minimum TP needed to fit the model. For small models, replica scaling beats TP.

---

*30-Day Series: LLM Inference Is Everything | Day 22 of 30 — Week 4: Scaling Beyond One GPU*  
*← [Day 21](../week-3/day-21-inference-throughput.md) | [Day 23 →](./day-23-pipeline-parallelism.md) | [Back to Index](../README.md)*
