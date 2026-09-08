# Day 8 — Why GPUs Are Critical for LLM Inference

> **Hook:** LLMs aren't just software — they are memory + compute problems.

---

## The Post

People think LLMs are a software problem.

They're not. They're a hardware problem first.

A 7B parameter model in FP16 = **14 GB of raw numbers** that must flow through silicon thousands of times per second. No software trick makes that cheap. You need hardware built for it.

That hardware is the GPU.

---

## CPU vs GPU — The Architecture Difference

```
  CPU (Central Processing Unit)
  ═════════════════════════════

  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  Core  Core  Core  Core  Core  Core  Core  Core       │
  │  ████  ████  ████  ████  ████  ████  ████  ████       │
  │                                                        │
  │  8-64 powerful cores                                   │
  │  Large cache (L1/L2/L3)                               │
  │  Optimized for: sequential logic, branching, I/O      │
  │  Clock speed: 3-5 GHz                                 │
  │  Memory bandwidth: ~50-100 GB/s (DDR5)                │
  │                                                        │
  │  Great at: web servers, databases, OS tasks            │
  │  Bad at: doing the same math op on 10,000 numbers     │
  │          simultaneously                               │
  └────────────────────────────────────────────────────────┘

  GPU (Graphics Processing Unit)
  ═══════════════════════════════

  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██     │
  │  ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██     │
  │  ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██     │
  │  ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██     │
  │  ... thousands of small CUDA cores ...                │
  │                                                        │
  │  NVIDIA L40S (our workshop): 18,176 CUDA cores        │
  │  Optimized for: parallel math on huge data arrays     │
  │  Memory bandwidth: ~864 GB/s (GDDR6)                  │
  │  VRAM: 48 GB dedicated on-die memory                  │
  │                                                        │
  │  Great at: matrix multiply on billions of weights     │
  │  Bad at: complex sequential business logic            │
  └────────────────────────────────────────────────────────┘
```

---

## Why LLM Inference Maps Perfectly to GPUs

```
  THE CORE OPERATION OF EVERY TRANSFORMER LAYER
  ══════════════════════════════════════════════

  Matrix Multiplication:  Y = X · W

  Where:
    X = token embeddings  [batch × seq_len × hidden_dim]
    W = weight matrix     [hidden_dim × hidden_dim]
    Y = output activations

  For Ministral-3-8B on a single forward pass:
  ┌──────────────────────────────────────────────────────┐
  │  Hidden dim: 4,096                                   │
  │  Layers: 32                                          │
  │  Per layer: 4096 × 4096 matrix multiply              │
  │  = ~16.7 million multiplications per layer           │
  │  × 32 layers = ~537 million multiplications          │
  │  × per output token × per request                   │
  │                                                      │
  │  This is ALL the GPU does. Over and over.            │
  │  Matrix multiply is exactly what GPUs are built for. │
  └──────────────────────────────────────────────────────┘

  CPU doing this: seconds per token
  GPU doing this: milliseconds per token
  Speedup: 100-1000×
```

---

## The Memory Bandwidth Problem

```
  WHY MEMORY BANDWIDTH MATTERS AS MUCH AS COMPUTE
  ═════════════════════════════════════════════════

  To do Y = X · W, the GPU must first LOAD the weights W from VRAM.

  Model weights must travel:

  VRAM ──────────────────────▶ GPU Cores
  (stored here)    memory bus    (compute here)

  Weight size:     ~7 GB (Ministral-3-8B, BF16)
  Must be loaded:  every forward pass, every decode step

  Memory bandwidth comparison:
  ┌────────────────────────────────────────┬───────────────┐
  │ Hardware                               │ Bandwidth     │
  ├────────────────────────────────────────┼───────────────┤
  │ CPU DDR5 RAM                           │   ~50 GB/s    │
  │ NVIDIA A10G (cloud GPU)                │  ~600 GB/s    │
  │ NVIDIA L40S (our workshop)             │  ~864 GB/s    │
  │ NVIDIA A100 80GB                       │ ~2,000 GB/s   │
  │ NVIDIA H100 80GB SXM                   │ ~3,350 GB/s   │
  └────────────────────────────────────────┴───────────────┘

  This is why the same model runs 4× faster on H100 vs A10G
  — not because of FLOPS, but because of memory bandwidth
  loading weights faster during the decode phase.
```

---

## GPU Hierarchy in Our Workshop

```
  AWS g6e INSTANCES — WHAT WE USED
  ══════════════════════════════════

  g6e.2xlarge  ──▶  1× NVIDIA L40S GPU
  g6e.12xlarge ──▶  4× NVIDIA L40S GPU (tensor parallel)

  NVIDIA L40S Spec Sheet:
  ┌─────────────────────────────────────────────────────────┐
  │  VRAM:          48 GB GDDR6                             │
  │  Memory BW:     864 GB/s                               │
  │  FP32 FLOPS:    91.6 TFLOPS                            │
  │  FP16 FLOPS:    183 TFLOPS                             │
  │  INT8 TOPS:     362 TOPS                               │
  │  CUDA cores:    18,176                                 │
  │  TDP:           350W                                   │
  │                                                         │
  │  Why L40S for inference?                               │
  │  ✓ 48 GB VRAM fits 7-13B models comfortably           │
  │  ✓ 864 GB/s memory bandwidth fast decode              │
  │  ✓ FP8 support for quantized inference                │
  │  ✓ NVLink for multi-GPU tensor parallelism            │
  └─────────────────────────────────────────────────────────┘

  GPU selection flowchart:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  Model fits in VRAM?                               │
  │       │                                            │
  │    YES │                          NO               │
  │       ▼                           ▼               │
  │  Use 1 GPU             Multi-GPU tensor parallel  │
  │  Optimize batch        (Module 800: Ray Serve)    │
  │  for throughput        or use quantization        │
  │                        to reduce model size       │
  └────────────────────────────────────────────────────┘
```

---

## What Happens Without a GPU

```
  RUNNING Ministral-3-8B ON CPU (realistic estimate)
  ════════════════════════════════════════════════════

  CPU memory bandwidth: ~50 GB/s
  Model weights: 7 GB (BF16)

  Time to load weights per forward pass:
  7 GB / 50 GB/s = ~140ms per decode step

  At 140ms per token:
  → ~7 tokens/second (just barely readable)
  → 1 request at a time (no concurrency)
  → No KV cache efficiency
  → CPU at 100% utilization

  Same model on L40S GPU:
  7 GB / 864 GB/s = ~8ms per decode step
  → ~125 tokens/second
  → Multiple concurrent requests
  → 17× faster than CPU

  This is why GPU is not optional for production LLM serving.
```

---

## Workshop Connection

```
  MODULE DEPENDENCY ON GPU
  ═════════════════════════

  Every module in our workshop required GPU:

  GPU Module  ──▶  Karpenter auto-provisions g6e instances
                   DCGM Exporter collects GPU metrics
                   Prometheus + Grafana visualize GPU util

  100-vllm    ──▶  vLLM uses CUDA for all inference ops
                   PagedAttention manages GPU VRAM

  300-bench   ──▶  GPU utilization is a primary metric
                   DCGM shows SM utilization, memory util

  400-lmcache ──▶  KV cache lives in GPU VRAM (L1)
                   Overflow to CPU RAM (L2) or Valkey (L3)

  800-ray     ──▶  Ray Serve manages GPU resources across pods
                   Tensor parallelism across multiple L40S GPUs
```

---

## Key Takeaway

> GPUs aren't just "faster CPUs."  
> They are purpose-built parallel math engines with dedicated high-bandwidth memory.  
> LLM inference is nothing but parallel math on large memory arrays.  
> The match is perfect — and the performance difference is 10-1000×.

---

*30-Day Series: LLM Inference Is Everything | Day 8 of 30 — Week 2 Begins*
*← [Day 7](../week-1/day-07-why-llm-feels-slow.md) | Next → [Day 9](./day-09-gpu-memory-bottleneck.md)*
