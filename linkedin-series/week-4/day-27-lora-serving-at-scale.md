# Day 27 — Serving LoRA Adapters at Scale

> **Hook:** You fine-tuned 10 different LoRA adapters for 10 different customers. Do you need 10 separate GPU servers? No — vLLM can serve all of them from one.

---

## The Post

Fine-tuning is powerful. Train a small LoRA adapter on customer-specific data and your 3.8B model behaves like a domain expert for that customer.

But fine-tuning creates a new operational problem: **adapter proliferation**.

You have one base model but 50 enterprise customers, each with their own LoRA adapter. In the naive approach, you need 50 separate GPU servers — one per adapter. That's 50× the cost, 50× the ops burden, and 50× the GPU fleet to manage.

**vLLM's multi-LoRA serving** eliminates this completely.

---

## Quick Recap: What LoRA Is

```
  LORA IN ONE DIAGRAM
  ════════════════════

  Base model (Ministral-3-8B, 3.8B params, frozen):
  ┌──────────────────────────────────────────────────────┐
  │  Attention layer weights W (4096 × 4096)             │
  │  These NEVER change                                  │
  └──────────────────────────────────────────────────────┘
              +
  LoRA adapter (tiny, trainable):
  ┌──────────────────────────────────────────────────────┐
  │  Matrix A (4096 × r)  →  r=16 typically             │
  │  Matrix B (r × 4096)                                 │
  │  ΔW = A × B  (rank-16 decomposition of the update)  │
  │  Trainable params: 2 × 4096 × 16 = 131,072          │
  │  File size: ~50-100 MB per adapter                  │
  └──────────────────────────────────────────────────────┘

  During inference:
  output = W × input + (A × B) × input × scaling_factor

  One GPU load:
  - Base model W: ~7.5 GB (frozen, shared)
  - Adapter A+B: ~50-100 MB (swapped per request)
  - Multiple adapters can be loaded simultaneously!
```

---

## The Multi-LoRA Problem (Without vLLM)

```
  NAIVE APPROACH: ONE SERVER PER ADAPTER
  ═══════════════════════════════════════

  Customer A: LoRA adapter (startup advisor)
  Customer B: LoRA adapter (legal document)
  Customer C: LoRA adapter (medical Q&A)
  Customer D: LoRA adapter (e-commerce support)
  ...
  Customer Z: LoRA adapter (custom domain)

  ┌──────────────────────────────────────────────────────┐
  │  Naive setup:                                        │
  │  26 customers × 1 GPU server = 26× g6e.2xlarge      │
  │  Cost: 26 × $1.65/hr = $43/hr = $31,000/month       │
  │                                                      │
  │  GPU utilization per server: ~5-20% (most are idle) │
  │  Total wasted GPU capacity: ~80%                    │
  └──────────────────────────────────────────────────────┘

  WITH vLLM multi-LoRA:
  ┌──────────────────────────────────────────────────────┐
  │  1 GPU server, all 26 adapters loaded simultaneously │
  │  Cost: 1-2 × $1.65/hr = $1.65–$3.30/hr              │
  │  GPU utilization: 70-85%                            │
  │  Savings: 90-95% vs naive approach                  │
  └──────────────────────────────────────────────────────┘
```

---

## How vLLM Multi-LoRA Works

```
  VLLM MULTI-LORA ARCHITECTURE
  ══════════════════════════════

  GPU VRAM Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  Base model weights (7.5 GB) — ALWAYS IN VRAM           │
  │  ┌────────────────────────────────────────────────────┐ │
  │  │  LoRA adapter pool (max_loras × adapter_size)      │ │
  │  │  max_loras=8, each ~100 MB → 800 MB reserved       │ │
  │  │                                                     │ │
  │  │  Slot 0: customer_A_adapter (active)               │ │
  │  │  Slot 1: customer_B_adapter (active)               │ │
  │  │  Slot 2: customer_C_adapter (active)               │ │
  │  │  Slot 3: customer_D_adapter (active)               │ │
  │  │  Slot 4: [empty]                                   │ │
  │  │  Slot 5: [empty]                                   │ │
  │  │  Slot 6: [empty]                                   │ │
  │  │  Slot 7: [empty]                                   │ │
  │  └────────────────────────────────────────────────────┘ │
  │  KV cache: remaining VRAM (~37 GB)                       │
  └──────────────────────────────────────────────────────────┘

  Request from customer_E (adapter not in pool):
  1. Evict least-recently-used adapter from pool (LRU)
  2. Load customer_E_adapter from S3 (~100ms cold load)
  3. Process request with customer_E adapter
  4. Keep adapter hot for subsequent customer_E requests
```

---

## Enabling Multi-LoRA in vLLM

```yaml
# From 600-finetuning/vllm-with-lora.yaml in this workshop
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-lora
spec:
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.21.0
        args:
        - --model
        - /model-storage/Ministral-3-8B-Instruct-2512
        - --enable-lora                    # Enable LoRA support
        - --max-loras                      # How many adapters in VRAM at once
        - "4"
        - --max-lora-rank                  # Max rank any adapter can have
        - "64"
        - --lora-modules                   # Pre-load named adapters
        - "startup-advisor=/adapters/startup-advisor-lora"
        - "legal-assistant=/adapters/legal-assistant-lora"
```

```bash
# Serve request with specific adapter via model parameter
curl http://vllm-service/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "startup-advisor",   # ← selects the adapter
    "messages": [{"role": "user", "content": "How should I price my SaaS?"}]
  }'
```

---

## Adapter Loading Performance

```
  LORA ADAPTER LOAD TIMES
  ═════════════════════════

  Adapter size: ~100 MB (rank-16, all attention layers)

  Load source          │ Time
  ─────────────────────┼──────────────────
  Already in VRAM      │ 0ms (instant)
  CPU RAM cache        │ ~8ms (PCIe transfer)
  Local NVMe SSD       │ ~30ms
  Amazon S3 (region)   │ ~80-200ms
  Amazon S3 (cold)     │ ~300-500ms

  For SLA-sensitive workloads:
  → Pre-warm popular adapters into VRAM at startup
  → Use --lora-modules to pre-load common adapters
  → Store adapters on node-local SSD for <30ms load
  → S3 only for adapter library, pull to local first

  With 8 adapters in VRAM, handling 100 different customers:
  → Top 8 adapters always hot: 0ms latency
  → LRU eviction kicks adapter 8 when #9 requested
  → Cold load from S3 for rarely-used adapters: 200ms
```

---

## Batching Across Adapters: The Hard Part

```
  THE BATCHING CHALLENGE WITH MULTIPLE ADAPTERS
  ══════════════════════════════════════════════

  vLLM continuous batching (Day 15) batches requests together.
  But each request may need a DIFFERENT adapter.

  Batch of 8 requests:
  Req 1: adapter=startup-advisor
  Req 2: adapter=legal-assistant
  Req 3: adapter=startup-advisor
  Req 4: adapter=medical-qa
  Req 5: adapter=startup-advisor
  Req 6: adapter=legal-assistant
  Req 7: adapter=startup-advisor
  Req 8: adapter=e-commerce-support

  Per-layer computation:
  For each transformer layer, vLLM must apply DIFFERENT
  adapter deltas to different requests in the same batch.

  vLLM's solution: "punica" CUDA kernel
  ┌────────────────────────────────────────────────────────┐
  │ Process base model computation for full batch (dense) │
  │ Then apply adapter-specific ΔW per request (sparse)  │
  │ Using batched matrix multiply with index selection   │
  │                                                        │
  │ Overhead vs single-adapter: ~10-15% throughput drop  │
  │ Savings vs 8 separate servers: 87%+ cost reduction   │
  └────────────────────────────────────────────────────────┘
```

---

## Multi-LoRA at Scale: S3 as Adapter Registry

```
  ADAPTER STORAGE PATTERN IN THIS WORKSHOP
  ═════════════════════════════════════════

  Amazon S3 (s3://genai-models-{ACCOUNT_ID}/)
  ├── Ministral-3-8B-Instruct-2512/      ← base model
  │   └── consolidated.safetensors
  └── lora-adapters/                      ← adapter registry
      ├── startup-advisor-lora/
      │   ├── adapter_config.json
      │   └── adapter_model.safetensors   (~65 MB)
      ├── legal-assistant-lora/
      │   └── ...
      └── domain-expert-v2-lora/
          └── ...

  vLLM loading pattern:
  1. On startup: pull --lora-modules adapters from S3 to node
  2. On request for unknown adapter: pull on-demand from S3
  3. LRU eviction from VRAM when pool full
  4. Node-local cache prevents repeated S3 pulls

  Adding a new adapter = upload to S3, hot-load via API:
  curl -X POST http://vllm/v1/load_lora_adapter \
    -d '{"lora_name": "new-customer", "lora_path": "s3://..."}'
  # No restart required. Zero downtime adapter addition.
```

---

## The Economics: Multi-LoRA at Production Scale

```
  COST COMPARISON: DEDICATED vs MULTI-LORA
  ═══════════════════════════════════════════

  Scenario: 50 enterprise customers, each with 1 LoRA adapter
  Avg traffic: 10 requests/sec per customer = 500 req/s total
  Each request: 512 input + 256 output tokens
  Hardware: g6e.2xlarge ($1.65/hr per GPU)

  DEDICATED SERVERS (1 GPU per customer):
  ┌──────────────────────────────────────────────────────────┐
  │  50 × $1.65/hr × 720 hr/month = $59,400/month           │
  │  Average GPU utilization: ~8% (most requests are idle)  │
  │  Wasted GPU value: ~$54,600/month                       │
  └──────────────────────────────────────────────────────────┘

  MULTI-LORA FLEET (autoscaled via Ray Serve):
  ┌──────────────────────────────────────────────────────────┐
  │  500 req/s ÷ 50 req/s/GPU = 10 GPUs needed at peak     │
  │  With autoscaling: avg 6 GPUs during business hours     │
  │  6 × $1.65 × 720 = $7,128/month                        │
  │  GPU utilization: ~75%                                  │
  └──────────────────────────────────────────────────────────┘

  Savings: $52,272/month (88% reduction)
  This is the multi-tenancy argument for multi-LoRA serving.
```

---

## Workshop Connection

```
  MODULE 600 FINE-TUNING → SERVING
  ════════════════════════════════

  The pipeline in this workshop:

  Step 1: Fine-tune (600-finetuning/)
    train_lora.py:
    → Trains LoRA adapter on anyvc-startup-dataset.jsonl
    → Output: adapter_model.safetensors + adapter_config.json
    → Pushed to S3

  Step 2: Serve (600-finetuning/vllm-with-lora.yaml)
    → vLLM deployment with --enable-lora
    → Adapter loaded from S3 at startup
    → Accessible via model="anyvc-startup-advisor"

  Step 3: Scale (800-ray/ integration)
    → Ray Serve wraps the multi-LoRA vLLM
    → Multiple replicas, each with same adapter pool
    → Autoscales based on traffic

  Test it:
  curl http://vllm-service/v1/chat/completions \
    -d '{"model": "anyvc-startup-advisor",
         "messages": [{"role": "user",
           "content": "We have 50 users. Should we raise a Seed?"}]}'
```

---

## Key Takeaway

> vLLM's multi-LoRA serving loads multiple fine-tuned adapters onto a single GPU — one base model serving many customers.  
> The adapter pool in VRAM holds the most-recently-used adapters; LRU eviction handles overflow.  
> Multi-LoRA turns what would be 50 dedicated GPU servers into a single autoscaled fleet.  
> Economics: 88% cost reduction for a 50-customer deployment. Fine-tuning as a feature, not a cost center.

---

*30-Day Series: LLM Inference Is Everything | Day 27 of 30 — Week 4: Scaling Beyond One GPU*  
*← [Day 26](./day-26-kv-cache-offloading.md) | [Day 28 →](./day-28-rag-vs-finetuning.md) | [Back to Index](../README.md)*
