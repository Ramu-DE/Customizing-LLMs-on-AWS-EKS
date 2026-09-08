# GPU Infrastructure – The Hardware Foundation for AI Inference

> New to AI inference? Read [CONCEPTS.md](../CONCEPTS.md) first for plain-English definitions of every term used here.

---

## What This Module Is About

Before any AI inference can happen, you need hardware that is physically capable of running a large language model (LLM) at speed. This module sets up that hardware on AWS EKS.

**The core idea from the Red Hat article:**
> "For inference to be successful, AI models need to do a lot of math in a short period of time. The hardware and software that support your inference capabilities can make or break your AI strategy."

This module is the answer to that requirement: a Kubernetes cluster on AWS with on-demand GPU nodes, a model loaded from S3, and a monitoring stack to watch it all.

---

## Why GPUs for AI Inference?

A language model doing inference is essentially performing enormous matrix multiplications, billions of times per second. Here is why GPUs are uniquely suited for this:

```
CPU (Central Processing Unit)
────────────────────────────
Design goal: Execute one complex task very fast (sequential)
Cores: 8–64 large, sophisticated cores
AI math speed: ~100–500 GFLOPS
Good for: Operating systems, databases, logic

GPU (Graphics Processing Unit)
────────────────────────────────
Design goal: Execute thousands of simple tasks at once (parallel)
Cores: 18,176 smaller CUDA cores (NVIDIA L40S)
AI math speed: 733 TOPS (INT8) – roughly 1,000× faster for AI math
Good for: Matrix multiplication → the core of neural networks

Why does this matter for inference?
When Ministral-3-8B generates one token, it performs roughly
3.8 billion multiply-add operations. At 100 GFLOPS (CPU) that
would take ~38 milliseconds per token. At 733 TOPS (GPU) it
takes ~0.005 milliseconds. The GPU makes real-time chat possible.
```

---

## NVIDIA L40S – Hardware Deep Dive

This workshop uses **g6e.2xlarge** instances (single L40S) and optionally **g6e.12xlarge** (4× L40S).

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NVIDIA L40S GPU Specifications                   │
├───────────────────────────────┬─────────────────────────────────────┤
│  VRAM (GPU Memory)            │  48 GB GDDR6                        │
│  CUDA Cores                   │  18,176                             │
│  Tensor Cores (4th gen)       │  568                                │
│  FP32 Performance             │  91.6 TFLOPS                        │
│  TF32 Performance             │  362.1 TFLOPS                       │
│  FP16/BF16 Performance        │  362.1 TFLOPS                       │
│  INT8 Performance             │  733.6 TOPS                         │
│  FP8 Performance              │  733.6 TOPS                         │
│  Memory Bandwidth             │  864 GB/s                           │
│  NVLink                       │  No (PCIe gen 4 only)               │
│  TDP (Power Draw)             │  300W                               │
└───────────────────────────────┴─────────────────────────────────────┘

How the 48 GB VRAM is used during Ministral-3-8B inference:
┌────────────────────────────────────────────────────────────────────┐
│  48 GB VRAM                                                         │
│  ┌────────────────────┐  7.5 GB  Model weights (BFloat16)          │
│  │  Model Weights     │         3.8B params × 2 bytes              │
│  ├────────────────────┤         ────────────────────────────────── │
│  │  KV Cache          │  40 GB  PagedAttention blocks              │
│  │  (PagedAttention)  │         Stores intermediate attention math  │
│  │                    │         for up to 256 concurrent requests   │
│  ├────────────────────┤         ────────────────────────────────── │
│  │  Activations /     │  0.5 GB Runtime buffers                    │
│  │  Runtime Buffers   │                                             │
│  └────────────────────┘                                             │
│  gpu_memory_utilization=0.90 → vLLM uses 43 GB, reserves 5 GB      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Full Infrastructure Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AWS ACCOUNT                                      │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         Amazon EKS Cluster                              │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────────┐  ┌────────────────────────────────┐  │  │
│  │  │      System Node Pool        │  │    GPU Node Pool (Karpenter)   │  │  │
│  │  │  Instance: m5.xlarge (CPU)   │  │  Instance: g6e.2xlarge         │  │  │
│  │  │                              │  │           g6e.12xlarge         │  │  │
│  │  │  Runs CPU-only workloads:    │  │  GPU: NVIDIA L40S (48 GB VRAM) │  │  │
│  │  │  - Open WebUI (chat UI)      │  │                                │  │  │
│  │  │  - Grafana (dashboards)      │  │  Runs GPU workloads:           │  │  │
│  │  │  - RAG Gradio UI             │  │  - vLLM inference server       │  │  │
│  │  │  - Strands Agent (FastAPI)   │  │  - LMCache (KV offloading)     │  │  │
│  │  │                              │  │  - LoRA fine-tuning jobs       │  │  │
│  │  │  No GPU taint                │  │  - Ray worker nodes            │  │  │
│  │  │                              │  │                                │  │  │
│  │  │                              │  │  Taint: nvidia.com/gpu:        │  │  │
│  │  │                              │  │  NoSchedule                    │  │  │
│  │  │                              │  │  (CPU pods cannot land here)   │  │  │
│  │  └──────────────────────────────┘  └────────────────────────────────┘  │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │                   Monitoring Stack                                │  │  │
│  │  │  Prometheus ◀── DCGM Exporter (GPU metrics) ◀── L40S hardware   │  │  │
│  │  │  Prometheus ◀── vLLM /metrics (inference metrics)               │  │  │
│  │  │  Prometheus ──▶ Grafana (dashboards, public ALB)                 │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │         EKS Pod Identity (IAM access without secrets)            │  │  │
│  │  │  ServiceAccount: model-storage-sa ──▶ IAM Role ──▶ S3 r/w       │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌──────────────────────────┐    ┌──────────────────────────────────────┐   │
│  │      Amazon S3           │    │   AWS Application Load Balancer      │   │
│  │  genai-models-<ACCT_ID>  │    │   - open-webui-ingress  (chat)       │   │
│  │  ├── Ministral-3-8B/     │    │   - grafana-ingress     (metrics)    │   │
│  │  ├── anyvc-startup-lora/ │    │   - rag-gradio-alb      (RAG UI)     │   │
│  │  └── benchmarks/         │    │   Scheme: internet-facing            │   │
│  └──────────────────────────┘    └──────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Karpenter: On-Demand GPU Provisioning

One of the biggest cost challenges in AI inference (as the Red Hat article notes) is that GPUs are expensive. Running a GPU node 24/7 even when no inference is happening wastes money.

**Karpenter** solves this by provisioning GPU nodes only when a pod needs one, and terminating them when they are idle.

```
How Karpenter works in this workshop:
──────────────────────────────────────

1. kubectl apply -f vllm-deployment.yml
   → Pod is created, requests nvidia.com/gpu: 1
   → No GPU node exists yet → pod status: Pending

2. Karpenter sees the pending pod
   → Picks the cheapest instance type that satisfies the request
   → Launches a g6e.2xlarge EC2 instance
   → Node joins the EKS cluster in ~90 seconds

3. Pod is scheduled onto the new GPU node
   → vLLM starts, loads model from S3 (~2–3 min)
   → Pod is Ready, inference begins

4. Pod is deleted (workshop done)
   → Karpenter detects idle node
   → consolidateAfter: 30s → terminates the EC2 instance
   → Cost stops immediately

NodePool configuration:
  instance-family: g6e
  instance-sizes: [2xlarge, 12xlarge]
  capacity-type: on-demand (not spot, for stability)
  taint: nvidia.com/gpu:NoSchedule
  consolidateAfter: 30s
  expireAfter: 720h (30 days max lifetime)
```

---

## Model Storage in Amazon S3

The model is NOT baked into the container image (which would make images 10+ GB). Instead, model weights live in S3 and are streamed at pod startup.

```
s3://genai-models-<ACCOUNT_ID>/Ministral-3-8B-Instruct-2512/

File                              Size    Purpose
──────────────────────────────────────────────────────────────────
consolidated.safetensors         10.4 GB  ALL model weights in one file
                                          SafeTensors = safe, fast, zero-copy mmap
                                          No pickle vulnerability unlike .pt files

config.json                       1.9 KB  Model architecture definition
                                          hidden_size=4096, num_layers=40,
                                          num_attention_heads=32, vocab_size=131072

params.json                       1.2 KB  Mistral-format architectural parameters
                                          Equivalent to config.json for Mistral format

tokenizer.json                   17.1 MB  Vocabulary + tokenization rules
                                          131,072 tokens (Tekken tokenizer)
                                          Maps text ↔ token IDs

tokenizer_config.json            21.2 KB  Tokenizer settings and chat template
                                          Defines [INST]/[/INST] format

tekken.json                      16.8 MB  Tekken tokenizer model data
                                          Mistral's custom BPE tokenizer

model.safetensors.index.json    103   KB  Weight shard lookup index
                                          Maps layer names → tensor offsets

generation_config.json           131   B  Default sampling parameters
                                          temperature=1.0, top_p=1.0 defaults

processor_config.json            976   B  Input processor configuration

SYSTEM_PROMPT.txt                 2.4 KB  Recommended system prompt for this model

README.md                        20.3 KB  Mistral's official model documentation
```

### Why SafeTensors?

```
Format comparison for model weight files:

SafeTensors (.safetensors):
  ✅ No arbitrary code execution (unlike pickle-based .pt files)
  ✅ Memory-mapped (mmap) – loaded directly into VRAM without copying
  ✅ Parallel loading – multiple threads read different parts simultaneously
  ✅ Zero-copy on GPU – no intermediate CPU buffer needed
  → Used in production (this workshop)

PyTorch (.pt / .pth):
  ⚠️  Uses Python pickle – can execute arbitrary code on load
  ✅  Native PyTorch format, widely supported
  → Used in research/development

HDF5 (.h5):
  ✅  Cross-platform, hierarchical
  ⚠️  Slower load than SafeTensors
  → Used in TensorFlow/Keras
```

### RunAI Streamer: Fast Model Loading

```
Standard loading (sequential):
  S3 → download → disk → load → GPU VRAM
  Time: 8–15 minutes for 10 GB model

RunAI Streamer (--load-format=runai_streamer):
  S3 ──┐
  S3 ──┤  16 parallel GET requests
  S3 ──┤  (--model-loader-extra-config={"concurrency":16})
  S3 ──┘
  Direct stream → GPU VRAM (zero disk, zero CPU copy)
  Time: 2–3 minutes for the same 10 GB model
```

---

## EKS Pod Identity – Secure S3 Access

Pods need to access S3 to load model weights. Hardcoding AWS credentials in pods is insecure. EKS Pod Identity provides temporary, automatically-rotated credentials.

```
How it works:
─────────────
1. Terraform creates: IAM Role with S3 permissions
2. Terraform creates: EKS Pod Identity Association
   → maps ServiceAccount "model-storage-sa" to that IAM Role
3. Pod spec sets: serviceAccountName: model-storage-sa
4. AWS SDK (boto3, RunAI Streamer) automatically picks up
   credentials from the Pod Identity token
5. No AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY needed anywhere

IAM permissions granted:
  s3:GetObject   → read model weights
  s3:PutObject   → write benchmark results, LoRA adapters
  s3:ListBucket  → list model files
  Resource: arn:aws:s3:::genai-models-<ACCOUNT_ID>/*
```

---

## DCGM Exporter – Monitoring GPU Health

DCGM (Data Center GPU Manager) is NVIDIA's tool for collecting GPU metrics. The DCGM Exporter DaemonSet runs one pod per GPU node and exposes metrics to Prometheus.

```
DaemonSet: one pod per GPU node (automatic)
Port: 9400 (Prometheus scrape target)
Helm values (dcgm-values.yaml):

serviceMonitor:
  enabled: true
  additionalLabels:
    release: kube-prometheus-stack   ← Prometheus Operator uses this label
                                        to discover the ServiceMonitor
  interval: 30s                      ← Scrape GPU metrics every 30 seconds
  honorLabels: true                  ← Keep the labels from the GPU pods

nodeSelector:
  karpenter.sh/nodepool: gpu         ← Only deploy on GPU nodes

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule               ← Must tolerate GPU taint to land on node

resources:
  requests: { cpu: 100m, memory: 256Mi }  ← Minimal CPU – just reads GPU counters
  limits:   { cpu: 500m, memory: 512Mi }

Key GPU metrics exposed:
  DCGM_FI_DEV_GPU_UTIL          GPU compute utilisation (%)
                                 0% = idle, 100% = fully used for inference
  DCGM_FI_DEV_MEM_COPY_UTIL    Memory copy utilisation (%)
                                 High value = lots of data moving to/from VRAM
  DCGM_FI_DEV_FB_FREE           Free framebuffer memory (MB)
                                 Watch this: if it hits 0, OOM crash
  DCGM_FI_DEV_FB_USED           Used framebuffer memory (MB)
  DCGM_FI_DEV_GPU_TEMP          GPU temperature (°C)
                                 L40S max: 83°C. High temp = throttling
  DCGM_FI_DEV_POWER_USAGE       Power draw (Watts)
                                 L40S TDP: 300W. Monitor cost
  DCGM_FI_DEV_SM_CLOCK          Streaming Multiprocessor clock (MHz)
  DCGM_FI_DEV_MEM_CLOCK         Memory clock (MHz)
```

---

## Monitoring Stack Overview

```
Data flow:
──────────────────────────────────────────────────────────────────────
GPU hardware
  └──▶ DCGM Exporter pod (:9400/metrics)
         └──▶ Prometheus (ServiceMonitor discovers it)
                └──▶ Grafana Dashboard: "NVIDIA GPU Metrics"

vLLM server process
  └──▶ /metrics endpoint (:8000/metrics)
         └──▶ Prometheus (ServiceMonitor: mistral-monitor)
                └──▶ Grafana Dashboard: "vLLM Benchmarking Dashboard"

Ray Serve head + worker pods
  └──▶ metrics port (:8080)
         └──▶ Prometheus (PodMonitor: ray-head-monitor, ray-workers-monitor)
                └──▶ Grafana Dashboard: "Ray Serve Inference Overview"
──────────────────────────────────────────────────────────────────────
Access Grafana: http://<grafana-ingress-ALB>/
Default login: admin / prom-operator  (or check Helm values)
```

---

## Instance Type Guide

```
┌──────────────────┬────────────┬────────────┬────────────┬───────────────────┐
│  Instance        │  GPUs      │  VRAM      │  vCPUs     │  Use in workshop  │
├──────────────────┼────────────┼────────────┼────────────┼───────────────────┤
│  g6e.2xlarge     │  1× L40S   │  48 GB     │  8         │  Single-GPU       │
│                  │            │            │            │  inference,       │
│                  │            │            │            │  fine-tuning      │
├──────────────────┼────────────┼────────────┼────────────┼───────────────────┤
│  g6e.12xlarge    │  4× L40S   │  192 GB    │  48        │  Multi-GPU        │
│                  │            │            │            │  tensor-parallel  │
│                  │            │            │            │  (larger models)  │
├──────────────────┼────────────┼────────────┼────────────┼───────────────────┤
│  m5.xlarge       │  None      │  N/A       │  4         │  CPU workloads:   │
│                  │            │            │            │  WebUI, Grafana,  │
│                  │            │            │            │  agents, RAG UI   │
└──────────────────┴────────────┴────────────┴────────────┴───────────────────┘
```

---

## Setup Commands

```bash
# 1. Set environment variables
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"
export AWS_REGION="us-east-1"

# 2. Verify model is in S3
aws s3 ls s3://${S3_BUCKET_NAME}/Ministral-3-8B-Instruct-2512/ --recursive --human-readable

# 3. Verify GPU nodes (after deploying a GPU workload)
kubectl get nodes -l karpenter.sh/nodepool=gpu
kubectl describe node <gpu-node-name> | grep -A5 "Allocatable"

# 4. Install DCGM Exporter
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm install --generate-name gpu-helm-charts/dcgm-exporter \
  -f GPU/dcgm-values.yaml -n monitoring --create-namespace

# 5. Check GPU metrics are flowing
kubectl port-forward svc/kube-prometheus-stack-prometheus 9090:9090 -n monitoring &
# Then open: http://localhost:9090 and query: DCGM_FI_DEV_GPU_UTIL

# Next: proceed to Module 100 to deploy the inference server
```

---

## What You Learn Here (AI Inference Connection)

| What you set up | What it means for AI inference |
|----------------|-------------------------------|
| GPU nodes (L40S 48 GB VRAM) | Hardware that makes real-time LLM inference possible |
| Karpenter auto-provisioning | Solves the **cost challenge** – pay only when inferring |
| S3 model storage + streaming | Decouples model from code; fast startup |
| Pod Identity (no hardcoded keys) | Secure, production-ready credential management |
| DCGM metrics → Grafana | Visibility into the **resource challenge** (GPU utilisation, memory) |
| Monitoring stack | Foundation for benchmarking (Module 300) |
