# Customizing LLMs on AWS EKS

> **A hands-on workshop** for deploying, optimizing, and customizing Large Language Models on Amazon Elastic Kubernetes Service (EKS) using GPU-accelerated infrastructure.

---

## Workshop Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CUSTOMIZING LLMs ON AWS EKS                             │
│                     Workshop Learning Path                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │   GPU    │───▶│  100     │───▶│  200     │───▶│  300     │             │
│   │  Stack   │    │  vLLM    │    │  Strands │    │ Bench-   │             │
│   │  Setup   │    │  Deploy  │    │  Agent   │    │ marking  │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                                                         │                    │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  800     │    │  700     │    │  600     │    │  400     │             │
│   │  Ray     │◀───│  RAG     │◀───│  Fine-   │◀───│ LMCache  │             │
│   │  Serve   │    │Pipeline  │    │  tuning  │    │ KV Cache │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Used: Ministral-3-8B-Instruct-2512

| Attribute | Value |
|-----------|-------|
| **Model Family** | Mistral AI – Ministral 3B |
| **Full Name** | Ministral-3-8B-Instruct-2512 |
| **Parameters** | ~3.8 Billion |
| **Precision** | BFloat16 / FP8 |
| **Context Length** | 8,192 tokens (configurable) |
| **Tokenizer** | Tekken (131k vocabulary) |
| **Format** | SafeTensors (consolidated) |
| **Storage** | Amazon S3 (`s3://genai-models-<ACCOUNT_ID>/Ministral-3-8B-Instruct-2512/`) |

### Model Files in S3

```
s3://${S3_BUCKET_NAME}/Ministral-3-8B-Instruct-2512/
├── consolidated.safetensors      # 10.4 GB  – All model weights (SafeTensors format)
├── config.json                   # 1.9 KB   – Model architecture configuration
├── params.json                   # 1.2 KB   – Architectural parameters (layers, heads)
├── tokenizer.json                # 17.1 MB  – Full tokenizer vocabulary + rules
├── tokenizer_config.json         # 21.2 KB  – Tokenizer configuration
├── tekken.json                   # 16.8 MB  – Tekken tokenizer data
├── model.safetensors.index.json  # 103 KB   – Weight shard index
├── generation_config.json        # 131 B    – Default generation parameters
├── processor_config.json         # 976 B    – Processor configuration
├── README.md                     # 20.3 KB  – Model documentation
└── SYSTEM_PROMPT.txt             # 2.4 KB   – Recommended system prompt
```

### Model Weight Format Comparison

| Format | Security | Speed | Memory | Use Case |
|--------|----------|-------|--------|----------|
| **SafeTensors** (.safetensors) | ✅ Highest | ✅ Fastest | ✅ Zero-copy | Production deployments |
| PyTorch (.pt/.pth) | ⚠️ Pickle risk | ✅ Fast | ✅ Good | Research & dev |
| TensorFlow SavedModel | ✅ Good | ✅ Good | ✅ Good | TF production |
| HDF5 (.h5) | ✅ Good | ⚠️ Moderate | ⚠️ Higher | Keras/legacy |

---

## Workshop Modules

| Module | Topic | Key Technology | Details |
|--------|-------|---------------|---------|
| [GPU](./GPU/ARCHITECTURE.md) | Infrastructure Setup | EKS + Karpenter + GPU nodes | Foundation stack |
| [100-vllm](./100-vllm/ARCHITECTURE.md) | LLM Serving | vLLM 0.21.0 + OpenWebUI | Model serving & chat UI |
| [200-strands-agent](./200-strands-agent/ARCHITECTURE.md) | AI Agents | Strands Agents SDK | Tool-calling agent |
| [300-benchmarking](./300-benchmarking/ARCHITECTURE.md) | Performance Testing | inference-perf + Grafana | Load testing & metrics |
| [400-lmcache](./400-lmcache/ARCHITECTURE.md) | KV Cache Offloading | LMCache + Valkey | Latency optimization |
| [600-finetuning](./600-finetuning/ARCHITECTURE.md) | Model Fine-tuning | LoRA + PEFT + TRL | Domain adaptation |
| [700-rag](./700-rag/ARCHITECTURE.md) | RAG Pipeline | S3 Vectors + Embeddings | Retrieval-augmented gen |
| [800-ray](./800-ray/ARCHITECTURE.md) | Distributed Serving | Ray Serve + vLLM | Autoscaling inference |

---

## Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AWS ACCOUNT                                        │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Amazon EKS Cluster                            │    │
│  │                                                                       │    │
│  │  ┌──────────────────────┐    ┌──────────────────────────────────┐   │    │
│  │  │   System Node Pool   │    │       GPU Node Pool (Karpenter)  │   │    │
│  │  │  m5.xlarge instances │    │  g6e.2xlarge / g6e.12xlarge      │   │    │
│  │  │  - Open WebUI        │    │  NVIDIA L40S GPU (48 GB VRAM)    │   │    │
│  │  │  - Grafana           │    │  - vLLM Inference                │   │    │
│  │  │  - RAG Gradio UI     │    │  - LMCache                       │   │    │
│  │  │  - Strands Agent     │    │  - Fine-tuning Jobs              │   │    │
│  │  └──────────────────────┘    │  - Ray Worker Nodes              │   │    │
│  │                               └──────────────────────────────────┘   │    │
│  │                                                                       │    │
│  │  ┌──────────────────────────────────────────────────────────────┐    │    │
│  │  │                    Monitoring Stack                           │    │    │
│  │  │  Prometheus (kube-prometheus-stack) → Grafana                │    │    │
│  │  │  DCGM Exporter → GPU Metrics → Dashboards                   │    │    │
│  │  └──────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌──────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐   │
│  │  Amazon S3   │  │  Amazon ElastiCache  │  │  AWS Load Balancer (ALB)│   │
│  │  Model Weights│  │  Valkey (Serverless) │  │  Open WebUI (public)    │   │
│  │  LoRA Adapters│  │  KV Cache (L2)       │  │  Grafana (public)       │   │
│  │  Benchmarks  │  │  Port: 6379 (TLS)    │  │  RAG Gradio (public)    │   │
│  │  RAG Vectors │  └──────────────────────┘  └─────────────────────────┘   │
│  └──────────────┘                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- AWS Account with EKS cluster deployed
- `kubectl` configured for the cluster
- `helm` v3+
- AWS CLI configured
- Karpenter installed on the cluster
- `model-storage-sa` service account with S3 access (via EKS Pod Identity)

---

## Quick Start

```bash
# 1. Set environment variables
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"
export AWS_REGION="us-east-1"   # or your region

# 2. Deploy base vLLM serving (Module 100)
kubectl apply -f 100-vllm/vllm-deployment.yml

# 3. Deploy Open WebUI
kubectl apply -f 100-vllm/openwebui.yml

# 4. Get the Open WebUI endpoint
kubectl get ingress open-webui-ingress
```

---

## Repository Structure

```
.
├── README.md                          # This file – workshop overview
├── GPU/
│   ├── ARCHITECTURE.md               # GPU node & EKS stack architecture
│   ├── gpu-initial-architecture.png  # GPU node diagram image
│   └── initial-stack-diagram.png     # Full initial stack diagram image
├── 100-vllm/
│   ├── ARCHITECTURE.md               # vLLM deployment architecture
│   ├── vllm-deployment.yml           # Main vLLM Deployment + Service
│   ├── openwebui.yml                 # Open WebUI Deployment + Ingress
│   ├── grafana-ingress.yaml          # Grafana ALB Ingress
│   ├── dcgm-values.yaml              # DCGM GPU metrics exporter config
│   └── vllm-servicemonitor.yaml      # Prometheus ServiceMonitor
├── 200-strands-agent/
│   ├── ARCHITECTURE.md               # Strands Agent architecture
│   ├── strands-agent.py              # FastAPI + Strands agent application
│   ├── Dockerfile                    # Container build definition
│   ├── requirements.txt              # Python dependencies
│   └── vllm-deployment-agents.yaml   # Deployment with prefix-caching enabled
├── 300-benchmarking/
│   ├── ARCHITECTURE.md               # Benchmarking pipeline architecture
│   ├── vllm-optimized-deployment.yml # Optimized vLLM for benchmarking
│   ├── baseline-values.yaml          # Baseline benchmark scenario
│   ├── saturation-values.yaml        # Saturation benchmark scenario
│   ├── vllm-benchmarking-dashboard-cr.yaml  # Grafana dashboard CRD
│   ├── vllm-benchmarking-dashboard.json     # Grafana dashboard definition
│   └── benchmark-charts/             # Helm chart for benchmark jobs
├── 400-lmcache/
│   ├── ARCHITECTURE.md               # LMCache KV offloading architecture
│   ├── cpu-ram-offloading/           # L1 CPU RAM cache setup
│   │   ├── lmcache-cpu-ram-pod.yaml  # Pod with CPU RAM offloading
│   │   └── lmcache-cpu-ram-configmap.yaml
│   └── remote-cache-sharing-with-valkey/   # L2 remote cache setup
│       ├── lmcache-valkey-pod.yaml   # Pod with Valkey remote cache
│       └── lmcache-valkey-configmap.yaml
├── 600-finetuning/
│   ├── ARCHITECTURE.md               # LoRA fine-tuning architecture
│   ├── train_lora.py                 # LoRA training script
│   ├── lora-training-job.yaml        # Kubernetes Job definition
│   ├── vllm-with-lora.yaml          # vLLM serving with LoRA adapter
│   └── anyvc-startup-dataset.jsonl  # Training dataset (AnyVC startup advisor)
├── 700-rag/
│   ├── ARCHITECTURE.md               # RAG pipeline architecture
│   ├── rag-service.yml               # RAG backend service
│   ├── rag-gradio-deploy.yml         # Gradio UI deployment + Ingress
│   ├── rag-document-job.yml          # Document processing job
│   ├── rag-processor.yml             # Embedding processor
│   ├── rag-serve.yml                 # RAG serving application
│   ├── rag-gradio-app.yml            # Gradio application config
│   └── electronics.jsonl             # Sample documents dataset
└── 800-ray/
    ├── ARCHITECTURE.md               # Ray Serve + vLLM architecture
    ├── ray-vllm-service.yaml         # RayService CRD definition
    ├── ray-podmonitor.yaml           # Prometheus PodMonitor for Ray
    ├── openwebui.yml                 # Open WebUI for Ray deployment
    └── generate-load.sh              # Load generation script
```

---

## Key Technologies

| Technology | Version | Purpose |
|-----------|---------|---------|
| **vLLM** | 0.21.0 | High-throughput LLM inference engine |
| **Ray Serve** | 2.56.1 | Distributed model serving framework |
| **LMCache** | 0.3.8 | KV cache offloading (CPU RAM + remote) |
| **Strands Agents** | 1.50.2 | AI agent SDK with tool calling |
| **Open WebUI** | v0.11.0 | Chat interface for LLM interaction |
| **Karpenter** | latest | GPU node auto-provisioning |
| **DCGM Exporter** | latest | NVIDIA GPU metrics collection |
| **Prometheus** | kube-prometheus-stack | Metrics aggregation |
| **Grafana** | kube-prometheus-stack | Metrics visualization |
| **inference-perf** | v0.6.1 | LLM benchmarking tool |
| **PEFT / LoRA** | 0.19.1 | Parameter-efficient fine-tuning |
| **TRL** | 1.5.1 | Transformer reinforcement learning |
| **FlashInfer** | latest | Optimized attention backend |

---

## GPU Hardware

| Instance Type | GPU | VRAM | Use Case |
|--------------|-----|------|---------|
| `g6e.2xlarge` | NVIDIA L40S | 48 GB | Single-GPU inference & fine-tuning |
| `g6e.12xlarge` | 4× NVIDIA L40S | 192 GB | Multi-GPU tensor parallel |

---

*Workshop maintained by Ramu-DE. See individual module ARCHITECTURE.md files for detailed diagrams.*
