# GPU Infrastructure Architecture

> Foundation stack for the entire workshop. Provisions an EKS cluster with GPU-accelerated node pools managed by Karpenter, loads the Ministral-3-8B model from S3, and wires up the monitoring stack.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AWS ACCOUNT                                      │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         Amazon EKS Cluster                              │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────────┐  ┌────────────────────────────────┐  │  │
│  │  │      System Node Pool        │  │    GPU Node Pool (Karpenter)   │  │  │
│  │  │  Instance: m5.xlarge         │  │  Instance: g6e.2xlarge         │  │  │
│  │  │  Role: Control plane workloads│  │           g6e.12xlarge         │  │  │
│  │  │  - Open WebUI (Chat UI)       │  │  GPU: NVIDIA L40S (48 GB VRAM) │  │  │
│  │  │  - Grafana Dashboard          │  │  Taint: nvidia.com/gpu=NoSched │  │  │
│  │  │  - RAG Gradio UI              │  │  Label: karpenter.sh/nodepool  │  │  │
│  │  │  - Strands Agent              │  │         = gpu                  │  │  │
│  │  └──────────────────────────────┘  └────────────────────────────────┘  │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     Monitoring Stack                              │  │  │
│  │  │  kube-prometheus-stack                                            │  │  │
│  │  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │  │  │
│  │  │  │  Prometheus  │◀──│ DCGM Exporter│   │       Grafana        │ │  │  │
│  │  │  │  Port: 9090  │   │  Port: 9400  │   │  Port: 3000          │ │  │  │
│  │  │  │  Scrape: 30s │   │  GPU Metrics │   │  ALB Ingress(public) │ │  │  │
│  │  │  └──────────────┘   └──────────────┘   └──────────────────────┘ │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │                  EKS Pod Identity / IRSA                          │  │  │
│  │  │  ServiceAccount: model-storage-sa  ──▶  IAM Role  ──▶  S3 r/w   │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌──────────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │          Amazon S3               │  │   AWS Application Load Balancer  │ │
│  │  Bucket: genai-models-<ACCT_ID>  │  │   - Open WebUI  (port 80/443)    │ │
│  │  ├── Ministral-3-8B-Instruct/    │  │   - Grafana     (port 80/443)    │ │
│  │  │   ├── consolidated.safetensors│  │   - RAG Gradio  (port 80/443)    │ │
│  │  │   ├── config.json             │  │   Scheme: internet-facing        │ │
│  │  │   ├── params.json             │  │   Target: ip                     │ │
│  │  │   ├── tokenizer.json          │  └──────────────────────────────────┘ │
│  │  │   ├── tekken.json             │                                        │
│  │  │   └── ...                     │                                        │
│  │  ├── anyvc-startup-lora/         │                                        │
│  │  └── benchmarks/                 │                                        │
│  └──────────────────────────────────┘                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## GPU Node Specification

```
┌─────────────────────────────────────────────────────────────┐
│                   GPU Node (g6e.2xlarge)                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 NVIDIA L40S GPU                      │    │
│  │  VRAM:       48 GB GDDR6                             │    │
│  │  CUDA Cores: 18,176                                  │    │
│  │  TF32 TOPS:  362.1                                   │    │
│  │  INT8 TOPS:  733.6                                   │    │
│  │  FP8 TOPS:   733.6                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Host Resources                      │    │
│  │  vCPUs:  8    RAM: 64 GB    Storage: NVMe SSD        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                NVIDIA Device Plugin                  │    │
│  │  Exposes: nvidia.com/gpu resource to Kubernetes      │    │
│  │  Taint:   nvidia.com/gpu:NoSchedule                  │    │
│  │  CUDA:    12.x + cuDNN                               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

Multi-GPU variant: g6e.12xlarge
  └── 4× NVIDIA L40S (192 GB total VRAM)
  └── Used for tensor-parallel-size=4 experiments
```

---

## Karpenter NodePool Configuration

```
NodePool: gpu
  ┌─────────────────────────────────────────────────────┐
  │  Requirements                                         │
  │  - karpenter.k8s.aws/instance-family: [g6e]          │
  │  - karpenter.k8s.aws/instance-size:                  │
  │      [2xlarge, 12xlarge]                             │
  │  - kubernetes.io/arch: amd64                         │
  │  - karpenter.sh/capacity-type: [on-demand]           │
  │                                                       │
  │  Taints                                               │
  │  - nvidia.com/gpu:NoSchedule                         │
  │    (GPU pods must have matching toleration)          │
  │                                                       │
  │  Disruption Policy                                   │
  │  - consolidateAfter: 30s                             │
  │  - expireAfter: 720h (30 days)                       │
  └─────────────────────────────────────────────────────┘
```

---

## Model Storage and Loading

### S3 Bucket Structure

```
s3://genai-models-<ACCOUNT_ID>/
├── Ministral-3-8B-Instruct-2512/          # Base model (10.4 GB weights)
│   ├── consolidated.safetensors           # All model weights – SafeTensors format
│   │   Size: 10.4 GB | Format: BFloat16/FP8
│   │   Zero-copy mmap loading via RunAI Streamer
│   ├── config.json                        # Model architecture
│   │   hidden_size, num_layers, num_heads, vocab_size
│   ├── params.json                        # Architectural parameters
│   ├── tokenizer.json                     # Full vocab (131k tokens, Tekken)
│   ├── tokenizer_config.json              # Tokenizer settings
│   ├── tekken.json                        # Tekken tokenizer data (16.8 MB)
│   ├── model.safetensors.index.json       # Weight shard index (103 KB)
│   ├── generation_config.json             # Default generation params
│   ├── processor_config.json              # Processor configuration
│   ├── README.md                          # Model documentation
│   └── SYSTEM_PROMPT.txt                  # Recommended system prompt
│
├── anyvc-startup-lora/                    # LoRA adapter (Module 600)
│   ├── adapter_model.safetensors         # LoRA weight deltas (~50–100 MB)
│   ├── adapter_config.json               # r=16, alpha=32, target_modules
│   └── tokenizer.*                       # Tokenizer copy
│
└── benchmarks/                           # Benchmark results (Module 300)
    └── <timestamp>/                      # Per-run result directories
```

### Model Loading Flow

```
Pod startup
    │
    ▼
┌───────────────────────────────────────────────────────┐
│         RunAI Streamer (--load-format=runai_streamer)  │
│                                                         │
│  1. Authenticate via EKS Pod Identity                  │
│     model-storage-sa → IAM Role → S3 Access            │
│                                                         │
│  2. Stream-load with concurrency=16                    │
│     16 parallel S3 GET requests                        │
│     ~2–3 min load time (vs 8+ min sequential)         │
│                                                         │
│  3. Zero-copy mmap into GPU VRAM                       │
│     consolidated.safetensors → NVIDIA L40S             │
│     BFloat16 → 48 GB VRAM utilization ~90%            │
│                                                         │
│  4. FlashInfer attention backend init                  │
│     VLLM_ATTENTION_BACKEND=FLASHINFER                  │
└───────────────────────────────────────────────────────┘
    │
    ▼
Model ready: /v1/models → { "id": "ministral" }
```

---

## DCGM GPU Metrics Exporter

```
┌─────────────────────────────────────────────────────────────┐
│                  DCGM Exporter DaemonSet                     │
│                                                               │
│  Deployment: DaemonSet (one pod per GPU node)                │
│  Image:      nvcr.io/nvidia/k8s/dcgm-exporter               │
│  Port:       9400 (Prometheus metrics)                       │
│                                                               │
│  Key GPU Metrics Exported:                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  DCGM_FI_DEV_GPU_UTIL         GPU utilization (%)     │  │
│  │  DCGM_FI_DEV_MEM_COPY_UTIL   Memory copy utilization  │  │
│  │  DCGM_FI_DEV_FB_FREE         Free framebuffer (MB)    │  │
│  │  DCGM_FI_DEV_FB_USED         Used framebuffer (MB)    │  │
│  │  DCGM_FI_DEV_GPU_TEMP        GPU temperature (°C)     │  │
│  │  DCGM_FI_DEV_POWER_USAGE     Power draw (W)           │  │
│  │  DCGM_FI_DEV_SM_CLOCK        SM clock (MHz)           │  │
│  │  DCGM_FI_DEV_MEM_CLOCK       Memory clock (MHz)       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  Helm values (dcgm-values.yaml):                             │
│    serviceMonitor.enabled: true                              │
│    serviceMonitor.additionalLabels:                          │
│      release: kube-prometheus-stack                          │
│    serviceMonitor.interval: 30s                              │
│    nodeSelector:                                             │
│      karpenter.sh/nodepool: gpu                              │
│    tolerations:                                              │
│      - key: nvidia.com/gpu | effect: NoSchedule              │
│    resources:                                                │
│      limits:   cpu: 500m, memory: 512Mi                     │
│      requests: cpu: 100m, memory: 256Mi                     │
└─────────────────────────────────────────────────────────────┘
```

---

## EKS Pod Identity (IAM Access)

```
┌──────────────────────────────────────────────────────────────────┐
│                   EKS Pod Identity Flow                           │
│                                                                   │
│  Kubernetes Pod                                                   │
│  serviceAccountName: model-storage-sa                            │
│         │                                                         │
│         ▼                                                         │
│  EKS Pod Identity Association                                     │
│         │ maps SA → IAM Role                                      │
│         ▼                                                         │
│  IAM Role: eks-model-storage-role                                 │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  Policy: AmazonS3FullAccess (scoped to bucket)            │   │
│  │  - s3:GetObject                                           │   │
│  │  - s3:PutObject                                           │   │
│  │  - s3:ListBucket                                          │   │
│  │  Resource: arn:aws:s3:::genai-models-<ACCOUNT_ID>/*       │   │
│  └───────────────────────────────────────────────────────────┘   │
│         │                                                         │
│         ▼                                                         │
│  Amazon S3: s3://genai-models-<ACCOUNT_ID>/                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Monitoring Stack Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│               kube-prometheus-stack (Helm Release)               │
│                                                                   │
│  ┌──────────────┐  scrape   ┌──────────────────────────────┐    │
│  │  Prometheus  │◀──────────│   ServiceMonitors /           │    │
│  │  Port: 9090  │           │   PodMonitors                 │    │
│  │  Retention:  │           │   - mistral-monitor (vLLM)    │    │
│  │  15 days     │           │   - dcgm-exporter             │    │
│  └──────┬───────┘           │   - ray-head-monitor          │    │
│         │                   │   - ray-workers-monitor       │    │
│         │ query             └──────────────────────────────┘    │
│         ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    Grafana                                │    │
│  │  Port: 3000 | ALB Ingress: grafana-ingress               │    │
│  │  Dashboards:                                              │    │
│  │  - NVIDIA GPU Metrics (DCGM)                             │    │
│  │  - vLLM Benchmarking Dashboard                           │    │
│  │  - Ray Serve Inference Overview                          │    │
│  │  - Kubernetes cluster overview                           │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Environment Variables

```bash
# Required for all modules
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"
export AWS_REGION="us-east-1"

# Model path
export MODEL_PATH="s3://${S3_BUCKET_NAME}/Ministral-3-8B-Instruct-2512/"

# Verify model files in S3
aws s3 ls s3://${S3_BUCKET_NAME}/Ministral-3-8B-Instruct-2512/ --recursive
```

---

## Reference Images

| Image | Description |
|-------|-------------|
| [gpu-initial-architecture.png](./gpu-initial-architecture.png) | GPU node hardware layout |
| [initial-stack-diagram.png](./initial-stack-diagram.png) | Full initial EKS stack topology |
