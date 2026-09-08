# Module 100 – vLLM Deployment Architecture

> Deploys the Ministral-3-8B-Instruct-2512 model using vLLM on a GPU node, exposes an OpenAI-compatible REST API, and provides a browser-based chat UI through Open WebUI. Prometheus metrics are scraped by DCGM Exporter and a dedicated ServiceMonitor.

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        EKS Cluster – default namespace                        │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  GPU Node Pool (g6e.2xlarge)  |  Taint: nvidia.com/gpu:NoSchedule    │  │
│  │                                                                        │  │
│  │  ┌───────────────────────────────────────────────────────────────┐   │  │
│  │  │                  Pod: mistral (Deployment)                     │   │  │
│  │  │  image: public.ecr.aws/deep-learning-containers/              │   │  │
│  │  │         vllm:0.21.0-gpu-py312-cu130-ubuntu22.04-ec2-v1.0-soci │   │  │
│  │  │  serviceAccount: model-storage-sa                              │   │  │
│  │  │  replicas: 1  |  strategy: Recreate                           │   │  │
│  │  │                                                                │   │  │
│  │  │  ┌──────────────────────────────────────────────────────┐    │   │  │
│  │  │  │               vLLM Server Process                     │    │   │  │
│  │  │  │  Port: 8000 (HTTP / OpenAI-compatible API)            │    │   │  │
│  │  │  │  /v1/completions    /v1/chat/completions              │    │   │  │
│  │  │  │  /v1/models         /metrics (Prometheus)             │    │   │  │
│  │  │  │  /health            /v1/models                        │    │   │  │
│  │  │  └──────────────────────────────────────────────────────┘    │   │  │
│  │  │                                                                │   │  │
│  │  │  Resources:                                                    │   │  │
│  │  │    requests: cpu=2, memory=16Gi, nvidia.com/gpu=1             │   │  │
│  │  │    limits:   cpu=4, memory=28Gi, nvidia.com/gpu=1             │   │  │
│  │  └───────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                          │                                    │
│                    Service: vllm-serve-svc (ClusterIP, port 8000)            │
│                                          │                                    │
│  ┌──────────────────────────────────────┼──────────────────────────────┐    │
│  │  System Node Pool (m5.xlarge)        │                               │    │
│  │                                      ▼                               │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │             Pod: open-webui (Deployment)                      │   │    │
│  │  │  image: ghcr.io/open-webui/open-webui:v0.11.0                │   │    │
│  │  │  Port: 8080  |  replicas: 1                                   │   │    │
│  │  │  nodeSelector: m5.xlarge                                      │   │    │
│  │  │                                                                │   │    │
│  │  │  ENV:                                                          │   │    │
│  │  │    OPENAI_API_BASE_URLS = http://vllm-serve-svc:8000/v1       │   │    │
│  │  │    OPENAI_API_KEY       = dummy                               │   │    │
│  │  │    WEBUI_AUTH           = False                               │   │    │
│  │  │    ENABLE_OLLAMA_API    = False                               │   │    │
│  │  │  Volume: emptyDir → /app/backend/data                        │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                             │                                         │    │
│  │     Service: open-webui (ClusterIP, port 80 → 8080)                  │    │
│  │                             │                                         │    │
│  │     Ingress: open-webui-ingress (ALB, internet-facing)               │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                monitoring namespace                                   │    │
│  │  ServiceMonitor: mistral-monitor                                      │    │
│  │    selector: model=mistral | port: http | interval: 30s              │    │
│  │    path: /metrics                                                     │    │
│  │    label: release=kube-prometheus-stack                               │    │
│  │                                                                       │    │
│  │  Ingress: grafana-ingress (ALB, internet-facing)                     │    │
│  │    service: kube-prometheus-stack-grafana:3000                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘

             Internet
                │
    ┌───────────┴────────────┐
    │   AWS ALB              │
    │   open-webui-ingress   │  ← http://<ALB_DNS>  (Chat UI)
    │   grafana-ingress      │  ← http://<ALB_DNS>  (Metrics)
    └────────────────────────┘
```

---

## vLLM Configuration (vllm-deployment.yml)

### Model and Weights

```
Model:   Ministral-3-8B-Instruct-2512
Source:  s3://genai-models-<ACCOUNT_ID>/Ministral-3-8B-Instruct-2512/
Format:  SafeTensors (consolidated.safetensors, 10.4 GB)
```

The model weights are stored in S3 and streamed directly into GPU VRAM at pod startup using RunAI Streamer – no persistent disk required.

### vLLM CLI Arguments (Full Parameter Reference)

```
--port=8000
    Listening port for the OpenAI-compatible HTTP server.

--model=s3://genai-models-<ACCOUNT_ID>/Ministral-3-8B-Instruct-2512/
    S3 URI of the model weights directory.
    Authenticated via EKS Pod Identity (model-storage-sa → IAM role).

--served-model-name=ministral
    The model name returned by /v1/models and used in API requests.
    Client must send: { "model": "ministral" }

--load-format=runai_streamer
    Uses RunAI Streamer for concurrent, streaming S3 loading.
    Bypasses local disk; streams directly into GPU VRAM.
    Dramatically faster than default loading (2-3 min vs 8+ min).

--model-loader-extra-config={"concurrency":16}
    Number of parallel S3 GET requests during model streaming.
    16 concurrent streams saturate typical S3 bandwidth.

--tokenizer_mode=mistral
    Uses the Mistral-native tokenizer backend.
    Required for the Tekken tokenizer (131k vocabulary).
    Falls back to HuggingFace AutoTokenizer otherwise.

--trust-remote-code
    Allows execution of custom model code from the repository.
    Required for Mistral-3 model class.

--gpu_memory_utilization=0.90
    Fraction of GPU VRAM reserved for the KV cache.
    0.90 → ~43 GB of the 48 GB L40S VRAM for KV cache.
    Remaining 10% (~5 GB) reserved for model weights + activations.

--max-model-len=8192
    Maximum sequence length (prompt + output combined).
    Matches the model's native 8,192 context window.
    Larger values increase KV cache memory requirements.

--tensor-parallel-size=1
    Number of GPUs for tensor parallelism.
    1 = single GPU (g6e.2xlarge with 1× L40S).
    Set to 4 for g6e.12xlarge with 4× L40S GPUs.

--max-num-batched-tokens=8192
    Maximum tokens processed in one scheduler step.
    Equal to max-model-len for single-sequence optimization.

--max-num-seqs=256
    Maximum number of sequences (requests) processed concurrently.
    Higher values increase throughput but use more KV cache memory.

--block-size=16
    Size of KV cache memory blocks (tokens per block).
    16 is the vLLM default for PagedAttention.

--enforce-eager
    Disables CUDA Graph capture.
    Required for some Mistral model variants.
    Slightly lower throughput vs CUDA Graphs but more stable.

--disable-custom-all-reduce
    Disables custom NCCL all-reduce kernels.
    Ensures compatibility in single-GPU mode.

--config-format=mistral
    Reads model configuration in Mistral JSON format (params.json).
    Rather than HuggingFace config.json format.

--enable-auto-tool-choice
    Enables automatic tool/function call detection in requests.
    Required for Strands Agent (Module 200) tool calling.

--tool-call-parser=mistral
    Uses Mistral's native tool call parsing format.
    Parses [TOOL_CALLS] tokens in model output.
```

### Environment Variables

```
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    Limits CUDA memory allocation fragment size to 512 MB.
    Reduces memory fragmentation in long-running inference servers.

VLLM_ATTENTION_BACKEND=FLASHINFER
    Uses FlashInfer as the attention computation backend.
    Faster than the default Triton backend for most workloads.
    Optimized for NVIDIA Ampere/Ada Lovelace architectures (L40S).
```

---

## API Endpoints

```
Base URL: http://vllm-serve-svc:8000   (internal)
          http://<ALB_DNS>:8000        (via port-forward or ingress)

OpenAI-Compatible REST API:
┌────────────────────────────────────────────────────────┐
│  GET  /v1/models                                        │
│       Returns: { "data": [{ "id": "ministral" }] }     │
│                                                         │
│  POST /v1/completions                                   │
│       Body: { "model": "ministral",                    │
│               "prompt": "...",                          │
│               "max_tokens": 256 }                       │
│                                                         │
│  POST /v1/chat/completions                             │
│       Body: { "model": "ministral",                    │
│               "messages": [{"role":"user","content":…}] │
│               "temperature": 0.7 }                      │
│                                                         │
│  GET  /metrics      Prometheus metrics                  │
│  GET  /health       Liveness probe                      │
└────────────────────────────────────────────────────────┘
```

---

## Readiness Probe

```
httpGet:
  path: /v1/models       # Returns HTTP 200 once model weights are loaded
  port: http (8000)
periodSeconds:    5      # Check every 5 seconds
timeoutSeconds:   2      # 2-second timeout per check
failureThreshold: 3      # Mark pod failed after 3 consecutive failures
```

The pod stays `NotReady` until vLLM has fully loaded model weights into VRAM and the API server is accepting requests. This typically takes 2–4 minutes (RunAI Streamer concurrent S3 loading).

---

## Monitoring Integration

### ServiceMonitor (vllm-servicemonitor.yaml)

```yaml
ServiceMonitor: mistral-monitor
  namespace: monitoring
  labels:
    release: kube-prometheus-stack    # Prometheus Operator discovery label

  spec:
    namespaceSelector:
      matchNames: [default]
    selector:
      matchLabels:
        model: mistral                # Matches Service label
    endpoints:
      - port: http                   # Port named "http" on the Service
        interval: 30s                # Scrape every 30 seconds
        path: /metrics               # vLLM Prometheus metrics endpoint
```

### Key vLLM Prometheus Metrics

```
vllm:num_requests_running         Active inference requests
vllm:num_requests_waiting         Requests in the scheduler queue
vllm:gpu_cache_usage_perc         KV cache utilization (%)
vllm:num_preemptions_total        KV cache evictions (memory pressure)
vllm:e2e_request_latency_seconds  End-to-end request latency histogram
vllm:time_to_first_token_seconds  TTFT latency histogram
vllm:time_per_output_token_seconds Inter-token latency
vllm:request_prompt_tokens        Prompt token count histogram
vllm:request_generation_tokens    Generation token count histogram
vllm:avg_generation_throughput    Tokens/second (generation)
```

---

## DCGM Values (dcgm-values.yaml)

```
serviceMonitor:
  enabled: true
  additionalLabels:
    release: kube-prometheus-stack    # Prometheus Operator discovery
  interval: 30s                       # GPU metrics scrape interval
  honorLabels: true

service:
  type: ClusterIP
  port: 9400                          # DCGM Prometheus metrics port

nodeSelector:
  karpenter.sh/nodepool: gpu         # Deploy only on GPU nodes

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule               # Must tolerate GPU node taint

resources:
  limits:   { cpu: 500m, memory: 512Mi }
  requests: { cpu: 100m, memory: 256Mi }
```

---

## Grafana ALB Ingress (grafana-ingress.yaml)

```
Namespace: monitoring
Service:   kube-prometheus-stack-grafana:3000
ALB Annotations:
  scheme: internet-facing           # Public ALB
  target-type: ip                   # Direct pod routing
  healthcheck-path: /api/health     # Grafana health endpoint
  load-balancer-name: grafana-ingress
  success-codes: 200-302
```

---

## Data Flow

```
User Browser
     │
     ▼ HTTP
┌─────────────────────────────────┐
│     AWS ALB: open-webui-ingress  │
│     DNS: <open-webui-alb>.elb.amazonaws.com
└─────────────────────────────────┘
     │ HTTP :80 → :8080
     ▼
┌─────────────────────────────────┐
│   Pod: open-webui (m5.xlarge)   │
│   Image: open-webui:v0.11.0     │
│   Streams response to browser   │
└─────────────────────────────────┘
     │ HTTP :8000
     │ POST /v1/chat/completions
     ▼
┌─────────────────────────────────┐
│   Service: vllm-serve-svc:8000  │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│   Pod: mistral (g6e.2xlarge, NVIDIA L40S 48GB)      │
│   vLLM 0.21.0                                        │
│   ┌───────────────────────────────────────────────┐ │
│   │  1. Tokenize input (Tekken tokenizer, 131k)   │ │
│   │  2. PagedAttention KV cache lookup            │ │
│   │  3. FlashInfer attention forward pass         │ │
│   │  4. Continuous batching scheduler             │ │
│   │  5. Token sampling (temperature, top_p, etc.) │ │
│   │  6. Streaming token output via SSE            │ │
│   └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
     │ S3 (model load at startup)
     ▼
┌─────────────────────────────────┐
│   Amazon S3                     │
│   genai-models-<ACCOUNT_ID>     │
│   Ministral-3-8B-Instruct-2512/ │
└─────────────────────────────────┘
```

---

## Quick Start

```bash
# Set environment variables
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"

# Deploy vLLM
kubectl apply -f vllm-deployment.yml

# Deploy Open WebUI + Ingress
kubectl apply -f openwebui.yml

# Deploy Grafana Ingress
kubectl apply -f grafana-ingress.yaml

# Install DCGM Exporter
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm install --generate-name gpu-helm-charts/dcgm-exporter \
  -f dcgm-values.yaml -n monitoring

# Apply ServiceMonitor
kubectl apply -f vllm-servicemonitor.yaml

# Watch pod start (model loading takes 2-4 min)
kubectl get pods -w

# Get the Open WebUI URL
kubectl get ingress open-webui-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Test the API
curl http://$(kubectl get svc vllm-serve-svc -o jsonpath='{.spec.clusterIP}'):8000/v1/models
```
