# Module 100 – vLLM: Your First Online Inference Server

> New to AI inference? Read [CONCEPTS.md](../CONCEPTS.md) first.
> This module is the heart of the workshop – everything else builds on it.

---

## What This Module Does

This module answers the question: **"How do I take a trained AI model and let people talk to it?"**

It deploys **vLLM** – an open-source inference server – on a GPU node in Kubernetes, loads the Ministral-3-8B model from S3, and exposes two things:
1. An **OpenAI-compatible REST API** that any application can call
2. A **browser-based chat interface** (Open WebUI) so you can talk to it immediately

This is **online inference** – the model responds in real time while you wait.

```
You type a message in the browser
        │
        ▼
Open WebUI (chat interface)
        │  POST /v1/chat/completions
        ▼
vLLM inference server (GPU node)
        │  runs the model
        ▼
Ministral-3-8B-Instruct-2512
        │  generates tokens one by one
        ▼
Response streams back to your browser
```

---

## AI Inference Concepts Demonstrated Here

### 1. Online Inference (Real-Time)

This is what the Red Hat article calls "dynamic inference" – the model responds while you wait. ChatGPT is an example. This workshop's Open WebUI is another.

The challenge: **latency**. Every millisecond the user waits feels long. This is why vLLM exists – to make every millisecond count.

### 2. Single-Model Inference Server

vLLM in this module runs one model (Ministral-3-8B) and specialises entirely around it. This means:
- Maximum efficiency for that model
- All GPU memory dedicated to it
- Optimised batching for its specific architecture

### 3. PagedAttention – How vLLM Eliminates Wasted GPU Memory

This is the single most important innovation in vLLM. Here is a simple explanation:

```
THE PROBLEM:
When you start a conversation, vLLM doesn't know how long it will be.
Naive servers reserve a large block of GPU memory upfront "just in case."
If your message is short, most of that block sits empty → wasted.

User A (short message): [████░░░░░░░░░░░░░░░░░░░░░░░░] 25% used, 75% wasted
User B (short message): [██░░░░░░░░░░░░░░░░░░░░░░░░░░] 12% used, 88% wasted
User C: REJECTED – not enough contiguous free memory

THE SOLUTION – PagedAttention:
GPU memory is split into small "pages" (like RAM pages in an OS).
Pages are assigned only as needed, never pre-allocated.

Page pool:  [free][free][free][free][free][free][free][free]

User A starts: give 1 page
  [  A ][free][free][free][free][free][free][free]

User B starts: give 1 page
  [  A ][  B ][free][free][free][free][free][free]

User C starts: give 1 page
  [  A ][  B ][  C ][free][free][free][free][free]

Users continue talking: give more pages as needed
  [  A ][  B ][  C ][  A ][  B ][  C ][  A ][  B ]

All 3 users fit! Zero waste!
```

### 4. Continuous Batching

Without continuous batching, the server processes one request at a time – like a cashier who helps one customer, then calls the next. GPU sits idle between requests.

With continuous batching, multiple users' tokens are processed **simultaneously** in one GPU pass – like a cashier who rings up several items from different customers at once.

```
Time →  ████████████████████████████████
        ─────────────────────────────────
No CB:  [User A ][idle][User B][idle][User C]
With CB:[A+B+C simultaneously processed  ]
        
Result: 2-3× higher throughput, same GPU
```

---

## Full Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        EKS Cluster – default namespace                        │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GPU Node (g6e.2xlarge) – NVIDIA L40S 48 GB VRAM                     │   │
│  │  Taint: nvidia.com/gpu:NoSchedule                                    │   │
│  │  Label: karpenter.sh/nodepool=gpu                                    │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Deployment: mistral (replicas: 1, strategy: Recreate)          │  │   │
│  │  │  ServiceAccount: model-storage-sa (S3 access via Pod Identity)  │  │   │
│  │  │                                                                  │  │   │
│  │  │  ┌────────────────────────────────────────────────────────┐    │  │   │
│  │  │  │  Container: vllm                                         │    │  │   │
│  │  │  │  Image: public.ecr.aws/deep-learning-containers/         │    │  │   │
│  │  │  │    vllm:0.21.0-gpu-py312-cu130-ubuntu22.04-ec2-v1.0-soci│    │  │   │
│  │  │  │                                                          │    │  │   │
│  │  │  │  Port: 8000 (OpenAI-compatible HTTP API)                │    │  │   │
│  │  │  │                                                          │    │  │   │
│  │  │  │  Startup: RunAI Streamer loads model from S3            │    │  │   │
│  │  │  │  Ready: /v1/models returns HTTP 200 (~2-3 min)          │    │  │   │
│  │  │  │                                                          │    │  │   │
│  │  │  │  Resources:                                              │    │  │   │
│  │  │  │    requests: cpu=2, memory=16Gi, nvidia.com/gpu=1       │    │  │   │
│  │  │  │    limits:   cpu=4, memory=28Gi, nvidia.com/gpu=1       │    │  │   │
│  │  │  └────────────────────────────────────────────────────────┘    │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                          │                                            │   │
│  │   Service: vllm-serve-svc (ClusterIP, port 8000)                     │   │
│  │   Labels: model=mistral (used by ServiceMonitor for scraping)        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                          │ HTTP :8000                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  System Node (m5.xlarge) – CPU only                                   │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Deployment: open-webui (replicas: 1)                           │  │   │
│  │  │  Image: ghcr.io/open-webui/open-webui:v0.11.0                  │  │   │
│  │  │  Port: 8080  Resources: cpu=500m–1000m, memory=500Mi–1Gi       │  │   │
│  │  │                                                                  │  │   │
│  │  │  Talks to vLLM via: OPENAI_API_BASE_URLS                        │  │   │
│  │  │    = http://vllm-serve-svc:8000/v1                              │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                          │                                            │   │
│  │   Service: open-webui (ClusterIP, port 80 → 8080)                   │   │
│  │   Ingress: open-webui-ingress (ALB, internet-facing) ◀── YOU        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  monitoring namespace                                                 │   │
│  │  ServiceMonitor: mistral-monitor → scrapes /metrics every 30s       │   │
│  │  Ingress: grafana-ingress (ALB) ◀── Grafana dashboards              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## vLLM Configuration: Every Parameter Explained

File: `vllm-deployment.yml`

### Container Image

```
image: public.ecr.aws/deep-learning-containers/
       vllm:0.21.0-gpu-py312-cu130-ubuntu22.04-ec2-v1.0-soci

Breakdown:
  public.ecr.aws/deep-learning-containers/  ← AWS pre-built AI containers (free)
  vllm:0.21.0                               ← vLLM version 0.21.0
  gpu                                        ← GPU-enabled build
  py312                                      ← Python 3.12
  cu130                                      ← CUDA 13.0 (matches L40S drivers)
  ubuntu22.04                                ← Ubuntu base OS
  ec2-v1.0                                   ← AWS EC2 optimised
  soci                                        ← Seekable OCI (lazy image pull)
                                               SOCI lets Kubernetes start pulling
                                               the container before it's fully
                                               downloaded → faster pod startup
```

### vLLM Command-Line Arguments (Every flag explained)

```bash
# MODEL LOADING
--port=8000
  # The HTTP port vLLM listens on for API requests.
  # All clients (OpenWebUI, Strands Agent, benchmarks) connect here.

--model=s3://genai-models-<ACCOUNT_ID>/Ministral-3-8B-Instruct-2512/
  # Where to find model weights. vLLM supports local paths AND S3 URIs.
  # Credentials come from the Pod Identity (model-storage-sa ServiceAccount).
  # No AWS keys needed in the YAML.

--served-model-name=ministral
  # The name clients use in API calls: { "model": "ministral" }
  # Without this, clients would need to use the full S3 path as the model name.
  # Acts as a human-friendly alias.

--load-format=runai_streamer
  # HOW to load the model weights.
  # runai_streamer = stream directly from S3 to GPU VRAM with 16 parallel
  # threads. No disk write, no CPU copy. ~2-3x faster than default loading.
  # Default (auto) would download to disk first → much slower.

--model-loader-extra-config={"concurrency":16}
  # Extra config for the runai_streamer loader.
  # concurrency:16 = 16 simultaneous S3 GET requests.
  # Higher = faster loading (up to S3 bandwidth limit).
  # Too high can hit S3 rate limits; 16 is a safe sweet spot.

--tokenizer_mode=mistral
  # Tells vLLM to use Mistral's native tokenizer backend.
  # The Ministral model uses the Tekken tokenizer (131k vocabulary).
  # Without this flag, vLLM falls back to the HuggingFace AutoTokenizer
  # which cannot fully handle Tekken's encoding rules.

--trust-remote-code
  # Allows vLLM to execute custom Python code bundled with the model.
  # Needed for Mistral-3 which has custom model class code.
  # Only enable this for models from trusted sources.

--config-format=mistral
  # The model's config file format. Ministral uses params.json (Mistral format)
  # rather than config.json (HuggingFace format).
  # Without this, vLLM cannot read the model architecture correctly.

# MEMORY MANAGEMENT
--gpu_memory_utilization=0.90
  # What fraction of GPU VRAM vLLM may use.
  # 0.90 = 90% of 48 GB = 43.2 GB available for model + KV cache.
  # Remaining 10% (~4.8 GB) reserved for CUDA runtime, activations.
  # Higher = more KV cache space = more concurrent users.
  # Too high = OOM (Out of Memory) crashes.
  # 0.90 is safe for Ministral-3-8B on the L40S.

--max-model-len=8192
  # Maximum total tokens in one conversation (prompt + response combined).
  # 8192 = Ministral's native context window.
  # Longer contexts = larger KV cache entries = fewer concurrent users.
  # You can reduce this (e.g., 4096) to serve more users with shorter contexts.

--block-size=16
  # Size of each KV cache memory block in tokens (PagedAttention setting).
  # 16 tokens per block = fine-grained memory allocation.
  # Default vLLM value. Smaller blocks = less waste but more overhead.
  # Larger blocks = less overhead but more potential waste.

# THROUGHPUT / BATCHING
--max-num-batched-tokens=8192
  # Maximum tokens processed in ONE scheduler step across ALL requests.
  # 8192 = allows one full-context request or many shorter ones simultaneously.
  # This controls how much the GPU does per "tick" of the scheduler.
  # Higher = higher throughput, higher memory pressure per step.

--max-num-seqs=256
  # Maximum number of sequences (requests) the scheduler handles at once.
  # 256 = up to 256 simultaneous conversations.
  # Limited by KV cache size. If you lower gpu_memory_utilization, lower this.
  # This is the key number for concurrent user capacity.

# COMPATIBILITY
--enforce-eager
  # Disables CUDA Graph capture.
  # CUDA Graphs pre-compile kernel launch sequences for speed.
  # Some Mistral model variants have variable computation graphs that
  # are incompatible with CUDA Graphs. enforce-eager = always use
  # standard (eager) CUDA execution. Slightly slower but more stable.

--disable-custom-all-reduce
  # Disables vLLM's custom NCCL all-reduce kernels.
  # all-reduce is a multi-GPU communication operation.
  # With tensor-parallel-size=1 (single GPU), this is irrelevant anyway.
  # Disabling prevents potential NCCL version conflicts.

# TOOL CALLING (for AI Agents - Module 200)
--enable-auto-tool-choice
  # Enables automatic detection of tool call intent in model output.
  # When the model decides to call a function, vLLM intercepts and
  # structures the output as a tool call JSON, not raw text.
  # Required for Module 200 (Strands Agent) to work.

--tool-call-parser=mistral
  # Mistral models output tool calls using [TOOL_CALLS] tokens in a
  # specific JSON format. This parser knows how to extract:
  #   - Which function to call
  #   - What arguments to pass
  # Without this, tool call output is raw text the agent cannot parse.
```

### Environment Variables

```yaml
env:
  - name: PYTORCH_CUDA_ALLOC_CONF
    value: "max_split_size_mb:512"
    # Controls PyTorch's CUDA memory allocator.
    # max_split_size_mb:512 = never split a free memory block larger than 512 MB.
    # This reduces memory fragmentation in long-running servers.
    # Without it, the allocator may split large blocks into small fragments
    # that cannot satisfy large allocation requests → OOM even with free memory.

  - name: VLLM_ATTENTION_BACKEND
    value: "FLASHINFER"
    # Selects the attention computation backend.
    # Options: FLASHINFER, FLASH_ATTN, XFORMERS, TORCH_SDPA
    # FLASHINFER = custom optimised attention kernels by FlashInfer project.
    # On NVIDIA Ada Lovelace (L40S) architecture, FLASHINFER is the fastest.
    # Produces correct outputs; this is a pure performance choice.
```

### Resource Requests and Limits

```yaml
resources:
  requests:
    cpu: 2          # Minimum CPUs guaranteed by Kubernetes
                    # vLLM uses CPU for tokenisation, scheduling, networking
    memory: 16Gi    # Minimum RAM guaranteed
                    # Used for: model metadata, CPU buffers, Python process
    nvidia.com/gpu: 1   # Exactly 1 GPU (whole GPU, not fractional)
                        # vLLM is not designed for GPU sharing/MIG
  limits:
    cpu: 4          # Maximum CPUs allowed (burst up to 4 during heavy load)
    memory: 28Gi    # Maximum RAM allowed
    nvidia.com/gpu: 1   # Same as requests - GPU is not overcommitted
```

### Readiness Probe

```yaml
readinessProbe:
  httpGet:
    path: /v1/models    # Endpoint that returns HTTP 200 only when model is loaded
    port: http          # Port 8000
  periodSeconds: 5      # Check every 5 seconds
  timeoutSeconds: 2     # Timeout each check after 2 seconds
  failureThreshold: 3   # Mark pod NotReady after 3 consecutive failures

# What this means in practice:
# 1. Pod starts, vLLM begins loading model from S3
# 2. /v1/models returns HTTP 503 while loading
# 3. After ~2-3 minutes, model is in GPU VRAM
# 4. /v1/models returns HTTP 200 → pod goes Ready
# 5. Kubernetes now sends real traffic to this pod
# 6. Open WebUI can connect and users can chat
```

---

## Open WebUI Configuration

File: `openwebui.yml`

```yaml
# What Open WebUI is:
# A browser-based chat interface compatible with OpenAI's API format.
# Think of it as a self-hosted ChatGPT interface that points to YOUR model.

image: ghcr.io/open-webui/open-webui:v0.11.0

env:
  OPENAI_API_BASE_URLS: "http://vllm-serve-svc:8000/v1"
  # Points WebUI to our vLLM server (by Kubernetes DNS name).
  # vllm-serve-svc resolves to the ClusterIP Service inside the cluster.
  # /v1 is the OpenAI API prefix vLLM uses.

  OPENAI_API_KEY: "dummy"
  # vLLM does not validate API keys by default.
  # Open WebUI requires a non-empty value, so we provide a placeholder.
  # To add real authentication, add --api-key=<secret> to vLLM args.

  WEBUI_AUTH: "False"
  # Disables Open WebUI's own login screen for workshop simplicity.
  # In production, set to "True" and create user accounts.

  ENABLE_OLLAMA_API: "False"
  # Open WebUI supports both OpenAI and Ollama APIs.
  # We disable Ollama since we only have a vLLM (OpenAI-compatible) server.

  ENABLE_EVALUATION_ARENA_MODELS: "False"
  # Disables the model comparison arena feature.
  # We only have one model; no need for head-to-head comparison UI.

# Node placement:
nodeSelector:
  node.kubernetes.io/instance-type: "m5.xlarge"
# Open WebUI is a web app, not AI compute.
# It runs on cheap CPU nodes, NOT on the expensive GPU node.
# The GPU is 100% dedicated to running the model.
```

---

## API Endpoints Reference

```
Base URL (internal):  http://vllm-serve-svc:8000
Base URL (external):  kubectl port-forward svc/vllm-serve-svc 8000:8000

┌────────────────────────────────────────────────────────────────────────┐
│  GET /v1/models                                                         │
│  Returns: { "data": [{ "id": "ministral", "object": "model" }] }      │
│  Use: Verify the model is loaded and ready                             │
├────────────────────────────────────────────────────────────────────────┤
│  POST /v1/chat/completions    ← Most common endpoint (chat format)     │
│  Body: {                                                                │
│    "model": "ministral",                                                │
│    "messages": [                                                        │
│      { "role": "system",    "content": "You are a helpful assistant" } │
│      { "role": "user",      "content": "Explain LLMs simply" }         │
│    ],                                                                   │
│    "temperature": 0.7,     ← 0=deterministic, 1=creative, 2=random   │
│    "max_tokens": 500,       ← Maximum response length in tokens        │
│    "stream": true           ← Stream tokens as they are generated      │
│  }                                                                      │
├────────────────────────────────────────────────────────────────────────┤
│  POST /v1/completions         ← Legacy endpoint (text completion)      │
│  Body: {                                                                │
│    "model": "ministral",                                                │
│    "prompt": "The capital of France is",                               │
│    "max_tokens": 10                                                     │
│  }                                                                      │
├────────────────────────────────────────────────────────────────────────┤
│  GET  /metrics                ← Prometheus metrics                     │
│  GET  /health                 ← Liveness probe (returns 200 if running)│
└────────────────────────────────────────────────────────────────────────┘
```

---

## Key Prometheus Metrics (What to Watch)

ServiceMonitor (`vllm-servicemonitor.yaml`) scrapes these from `/metrics`:

```
vllm:num_requests_running
  How many requests are being generated RIGHT NOW.
  Normal: 0–50 depending on load.
  If stuck at max-num-seqs: your server is saturated.

vllm:num_requests_waiting
  Requests in the queue, waiting for a free slot.
  If this grows: add more GPU replicas (Module 800).

vllm:gpu_cache_usage_perc
  What % of the KV cache is currently in use.
  0% = no users. 100% = all cache pages occupied.
  If consistently > 90%: consider reducing max-model-len
  or adding Module 400 (LMCache) to offload KV.

vllm:e2e_request_latency_seconds
  Histogram of end-to-end request latency.
  Use: track P50/P95/P99 latency as load increases.

vllm:time_to_first_token_seconds
  TTFT: Time from request received to first token generated.
  This is what the user perceives as "response time."
  Acceptable: < 1 second. Concerning: > 3 seconds.

vllm:avg_generation_throughput
  Tokens generated per second (aggregate, all users).
  Benchmark target: see Module 300.
```

---

## Monitoring: ServiceMonitor

File: `vllm-servicemonitor.yaml`

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mistral-monitor
  namespace: monitoring
  labels:
    release: kube-prometheus-stack
    # ↑ This label is critical. The Prometheus Operator only picks up
    #   ServiceMonitors that have this exact label. Without it,
    #   Prometheus never discovers this scrape target.

spec:
  namespaceSelector:
    matchNames: [default]      # Look for Services in the "default" namespace

  selector:
    matchLabels:
      model: mistral           # Match the Service that has label model=mistral
                               # This is set on vllm-serve-svc

  endpoints:
    - port: http               # Scrape the port named "http" (port 8000)
      interval: 30s            # Scrape every 30 seconds
      path: /metrics           # vLLM's Prometheus metrics endpoint
```

---

## DCGM Exporter Values

File: `dcgm-values.yaml`

```yaml
serviceMonitor:
  enabled: true
  additionalLabels:
    release: kube-prometheus-stack  # Prometheus Operator discovery label
  interval: 30s                     # GPU metric scrape frequency
  honorLabels: true                 # Keep labels from GPU pod

service:
  enable: true
  type: ClusterIP                   # Internal only
  port: 9400                        # Prometheus scrape port

nodeSelector:
  karpenter.sh/nodepool: gpu        # Only run on GPU nodes

tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"            # Must tolerate the GPU taint

resources:
  limits:   { cpu: 500m, memory: 512Mi }
  requests: { cpu: 100m, memory: 256Mi }
```

---

## Data Flow: User Message to Response

```
User types: "What is quantum computing?"
      │
      ▼  Browser POST to ALB
┌─────────────────────────────────────┐
│  AWS ALB: open-webui-ingress         │
│  Routes to: open-webui Service :80  │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  Open WebUI Pod (m5.xlarge)          │
│  Forwards: POST /v1/chat/completions │
│  To: http://vllm-serve-svc:8000/v1  │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  vLLM Server (GPU Node, L40S 48 GB)                          │
│                                                              │
│  1. Tokenise: "What is quantum computing?"                  │
│     → [1, 1724, 349, 12256, 21237, 28804]  (6 tokens)      │
│                                                              │
│  2. PagedAttention: assign KV cache pages for this request  │
│                                                              │
│  3. Batch with other active requests (continuous batching)  │
│                                                              │
│  4. Forward pass through Ministral-3-8B layers              │
│     (transformer attention + feed-forward × 32 layers)      │
│     → logits for next token probability distribution        │
│                                                              │
│  5. Sample next token (temperature=0.7)                     │
│     → "Quantum" (token ID: 28984)                           │
│                                                              │
│  6. Stream token back → repeat from step 4 until EOS       │
│     "Quantum" → "computing" → "is" → "a" → ...             │
└─────────────────────────────────────────────────────────────┘
      │ Server-Sent Events (streaming)
      ▼
Open WebUI renders tokens as they arrive (typewriter effect)
```

---

## Quick Start

```bash
# Set variables
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"

# Deploy vLLM inference server
kubectl apply -f 100-vllm/vllm-deployment.yml

# Deploy Open WebUI chat interface
kubectl apply -f 100-vllm/openwebui.yml

# Deploy Grafana public ingress
kubectl apply -f 100-vllm/grafana-ingress.yaml

# Apply vLLM Prometheus ServiceMonitor
kubectl apply -f 100-vllm/vllm-servicemonitor.yaml

# Watch the pod come up (takes 2-4 min while model loads from S3)
kubectl get pods -w

# Get the chat UI URL
kubectl get ingress open-webui-ingress \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Test the API directly
curl http://localhost:8000/v1/models  # after port-forward
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ministral","messages":[{"role":"user","content":"Hello!"}]}'
```
