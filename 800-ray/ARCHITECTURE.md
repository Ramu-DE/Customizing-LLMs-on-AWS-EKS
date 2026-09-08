# Module 800 – Ray Serve: Distributed and Autoscaling AI Inference

> New to AI inference? Read [CONCEPTS.md](../CONCEPTS.md) first.
> This module is the most advanced. Prerequisites: Module 100 (vLLM) concepts understood.

---

## What This Module Does

Module 100 ran one vLLM server on one GPU. That is fine for a workshop with a few users. In production, you need:
- **Autoscaling**: automatically add GPU replicas when demand spikes, remove them when idle
- **Distributed inference**: spread computation across multiple GPUs for larger models
- **Observability**: rich metrics on queue depth, replica count, and throughput

This module deploys vLLM as a **Ray Serve application** – a distributed serving framework that turns your single-GPU vLLM into a horizontally-scalable, self-managing inference cluster.

---

## AI Inference Concepts Demonstrated Here

### Distributed Inference (The Red Hat Article Concept)

From the Red Hat article:
> "Distributed inference lets AI models process workloads more efficiently by dividing the labor of inference across a group of interconnected devices. Think of it as the software equivalent of the saying 'many hands make light work.'"

There are two distinct ways to distribute inference:

```
Type 1: Tensor Parallelism (one BIG model across multiple GPUs)
──────────────────────────────────────────────────────────────
Use case: Model weights DON'T FIT on one GPU
Example: Llama-3-70B requires 140 GB; one L40S only has 48 GB

GPU 0 [Layers 0–15]  ──┐
GPU 1 [Layers 16–31] ──┤── All GPUs cooperate on EVERY token
GPU 2 [Layers 32–47] ──┤   via NVLink/InfiniBand communication
GPU 3 [Layers 48–63] ──┘

vLLM flag: --tensor-parallel-size=4
Ray worker: num-gpus=4 (with g6e.12xlarge = 4× L40S)

Type 2: Replica Scaling (multiple copies of the SAME model)
────────────────────────────────────────────────────────────
Use case: Model fits on one GPU, but demand exceeds one GPU's capacity

GPU 0 [Full Ministral-3-8B] ← serves User A, C, E
GPU 1 [Full Ministral-3-8B] ← serves User B, D, F
GPU 2 [Full Ministral-3-8B] ← serves overflow (auto-provisioned)

Ray Serve handles the load balancing and autoscaling automatically.
This workshop demonstrates Type 2 (replica scaling).
```

### Autoscaling: The Solution to the Cost Challenge

From the Red Hat article, cost is one of the three biggest challenges:
> "Whether your goal is to scale or to transition to the latest AI-supported hardware, the resources it takes can be extensive."

Ray Serve's autoscaler directly addresses this:

```
Without autoscaling (Module 100 approach):
  Traffic spike at 2 PM: 1 GPU is overloaded, users get slow responses
  Quiet at 3 AM: 1 GPU sits idle, you pay ~$3/hr for nothing

With Ray Serve autoscaling:
  2 PM: 5 concurrent users → scale from 1 to 3 replicas automatically
  3 AM: 0 users → scale down to 1 replica (min_replicas=1 keeps 1 warm)

  Cost: You pay for compute only when it's used
  Latency: No cold start for the first user (min_replicas=1)
```

### How Ray Serve Autoscaling Works

```
The autoscaler watches: average ongoing requests per replica

target_ongoing_requests: 2   ← The target (desired) queue depth per replica
max_ongoing_requests: 5      ← Hard cap before new requests queue at the router

Decision logic:
  avg_ongoing = total_in_flight / current_replicas

  If avg_ongoing > target_ongoing_requests:
    Add a replica (scale up)
    New replica = new vLLM server on a new GPU node
    Karpenter provisions the GPU node automatically

  If avg_ongoing < target_ongoing_requests for sustained period:
    Remove a replica (scale down)
    Karpenter terminates the idle GPU node

Example:
  20 concurrent requests, 2 replicas
  avg_ongoing = 20/2 = 10 > target(2)
  → Scale up to 10 replicas
  → Karpenter provisions 8 more g6e.2xlarge nodes
```

---

## Full Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EKS Cluster – default namespace                                              │
│                                                                               │
│  RayService CRD: vllm  (managed by KubeRay Operator)                        │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Ray Head Pod  (GPU node, g6e.2xlarge)                                │   │
│  │  Ray Version: 2.56.1                                                  │   │
│  │                                                                       │   │
│  │  Ports:                                                               │   │
│  │    6379   GCS (Global Control Service) – cluster state & scheduling  │   │
│  │    8265   Ray Dashboard (web UI + REST API)                          │   │
│  │    10001  Ray Client (Python SDK connections)                         │   │
│  │    8000   Ray Serve HTTP endpoint (OpenAI-compatible API)             │   │
│  │    52365  Dashboard Agent (health checks)                             │   │
│  │    8080   Prometheus metrics                                          │   │
│  │    44217  Autoscaler metrics                                         │   │
│  │    44227  Dashboard metrics                                           │   │
│  │                                                                       │   │
│  │  Resources: { cpu: 0.5-1, memory: 4-6 Gi }  NO GPU (head is CPU)    │   │
│  │  Startup: installs ray[llm]==2.56.1, vllm==0.22.0, then starts GCS  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                            │ Ray Cluster Bus (:6379)                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Ray Worker Pod  (GPU node, g6e.2xlarge)                              │   │
│  │  replicas: min=1, max=1  (set max higher for real autoscaling)        │   │
│  │                                                                       │   │
│  │  numCPUs: 2  numGPUs: 1                                              │   │
│  │  Resources: { cpu: 2-3, memory: 20-22 Gi, nvidia.com/gpu: 1 }       │   │
│  │                                                                       │   │
│  │  /dev/shm (Memory emptyDir 10Gi): Shared memory for tensor comms     │   │
│  │  /tmp/ray (emptyDir): Ray object store + logs                        │   │
│  │                                                                       │   │
│  │  Runs: vLLM engine (ray.serve.llm:build_openai_app)                  │   │
│  │  Handles: actual token generation for user requests                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  Service: vllm (headService – routes :8000 to Ray Serve HTTP proxy)         │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Open WebUI (m5.xlarge, CPU) → ALB Ingress ← users browse here      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  PodMonitor: ray-head-monitor  (scrapes head metrics ports)          │   │
│  │  PodMonitor: ray-workers-monitor  (scrapes worker metrics)           │   │
│  │  Both in monitoring namespace, label: release=kube-prometheus-stack  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## RayService Configuration: Every Parameter Explained

File: `ray-vllm-service.yaml`

### Serve Configuration

```yaml
serveConfigV2: |
  applications:
    - name: mistral
      # Name of this Serve application. Appears in the Ray Dashboard.
      # Can run multiple applications on the same Ray cluster.

      import_path: ray.serve.llm:build_openai_app
      # Ray Serve v2.56.1 includes a built-in LLM serving function.
      # build_openai_app: wraps vLLM with OpenAI-compatible routing,
      # health checking, and autoscaling. No custom code needed.

      route_prefix: "/"
      # All HTTP requests to / are routed to this application.
      # To run multiple models: set different route prefixes.
      # e.g., /ministral/, /llama/

      args:
        llm_configs:
          - model_loading_config:
              model_id: ministral
              # Name used in API requests: { "model": "ministral" }
              # Same as --served-model-name in direct vLLM deployment.

              model_source: s3://${S3_BUCKET_NAME}/Ministral-3-8B-Instruct-2512/
              # S3 URI for model weights.
              # Authenticated via model-storage-sa Pod Identity.
```

### Engine Configuration

```yaml
engine_kwargs:
  dtype: bfloat16
  # Data type for model weights during inference.
  # bfloat16 = 2 bytes per parameter, good precision/speed balance.
  # Other options: float16 (slightly faster, less numerically stable),
  #                float32 (double the memory, highest precision)
  # BF16 is recommended for modern NVIDIA GPUs (Ampere/Ada Lovelace).

  max_model_len: 8192
  # Maximum total tokens (prompt + response).
  # Matches Ministral's native 8K context window.
  # Larger = more KV cache per request = fewer concurrent users.

  gpu_memory_utilization: 0.9
  # 90% of 48 GB = 43.2 GB available for model weights + KV cache.
  # Same as the direct vLLM deployment (Module 100).

  trust_remote_code: true
  # Required for Mistral-3 model class.
  # Allows execution of model-bundled Python code.

  enable_chunked_prefill: true
  # Splits long prefill computations across multiple scheduler iterations.
  #
  # Without chunked prefill:
  #   Long prompt (4096 tokens) → one giant prefill step
  #   Blocks the GPU for ~500ms → all other users' TTFT spikes
  #
  # With chunked prefill:
  #   Long prompt → multiple smaller prefill steps interleaved with decoding
  #   Other users' decoding continues while long prefill progresses
  #   Result: fairer TTFT distribution under concurrent load
  #
  # Critical for production fairness when users send variable-length prompts.

  max_num_seqs: 4
  # Maximum sequences the vLLM engine processes simultaneously.
  # Intentionally low (4) in this workshop.
  # WHY: Works WITH the Ray Serve autoscaler:
  #   - Each replica handles max 4 sequences at once
  #   - max_ongoing_requests: 5 ensures no more than 5 queue at the replica
  #   - When avg_ongoing > 2: autoscaler adds more replicas
  #   - This creates visible queue pressure for the Grafana dashboard demo
  #
  # In production: increase to 256 (matching direct vLLM) and raise
  # target_ongoing_requests accordingly.
```

### Deployment (Autoscaling) Configuration

```yaml
deployment_config:
  autoscaling_config:
    min_replicas: 1
    # Always keep at least 1 replica running.
    # Prevents cold start for the first request.
    # In the workshop: this means 1 GPU node always running.
    # For overnight cost savings: set to 0 (first request takes ~3 min)

    max_replicas: 1
    # Maximum replicas to scale up to.
    # Set to 1 for this workshop (cost management).
    # For real production: set to match your SLA capacity
    # e.g., max_replicas: 10 → up to 10 GPU nodes, 10 Ministral instances

    target_ongoing_requests: 2
    # Ray Serve watches: avg ongoing requests per replica
    # When avg > 2: scale up (add more replicas)
    # When avg < 2 for sustained period: scale down
    # With max_num_seqs=4 and max_ongoing_requests=5:
    #   2 users → 1 replica (comfortable)
    #   5 users → scale up to 3 replicas (2 per replica = target met)

  max_ongoing_requests: 5
  # Hard limit on requests queued at THIS replica.
  # Requests above 5 are queued at the Ray Serve router (not at this replica).
  # The router's queue triggers autoscaling.
  # At 5: new requests wait at router → avg_ongoing rises → autoscaler acts.
  #
  # Relationship:
  #   max_num_seqs: 4       Engine processes this many simultaneously
  #   max_ongoing_requests: 5  Router sends at most this many at once
  #   target_ongoing: 2        Autoscaler fires when avg exceeds this
```

---

## Ray Cluster Components

### Head Node: The Control Plane

```bash
# What runs on the head node:

python3 -m pip install -q 'ray[llm]==2.56.1' 'vllm==0.22.0'
# ray[llm] = Ray core + Ray Serve + LLM utilities
# vllm==0.22.0 NOTE: this is 0.22.0, not 0.21.0 as in Module 100.
#              Ray Serve's build_openai_app is designed for 0.22.0.

ray start \
  --head \
  --dashboard-host=0.0.0.0 \    # Listen on all interfaces (required in k8s)
  --dashboard-port=8265 \        # Web dashboard port
  --num-cpus=1 \                 # Head claims 1 CPU for itself
  --num-gpus=0 \                 # Head claims NO GPU
                                 # WHY: GPU resources are for worker pods only.
                                 # Head node doing GPU work would conflict.
  --metrics-export-port=8080 \  # Prometheus metrics endpoint
  --block \                      # Keep process in foreground (required in containers)
  --dashboard-agent-listen-port=52365 \  # Health check port
  --no-monitor                   # Disable built-in autoscaler (Ray Serve handles it)
```

### Worker Node: The GPU Inference Engine

```bash
# Build tools required for vLLM compilation on RHEL-based container:
dnf install -y -q gcc gcc-c++ make which

python3 -m pip install -q 'ray[llm]==2.56.1' 'vllm==0.22.0'

ray start \
  --address=${RAY_ADDRESS:-${FQ_RAY_IP}:6379} \  # Connect to head GCS
  --metrics-export-port=8080 \   # Worker's own Prometheus metrics
  --block \
  --num-cpus=2 \                 # Worker claims 2 CPUs
  --num-gpus=1 \                 # Worker claims 1 GPU (the L40S)
                                 # Ray Serve vLLM uses this 1 GPU
  --no-monitor

# Worker-specific environment:
CC=/usr/bin/gcc
# Tells build tools where to find the C compiler.
# Required for some vLLM extension compilation steps.

VLLM_USE_FLASHINFER_SAMPLER=0
# Disable FlashInfer sampling kernel in this Ray variant.
# Compatibility setting for vLLM 0.22.0 on this container image.
```

### Shared Memory Volume

```yaml
volumes:
  - name: shm
    emptyDir:
      medium: Memory    # RAM-backed tmpfs, not disk
      sizeLimit: 10Gi   # 10 GB of shared memory

mountPath: /dev/shm

# WHY 10 GB shared memory?
# /dev/shm is used for:
#   1. PyTorch inter-process tensor sharing (multi-worker data loading)
#   2. NCCL communication buffers (for multi-GPU, even on single node)
#   3. Ray's object store (for passing tensors between Ray actors)
#
# Without this: default /dev/shm is 64 MB → NCCL OOM → crashes
# Container default /dev/shm is much smaller than bare-metal default (typically 64 MB)
# 10 Gi ensures ample space for all IPC operations.
```

---

## Monitoring: PodMonitors

Unlike Module 100 which uses a ServiceMonitor, Ray Serve needs PodMonitors because metrics come from multiple ports on the same pod.

```yaml
PodMonitor: ray-head-monitor
  selector: ray.io/node-type=head    # Targets only head pods

  podMetricsEndpoints:
    - port: metrics     # port 8080 – vLLM engine metrics + Ray Serve metrics
      relabelings:
        - sourceLabels: [__meta_kubernetes_pod_label_ray_io_cluster]
          targetLabel: ray_io_cluster    # Add cluster name label to all metrics

    - port: as-metrics  # port 44217 – Autoscaler-specific metrics
                        # Tracks scale-up/down events, replica count changes
      relabelings: ...

    - port: dash-metrics # port 44227 – Dashboard metrics
                         # Tracks object store size, worker node health
      relabelings: ...
```

### Key Ray Serve Metrics to Watch

```
serve_deployment_request_counter_total
  Total requests received by the deployment.
  Watch per deployment name for traffic distribution.

serve_num_ongoing_requests  (GAUGE)
  Current in-flight requests at a replica.
  Watch this to understand actual load per replica.
  Should stay near target_ongoing_requests (2) when scaling works.

serve_num_pending_requests  (GAUGE)
  Requests queued at the Ray router, not yet assigned to a replica.
  Growing queue = autoscaler is about to add a replica.
  This is the key metric for observing autoscaling in action.

serve_deployment_replica_starts_total
  Counter of replica additions.
  Each increment = one new vLLM instance started on a new GPU.

# vLLM metrics (same as Module 100):
vllm:num_requests_running
vllm:num_requests_waiting
vllm:gpu_cache_usage_perc
vllm:e2e_request_latency_seconds
vllm:time_to_first_token_seconds
```

---

## Load Generation Script (generate-load.sh)

This script creates enough concurrent requests to make the autoscaling metrics visible in Grafana.

```bash
# Why you need this script (and can't just use the chat UI):
# The vLLM engine batches at most max_num_seqs=4 simultaneously.
# Ray Serve routes at most max_ongoing_requests=5 to each replica.
# Nothing queues until > 5 requests arrive AT THE SAME TIME.
# A human typing in chat has 1 request in flight at a time.
# Queue depth stays 0 → dashboard panels are flat.

# Usage:
bash 800-ray/generate-load.sh [DURATION_SECONDS] [CONCURRENCY]
# Defaults: 180 seconds, 8 concurrent requests

# Prerequisite: port-forward in another terminal
kubectl port-forward svc/vllm-serve-svc 8000:8000

# What the script does:
# For 180 seconds:
#   Launch 8 concurrent curl requests simultaneously
#   Each request sends a long prompt (expecting ~600 token response)
#   All 8 run in background (& ) then wait (wait)
#   Repeat immediately: 8 more requests
#
# Effect on metrics:
#   serve_num_ongoing_requests → spikes to 5 (max_ongoing_requests)
#   serve_num_pending_requests → grows (requests queue at router)
#   vllm:num_requests_running  → stays at 4 (max_num_seqs)
#   vllm:num_requests_waiting  → grows (engine queue fills up)
#
# Watch these panels in Grafana while script runs.

CONCURRENCY=8  # Above max_num_seqs(4) AND max_ongoing_requests(5)
               # Guarantees visible queue pressure

DURATION=180s  # 3 minutes → enough time for autoscaler to react
               # AND for Prometheus (30s scrape) to capture 6 data points

PROMPT: "Explain how Kubernetes HPA works, step by step, ~400 words"
        # Long enough to take 3-8 seconds per request
        # Creates sustained queue pressure throughout the window
max_tokens: 600  # Forces the model to generate a full long response
```

---

## Module 100 vs Module 800: When to Use Which

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Feature              │  Module 100 (Direct vLLM)   │  Module 800 (Ray Serve)│
├──────────────────────────────────────────────────────────────────────────────┤
│  Setup complexity     │  Simple (1 Deployment)      │  Complex (RayService CRD│
│                       │                             │  + operator required)   │
│  Autoscaling          │  Manual (HPA CPU-based)     │  Built-in, request-based│
│  GPU replicas         │  Fixed                      │  Dynamic (min→max)      │
│  Multi-model          │  Separate deployments       │  Multiple apps, 1 cluster│
│  Load balancing       │  K8s Service (round-robin)  │  Smart (Ray router)     │
│  Queue visibility     │  Limited                    │  Full (num_pending_reqs) │
│  Tensor parallelism   │  Manual flag                │  Integrated with cluster │
│  Rolling updates      │  K8s rolling update         │  Ray Serve deployment    │
│                       │                             │  versioning             │
│  Best for             │  Single-team workshop,      │  Production multi-team  │
│                       │  predictable load           │  autoscaling systems     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"
export AWS_REGION="us-east-1"

# Step 1: Install KubeRay Operator (prerequisite)
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator \
  -n ray-system --create-namespace

# Step 2: Deploy RayService
envsubst < 800-ray/ray-vllm-service.yaml | kubectl apply -f -

# Watch Ray cluster come up (head first, then worker, ~5-8 min)
kubectl get rayservice vllm -w
kubectl get pods -w

# Step 3: Apply Prometheus PodMonitors
kubectl apply -f 800-ray/ray-podmonitor.yaml

# Step 4: Deploy Open WebUI (Ray Serve variant)
kubectl apply -f 800-ray/openwebui.yml

# Step 5: Get Open WebUI URL
kubectl get ingress open-webui-ingress \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Step 6: Test API
kubectl port-forward svc/vllm-serve-svc 8000:8000 &
curl http://localhost:8000/v1/models

# Step 7: Access Ray Dashboard
kubectl port-forward svc/vllm 8265:8265 &
open http://localhost:8265   # or copy URL to browser

# Step 8: Generate load and watch autoscaling metrics
bash 800-ray/generate-load.sh 180 8
# While running: open Grafana → Ray Serve Inference Overview dashboard
# Watch: serve_num_pending_requests climb, serve_num_ongoing_requests spike

# Clean up
kubectl delete rayservice vllm
# Karpenter automatically terminates the idle GPU nodes → billing stops
```
