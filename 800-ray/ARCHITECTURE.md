# Module 800 – Ray Serve + vLLM Architecture

> Deploys Ministral-3-8B-Instruct-2512 as a distributed, autoscaling inference service using Ray Serve's `build_openai_app` helper backed by vLLM. A RayService CRD manages the full Ray cluster lifecycle (head + worker pods) on EKS, with Prometheus monitoring via PodMonitors and load-tested via a provided shell script.

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        EKS Cluster – default namespace                        │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  RayService CRD: vllm  (ray.io/v1)                                    │   │
│  │  Managed by: KubeRay Operator                                          │   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  Ray Head Node  (GPU Node, g6e.2xlarge)                          │ │   │
│  │  │  image: public.ecr.aws/deep-learning-containers/ray:             │ │   │
│  │  │         serve-ml-cuda-v1.3                                       │ │   │
│  │  │  Installs: ray[llm]==2.56.1, vllm==0.22.0                        │ │   │
│  │  │                                                                   │ │   │
│  │  │  Ports:                                                           │ │   │
│  │  │    6379  – GCS (Global Control Service, cluster state)           │ │   │
│  │  │    8265  – Ray Dashboard + REST API                              │ │   │
│  │  │    10001 – Ray Client (Python remote connection)                  │ │   │
│  │  │    8000  – Ray Serve HTTP endpoint (OpenAI API)                   │ │   │
│  │  │    52365 – Dashboard Agent                                        │ │   │
│  │  │    8080  – Prometheus metrics                                     │ │   │
│  │  │    44217 – Autoscaler metrics                                     │ │   │
│  │  │    44227 – Dashboard metrics                                      │ │   │
│  │  │                                                                   │ │   │
│  │  │  Resources:                                                       │ │   │
│  │  │    requests: { cpu: "500m", memory: "4Gi" }                      │ │   │
│  │  │    limits:   { cpu: "1",    memory: "6Gi" }                      │ │   │
│  │  │  (Head does NOT hold a GPU – GPU reserved for workers)           │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  │                                │                                      │   │
│  │          Ray Cluster Bus (GCS: port 6379)                             │   │
│  │                                │                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  Ray Worker Node  (GPU Node, g6e.2xlarge)                        │ │   │
│  │  │  image: public.ecr.aws/deep-learning-containers/ray:             │ │   │
│  │  │         serve-ml-cuda-v1.3                                       │ │   │
│  │  │  Installs: ray[llm]==2.56.1, vllm==0.22.0                        │ │   │
│  │  │                                                                   │ │   │
│  │  │  replicas: min=1, max=1                                           │ │   │
│  │  │  groupName: gpu-group                                             │ │   │
│  │  │  numCPUs: 2  |  numGPUs: 1                                       │ │   │
│  │  │                                                                   │ │   │
│  │  │  Resources:                                                       │ │   │
│  │  │    requests: { cpu: "2",  memory: "20Gi", nvidia.com/gpu: "1" } │ │   │
│  │  │    limits:   { cpu: "3",  memory: "22Gi", nvidia.com/gpu: "1" } │ │   │
│  │  │                                                                   │ │   │
│  │  │  Volumes:                                                         │ │   │
│  │  │    /tmp/ray  – emptyDir (Ray logs)                               │ │   │
│  │  │    /dev/shm  – Memory-backed emptyDir (10Gi)                     │ │   │
│  │  │               (shared memory for NCCL/IPC between processes)     │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                              │
│   Service: vllm (ClusterIP, headService) – proxies to port 8000             │
│                                │                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  System Node Pool (m5.xlarge)                                         │   │
│  │  Deployment: open-webui (v0.11.0) → Service → Ingress (ALB)          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  monitoring namespace                                                 │   │
│  │  PodMonitor: ray-head-monitor                                         │   │
│  │    ports: [metrics:8080, as-metrics:44217, dash-metrics:44227]       │   │
│  │    selector: ray.io/node-type=head                                    │   │
│  │  PodMonitor: ray-workers-monitor                                      │   │
│  │    ports: [metrics:8080]                                              │   │
│  │    selector: ray.io/node-type=worker                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘

Internet
  │
  ▼ HTTP
┌──────────────────────────────┐
│  ALB: open-webui-ingress      │  ← http://<ALB_DNS>  (Chat UI)
│  scheme: internet-facing      │
└──────────────────────────────┘
```

---

## RayService Serve Configuration

```yaml
serveConfigV2: |
  applications:
    - name: mistral
      import_path: ray.serve.llm:build_openai_app
      route_prefix: "/"
      args:
        llm_configs:
          - model_loading_config:
              model_id: ministral
              model_source: s3://${S3_BUCKET_NAME}/Ministral-3-8B-Instruct-2512/
```

### `build_openai_app` Parameters

```
model_id: ministral
    Served model name. Clients use "model": "ministral" in API requests.
    Same as vLLM's --served-model-name in Module 100.

model_source: s3://${S3_BUCKET_NAME}/Ministral-3-8B-Instruct-2512/
    S3 URI for model weights.
    Authenticated via EKS Pod Identity (model-storage-sa → IAM role).
```

### Engine Configuration (engine_kwargs)

```
dtype: bfloat16
    Model weights precision. BFloat16 = 2 bytes/param.
    3.8B × 2 bytes ≈ 7.5 GB VRAM for weights.

max_model_len: 8192
    Maximum sequence length (prompt + output).
    Matches Ministral's native 8K context window.

gpu_memory_utilization: 0.9
    90% of 48 GB GPU VRAM for KV cache (~43 GB).

trust_remote_code: true
    Required for Mistral model custom classes.

enable_chunked_prefill: true
    Splits long prefill phases across multiple scheduler steps.
    Prevents head-of-line blocking when serving long prompts.
    Improves TTFT fairness across concurrent requests.

max_num_seqs: 4
    Maximum simultaneous sequences in the engine.
    Intentionally low (4) to stay within:
      target_ongoing_requests: 2 (Ray Serve autoscaler trigger)
      max_ongoing_requests: 5   (Ray Serve replica cap)
```

### Deployment (Autoscaling) Configuration

```
autoscaling_config:
  min_replicas: 1
      Always keep at least 1 vLLM replica warm.
      No cold-start latency for the first request.

  max_replicas: 1
      Single replica for this workshop (cost management).
      Increase for production multi-GPU autoscaling.

  target_ongoing_requests: 2
      Scale up when average concurrent requests > 2.
      Scale down when < 2 requests in flight.

max_ongoing_requests: 5
    Hard cap per replica.
    Requests over 5 are queued at the Ray Serve router level.
    Prevents engine overload when batch cap (max_num_seqs=4) is hit.
```

---

## Ray Cluster Components

### Head Node Startup

```bash
python3 -m pip install -q 'ray[llm]==2.56.1' 'vllm==0.22.0'

ray start \
  --head \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-cpus=1 \
  --num-gpus=0 \                   # Head holds no GPU
  --metrics-export-port=8080 \
  --block \
  --dashboard-agent-listen-port=52365 \
  --no-monitor
```

### Worker Node Startup

```bash
dnf install -y -q gcc gcc-c++ make which   # Build tools for vLLM

python3 -m pip install -q 'ray[llm]==2.56.1' 'vllm==0.22.0'

ray start \
  --address=${RAY_ADDRESS:-${FQ_RAY_IP}:6379} \   # Connect to head GCS
  --metrics-export-port=8080 \
  --block \
  --num-cpus=2 \
  --num-gpus=1 \                   # Worker holds 1 GPU
  --no-monitor

# Worker ENV:
CC=/usr/bin/gcc                    # C compiler for native extensions
VLLM_USE_FLASHINFER_SAMPLER=0      # Disable FlashInfer sampler (compatibility)

# Worker Volumes:
/tmp/ray  – emptyDir (Ray object store + logs)
/dev/shm  – Memory emptyDir 10Gi  (NCCL shared memory for tensor communication)
```

---

## Probes

### Head Node Probes

```
livenessProbe:
  command: bash -c "curl -sf http://localhost:52365/api/local_raylet_healthz |
                    grep -q success && curl -sf http://localhost:8265/api/gcs_healthz |
                    grep -q success"
  # Checks both: raylet health AND GCS health
  initialDelaySeconds: 30 | periodSeconds: 5 | failureThreshold: 120

readinessProbe:
  command: bash -c "ray health-check --address 127.0.0.1:6379 > /dev/null 2>&1"
  # Cluster is ready only when GCS is accepting connections
  initialDelaySeconds: 10 | periodSeconds: 5 | failureThreshold: 120
```

### Worker Node Probes

```
livenessProbe:
  command: bash -c "ray status > /dev/null 2>&1"
  # Worker is alive if it can communicate with the cluster
  initialDelaySeconds: 30 | periodSeconds: 5 | failureThreshold: 120

readinessProbe:
  command: bash -c "ray status > /dev/null 2>&1"
  initialDelaySeconds: 10 | periodSeconds: 5 | failureThreshold: 10
```

---

## Monitoring: PodMonitors (ray-podmonitor.yaml)

### Head Monitor

```yaml
PodMonitor: ray-head-monitor
  namespace: monitoring
  labels:
    release: kube-prometheus-stack    # Prometheus Operator discovery

  selector: ray.io/node-type=head

  podMetricsEndpoints:
    - port: metrics      # port 8080 – vLLM + Ray Serve metrics
      relabelings:
        - action: replace
          sourceLabels: [__meta_kubernetes_pod_label_ray_io_cluster]
          targetLabel: ray_io_cluster

    - port: as-metrics   # port 44217 – Autoscaler metrics
      relabelings: [ray_io_cluster label]

    - port: dash-metrics # port 44227 – Dashboard metrics
      relabelings: [ray_io_cluster label]
```

### Worker Monitor

```yaml
PodMonitor: ray-workers-monitor
  namespace: monitoring
  labels:
    release: kube-prometheus-stack

  selector: ray.io/node-type=worker

  podMetricsEndpoints:
    - port: metrics       # port 8080 – Worker GPU + inference metrics
      relabelings: [ray_io_cluster label]
```

### Key Ray Serve Prometheus Metrics

```
serve_deployment_request_counter_total         Total requests per deployment
serve_deployment_error_counter_total           Failed requests
serve_deployment_replica_starts_total          Autoscaling events
serve_deployment_processing_latency_ms         End-to-end processing latency
serve_replica_processing_latency_ms_bucket     Latency histogram
serve_num_ongoing_requests                     Current in-flight requests (gauge)
serve_num_pending_requests                     Queued requests at router (gauge)

# vLLM metrics (same as Module 100, via worker pod)
vllm:gpu_cache_usage_perc
vllm:num_requests_running
vllm:num_requests_waiting
vllm:e2e_request_latency_seconds_bucket
```

---

## Load Generation Script (generate-load.sh)

```
Purpose:
  Generate sustained concurrent inference load so that Ray Serve
  queueing and concurrency panels in Grafana show meaningful data.

  A person typing in a chat has ≤1 request in flight at a time.
  Queue panels stay flat at 0 unless multiple concurrent requests
  arrive simultaneously and continuously.

Usage:
  bash 800-ray/generate-load.sh [DURATION_SECONDS] [CONCURRENCY]
  Default: 180 seconds at 8 concurrent requests

Pre-requisite:
  kubectl port-forward svc/vllm-serve-svc 8000:8000

Parameters:
  DURATION    = 180s   (3 minutes of sustained load)
  CONCURRENCY = 8      (above engine cap of max_num_seqs=4)
                        and Serve per-replica cap of max_ongoing_requests=5
  ENDPOINT    = http://localhost:8000/v1/chat/completions
  MODEL       = ministral

Prompt used:
  "Explain how Kubernetes horizontal pod autoscaling works, then walk
   through what happens step by step when CPU usage spikes on a
   deployment with three replicas. Aim for about 400 words."
  max_tokens: 600

Behavior:
  while time < deadline:
    for i in 1..CONCURRENCY:
      curl POST /v1/chat/completions &   # Fire and forget
    wait                                  # Wait for all CONCURRENCY to finish
    sent += CONCURRENCY
    echo "sent N requests (Xs elapsed)"
```

---

## Ray Serve vs Direct vLLM (Comparison)

```
┌──────────────────┬────────────────────────────┬────────────────────────────┐
│  Feature         │  Module 100 (vLLM direct)   │  Module 800 (Ray Serve)   │
├──────────────────┼────────────────────────────┼────────────────────────────┤
│  Autoscaling     │  Manual (HPA or manual)     │  Built-in, request-based  │
│  Multi-model     │  One Deployment             │  Multiple apps per cluster │
│  Load balancing  │  Kubernetes Service         │  Ray router (smart)       │
│  Batching        │  vLLM only                  │  Ray + vLLM combined      │
│  Monitoring      │  ServiceMonitor             │  PodMonitor (multi-port)  │
│  Failure recovery│  Pod restart                │  Replica restart + re-route│
│  Throughput      │  High (direct engine)       │  High (distributed)       │
│  Complexity      │  Low                        │  Higher (Ray cluster)     │
└──────────────────┴────────────────────────────┴────────────────────────────┘
```

---

## Quick Start

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"

# Deploy KubeRay Operator (prerequisite)
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator -n ray-system --create-namespace

# Deploy RayService (vLLM + Ray Serve)
envsubst < ray-vllm-service.yaml | kubectl apply -f -

# Watch cluster come up (head first, then worker ~3-5 min)
kubectl get rayservice vllm -w
kubectl get pods -w

# Apply PodMonitors for Prometheus
kubectl apply -f ray-podmonitor.yaml

# Deploy Open WebUI (same UI but points to Ray Serve endpoint)
kubectl apply -f openwebui.yml

# Get Open WebUI URL
kubectl get ingress open-webui-ingress \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Test the API directly
kubectl port-forward svc/vllm-serve-svc 8000:8000 &
curl http://localhost:8000/v1/models

# Generate load for Grafana dashboard
bash generate-load.sh 180 8

# View Ray Dashboard
kubectl port-forward svc/vllm 8265:8265 &
# Open: http://localhost:8265
```
