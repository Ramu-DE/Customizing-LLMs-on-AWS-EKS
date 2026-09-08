# Module 300 – Benchmarking Pipeline Architecture

> Systematically measures vLLM inference performance using `inference-perf` (v0.6.1). Runs Kubernetes Jobs (via Helm chart) against an optimized vLLM deployment, stores results in S3, and visualizes them in a custom Grafana dashboard.

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        EKS Cluster – default namespace                        │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GPU Node Pool (g6e.2xlarge)                                          │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │       Pod: mistral (Optimized vLLM Deployment)                  │  │   │
│  │  │  vllm-optimized-deployment.yml                                  │  │   │
│  │  │  Additional: --chunked-prefill, tuned batching params           │  │   │
│  │  │  Port: 8000  |  labels: model=mistral                           │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                             │                                         │   │
│  │   Service: vllm-serve-svc (ClusterIP, port 8000)                     │   │
│  │                             │                                         │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │   Pod: inference-perf (Kubernetes Job – benchmark-charts)       │  │   │
│  │  │   image: quay.io/inference-perf/inference-perf:v0.6.1          │  │   │
│  │  │   serviceAccount: model-storage-sa                              │  │   │
│  │  │   Affinity: co-located with model=mistral pods                  │  │   │
│  │  │   Scenarios: baseline | saturation | sweep | production          │  │   │
│  │  │                                                                  │  │   │
│  │  │   Flow: send requests → collect latency → upload to S3          │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  monitoring namespace                                                 │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  GrafanaDashboard: vllm-benchmarking-dashboard                  │  │   │
│  │  │  CRD: grafana.integreatly.org/v1beta1                           │  │   │
│  │  │  ConfigMap: vllm-benchmarking-dashboard-config                  │  │   │
│  │  │  Panels: TTFT, throughput, latency percentiles, KV cache util   │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Amazon S3: s3://genai-models-<ACCOUNT_ID>/benchmarks/               │   │
│  │  Benchmark results (JSON) stored per run                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Benchmark Scenarios

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         4 Benchmark Scenarios                                 │
├──────────────────────┬───────────────────────────────────────────────────────┤
│  SCENARIO 1          │  BASELINE                                              │
│  baseline-values.yaml│  Goal: Establish optimal performance (zero contention) │
│                      │                                                         │
│                      │  Input tokens:  mean=512, stdDev=0 (fixed 512)         │
│                      │  Output tokens: mean=128, stdDev=0 (fixed 128)         │
│                      │  Load type:     constant                                │
│                      │  Workers:       4                                       │
│                      │  Rate:          1 req/s  Duration: 60s                 │
├──────────────────────┼───────────────────────────────────────────────────────┤
│  SCENARIO 2          │  SATURATION                                            │
│  saturation-values.  │  Goal: Determine maximum sustainable throughput        │
│  yaml                │                                                         │
│                      │  Input tokens:  mean=512, stdDev=128 (128–2048)        │
│                      │  Output tokens: mean=256, stdDev=64  (32–512)          │
│                      │  Load type:     constant (multi-stage ramp)             │
│                      │  Workers:       8                                       │
│                      │  Stages:                                                │
│                      │    rate=5   → duration=60s  (warm-up)                  │
│                      │    rate=10  → duration=60s  (medium load)              │
│                      │    rate=20  → duration=60s  (high load)                │
│                      │  Metrics:   Scrape /metrics every 5s during run        │
│                      │  Labels:    deployment=mistral, scenario=saturation    │
├──────────────────────┼───────────────────────────────────────────────────────┤
│  SCENARIO 3          │  SWEEP (Auto-saturation detection)                     │
│  (values.yaml)       │  Goal: Automated capacity discovery                    │
│                      │                                                         │
│                      │  Input tokens:  mean=512, stdDev=128 (128–2048)        │
│                      │  Output tokens: mean=256, stdDev=64  (32–512)          │
│                      │  Load type:     geometric sweep                         │
│                      │  Workers:       8                                       │
│                      │  numRequests:   2000 | timeout: 60s | numStages: 5    │
│                      │  stageDuration: 180s | saturationPercentile: 95        │
├──────────────────────┼───────────────────────────────────────────────────────┤
│  SCENARIO 4          │  PRODUCTION (Realistic simulation)                     │
│  (values.yaml)       │  Goal: Realistic traffic with variable sizes           │
│                      │                                                         │
│                      │  Input tokens:  mean=1024, stdDev=512 (128–4096)       │
│                      │  Output tokens: mean=512,  stdDev=256 (50–2048)        │
│                      │  Load type:     poisson (bursty arrivals)               │
│                      │  Workers:       8                                       │
│                      │  Rate:          15 req/s  Duration: 600s               │
└──────────────────────┴───────────────────────────────────────────────────────┘
```

---

## Benchmark Helm Chart Configuration (benchmark-charts/)

```
Chart.yaml:
  name: benchmark-charts
  apiVersion: v2

values.yaml (default configuration):
┌─────────────────────────────────────────────────────────────────┐
│  benchmark:                                                      │
│    enabled: true                                                 │
│    image:                                                        │
│      repository: quay.io/inference-perf/inference-perf          │
│      tag: v0.6.1                                                 │
│      pullPolicy: IfNotPresent                                    │
│                                                                  │
│    serviceAccount:                                               │
│      create: false                                               │
│      name: model-storage-sa   # Uses EKS Pod Identity for S3   │
│                                                                  │
│    job:                                                          │
│      backoffLimit: 2                                             │
│      ttlSecondsAfterFinished: 3600   # Auto-delete after 1hr   │
│                                                                  │
│    target:                                                       │
│      serverType: vllm                                            │
│      modelName: ministral                                        │
│      baseUrl: http://vllm-serve-svc.default.svc.cluster.local:8000
│      tokenizerPath: mistralai/Mistral-Nemo-Instruct-2407        │
│        # Proxy tokenizer: same Tekken 131k vocab,               │
│        # but HuggingFace-compatible tokenizer_config.json       │
│      ignoreEos: true   # Don't stop at EOS for fixed-length     │
│                                                                  │
│    api:                                                          │
│      type: completion   # Uses /v1/completions                  │
│      streaming: true    # SSE streaming mode                    │
│                                                                  │
│    storage:                                                      │
│      s3:                                                         │
│        bucketName: genai-models-<ACCOUNT_ID>                    │
│        pathPrefix: benchmarks                                    │
│                                                                  │
│    dependencies:                                                 │
│      packages: [sentencepiece, protobuf]  # Mistral tokenizer   │
│                                                                  │
│    resources:                                                    │
│      requests: { cpu: "2", memory: "4Gi" }                      │
│      limits:   { cpu: "4", memory: "8Gi" }                      │
│                                                                  │
│    affinity:                                                     │
│      enabled: true                                               │
│      targetLabels: { model: mistral }  # Co-locate w/ vLLM pod  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Optimized vLLM Deployment (vllm-optimized-deployment.yml)

```
Additional optimizations over baseline Module 100 deployment:

--chunked-prefill (if enabled)
    Splits long prefill computations across multiple steps.
    Prevents long prompts from blocking the scheduler.
    Improves TTFT for concurrent requests.

Resource allocation (same as Module 100):
  requests: { cpu: "2", memory: "16Gi", nvidia.com/gpu: "1" }
  limits:   { cpu: "4", memory: "28Gi", nvidia.com/gpu: "1" }

Labels:
  model: mistral   ← Required for benchmark affinity scheduling
```

---

## Metrics Collected

```
inference-perf collects per-request and aggregate metrics:

Per-Request Metrics:
  ┌──────────────────────────────────────────────────────────┐
  │  TTFT (Time to First Token)                               │
  │    - P50, P90, P95, P99 latency percentiles (ms)         │
  │                                                           │
  │  ITL (Inter-Token Latency)                                │
  │    - Time between consecutive output tokens (ms)         │
  │                                                           │
  │  E2E Latency                                              │
  │    - Total request duration (ms)                         │
  │                                                           │
  │  Prompt/Completion Tokens                                 │
  │    - Input token count, output token count               │
  └──────────────────────────────────────────────────────────┘

Aggregate Metrics:
  ┌──────────────────────────────────────────────────────────┐
  │  Throughput (tokens/second)                               │
  │  Request Rate (req/s, successful vs failed)              │
  │  Error Rate (%)                                          │
  │  GPU Cache Hit Rate (from /metrics endpoint)             │
  │  KV Cache Utilization (%)                                │
  └──────────────────────────────────────────────────────────┘

Saturation Metrics (from vLLM /metrics during saturation run):
  Labels: deployment=mistral, scenario=saturation
  Interval: 5s scrape
  vllm:num_requests_running
  vllm:num_requests_waiting
  vllm:gpu_cache_usage_perc
  vllm:num_preemptions_total
```

---

## Grafana Dashboard

```
Dashboard: vllm-benchmarking-dashboard.json
CRD:       GrafanaDashboard (grafana.integreatly.org/v1beta1)

Deployment via CRD (vllm-benchmarking-dashboard-cr.yaml):
  instanceSelector:
    matchLabels:
      dashboards: "external-grafana"
  configMapRef:
    name: vllm-benchmarking-dashboard-config
    key:  vllm-benchmarking-dashboard.json

Dashboard Panels:
  ┌──────────────────────────────────────────────────────────┐
  │  Row 1: Request Throughput                                │
  │  Row 2: TTFT Latency (P50/P90/P99)                      │
  │  Row 3: Inter-Token Latency                              │
  │  Row 4: KV Cache Utilization                             │
  │  Row 5: Queue Depth (waiting vs running requests)        │
  │  Row 6: GPU Utilization + Memory                        │
  │  Row 7: Token Throughput (tokens/s)                      │
  └──────────────────────────────────────────────────────────┘
```

---

## Benchmark Execution Flow

```
helm install baseline benchmark-charts/ -f baseline-values.yaml
                    │
                    ▼
        Kubernetes Job created
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│  inference-perf Pod                                            │
│                                                                │
│  1. Install dependencies                                       │
│     pip install sentencepiece protobuf                        │
│                                                                │
│  2. Download tokenizer                                         │
│     HuggingFace: mistralai/Mistral-Nemo-Instruct-2407         │
│     (proxy for Tekken tokenizer, same 131k vocab)             │
│                                                                │
│  3. Generate synthetic prompts                                 │
│     Distribution: normal(mean=512, stdDev=0)                  │
│     Fixed 512-token inputs for baseline                       │
│                                                                │
│  4. Execute load test                                          │
│     4 workers × 1 req/s × 60s = ~60 requests                 │
│     POST /v1/completions (streaming=true)                     │
│     Target: http://vllm-serve-svc.default:8000                │
│                                                                │
│  5. Collect & aggregate metrics                               │
│     Per-request: TTFT, ITL, E2E, tokens                       │
│     Aggregate: P50/P90/P99 latencies, throughput              │
│                                                                │
│  6. Upload results to S3                                       │
│     s3://genai-models-<ACCOUNT_ID>/benchmarks/<timestamp>/    │
│     Files: benchmark_results.json, summary.json               │
└───────────────────────────────────────────────────────────────┘
       │
       ▼
Job completes (ttlSecondsAfterFinished: 3600 → auto-delete in 1hr)
       │
       ▼
Results visible in Grafana vLLM Benchmarking Dashboard
```

---

## Quick Start

```bash
# Set variables
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"

# Deploy optimized vLLM (if not already running from Module 100)
kubectl apply -f vllm-optimized-deployment.yml

# Apply Grafana dashboard
kubectl apply -f vllm-benchmarking-dashboard-cr.yaml

# Run baseline benchmark
helm install baseline benchmark-charts/ \
  -f baseline-values.yaml \
  --set benchmark.storage.s3.bucketName=${S3_BUCKET_NAME}

# Watch benchmark job
kubectl get jobs -w
kubectl logs -f job/baseline-benchmark-job

# Run saturation benchmark
helm install saturation benchmark-charts/ \
  -f saturation-values.yaml \
  --set benchmark.storage.s3.bucketName=${S3_BUCKET_NAME}

# View results in Grafana
kubectl get ingress grafana-ingress -n monitoring \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```
