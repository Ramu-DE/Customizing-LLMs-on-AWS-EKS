# Module 300 – Benchmarking: Measuring AI Inference Performance

> New to AI inference? Read [CONCEPTS.md](../CONCEPTS.md) first.
> Prerequisite: Module 100 (vLLM) must be running.

---

## What This Module Does

The Red Hat article identifies **cost** and **resources** as the two biggest challenges of AI inference. Before you can address them, you need to **measure** where you actually stand.

This module runs systematic load tests against your vLLM inference server to answer:
- What is the maximum number of requests per second my server can handle?
- How fast does the model respond (latency)?
- At what point does performance degrade?
- How is the GPU being utilised?

Without benchmarking, you are guessing. With benchmarking, you have data to make decisions: when to add more GPUs, when to enable LMCache (Module 400), and when to switch to distributed inference (Module 800).

---

## AI Inference Concepts Demonstrated Here

### The 3 Key Inference Performance Metrics

```
1. TTFT – Time To First Token
   ─────────────────────────
   The time from when the user sends a request until they see
   the first word of the response.
   
   This is what the user experiences as "response time."
   Even if the full answer takes 10 seconds to generate,
   if the first word appears in 200ms, the experience feels fast.
   
   Affected by:
   - Prefill computation (processing the input prompt)
   - How busy the GPU is with other requests
   - KV cache availability (Module 400 improves this)
   
   Target: < 1 second for interactive use cases
   Acceptable: < 3 seconds
   Poor: > 5 seconds

2. Throughput – Tokens Per Second
   ────────────────────────────────
   How many output tokens the server generates per second,
   across ALL concurrent users combined.
   
   This is the "production capacity" number.
   High throughput = serving more users at lower cost per user.
   
   Affected by:
   - GPU compute speed
   - Batch size (continuous batching efficiency)
   - Model size (smaller model = higher throughput)
   
   Ministral-3-8B on L40S: typically 800–1500 tokens/second
   under moderate load.

3. E2E Latency – End-to-End Request Latency
   ──────────────────────────────────────────
   Total time from request sent to full response received.
   = TTFT + (output_tokens × inter_token_latency)
   
   For a 200-token response at 100ms TTFT and 15ms/token:
   E2E = 100ms + (200 × 15ms) = 3,100ms = 3.1 seconds
```

### Batch vs Online Inference in Benchmarking

The benchmarking tool (`inference-perf`) uses **batch inference** semantics – it sends many requests as fast as possible and measures aggregate performance. This simulates production load patterns.

The underlying vLLM server still does **online inference** (PagedAttention, continuous batching) – the distinction is at the client level:
- Benchmark client: fires many requests quickly (batch-like pattern)
- vLLM server: handles each request in real-time (online inference)

### Understanding the Saturation Point

```
Throughput vs Request Rate:

tokens/sec
   ↑
   │                    ●──── SATURATION POINT
   │               ●──/
   │           ●──/
   │       ●──/
   │   ●──/
   │──/
   └───────────────────────────────▶ request rate (req/s)
   0    5    10   15   20   25   30

Linear zone:  GPU has capacity; adding requests → proportional throughput
Saturation:   GPU is fully occupied; more requests → queue grows → latency spikes
              (throughput stops growing; TTFT blows up)

The saturation scenario (Module 300) finds this knee point.
Operating just below saturation = maximum efficiency.
```

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EKS Cluster – default namespace                                              │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GPU Node (g6e.2xlarge) – NVIDIA L40S 48 GB                          │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Deployment: mistral (vllm-optimized-deployment.yml)            │  │   │
│  │  │  Service: vllm-serve-svc:8000  Label: model=mistral             │  │   │
│  │  │  (Same vLLM config as Module 100, optimised for benchmarking)   │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Job: inference-perf (Kubernetes Job, co-located via affinity)  │  │   │
│  │  │  Image: quay.io/inference-perf/inference-perf:v0.6.1           │  │   │
│  │  │  ServiceAccount: model-storage-sa (for S3 results upload)       │  │   │
│  │  │                                                                  │  │   │
│  │  │  Affinity: targetLabels: { model: mistral }                     │  │   │
│  │  │  → Scheduled on SAME node as vLLM pod                          │  │   │
│  │  │  → Eliminates network latency from measurements                 │  │   │
│  │  │                                                                  │  │   │
│  │  │  Resources:                                                      │  │   │
│  │  │    requests: { cpu: "2", memory: "4Gi" }                        │  │   │
│  │  │    limits:   { cpu: "4", memory: "8Gi" }                        │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Amazon S3: genai-models-<ACCOUNT_ID>/benchmarks/<timestamp>/        │   │
│  │  Stores: benchmark_results.json, summary.json per run                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  monitoring namespace                                                 │   │
│  │  GrafanaDashboard: vllm-benchmarking-dashboard                       │   │
│  │  → Visualises TTFT, throughput, latency, KV cache utilisation        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## The 4 Benchmark Scenarios

### Scenario 1: Baseline (baseline-values.yaml)

```
Goal: Understand the best-case performance with zero contention.
      One request at a time, fixed message size.

data:
  input:
    mean: 512     ← Every prompt is exactly 512 tokens
    stdDev: 0     ← No variation (deterministic test)
    min: 512
    max: 512
  output:
    mean: 128     ← Model generates exactly 128 tokens per response
    stdDev: 0
    min: 128
    max: 128

load:
  type: constant      ← Steady, fixed rate (no ramping)
  numWorkers: 4       ← 4 concurrent client workers
  stages:
    - rate: 1         ← 1 request per second
      duration: 60s   ← Run for 60 seconds

What you learn:
  - Single-request TTFT when GPU is uncontested
  - Maximum tokens/second for this model configuration
  - Whether the deployment is stable at light load
  - Establishes the BEST CASE performance baseline
```

### Scenario 2: Saturation (saturation-values.yaml)

```
Goal: Find the breaking point – maximum sustainable throughput.

data:
  input:
    mean: 512
    stdDev: 128   ← Variable input length (realistic workload)
    min: 128      ← Some very short prompts
    max: 2048     ← Some very long prompts
  output:
    mean: 256
    stdDev: 64
    min: 32
    max: 512

load:
  type: constant
  numWorkers: 8     ← More workers = more concurrent pressure
  stages:           ← Ramps up rate over time (step-stress test)
    - rate: 5       ← 5 req/s for 60 seconds  (moderate load)
      duration: 60
    - rate: 10      ← 10 req/s for 60 seconds (high load)
      duration: 60
    - rate: 20      ← 20 req/s for 60 seconds (very high – likely saturates)
      duration: 60

metrics:            ← Simultaneously scrapes vLLM /metrics during the run
  enabled: true
  endpoint: "http://vllm-serve-svc.default:8000/metrics"
  interval: 5s      ← Every 5 seconds (not 30s like ServiceMonitor)
  labels:
    deployment: mistral
    scenario: saturation

What you learn:
  - At what request rate does TTFT start climbing?
  - At what rate does the waiting queue (num_requests_waiting) grow?
  - The SATURATION POINT: maximum useful request rate
  - Whether KV cache fills up (vllm:gpu_cache_usage_perc → 100%)
```

### Scenario 3: Sweep (values.yaml)

```
Goal: Automated saturation discovery (no guessing the right rate).
      Geometric sweep finds the knee point automatically.

load:
  type: constant
  numWorkers: 8
  stages: []      ← stages are AUTO-GENERATED by the sweep config
  sweep:
    type: geometric     ← Exponentially increasing rates
    numRequests: 2000   ← Total requests across all stages
    timeout: 60s        ← Max time per rate stage
    numStages: 5        ← Test 5 different rate points
    stageDuration: 180s ← 3 min per stage for stable measurements
    saturationPercentile: 95   ← Mark as saturated when P95 latency
                                  exceeds 2× the low-load baseline

What you learn:
  - The exact saturation point without manual rate guessing
  - A throughput curve you can use for capacity planning
```

### Scenario 4: Production (values.yaml)

```
Goal: Simulate realistic production traffic patterns.

data:
  input:  mean=1024, stdDev=512, min=128, max=4096
  output: mean=512,  stdDev=256, min=50,  max=2048
  (Highly variable – real users ask very different length questions)

load:
  type: poisson           ← Poisson distribution = realistic bursty arrivals
                            Not a constant rate; bursts and quiet periods
  numWorkers: 8
  stages:
    - rate: 15            ← 15 req/s sustained for 10 minutes
      duration: 600s

What you learn:
  - Performance under realistic (bursty) traffic
  - P99 latency under production conditions (the worst-case user experience)
  - Whether the server degrades gracefully under overload
```

---

## Helm Chart Configuration (benchmark-charts/)

### Core Settings (values.yaml)

```yaml
benchmark:
  image:
    repository: quay.io/inference-perf/inference-perf
    tag: v0.6.1
    # inference-perf: open-source LLM benchmarking tool.
    # Designed specifically for testing OpenAI-compatible APIs.
    pullPolicy: IfNotPresent    # Use cached image if available

  serviceAccount:
    create: false
    name: model-storage-sa
    # Reuses the pre-existing ServiceAccount that has S3 write access.
    # Benchmark results are uploaded to S3 automatically.

  job:
    backoffLimit: 2               # Retry up to 2 times if job fails
    ttlSecondsAfterFinished: 3600 # Auto-delete job pod after 1 hour
                                  # Prevents stale pods accumulating

  target:
    serverType: vllm              # Inference server type (affects API format)
    modelName: ministral          # Must match --served-model-name
    baseUrl: http://vllm-serve-svc.default.svc.cluster.local:8000
    # Full Kubernetes DNS name. More explicit than just "vllm-serve-svc:8000".
    # Format: <service>.<namespace>.svc.cluster.local:<port>

    tokenizerPath: mistralai/Mistral-Nemo-Instruct-2407
    # WHY NOT the actual Ministral tokenizer?
    # inference-perf needs to count tokens accurately to generate
    # prompts with the right number of tokens.
    # Ministral-3-8B's tokenizer_config.json uses "TokenizersBackend"
    # which HuggingFace AutoTokenizer cannot load.
    # Mistral-Nemo uses the SAME Tekken 131k vocabulary.
    # Token counts are identical → accurate prompt generation.

    ignoreEos: true
    # Do NOT stop generating when the model produces an EOS (End of String) token.
    # Benchmarks need to generate EXACTLY the configured output token count.
    # Without this, short responses skew latency measurements.

  api:
    type: completion    # Use /v1/completions (not /v1/chat/completions)
                        # Simpler format; inference-perf v0.6.1 supports
                        # synthetic data generation only with this API type
    streaming: true     # SSE streaming (measures TTFT correctly)
                        # Without streaming, you only measure E2E latency,
                        # not TTFT (can't see first token until all are done)

  storage:
    s3:
      bucketName: "genai-models-<ACCOUNT_ID>"
      pathPrefix: "benchmarks"
      # Results stored at: s3://genai-models-<ACCOUNT_ID>/benchmarks/<timestamp>/
      # Files: benchmark_results.json (per-request), summary.json (aggregates)

  dependencies:
    packages: [sentencepiece, protobuf]
    # Required for the Mistral proxy tokenizer to function correctly.
    # sentencepiece: the underlying tokenizer library
    # protobuf: serialisation format used by tokenizer models

  affinity:
    enabled: true
    targetLabels: { model: mistral }
    # Kubernetes pod affinity: schedule the benchmark job on the SAME node
    # as the vLLM pod (which has label model=mistral).
    # WHY: Eliminates network latency from measurements.
    # Without this: ~1ms network overhead per request (skews latency metrics).
    # With this: direct local network → measurements reflect GPU performance only.
```

---

## Results Interpretation Guide

```
After running baseline scenario, you should see results like:

METRIC                    EXPECTED (good)    CONCERNING
──────────────────────────────────────────────────────────────────────
TTFT P50                  100–300 ms         > 1000 ms
TTFT P99                  300–800 ms         > 3000 ms
Throughput                800–1500 tok/s     < 300 tok/s
Error rate                0%                 > 1%
KV cache utilisation      20–60%             > 90%

After running saturation scenario at rate=20:
  If TTFT P99 > 5× the baseline P99: you've found the saturation point
  If vllm:num_requests_waiting > 0 consistently: queue is building up
  
Decisions based on results:
  TTFT too high at moderate load?     → Enable LMCache (Module 400)
  Throughput too low for your needs?  → Add Ray autoscaling (Module 800)
  High error rate?                    → Check GPU OOM, reduce max-num-seqs
  KV cache at 100%?                   → Reduce max-model-len or add LMCache
```

---

## Grafana Dashboard

```
File: vllm-benchmarking-dashboard.json
CRD:  vllm-benchmarking-dashboard-cr.yaml

Deployed via GrafanaDashboard CRD (requires Grafana Operator):
  namespace: monitoring
  instanceSelector: dashboards=external-grafana
  configMapRef: vllm-benchmarking-dashboard-config

Dashboard rows:
  Row 1: Request Rate & Throughput  → tokens/sec, req/sec over time
  Row 2: TTFT Latency               → P50, P90, P99 time series
  Row 3: End-to-End Latency         → Full request duration
  Row 4: KV Cache                   → gpu_cache_usage_perc
  Row 5: Queue Depth                → num_requests_running vs waiting
  Row 6: GPU Hardware               → utilisation %, memory used, temperature
  Row 7: Token Counts               → avg input/output tokens per request
```

---

## Quick Start

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"

# Make sure vLLM is running from Module 100
kubectl get pods | grep mistral

# Apply Grafana dashboard
kubectl apply -f vllm-benchmarking-dashboard-cr.yaml

# Run Scenario 1: Baseline
helm install baseline 300-benchmarking/benchmark-charts/ \
  -f 300-benchmarking/baseline-values.yaml \
  --set benchmark.storage.s3.bucketName=${S3_BUCKET_NAME}

# Watch the job
kubectl logs -f job/baseline-benchmark-job

# Run Scenario 2: Saturation
helm install saturation 300-benchmarking/benchmark-charts/ \
  -f 300-benchmarking/saturation-values.yaml \
  --set benchmark.storage.s3.bucketName=${S3_BUCKET_NAME}

# View results in Grafana
kubectl get ingress grafana-ingress -n monitoring \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Check raw results in S3
aws s3 ls s3://${S3_BUCKET_NAME}/benchmarks/ --recursive

# Clean up completed jobs
helm uninstall baseline
helm uninstall saturation
```
