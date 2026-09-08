# Day 24 — Ray Serve for LLM Autoscaling

> **Hook:** vLLM handles one server. Ray Serve handles a fleet of them — and scales it up or down automatically based on demand.

---

## The Post

So far in this series, we've talked about optimizing a single vLLM instance.

But what happens when:
- Traffic spikes 10× on Monday morning
- You want to serve multiple models from one endpoint
- You need zero-downtime model updates
- Your team can't babysit GPU nodes 24/7

That's where **Ray Serve** comes in. It's the infrastructure layer above vLLM that turns a single optimized inference engine into a production-grade, autoscaling service.

---

## What Ray Serve Actually Does

```
  RAY SERVE: THE LAYER ABOVE vLLM
  ════════════════════════════════

  WITHOUT Ray Serve:
  ┌──────────────────────────────────────────────────┐
  │  Traffic spike → vLLM queue fills → latency 10×  │
  │  You: manually kubectl scale deployment replicas  │
  │  Night traffic: GPU still running at $1.65/hr     │
  │  Model update: restart pod → downtime 2 min       │
  └──────────────────────────────────────────────────┘

  WITH Ray Serve:
  ┌──────────────────────────────────────────────────┐
  │  Traffic spike → Ray detects → spins up replicas │
  │  automatically (new GPU via Karpenter)            │
  │  Night traffic: scales to 0, GPU terminates       │
  │  Model update: rolling update, zero downtime      │
  └──────────────────────────────────────────────────┘

  Ray Serve sits between your load balancer and vLLM:

  Client
    │
    ▼
  ALB / Ingress
    │
    ▼
  ┌──────────────────────────────────────────┐
  │  Ray Serve Head                          │
  │  - Request routing                       │
  │  - Autoscaling controller                │
  │  - Health monitoring                     │
  └──────────────────┬───────────────────────┘
                     │ dispatches to
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ vLLM     │ │ vLLM     │ │ vLLM     │
    │ Replica 1│ │ Replica 2│ │ Replica 3│
    │ GPU node │ │ GPU node │ │ GPU node │
    └──────────┘ └──────────┘ └──────────┘
```

---

## The RayService CRD — Kubernetes Native

Ray Serve on EKS is managed through the **KubeRay Operator**, which introduces a `RayService` Custom Resource Definition (CRD). Your entire serving config lives in a single YAML.

```yaml
# Simplified from 800-ray/ray-vllm-service.yaml in this workshop
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: vllm-serve
spec:
  serveConfigV2: |
    applications:
    - name: llm
      route_prefix: /
      import_path: serve_app:deployment
      deployments:
      - name: VLLMDeployment
        num_replicas: 1           # starting replicas
        max_ongoing_requests: 10  # queue depth before new replica
        autoscaling_config:
          min_replicas: 1
          max_replicas: 4
          target_ongoing_requests: 5
  rayClusterConfig:
    headGroupSpec:
      replicas: 1
      template:
        spec:
          containers:
          - resources:
              limits:
                cpu: "4"
                memory: "16Gi"
    workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 1
      minReplicas: 1
      maxReplicas: 4
      template:
        spec:
          containers:
          - resources:
              limits:
                nvidia.com/gpu: "1"   # 1 GPU per vLLM replica
```

---

## Autoscaling: How It Actually Works

```
  RAY SERVE AUTOSCALING ALGORITHM
  ════════════════════════════════

  Metric: ongoing_requests per replica
  Target: target_ongoing_requests = 5

  Every scaling evaluation interval (10s default):

  Current ongoing requests = 47
  Current replicas = 4
  Requests per replica = 47 / 4 = 11.75

  Target requests per replica = 5
  Desired replicas = 47 / 5 = 9.4 → rounds up to 10

  Action: scale from 4 → 10 replicas
  Trigger: Karpenter provisions new GPU node(s)
  Time to new GPU node: ~2-4 minutes (g6e.2xlarge cold start)

  ┌────────────────────────────────────────────────────────────┐
  │  Traffic pattern:                                          │
  │                                                            │
  │  Replicas:                          ●──●──●               │
  │  ●──●──●──●──●──●──●──●──●──●──●──       ●──●──●         │
  │                                                            │
  │  Requests:    ●──●──●──●──●          ●──●──●              │
  │  ●──●                   ●──●──●──●──       ●──●──●──●    │
  │  ────────────────────────────────────────────────── time  │
  │  6am        9am       12pm        3pm        6pm          │
  │                                                            │
  │  Scale-up trigger:   ongoing_requests > target × replicas  │
  │  Scale-down trigger: 5 min cooldown below target           │
  │  Scale-to-zero:      min_replicas=0 (if acceptable latency)│
  └────────────────────────────────────────────────────────────┘
```

---

## Load Balancing Strategies

```
  HOW RAY SERVE ROUTES REQUESTS
  ══════════════════════════════

  Default: Round-robin across all healthy replicas

  ┌─────────────────────────────────────────────────────────┐
  │  Request 1  ──────────────────────────▶  Replica 1      │
  │  Request 2  ──────────────────────────▶  Replica 2      │
  │  Request 3  ──────────────────────────▶  Replica 3      │
  │  Request 4  ──────────────────────────▶  Replica 1      │
  └─────────────────────────────────────────────────────────┘

  Power-of-two choices (default in newer Ray versions):
  ┌─────────────────────────────────────────────────────────┐
  │  Request 1: pick 2 random replicas → send to least busy │
  │  → Avoids hot spots when replicas have uneven queue depth│
  └─────────────────────────────────────────────────────────┘

  Sticky routing (for stateful adapters, e.g., LoRA):
  ┌─────────────────────────────────────────────────────────┐
  │  Route by model_name header → always to adapter replica  │
  │  Useful when different replicas serve different adapters  │
  └─────────────────────────────────────────────────────────┘
```

---

## Real Numbers: Ray Serve + vLLM on EKS

```
  BENCHMARKED IN THIS WORKSHOP (Module 800)
  ═══════════════════════════════════════════

  Configuration:
  - Model: Ministral-3-8B-Instruct
  - GPUs: g6e.2xlarge (1× L40S, 48 GB each)
  - Starting replicas: 1
  - Max replicas: 4
  - Load: gradual ramp from 1 to 128 concurrent users

  Results (from generate-load.sh in 800-ray/):

  ┌────────────────┬──────────────┬──────────────┬──────────────┐
  │ Concurrent     │ Active       │ Throughput   │ P95 TTFT     │
  │ Users          │ Replicas     │ (tok/s)      │              │
  ├────────────────┼──────────────┼──────────────┼──────────────┤
  │ 1-10           │ 1            │ ~2,200       │ ~150ms       │
  │ 11-30          │ 2            │ ~4,400       │ ~180ms       │
  │ 31-60          │ 3            │ ~6,600       │ ~210ms       │
  │ 61-128         │ 4            │ ~8,800       │ ~250ms       │
  └────────────────┴──────────────┴──────────────┴──────────────┘

  Scale-up time (new Karpenter node): ~2.5 minutes
  Scale-down time (pod termination): ~30 seconds
  GPU idle at night (scale-to-0): $0/hr vs $1.65/hr always-on
```

---

## Zero-Downtime Model Updates

```
  ROLLING UPDATE WITH RAY SERVE
  ══════════════════════════════

  Before update:
  [Replica 1: v1.0] [Replica 2: v1.0] [Replica 3: v1.0]
         ↑                ↑                ↑
  All serving requests normally

  During update (new RayService YAML applied):
  Step 1: Start Replica 4 with v1.1, run health checks
  [Replica 1: v1.0] [Replica 2: v1.0] [Replica 3: v1.0] [Replica 4: v1.1 (warming)]

  Step 2: Replica 4 healthy → shift traffic
  [Replica 1: v1.0] [Replica 2: v1.0] [Replica 3: v1.0] → drain
                                                           [Replica 4: v1.1] ← serving

  Step 3: Old replicas drained → terminate
                                [Replica 4: v1.1] [Replica 5: v1.1] [Replica 6: v1.1]

  Total downtime: 0ms
  Users during update: unaffected

  Rollback: re-apply old RayService YAML → same process in reverse
```

---

## Ray Dashboard & Observability

```
  MONITORING WITH PROMETHEUS + GRAFANA
  ══════════════════════════════════════

  Ray exposes metrics via /metrics endpoint.
  The 800-ray/ray-podmonitor.yaml scrapes them.

  Key metrics to watch:
  ┌─────────────────────────────────────────────────────────┐
  │  ray_serve_num_ongoing_requests_per_replica             │
  │  → Triggers autoscaling. Target: your configured value  │
  │                                                         │
  │  ray_serve_deployment_replica_starts_total              │
  │  → Count of replica scale-ups                           │
  │                                                         │
  │  ray_serve_request_latency_ms_bucket                    │
  │  → Histogram of end-to-end latency (P50, P95, P99)     │
  │                                                         │
  │  ray_serve_deployment_error_rate                        │
  │  → Trigger alerts when error rate > threshold           │
  │                                                         │
  │  vllm:gpu_cache_usage_perc                             │
  │  → KV cache fill rate per replica                       │
  └─────────────────────────────────────────────────────────┘

  All visible in Grafana dashboards deployed by this workshop.
```

---

## The Cost Model: Why This Matters

```
  AUTOSCALING vs FIXED FLEET (per month)
  ═══════════════════════════════════════

  Traffic profile: peak 9am-6pm weekdays, quiet nights/weekends
  Actual busy hours: ~45 hrs/week = ~195 hrs/month
  Peak GPUs needed: 4 (g6e.2xlarge @ $1.65/hr each)

  FIXED FLEET (always-on 4 GPUs):
  4 × $1.65 × 720 hrs = $4,752/month

  AUTOSCALING (Ray Serve + Karpenter):
  Peak (195 hrs): 4 × $1.65 × 195 = $1,287
  Off-peak (525 hrs): 1 GPU minimum × $1.65 × 525 = $866
  Total: ~$2,153/month

  SAVINGS: $2,599/month (55% reduction)
  ↑ This is why autoscaling is not optional at production scale.
```

---

## Workshop Connection — Module 800

```
  RAY SERVE IN THIS WORKSHOP
  ═══════════════════════════

  Files in 800-ray/:
  ┌─────────────────────────────────────────────────────────────┐
  │  ray-vllm-service.yaml                                       │
  │  → RayService CRD with vLLM integration                     │
  │  → Autoscaling config, GPU resource limits                   │
  │  → Ministral-3-8B serving with TP config                    │
  │                                                              │
  │  ray-podmonitor.yaml                                         │
  │  → Prometheus scrape config for Ray metrics                  │
  │  → Enables Grafana dashboards for autoscaling visibility     │
  │                                                              │
  │  openwebui.yml                                               │
  │  → OpenWebUI pointing to Ray Serve endpoint                  │
  │  → Users see same chat UI, now backed by autoscaling fleet   │
  │                                                              │
  │  generate-load.sh                                            │
  │  → Load test script that triggers autoscaling                │
  │  → Watch Grafana: replicas climb as load increases           │
  └─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaway

> Ray Serve is the orchestration layer that turns a vLLM instance into a production fleet.  
> It autoscales replicas based on queue depth, routes traffic intelligently, and enables zero-downtime updates.  
> Combined with Karpenter, it scales GPU nodes automatically — no human babysitting required.  
> The cost difference between a fixed fleet and autoscaled fleet is often 50%+ for real traffic patterns.

---

*30-Day Series: LLM Inference Is Everything | Day 24 of 30 — Week 4: Scaling Beyond One GPU*  
*← [Day 23](./day-23-pipeline-parallelism.md) | [Day 25 →](./day-25-multi-node-inference.md) | [Back to Index](../README.md)*
