# Module 400 – LMCache: Solving the KV Cache Resource Challenge

> New to AI inference? Read [CONCEPTS.md](../CONCEPTS.md) first.
> Prerequisite: Module 100 (vLLM) must be running.

---

## What This Module Does

Module 300 (benchmarking) showed you where performance breaks down. One of the most common bottlenecks: **GPU VRAM fills up with KV cache**, limiting how many concurrent users you can serve.

This module adds **LMCache** (version 0.3.8) to vLLM. LMCache extends the KV cache beyond the 48 GB GPU VRAM into:
- **L1**: Local CPU RAM on the same node (4 GB, microsecond access)
- **L2**: Remote Valkey/Redis cache in ElastiCache (unlimited, millisecond access)

Result: the same GPU can serve more users, and repeated prompts (like system prompts, RAG context, or agent tool schemas) never need to be recomputed.

---

## AI Inference Concepts Demonstrated Here

### The Resources Challenge: GPU VRAM is Finite

From the Red Hat article:
> "More complex models will require specialized hardware and software to support the vast amount of data processing. A key component of these resources is CPU memory."

Let's make this concrete. With Ministral-3-8B on the L40S:

```
48 GB GPU VRAM breakdown:
  Model weights:  ~7.5 GB  (fixed)
  Runtime overhead: ~2.5 GB (fixed)
  KV cache: ~38 GB (used by vLLM PagedAttention)

One KV cache entry = 2 × num_layers × num_heads × head_dim × 2 bytes (BF16)
For Ministral-3-8B: 2 × 32 layers × 32 heads × 128 dim × 2 = ~524 KB per token

For a 512-token request: 512 × 524 KB ≈ 256 MB of KV cache
38 GB / 256 MB ≈ 148 concurrent 512-token requests

Without LMCache: once you hit 148 concurrent users, vLLM EVICTS old KV entries.
Evicted entries = recompute the prefix next time → higher TTFT.
With LMCache: evicted entries go to CPU RAM or Valkey instead of being deleted.
```

### The KV Cache Is a Database

Think of the KV cache as a database of "I've already done this math":
```
Key:   hash of the token sequence prefix
Value: the KV tensors computed for that prefix

Cache lookup:
  New request: "You are a helpful assistant. [system prompt] User: Hello!"
  Hash the first 200 tokens (system prompt) → look up in cache
  Cache HIT?  → Load KV tensors from cache → skip prefill computation
  Cache MISS? → Compute KV tensors on GPU → store in cache
```

This is especially powerful for:
- **Agents** (Module 200): system prompt + tool schemas repeated every call
- **RAG** (Module 700): same retrieved documents passed to many users
- **Chatbots**: same system personality prompt in every conversation

---

## Cache Tier Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  GPU Memory (48 GB) – TIER 0: Hot KV Cache                                   │
│  Fastest access: nanoseconds                                                  │
│  Managed by: vLLM PagedAttention (native)                                     │
│  Contents: KV tensors for currently active requests                          │
│                                                                               │
│  When a page is evicted by vLLM (not enough space):                          │
│          │                                                                    │
│          ▼  LMCacheConnectorV1 intercepts the eviction                        │
│                                                                               │
│  CPU RAM (4 GB reserved) – TIER 1: L1 Local Cache                            │
│  Access speed: ~1 ms (PCIe data transfer)                                     │
│  Managed by: LMCache (LMCACHE_LOCAL_CPU=True)                                 │
│  Capacity: LMCACHE_MAX_LOCAL_CPU_SIZE=4.0 (GB)                               │
│  Contents: Recently evicted KV tensors, chunked at 256 tokens                │
│                                                                               │
│  When L1 is full and a new entry arrives:                                     │
│          │                                                                    │
│          ▼  LMCache writes to remote Valkey (L2 variant only)                 │
│                                                                               │
│  Amazon ElastiCache (Valkey) – TIER 2: L2 Remote Cache                       │
│  Access speed: ~5–20 ms (network round-trip)                                  │
│  Managed by: LMCache (LMCACHE_REMOTE_URL)                                     │
│  Capacity: Serverless (auto-scales with usage)                               │
│  Contents: Evicted L1 entries                                                 │
│  Key feature: SHARED across ALL vLLM pods in the cluster                     │
│               Pod A evicts → Valkey stores → Pod B loads → cache hit!        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Sub-module 1: CPU RAM Offloading (L1 Cache Only)

### When to Use This

Use the L1-only configuration when:
- You have a single vLLM pod (no sharing needed)
- You want simple setup without external dependencies
- Use cases with many repeated prefixes (chatbots, RAG, agents)
- Available CPU RAM on the node > 4 GB spare

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  GPU Node (g6e.2xlarge) – karpenter.sh/nodepool=gpu                          │
│  Taint: nvidia.com/gpu:NoSchedule                                            │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Pod: lmcache-cpu-ram  (restartPolicy: Never)                          │  │
│  │  ServiceAccount: model-storage-sa                                       │  │
│  │                                                                         │  │
│  │  Container: vllm-server                                                 │  │
│  │  Image: public.ecr.aws/deep-learning-containers/                        │  │
│  │         vllm:0.21.0-gpu-py312-cu130-ubuntu22.04-ec2-v1.0-soci          │  │
│  │                                                                         │  │
│  │  Port: 8080  (different from Module 100's 8000, for coexistence)       │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │  NVIDIA L40S 48 GB VRAM                                          │   │  │
│  │  │  ┌────────────────────────────┐                                  │   │  │
│  │  │  │  GPU KV Cache (PagedAttn)  │ ← TIER 0: hot active tensors    │   │  │
│  │  │  │  ~38 GB                    │                                  │   │  │
│  │  │  └────────────┬───────────────┘                                  │   │  │
│  │  │               │ evict (LMCacheConnectorV1)                        │   │  │
│  │  │               ▼                                                   │   │  │
│  │  │  ┌────────────────────────────┐                                  │   │  │
│  │  │  │  CPU RAM (4 GB)             │ ← TIER 1: L1 local cache        │   │  │
│  │  │  │  LMCACHE_LOCAL_CPU=True    │                                  │   │  │
│  │  │  │  chunks of 256 tokens       │                                  │   │  │
│  │  │  └────────────────────────────┘                                  │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                         │  │
│  │  Resources:                                                             │  │
│  │    requests: { cpu: "2", memory: "20Gi",  nvidia.com/gpu: "1" }        │  │
│  │    limits:   { cpu: "4", memory: "28Gi",  nvidia.com/gpu: "1" }        │  │
│  │    20Gi RAM = 7.5 GB model metadata + 4 GB LMCache L1 + 8.5 GB OS     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Environment Variables Explained

```
LMCACHE_LOCAL_CPU=True
  Tells LMCache to enable CPU RAM as the L1 cache tier.
  Without this, KV tensors evicted from GPU VRAM are simply deleted.
  With this, they are moved to CPU RAM for potential reuse.

LMCACHE_MAX_LOCAL_CPU_SIZE=4.0
  Maximum gigabytes reserved for the L1 CPU RAM cache.
  4.0 GB = ~16 entries of 512-token requests (4 × 256 MB each).
  Increase if you have spare RAM and want to cache more entries.
  The node has 64 GB RAM total; OS + model needs ~16 GB; rest available.

LMCACHE_CHUNK_SIZE=256
  KV tensors are stored in chunks of 256 tokens.
  A 1024-token prompt = 4 chunks of 256 tokens each.
  Finer granularity = more reuse opportunities (partial prefix matches).
  Coarser granularity = less metadata overhead.
  256 is the recommended default for most workloads.
```

---

## Sub-module 2: Remote Cache Sharing via Valkey (L1 + L2)

### When to Use This

Use the L1 + L2 configuration when:
- You run multiple vLLM pods (horizontal scaling with Ray, Module 800)
- You want KV cache to survive pod restarts (warm restarts)
- Many users share the same long system prompts or RAG context
- You need cluster-wide cache sharing

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EKS Cluster                                                                  │
│                                                                               │
│  GPU Node 1: lmcache-valkey pod          GPU Node 2: lmcache-valkey pod       │
│  ┌──────────────────────────────┐        ┌──────────────────────────────┐    │
│  │  GPU KV Cache (38 GB)        │        │  GPU KV Cache (38 GB)        │    │
│  │       ↕ evict/load           │        │       ↕ evict/load           │    │
│  │  CPU RAM Cache (4 GB, L1)    │        │  CPU RAM Cache (4 GB, L1)    │    │
│  │       ↕ evict/load           │        │       ↕ evict/load           │    │
│  └──────────────┬───────────────┘        └──────────────┬───────────────┘    │
│                 │ RESP3/TLS port 6379                    │                    │
│                 └──────────────────┬───────────────────--┘                    │
│                                    ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  Amazon ElastiCache Serverless – Valkey (L2 Remote Cache)             │   │
│  │  Endpoint: ${VALKEY_ENDPOINT}:6379                                    │   │
│  │  Protocol: rediss:// (RESP3 over TLS – encrypted in transit)          │   │
│  │  Capacity: Auto-scales with usage (serverless)                        │   │
│  │  Shared: ALL vLLM pods read/write the same cache                      │   │
│  │                                                                       │   │
│  │  Cache key = deterministic hash of token sequence                    │   │
│  │  Cache value = KV tensor bytes (raw, no compression)                 │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘

Cross-pod sharing example:
  User A hits Pod 1: "Explain Python" (with 500-token system prompt)
    → Pod 1 computes KV for system prompt → stores in Valkey
  
  User B hits Pod 2 (same system prompt):
    → Pod 2 checks Valkey → CACHE HIT
    → Downloads KV tensors from Valkey → loads into Pod 2 GPU
    → Skips 500-token prefill computation entirely
    → TTFT dramatically reduced for Pod 2
```

### Additional Environment Variables (Valkey variant)

```
LMCACHE_REMOTE_URL=rediss://${VALKEY_ENDPOINT}:6379
  Connection URL for the remote Valkey/Redis cache.
  
  rediss://   = Redis Serialization Protocol + TLS encryption
               (two 's' = secure, like 'https' vs 'http')
               AWS ElastiCache Serverless REQUIRES TLS.
               
  ${VALKEY_ENDPOINT} = The ElastiCache Serverless endpoint DNS name.
                        Found in AWS Console → ElastiCache → Serverless cache.
  
  :6379       = Standard Redis/Valkey port.

LMCACHE_REMOTE_SERDE=naive
  Serialisation format for KV tensors stored in Valkey.
  
  naive  = Raw tensor bytes. No compression, no encoding.
           Fastest write/read. Uses most network bandwidth.
           
  Other options: pickle, msgpack (not used here).
  For LAN/VPC connections (low latency, high bandwidth), naive is optimal.
  For high-latency WAN connections, compression might help.

PYTHONHASHSEED=0
  THIS IS CRITICAL for cross-pod cache sharing.
  
  Python's built-in hash() function is non-deterministic by default.
  Between different Python processes, hash("same string") gives DIFFERENT results.
  
  LMCache uses Python hashes to generate cache keys from token sequences.
  Without PYTHONHASHSEED=0:
    Pod 1: hash([1,2,3,4]) = 0xA1B2...  → stores key 0xA1B2 in Valkey
    Pod 2: hash([1,2,3,4]) = 0xF3E4...  → looks up key 0xF3E4 → MISS!
    The pods use different keys for the same data → no sharing.
  
  With PYTHONHASHSEED=0:
    Pod 1: hash([1,2,3,4]) = 0x1234...  (deterministic)
    Pod 2: hash([1,2,3,4]) = 0x1234...  (identical)
    → Cache key matches → HIT! Cross-pod sharing works.
```

---

## The LMCache Connector: --kv-transfer-config

Both sub-modules pass this argument to vLLM:

```
--kv-transfer-config='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

kv_connector: "LMCacheConnectorV1"
  The name of the LMCache plugin registered in vLLM's plugin system.
  This tells vLLM: "when you want to evict or reload a KV block,
  call LMCache instead of just deleting it."
  LMCache registers itself at startup via Python entry points.

kv_role: "kv_both"
  What this pod is allowed to do with the KV cache.
  
  "kv_both"     = This pod can SEND (evict) AND RECEIVE (reload) KV tensors.
                  Both flows enabled. Normal for single-pod or symmetric setups.
  
  "kv_sender"   = Only SEND (evict GPU → cache). Used for a dedicated
                  "prefill" pod that fills the cache for other pods.
  
  "kv_receiver" = Only RECEIVE (load cache → GPU). Used for a "decode"
                  pod that relies on a sender pod to pre-fill.
  
  kv_both is the right choice for this workshop (each pod is self-sufficient).
```

---

## Probe Configuration (Why 3 Different Probes)

```yaml
startupProbe:
  httpGet: { path: /health, port: 8080 }
  periodSeconds: 5
  failureThreshold: 60    # 5 min grace (5s × 60 = 300s)
  # PURPOSE: Block readiness/liveness probes from firing until
  #          the model is fully loaded from S3.
  #          Model loading takes 2-3 min.
  #          Without this, liveness probe kills the pod before it's ready.
  #          With this, Kubernetes waits up to 5 min before considering
  #          the pod unhealthy.

readinessProbe:
  httpGet: { path: /health, port: 8080 }
  periodSeconds: 10
  failureThreshold: 3     # Remove from Service after 30 seconds unresponsive
  # PURPOSE: Only send traffic to this pod when the model is ready.
  #          Pod stays "not ready" during S3 loading.
  #          Once /health returns 200, pod joins the Service endpoints.

livenessProbe:
  httpGet: { path: /health, port: 8080 }
  periodSeconds: 30
  failureThreshold: 3     # Restart pod after 90 seconds unresponsive
  # PURPOSE: Restart the pod if the model process crashes or hangs.
  #          Less frequent than readiness (30s vs 10s) to avoid
  #          killing a slow-but-alive model mid-generation.
```

---

## LMCache Startup Sequence

```bash
# What the pod container runs:

# Step 1: Check if LMCache is pre-installed
python3 -c "import lmcache; print('lmcache version:', lmcache.__version__)"
# The AWS Deep Learning Container (DLC) for vLLM includes LMCache.
# If not found, install it:
pip install --no-cache-dir lmcache==0.3.8 openai==1.58.1

# Step 2: Start vLLM with LMCache connector
python3 -m vllm.entrypoints.openai.api_server \
  --port=8080 \
  --model=s3://${S3_BUCKET_NAME}/Ministral-3-8B-Instruct-2512/ \
  --kv-transfer-config='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
  # All other flags same as Module 100
```

---

## When Does Caching Actually Help?

```
Cache helps a LOT when:
  ✅ Same system prompt used by many users
     (e.g., "You are a customer service agent for Acme Corp...")
  ✅ RAG: Same retrieved documents passed to many users with similar queries
  ✅ Agents: Same tool schemas sent on every LLM call
  ✅ Documents: Multiple questions about the same document

Cache helps LESS when:
  ❌ Every user has a completely unique prompt (no shared prefix)
  ❌ Single-user system (no benefit from sharing across users)
  ❌ Very short prompts where prefill is already fast

Cache HURTS slightly:
  - 1-5 ms overhead to check cache on every new request
  - L2 (Valkey) adds ~10-20 ms when fetching from remote cache
    (still faster than full prefill for prompts > 100 tokens)
```

---

## Testing LMCache Effectiveness

```bash
# Deploy L1 CPU cache pod
kubectl apply -f 400-lmcache/cpu-ram-offloading/lmcache-cpu-ram-configmap.yaml
kubectl apply -f 400-lmcache/cpu-ram-offloading/lmcache-cpu-ram-pod.yaml

# Wait for it to be ready (model loads from S3, ~3 min)
kubectl wait pod/lmcache-cpu-ram --for=condition=Ready --timeout=600s

# Port-forward to the LMCache pod
kubectl port-forward pod/lmcache-cpu-ram 8080:8080 &

# Test: First request (cache MISS – must compute prefix)
time curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ministral","messages":[
    {"role":"system","content":"You are an expert Python teacher with 20 years experience. Always provide detailed examples. Explain concepts step by step. Use analogies. Be patient and thorough."},
    {"role":"user","content":"What is a Python list?"}
  ]}' | python3 -m json.tool | grep '"content"'
# Note the TTFT

# Test: Second request with SAME system prompt (cache HIT – skip prefill)
time curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ministral","messages":[
    {"role":"system","content":"You are an expert Python teacher with 20 years experience. Always provide detailed examples. Explain concepts step by step. Use analogies. Be patient and thorough."},
    {"role":"user","content":"What is a Python dictionary?"}
  ]}' | python3 -m json.tool | grep '"content"'
# TTFT should be noticeably faster

# Deploy Valkey variant
export VALKEY_ENDPOINT="<your-elasticache-endpoint>"
kubectl apply -f 400-lmcache/remote-cache-sharing-with-valkey/lmcache-valkey-configmap.yaml
envsubst < 400-lmcache/remote-cache-sharing-with-valkey/lmcache-valkey-pod.yaml | kubectl apply -f -
kubectl logs -f lmcache-valkey   # Confirm Valkey connection established
```
