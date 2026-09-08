# Module 400 – LMCache KV Cache Offloading Architecture

> Extends vLLM with LMCache (v0.3.8) to offload KV cache tensors beyond the GPU VRAM. Implements a two-tier cache hierarchy: L1 (local CPU RAM on the same node) and L2 (remote Valkey/Redis via Amazon ElastiCache Serverless). This dramatically reduces TTFT for repeated or similar prompts.

---

## Cache Hierarchy Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         KV Cache Tier Hierarchy                               │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  TIER 0 – GPU VRAM (KV Cache, fastest)                                 │  │
│  │  Device:    NVIDIA L40S                                                 │  │
│  │  Capacity:  ~43 GB (90% of 48 GB, set by gpu_memory_utilization=0.90) │  │
│  │  Access:    nanoseconds                                                 │  │
│  │  Managed:   vLLM PagedAttention (native)                               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                    │ cache miss → offload/fetch                               │
│                    ▼                                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  TIER 1 – CPU RAM (L1 Local Cache) ← LMCacheConnectorV1               │  │
│  │  Config:    LMCACHE_LOCAL_CPU=True                                     │  │
│  │  Capacity:  4.0 GB (LMCACHE_MAX_LOCAL_CPU_SIZE=4.0)                   │  │
│  │  Chunk:     256 tokens (LMCACHE_CHUNK_SIZE=256)                        │  │
│  │  Access:    microseconds (PCIe DMA transfer)                           │  │
│  │  Locality:  Node-local (same pod / same host)                          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                    │ cache miss → fetch from L2                               │
│                    ▼                                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  TIER 2 – Remote Cache (L2 Valkey) ← only in valkey variant           │  │
│  │  Service:   Amazon ElastiCache Serverless (Valkey)                     │  │
│  │  Config:    LMCACHE_REMOTE_URL=rediss://${VALKEY_ENDPOINT}:6379        │  │
│  │  Protocol:  RESP3 over TLS (rediss://)                                 │  │
│  │  Serde:     naive (tensor bytes, LMCACHE_REMOTE_SERDE=naive)           │  │
│  │  Access:    milliseconds (network round-trip)                          │  │
│  │  Key:       deterministic (PYTHONHASHSEED=0 for cross-pod consistency)│  │
│  │  Locality:  Shared across all vLLM pods in the cluster                │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Sub-module 1: CPU RAM Offloading (L1 Only)

```
Folder: cpu-ram-offloading/
Files:
  lmcache-cpu-ram-pod.yaml       # Pod with L1 CPU cache
  lmcache-cpu-ram-configmap.yaml # Test scripts ConfigMap
```

### Architecture Diagram (L1 Only)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  GPU Node (g6e.2xlarge)  |  karpenter.sh/nodepool=gpu                        │
│  Taint: nvidia.com/gpu:NoSchedule                                            │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Pod: lmcache-cpu-ram (restartPolicy: Never)                           │  │
│  │  serviceAccount: model-storage-sa                                       │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Container: vllm-server                                           │  │  │
│  │  │  image: public.ecr.aws/deep-learning-containers/                  │  │  │
│  │  │         vllm:0.21.0-gpu-py312-cu130-ubuntu22.04-ec2-v1.0-soci    │  │  │
│  │  │                                                                   │  │  │
│  │  │  ┌───────────────────────────────────────────────────────────┐   │  │  │
│  │  │  │  vLLM + LMCache L1 (CPU RAM)                               │   │  │  │
│  │  │  │                                                             │   │  │  │
│  │  │  │  Port: 8080                                                 │   │  │  │
│  │  │  │  --kv-transfer-config=                                      │   │  │  │
│  │  │  │    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'│  │  │
│  │  │  │                                                             │   │  │  │
│  │  │  │  GPU VRAM (48 GB, 90%):                                     │   │  │  │
│  │  │  │  ┌─────────────────────────────────────────────────────┐   │   │  │  │
│  │  │  │  │  Hot KV Cache (active sequences)                     │   │   │  │  │
│  │  │  │  │  Model Weights (~7.5 GB @ BF16)                      │   │   │  │  │
│  │  │  │  └─────────────────────────────────────────────────────┘   │   │  │  │
│  │  │  │            ↕ LMCache evict/reload                           │   │  │  │
│  │  │  │  CPU RAM (4 GB reserved):                                   │   │  │  │
│  │  │  │  ┌─────────────────────────────────────────────────────┐   │   │  │  │
│  │  │  │  │  LMCACHE_MAX_LOCAL_CPU_SIZE = 4.0 GB                 │   │   │  │  │
│  │  │  │  │  LMCACHE_CHUNK_SIZE = 256 tokens per chunk           │   │   │  │  │
│  │  │  │  │  Evicted KV tensors stored here                      │   │   │  │  │
│  │  │  │  └─────────────────────────────────────────────────────┘   │   │  │  │
│  │  │  └───────────────────────────────────────────────────────────┘   │  │  │
│  │  │                                                                   │  │  │
│  │  │  Resources:                                                       │  │  │
│  │  │    requests: { cpu: "2", memory: "20Gi", nvidia.com/gpu: "1" }   │  │  │
│  │  │    limits:   { cpu: "4", memory: "28Gi", nvidia.com/gpu: "1" }   │  │  │
│  │  │                                                                   │  │  │
│  │  │  Volume: ConfigMap cpu-ram-cm → /app (test scripts)              │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### LMCache CPU RAM Environment Variables

```
LMCACHE_LOCAL_CPU=True
    Enable local CPU RAM as the L1 KV cache tier.
    LMCache will allocate a memory buffer on host RAM.

LMCACHE_MAX_LOCAL_CPU_SIZE=4.0
    Maximum CPU RAM allocated to L1 KV cache: 4.0 GB.
    Higher values allow caching more KV tensors locally.
    Trade-off: competes with OS page cache and model loading.

LMCACHE_CHUNK_SIZE=256
    Granularity of KV cache chunks: 256 tokens per chunk.
    Smaller chunks = more granular reuse, more metadata overhead.
    Larger chunks = less metadata, requires exact prefix match.
```

---

## Sub-module 2: Remote Cache Sharing via Valkey (L1 + L2)

```
Folder: remote-cache-sharing-with-valkey/
Files:
  lmcache-valkey-pod.yaml       # Pod with L1 CPU + L2 Valkey remote cache
  lmcache-valkey-configmap.yaml # Test scripts ConfigMap
```

### Architecture Diagram (L1 + L2 Valkey)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EKS Cluster – default namespace                                              │
│                                                                               │
│  ┌───────────────────────────┐    ┌───────────────────────────────────────┐  │
│  │  Pod: lmcache-valkey      │    │  Pod: lmcache-valkey (replica 2)       │  │
│  │  (vLLM + LMCache)         │    │  (vLLM + LMCache)                     │  │
│  │                           │    │                                         │  │
│  │  L0: GPU VRAM (43 GB)     │    │  L0: GPU VRAM (43 GB)                  │  │
│  │  L1: CPU RAM  (4 GB)      │    │  L1: CPU RAM  (4 GB)                   │  │
│  │         │                 │    │         │                               │  │
│  │         └─────────────────┼────┼─────────┘                               │  │
│  │                           │    │         │                               │  │
│  └───────────────────────────┘    └─────────┼───────────────────────────────┘  │
│                                             │ RESP3 over TLS (port 6379)        │
│                                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Amazon ElastiCache Serverless – Valkey                                   │  │
│  │  Endpoint: ${VALKEY_ENDPOINT}:6379                                        │  │
│  │  Protocol: rediss:// (Redis Serialization Protocol + TLS)                 │  │
│  │  Mode:     Serverless (auto-scales, no cluster management)                │  │
│  │  Purpose:  L2 shared KV cache across all vLLM pods                        │  │
│  │                                                                            │  │
│  │  Stores: KV tensor chunks (key → tensor bytes)                             │  │
│  │  Key:    deterministic hash of token sequence prefix                       │  │
│  │  Serde:  naive (raw bytes, fastest, no compression)                        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### LMCache Valkey Environment Variables

```
# L1 (same as CPU RAM variant):
LMCACHE_LOCAL_CPU=True                 Enable CPU RAM cache
LMCACHE_MAX_LOCAL_CPU_SIZE=4.0         4 GB local buffer
LMCACHE_CHUNK_SIZE=256                 256 tokens per chunk

# L2 (Valkey remote):
LMCACHE_REMOTE_URL=rediss://${VALKEY_ENDPOINT}:6379
    Remote cache endpoint (ElastiCache Serverless).
    rediss:// = Redis protocol with TLS encryption (required for ElastiCache).
    Port 6379 is the standard Redis/Valkey port.

LMCACHE_REMOTE_SERDE=naive
    Serialization format for remote KV tensors.
    "naive" = raw bytes (fastest, lowest overhead).
    No compression applied; network bandwidth is the bottleneck.

PYTHONHASHSEED=0
    Forces deterministic Python hash values.
    Critical: LMCache uses token sequence hashes as cache keys.
    Without this, different Python processes generate different hash values
    for the same token sequence → cache misses across pods.
    Setting PYTHONHASHSEED=0 ensures cross-pod cache key consistency.

# Other:
CUDA_LAUNCH_BLOCKING=1
    Synchronous CUDA kernel execution.
    Enables precise GPU operation profiling during tests.
    Slight performance overhead; used for debugging/testing only.

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    Limits CUDA memory fragment size.
    Reduces allocator fragmentation in long-running processes.

VLLM_ATTENTION_BACKEND=FLASHINFER
    FlashInfer attention backend (faster than Triton).
```

---

## LMCache Connector: kv-transfer-config

Both variants use the same connector configuration:

```json
--kv-transfer-config='{
  "kv_connector": "LMCacheConnectorV1",
  "kv_role":      "kv_both"
}'
```

```
kv_connector: "LMCacheConnectorV1"
    The LMCache plugin registered with vLLM's KV transfer API.
    Intercepts vLLM's KV eviction/load calls and routes to LMCache.

kv_role: "kv_both"
    This pod can both:
      - SEND:    evict KV tensors from GPU VRAM to cache
      - RECEIVE: reload KV tensors from cache into GPU VRAM
    
    Other roles:
      "kv_sender"   - write-only (dedicated cache filler)
      "kv_receiver" - read-only (prefill-free decoder)
```

---

## vLLM Startup Sequence (Both Variants)

```bash
# 1. Check if LMCache is pre-installed in the AWS DLC image
python3 -c "import lmcache; print('lmcache version:', lmcache.__version__)"

# 2. If not installed, pip install
pip install --no-cache-dir lmcache==0.3.8 openai==1.58.1

# 3. Start vLLM with LMCache via Python API (not CLI entry point)
python3 -m vllm.entrypoints.openai.api_server \
  --port=8080 \
  --model=s3://${S3_BUCKET_NAME}/Ministral-3-8B-Instruct-2512/ \
  --served-model-name=ministral \
  --load-format=runai_streamer \
  --model-loader-extra-config='{"concurrency":16}' \
  --tokenizer_mode=mistral \
  --trust-remote-code \
  --gpu_memory_utilization=0.90 \
  --max-model-len=8192 \
  --tensor-parallel-size=1 \
  --max-num-batched-tokens=8192 \
  --max-num-seqs=256 \
  --block-size=16 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --config-format=mistral \
  --enable-auto-tool-choice \
  --tool-call-parser=mistral \
  --kv-transfer-config='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

Note: Port is `8080` in LMCache pods (vs `8000` in standard vLLM) to allow both to coexist on the same node during testing.

---

## Probe Configuration

```
startupProbe:         # Gates readiness/liveness until model fully loaded
  httpGet: /health :8080
  periodSeconds:    5
  timeoutSeconds:   5
  failureThreshold: 60   # 5min × 60 = 5-minute grace period

readinessProbe:       # Pod removed from Service if model becomes unresponsive
  httpGet: /health :8080
  periodSeconds:    10
  timeoutSeconds:   5
  failureThreshold: 3

livenessProbe:        # Pod restarted if model crashes
  httpGet: /health :8080
  periodSeconds:    30
  timeoutSeconds:   10
  failureThreshold: 3
```

---

## Cache Hit / Miss Flow

```
Request arrives: "What is the capital of France? [512 token system prompt...]"
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  vLLM Scheduler (PagedAttention)                                  │
│                                                                   │
│  1. Hash prefix tokens → cache_key = hash(tokens[:N])            │
│                                                                   │
│  2. Check L0 (GPU VRAM KV Cache)                                 │
│     HIT → use cached KV, skip prefill for those tokens            │
│     MISS → check L1                                               │
│                                                                   │
│  3. LMCacheConnectorV1: Check L1 (CPU RAM, 4 GB)                 │
│     HIT → DMA transfer CPU → GPU VRAM, ~1ms                      │
│     MISS → check L2 (if Valkey variant)                          │
│                                                                   │
│  4. LMCacheConnectorV1: Check L2 (Valkey/ElastiCache)            │
│     HIT → network fetch → CPU RAM → GPU VRAM, ~5–20ms            │
│     MISS → full prefill computation required                      │
│                                                                   │
│  5. Full prefill (cache MISS all levels)                          │
│     Compute KV for all prompt tokens on GPU                       │
│     Store result: GPU VRAM → L1 (eviction) → L2 (async write)   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Performance Impact

```
Scenario: Repeated system prompt (e.g., same RAG context, same agent instructions)

Without LMCache:
  TTFT = full prefill time for all prompt tokens every request
  512 tokens × N req/s = high GPU compute pressure

With LMCache (L1 CPU RAM):
  TTFT = ~50–80% reduction for cache-hit tokens
  GPU freed for token generation instead of re-prefilling

With LMCache (L1 + L2 Valkey, multi-pod):
  TTFT = cache hit even across pod restarts or different replicas
  Warm-up cost amortized across entire cluster
  Pod A prefills once → stores in Valkey → Pod B hits cache directly
```

---

## Quick Start

```bash
export S3_BUCKET_NAME="genai-models-$(aws sts get-caller-identity --query Account --output text)"

# Sub-module 1: CPU RAM offloading (L1 only)
kubectl apply -f cpu-ram-offloading/lmcache-cpu-ram-configmap.yaml
kubectl apply -f cpu-ram-offloading/lmcache-cpu-ram-pod.yaml

# Wait for startup (model load ~3 min)
kubectl wait pod/lmcache-cpu-ram --for=condition=Ready --timeout=600s

# Test
kubectl exec -it lmcache-cpu-ram -- python3 /app/test_cache.py

# Sub-module 2: Valkey remote cache (L1 + L2)
export VALKEY_ENDPOINT="<your-elasticache-serverless-endpoint>"

kubectl apply -f remote-cache-sharing-with-valkey/lmcache-valkey-configmap.yaml
kubectl apply -f remote-cache-sharing-with-valkey/lmcache-valkey-pod.yaml

# Watch logs to confirm LMCache init + Valkey connection
kubectl logs -f lmcache-valkey
```
