# Module 700 – RAG: Grounding AI Inference with Your Own Data

> New to AI inference? Read [CONCEPTS.md](../CONCEPTS.md) first.
> Prerequisite: Module 100 (vLLM) must be running.

---

## What This Module Does

The Ministral-3-8B model was trained on data up to a certain date. It knows general knowledge, but it does not know:
- Your company's products and prices
- Your internal policies
- Recent news or events
- Domain-specific documents you have collected

**Retrieval-Augmented Generation (RAG)** solves this by searching your documents at query time and passing the relevant ones to the model as context.

This module builds a complete RAG pipeline for a consumer electronics Q&A system, using Amazon S3 Vectors as the vector database and a Gradio web interface.

---

## AI Inference Concepts Demonstrated Here

### What is RAG? (Plain English)

Think of RAG like an "open-book exam" for the AI:

```
Closed-book (standard inference):
  Examiner: "What is the best 4K TV under $800 in 2026?"
  Student (model): Answers from memory. Training cutoff = uncertain.
  Problem: Training data may be outdated. May hallucinate products.

Open-book (RAG inference):
  Examiner: "What is the best 4K TV under $800 in 2026?"
  System: Searches the electronics catalog. Finds 3 relevant entries.
  System: Puts those entries in front of the student (model).
  Student: "Based on our catalog, the LG C3 OLED at $749..."
  Benefit: Answer grounded in actual current data. No hallucination.
```

### The 4-Step RAG Process

```
Phase 1 (one-time, offline): DOCUMENT INGESTION
─────────────────────────────────────────────────
Your documents
  → Split into chunks (200-512 tokens each)
  → Embed each chunk (convert text to a list of numbers)
  → Store (chunk text + embedding vector) in S3 Vectors

Phase 2 (real-time, per query): RETRIEVAL + GENERATION
────────────────────────────────────────────────────────
User question
  → Embed the question (same embedding model)
  → Search S3 Vectors for most similar chunks
  → Top-K chunks retrieved (e.g., K=3)
  → Add chunks to the prompt as context
  → Send augmented prompt to vLLM
  → Model generates answer grounded in retrieved context
  → Return answer to user
```

### What is an Embedding?

An embedding converts text into a list of numbers (a vector) that captures meaning:

```
Text → Embedding Model → Vector (list of ~384 numbers)

"OLED television"  → [0.12, -0.34, 0.56, 0.21, ...]   (384 dimensions)
"TV display panel" → [0.11, -0.31, 0.58, 0.19, ...]   (very similar!)
"Python tutorial"  → [-0.45, 0.78, -0.12, 0.34, ...]  (very different)

The numbers capture MEANING, not just characters.
Similar meanings → similar vectors → close in vector space
```

Similarity between vectors is measured by **cosine similarity** (angle between vectors):
- 1.0 = identical meaning
- 0.0 = completely unrelated
- -1.0 = opposite meaning (rare in practice)

### Amazon S3 Vectors – Vector Database Built Into S3

Traditional RAG uses a separate vector database (Pinecone, Weaviate, pgvector, etc.) which adds infrastructure complexity.

Amazon S3 Vectors is a native vector search capability built directly into Amazon S3:
- No separate server to manage
- Serverless (no running cost when idle)
- Scales automatically
- IAM-integrated access control
- Data stays in your S3 environment

---

## Full Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EKS Cluster – default namespace                                              │
│                                                                               │
│  ─── PHASE 1: Document Ingestion (run once) ──────────────────────────────  │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Job: rag-document-processor                                          │   │
│  │  Image: python:3.11-slim (CPU only – no GPU needed for embedding)    │   │
│  │  ServiceAccount: s3-access-sa                                         │   │
│  │                                                                       │   │
│  │  1. Read: electronics.jsonl (22 KB, 25 product documents)            │   │
│  │  2. Embed: sentence-transformers (384-dim vectors, CPU)              │   │
│  │  3. Store: S3 Vectors API → vector index                             │   │
│  │                                                                       │   │
│  │  Resources: { cpu: "1-2", memory: "2-4Gi" } (CPU-only)              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                   │ put_vectors()                            │
│                                   ▼                                          │
│  ─── PHASE 2: Real-time Inference ─────────────────────────────────────── │
│                                                                               │
│  User                                                                        │
│   ▼ browser                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Deployment: rag-gradio-interface (m5.xlarge CPU node)               │   │
│  │  Image: python:3.11-slim  Port: 7860  Deps: gradio==5.49.1           │   │
│  │                                                                       │   │
│  │  ENV: RAG_SERVICE_HOST=rag-service  RAG_SERVICE_PORT=80             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│   │ Ingress: rag-gradio-alb (ALB, internet-facing)                          │
│   │ POST http://rag-service:80/query                                        │
│   ▼                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Deployment: rag-service (FastAPI)  Port: 8080                        │   │
│  │  ServiceAccount: s3-access-sa (S3 Vectors access)                    │   │
│  │                                                                       │   │
│  │  1. Embed query → 384-dim vector                                     │   │
│  │  2. query_vectors() → top-3 documents from S3 Vectors               │   │
│  │  3. Build augmented prompt (question + context)                       │   │
│  │  4. POST /v1/chat/completions → vLLM                                 │   │
│  │  5. Return answer + source documents                                  │   │
│  │                                                                       │   │
│  │  Deps: boto3, sentence_transformers, fastapi, aiohttp                │   │
│  │  Resources: { cpu: "2", memory: "4Gi" }                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│   │ POST /v1/chat/completions                                               │
│   ▼                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GPU Node: mistral vLLM Deployment (Module 100)                       │   │
│  │  Generates the final answer using retrieved context                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  Amazon S3 Vectors                                                           │
│  Bucket: ${S3_VECTOR_BUCKET_NAME}                                            │
│  Index:  ${S3_VECTOR_INDEX_NAME}                                             │
│  Stores: 25 product embeddings (384-dim float32 vectors + metadata)         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## RAG Query Flow (Step-by-Step)

```
User: "Which TVs have good HDR performance under $1000?"

Step 1 – Query Embedding (rag-service)
───────────────────────────────────────
sentence_transformers.encode("Which TVs have good HDR performance under $1000?")
→ [0.15, -0.22, 0.48, ..., 0.31]   (384 numbers)
Time: ~20ms on CPU (sentence-transformers is fast)

Step 2 – Vector Search (S3 Vectors API)
────────────────────────────────────────
boto3 s3vectors client:
query_vectors(
  vectorBucketName = "rag-vectors-<ACCOUNT_ID>",
  indexName        = "electronics-index",
  queryVector      = {"float32": [0.15, -0.22, ...]},
  topK             = 3,                    ← Return top 3 most similar
  returnMetadata   = True                  ← Include original text in response
)

S3 Vectors computes cosine similarity between query vector
and all stored product vectors. Returns top 3 matches:

Match 1 (score: 0.91): LG C3 OLED 55" – $749
  "Exceptional HDR performance with Dolby Vision IQ and HDR10+..."

Match 2 (score: 0.87): Sony A80L OLED 55" – $899
  "Outstanding HDR with XR OLED Contrast Pro, 800 nits peak brightness..."

Match 3 (score: 0.83): Samsung QN90C QLED 65" – $997
  "Quantum HDR 32× with Neo QLED, 2000 nits peak brightness..."

Step 3 – Prompt Augmentation (rag-service)
──────────────────────────────────────────
augmented_prompt = f"""
You are a helpful electronics advisor. Answer the user's question
using ONLY the provided product information. If the answer is not
in the context, say so.

Product Information:
──────────────────
{match1.text}
{match2.text}
{match3.text}
──────────────────

User Question: Which TVs have good HDR performance under $1000?
"""

Step 4 – LLM Generation (vLLM)
────────────────────────────────
POST http://vllm-serve-svc:8000/v1/chat/completions
{
  "model": "ministral",
  "messages": [
    {"role": "user", "content": augmented_prompt}
  ]
}

Step 5 – Response
──────────────────
"Based on our catalog, three excellent HDR TVs under $1000:

1. LG C3 OLED 55" ($749): Best value HDR with Dolby Vision IQ
   and HDR10+. OLED technology delivers perfect blacks...

2. Sony A80L OLED 55" ($899): Premium HDR with XR OLED Contrast Pro,
   800 nits peak brightness...

3. Samsung QN90C QLED 65" ($997): Brightest option at 2000 nits
   peak brightness, ideal for well-lit rooms..."
```

---

## Document Ingestion Job (rag-document-job.yml)

```yaml
Job: rag-document-processor
  serviceAccount: s3-access-sa        # Needs S3 Vectors write permission
  backoffLimit: 2                     # Retry up to 2 times on failure
  restartPolicy: Never                # Each attempt starts fresh

Container image: python:3.11-slim
Install commands:
  apt-get install curl python3-dev build-essential
  # build-essential needed to compile some sentence_transformers dependencies

  pip install boto3==1.43.43
  # AWS SDK: for S3 Vectors API calls (put_vectors)

  pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
  # CPU-only PyTorch. Full torch (~2 GB) but no GPU kernels.
  # Embedding generation doesn't need GPU (it's fast on CPU for small batches).

  pip install sentence-transformers==5.6.0
  # HuggingFace library for text embeddings.
  # Auto-downloads the embedding model (e.g., all-MiniLM-L6-v2).
  # all-MiniLM-L6-v2: 80 MB, 384 dimensions, good quality/speed balance.

Environment:
  S3_VECTOR_BUCKET_NAME  The S3 bucket containing the vector index
  S3_VECTOR_INDEX_NAME   The vector index to write embeddings to
  S3_BUCKET_NAME         Source data bucket (electronics.jsonl)
  AWS_REGION             AWS region

Resources:
  requests: { cpu: "1", memory: "2Gi" }
  limits:   { cpu: "2", memory: "4Gi" }
  # CPU-only. No GPU needed. Embedding 25 documents takes ~30 seconds.
```

---

## RAG Service Deployment (rag-service.yml)

```yaml
Deployment: rag-service
  serviceAccount: s3-access-sa

  Environment:
    ACCOUNT_ID            AWS account ID (for S3 Vectors ARN construction)
    S3_VECTOR_BUCKET_NAME Vector database bucket
    S3_VECTOR_INDEX_NAME  Vector index name
    MODEL_ID              ministral (vLLM served model name)
    MODEL_ENDPOINT        http://vllm-serve-svc:8000/v1
    AWS_REGION            us-east-1

  Dependencies installed at startup:
    boto3==1.43.44          AWS SDK (S3 Vectors query_vectors API)
    aiohttp==3.14.1         Async HTTP client for vLLM calls
                            (async = doesn't block while waiting for LLM)
    uvicorn==0.51.0         ASGI server for FastAPI
    fastapi==0.139.0        Web framework (REST API endpoints)
    sentence_transformers==5.6.0  Query embedding

  Resources: { cpu: "2", memory: "4Gi" } both requests and limits
  # Memory 4 GB: ~80 MB model + ~2 GB framework overhead + buffer.
  # CPU 2: embedding is compute-intensive, 2 cores improves latency.

  Readiness probe:
    tcpSocket: port 8080
    initialDelaySeconds: 30   # Wait 30s before probing (install + model download)
    periodSeconds: 10
    failureThreshold: 60      # 10 min grace period for model download
```

---

## Gradio UI Deployment (rag-gradio-deploy.yml)

```yaml
Deployment: rag-gradio-interface
  nodeSelector:
    node.kubernetes.io/instance-type: "m5.xlarge"
    # CPU-only node. Gradio is a web app, not AI compute.

  Container:
    apt-get install curl                  # For health checks
    pip install requests pandas gradio==5.49.1
    python /app/rag-gradio-app.py         # The Gradio application

  ENV:
    RAG_SERVICE_HOST: "rag-service"       # Kubernetes DNS name
    RAG_SERVICE_PORT: "80"                # Service port

  Resources:
    requests: { cpu: "500m", memory: "1Gi" }
    limits:   { cpu: "1000m", memory: "2Gi" }
    # 1 Gi is enough for Gradio + Python process.

  Port: 7860  (Gradio's default)

Service: rag-gradio-interface
  ClusterIP: port 80 → 7860

Ingress: rag-gradio-alb
  Annotations:
    scheme: internet-facing              Public internet access
    target-type: ip                      Direct pod routing (no kube-proxy hop)
    healthcheck-path: /                  Gradio root path for health checks
    success-codes: 200-302,307,404       Gradio redirects on initial load
    load-balancer-name: rag-gradio-alb   Deterministic ALB name
```

---

## Amazon S3 Vectors – How It Works

```
Vector bucket: separate from regular S3 buckets
Vector index:  a searchable collection of vectors within a bucket

Stored structure per vector:
  {
    "key": "electronics_001_chunk_0",    ← Unique identifier
    "data": {
      "float32": [0.12, -0.34, ...]      ← 384 embedding dimensions
    },
    "metadata": {
      "text": "LG C3 OLED 55 inch...",  ← Original text (retrieved for prompt)
      "source": "electronics.jsonl",
      "doc_id": "001"
    }
  }

Operations used:
  put_vectors()    → Write during ingestion
  query_vectors()  → Read during inference (similarity search)
  
  Both operations use IAM authentication via s3-access-sa Pod Identity.
  No API keys, no connection strings. Standard AWS SDK calls.

Pricing model:
  Charged per vector stored + per query
  No server to run → costs $0 when idle
  Scales automatically with data volume
```

---

## IAM Permissions for s3-access-sa

```
ServiceAccount: s3-access-sa
→ EKS Pod Identity Association
→ IAM Role: rag-s3-access-role

Permissions required:
  s3:GetObject                     Download documents from S3
  s3:PutObject                     Upload processed data
  s3:ListBucket                    List vector bucket contents
  
  s3vectors:PutVectors             Write embeddings during ingestion
  s3vectors:QueryVectors           Similarity search during inference
  s3vectors:GetVectors             Retrieve specific vectors by key
  
  Resources:
    arn:aws:s3:::${S3_VECTOR_BUCKET_NAME}/*
    arn:aws:s3vectors::<ACCOUNT_ID>:bucket/${S3_VECTOR_BUCKET_NAME}/*
```

---

## Quick Start

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_VECTOR_BUCKET_NAME="rag-vectors-${AWS_ACCOUNT_ID}"
export S3_VECTOR_INDEX_NAME="electronics-index"
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"
export MODEL_ENDPOINT="http://vllm-serve-svc:8000/v1"
export MODEL_ID="ministral"
export AWS_REGION="us-east-1"
export ACCOUNT_ID="${AWS_ACCOUNT_ID}"

# Step 1: Run document ingestion job
envsubst < 700-rag/rag-document-job.yml | kubectl apply -f -
kubectl wait --for=condition=complete job/rag-document-processor --timeout=600s
kubectl logs job/rag-document-processor   # Verify success

# Step 2: Deploy RAG service
envsubst < 700-rag/rag-service.yml | kubectl apply -f -

# Step 3: Deploy Gradio UI
envsubst < 700-rag/rag-gradio-deploy.yml | kubectl apply -f -

# Step 4: Wait for deployments
kubectl rollout status deployment/rag-service
kubectl rollout status deployment/rag-gradio-interface

# Step 5: Get public URL
kubectl get ingress rag-gradio-alb \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Step 6: Test the RAG API directly
kubectl port-forward svc/rag-service 8080:80 &
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What TVs have the best picture quality?"}'

# Clean up
kubectl delete job rag-document-processor
```
