# Module 700 – RAG Pipeline Architecture

> Implements a Retrieval-Augmented Generation (RAG) pipeline using Amazon S3 Vectors as the vector database. Documents are embedded with `sentence-transformers`, stored in an S3 vector index, and queried at inference time to provide relevant context to the Ministral-3-8B model. A Gradio web UI provides a public-facing interface.

---

## End-to-End Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        EKS Cluster – default namespace                        │
│                                                                               │
│  PHASE 1: Document Ingestion (one-time job)                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Job: rag-document-processor  (batch/v1)                              │   │
│  │  serviceAccount: s3-access-sa                                         │   │
│  │  image: python:3.11-slim                                               │   │
│  │                                                                       │   │
│  │  1. Reads: electronics.jsonl (sample document corpus)                │   │
│  │  2. Embeds: sentence-transformers (all-MiniLM-L6-v2 or similar)      │   │
│  │  3. Stores: embeddings → S3 Vector Index                             │   │
│  │                                                                       │   │
│  │  Dependencies:                                                        │   │
│  │    boto3==1.43.43                                                     │   │
│  │    torch==2.13.0+cpu (CPU-only, no GPU needed for embedding)          │   │
│  │    sentence-transformers==5.6.0                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                              │
│                                ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Amazon S3 Vector Index                                               │   │
│  │  Bucket: ${S3_VECTOR_BUCKET_NAME}                                    │   │
│  │  Index:  ${S3_VECTOR_INDEX_NAME}                                     │   │
│  │  Region: ${AWS_REGION}                                               │   │
│  │  Stores: (doc_id, embedding_vector, metadata, text_chunk)            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                              │
│  PHASE 2: Inference (real-time)│                                              │
│                                ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Deployment: rag-service  (FastAPI)                                   │   │
│  │  serviceAccount: s3-access-sa                                         │   │
│  │  image: python:3.11-slim                                               │   │
│  │  Port: 8080  |  replicas: 1                                           │   │
│  │  nodeSelector: (default – system node)                                │   │
│  │                                                                       │   │
│  │  Dependencies:                                                        │   │
│  │    boto3==1.43.44           S3 Vectors API                           │   │
│  │    sentence_transformers==5.6.0  Query embedding                     │   │
│  │    fastapi==0.139.0         REST API framework                        │   │
│  │    uvicorn==0.51.0          ASGI server                               │   │
│  │    aiohttp==3.14.1          Async HTTP (vLLM calls)                  │   │
│  │                                                                       │   │
│  │  ENV:                                                                 │   │
│  │    ACCOUNT_ID              AWS account ID                            │   │
│  │    S3_VECTOR_BUCKET_NAME   S3 bucket with vector index               │   │
│  │    S3_VECTOR_INDEX_NAME    Name of the S3 vector index               │   │
│  │    MODEL_ID                ministral                                  │   │
│  │    MODEL_ENDPOINT          http://vllm-serve-svc:8000/v1             │   │
│  │    AWS_REGION              us-east-1                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                              │
│   Service: rag-service (ClusterIP, port 80 → 8080)                          │
│                                │                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Deployment: rag-gradio-interface  (Gradio UI)                        │   │
│  │  image: python:3.11-slim                                               │   │
│  │  Port: 7860  |  nodeSelector: m5.xlarge                               │   │
│  │                                                                       │   │
│  │  Dependencies:                                                        │   │
│  │    gradio==5.49.1   requests==2.33.1   pandas==2.3.3                 │   │
│  │                                                                       │   │
│  │  ENV:                                                                 │   │
│  │    RAG_SERVICE_HOST = rag-service                                    │   │
│  │    RAG_SERVICE_PORT = 80                                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                              │
│   Service: rag-gradio-interface (ClusterIP, port 80 → 7860)                 │
│   Ingress: rag-gradio-alb (ALB, internet-facing)                            │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GPU Node Pool – Pod: mistral (vLLM, Module 100)                     │   │
│  │  Service: vllm-serve-svc:8000                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘

Internet
  │
  ▼ HTTP
┌──────────────────────┐
│  ALB: rag-gradio-alb  │  ← http://<ALB_DNS>
│  inbound: 0.0.0.0/0  │
└──────────────────────┘
```

---

## RAG Query Flow (Real-time)

```
User types question: "What are the best OLED TV recommendations under $1000?"
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Gradio UI (port 7860)                                                │
│  POST http://rag-service:80/query { "question": "..." }              │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RAG Service (rag_serve.py)                                           │
│                                                                       │
│  Step 1: Embed the query                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  sentence_transformers.encode("What are the best OLED...")   │   │
│  │  Output: float32 vector of dimension D (e.g., 384-dim)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  Step 2: Vector similarity search (S3 Vectors)                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  boto3 s3vectors client                                       │   │
│  │  query_vectors(                                               │   │
│  │    vectorBucketName = S3_VECTOR_BUCKET_NAME,                  │   │
│  │    indexName        = S3_VECTOR_INDEX_NAME,                   │   │
│  │    queryVector      = [0.12, -0.34, 0.56, ...],              │   │
│  │    topK             = 3,                                      │   │
│  │    returnMetadata   = True                                    │   │
│  │  )                                                            │   │
│  │  Returns: top-3 document chunks by cosine similarity         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  Step 3: Build augmented prompt                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  context = "\n".join([doc.text for doc in top3_docs])        │   │
│  │  augmented_prompt = f"""                                      │   │
│  │    Context: {context}                                         │   │
│  │                                                               │   │
│  │    Question: {user_question}                                  │   │
│  │                                                               │   │
│  │    Answer based on the context above:                        │   │
│  │  """                                                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  Step 4: LLM generation                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  aiohttp POST http://vllm-serve-svc:8000/v1/chat/completions │   │
│  │  {                                                            │   │
│  │    "model": "ministral",                                      │   │
│  │    "messages": [{"role":"user","content": augmented_prompt}] │   │
│  │  }                                                            │   │
│  │  Returns: streamed response text                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
Gradio UI renders the response with retrieved document sources
```

---

## Document Ingestion Flow (One-time Job)

```
File: electronics.jsonl (22 KB sample corpus)
Format: JSONL, one document per line
Domain: Consumer electronics product data/reviews

┌──────────────────────────────────────────────────────────────────────┐
│  rag-document-processor Job (rag-processor.py)                        │
│                                                                       │
│  1. Read documents from electronics.jsonl                            │
│     Each line: { "id": "...", "text": "...", "metadata": {...} }     │
│                                                                       │
│  2. Chunk documents (if long)                                         │
│     Split at sentence boundaries, ~200–512 tokens per chunk          │
│                                                                       │
│  3. Embed each chunk                                                  │
│     sentence_transformers.encode(chunks)                             │
│     Model: all-MiniLM-L6-v2 (384 dimensions, CPU-only)              │
│     Batch embedding for efficiency                                    │
│                                                                       │
│  4. Upload to S3 Vector Index                                         │
│     boto3 s3vectors.put_vectors(                                     │
│       vectorBucketName = S3_VECTOR_BUCKET_NAME,                      │
│       indexName        = S3_VECTOR_INDEX_NAME,                       │
│       vectors = [                                                     │
│         { "key": "doc_001_chunk_0",                                  │
│           "data": {"float32": [0.12, -0.34, ...]},                  │
│           "metadata": {"text": "chunk text...", "source": "..."}     │
│         },                                                            │
│         ...                                                           │
│       ]                                                               │
│     )                                                                 │
│                                                                       │
│  Resources:                                                           │
│    requests: { cpu: "1",  memory: "2Gi" }                            │
│    limits:   { cpu: "2",  memory: "4Gi" }                            │
│    (No GPU required – CPU embedding only)                            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Amazon S3 Vectors

```
Amazon S3 Vectors (new AWS service):
  A native vector database built into S3.
  No separate vector DB infrastructure to manage.
  Scales automatically with S3 storage.

Configuration:
  vectorBucketName: ${S3_VECTOR_BUCKET_NAME}
  indexName:        ${S3_VECTOR_INDEX_NAME}

Operations used:
  put_vectors()    → store embeddings during ingestion
  query_vectors()  → similarity search at inference time

Key advantages over self-managed vector DBs:
  - No running cost when idle
  - Serverless (no cluster to manage)
  - IAM-integrated access control via EKS Pod Identity
  - Same S3 data lake, no data silos
```

---

## Kubernetes Resources

### RAG Service Deployment (rag-service.yml)

```
Deployment: rag-service
  serviceAccount: s3-access-sa       # IAM role for S3 Vectors access
  image: python:3.11-slim
  Port: 8080 | requests: {cpu:"2", memory:"4Gi"} | limits: {cpu:"2", memory:"4Gi"}

  Volume: ConfigMap rag-serve-script → /app/rag_serve.py

Service: rag-service
  type: ClusterIP | port: 80 → 8080
```

### Gradio UI Deployment (rag-gradio-deploy.yml)

```
Deployment: rag-gradio-interface
  nodeSelector: m5.xlarge             # CPU-only node
  image: python:3.11-slim
  Port: 7860
  requests: {cpu:"500m", memory:"1Gi"} | limits: {cpu:"1000m", memory:"2Gi"}

  Volume: ConfigMap rag-gradio-app → /app/rag-gradio-app.py

Service: rag-gradio-interface
  type: ClusterIP | port: 80 → 7860

Ingress: rag-gradio-alb
  scheme: internet-facing
  target-type: ip
  healthcheck-path: /
  success-codes: 200-302,307,404    # Gradio redirects on first load
  load-balancer-name: rag-gradio-alb
```

### Document Processor Job (rag-document-job.yml)

```
Job: rag-document-processor
  serviceAccount: s3-access-sa       # S3 Vectors write access
  backoffLimit: 2
  restartPolicy: Never

  image: python:3.11-slim
  install: boto3==1.43.43, torch==2.13.0+cpu, sentence-transformers==5.6.0

  Volume: ConfigMap rag-processor-script → /app/rag-processor.py
  requests: {cpu:"1", memory:"2Gi"} | limits: {cpu:"2", memory:"4Gi"}
```

---

## IAM Access (s3-access-sa)

```
ServiceAccount: s3-access-sa
  → EKS Pod Identity Association
  → IAM Role with permissions:
       s3:GetObject, s3:PutObject, s3:ListBucket
         Resource: arn:aws:s3:::${S3_VECTOR_BUCKET_NAME}/*

       s3vectors:PutVectors, s3vectors:QueryVectors, s3vectors:GetVectors
         Resource: arn:aws:s3vectors:::<account>:bucket/${S3_VECTOR_BUCKET_NAME}
                   /index/${S3_VECTOR_INDEX_NAME}
```

---

## Component Comparison: With vs Without RAG

```
Without RAG (Module 100 – base vLLM):
  User: "Best OLED TV under $1000?"
  Model: Answers from training data (may be outdated)
  Limitation: Knowledge cutoff, no product-specific data

With RAG (Module 700):
  User: "Best OLED TV under $1000?"
  System:
    1. Embed query → search electronics corpus
    2. Retrieve: "LG C3 OLED 55" - $899, rated 9.2/10"
                 "Sony A80L - $950, great HDR performance"
    3. Augment prompt with retrieved context
    4. Model answers with real, up-to-date product data
  Benefit: Grounded, accurate, domain-specific responses
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

# Step 1: Create S3 vector index (via AWS CLI or console)
aws s3vectors create-vector-bucket --vector-bucket-name ${S3_VECTOR_BUCKET_NAME}
aws s3vectors create-index \
  --vector-bucket-name ${S3_VECTOR_BUCKET_NAME} \
  --index-name ${S3_VECTOR_INDEX_NAME} \
  --dimension 384 --metric-type cosine

# Step 2: Run document ingestion job
envsubst < rag-document-job.yml | kubectl apply -f -
kubectl wait --for=condition=complete job/rag-document-processor --timeout=600s

# Step 3: Deploy RAG service
envsubst < rag-service.yml | kubectl apply -f -

# Step 4: Deploy Gradio UI
envsubst < rag-gradio-deploy.yml | kubectl apply -f -

# Step 5: Get public URL
kubectl get ingress rag-gradio-alb \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```
