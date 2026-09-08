# Module 600 – LoRA Fine-Tuning: Domain Adaptation for AI Inference

> New to AI inference? Read [CONCEPTS.md](../CONCEPTS.md) first.
> Prerequisite: Module 100 (vLLM) must be running. GPU node required.

---

## What This Module Does

The Red Hat article says: "If the model is struggling to make accurate inferences after training, fine-tuning can add knowledge and improve accuracy."

This module fine-tunes Ministral-3-8B to become an expert Venture Capital (VC) advisor called "AnyVC". After fine-tuning, when you ask it about startup evaluations, pitch decks, and investment decisions, it responds with VC-specific vocabulary, frameworks, and depth that the general-purpose model lacks.

The approach is **LoRA** (Low-Rank Adaptation) – a parameter-efficient fine-tuning technique that trains only 0.12% of the model's parameters while achieving nearly the same result as full fine-tuning at 1% of the cost.

---

## AI Inference Concepts Demonstrated Here

### Fine-Tuning vs RAG: When to Choose Which

The Red Hat article links to "RAG vs. fine-tuning" as a key topic. Here is the practical decision framework:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  RAG (Module 700)               │  Fine-Tuning (Module 600)             │
├─────────────────────────────────┼───────────────────────────────────────┤
│  HOW: Retrieve documents at     │  HOW: Adjust model weights on a       │
│       query time, add to prompt │       domain-specific dataset         │
│                                 │                                       │
│  Best for:                      │  Best for:                            │
│  ✅ Dynamic/changing data       │  ✅ Stable domain knowledge           │
│     (product catalog, news)     │     (VC evaluation methodology)       │
│  ✅ Large document corpus       │  ✅ Specific response style/tone      │
│  ✅ No training budget          │  ✅ Domain-specific vocabulary        │
│  ✅ Immediate deployment        │  ✅ Latency-critical (no retrieval)   │
│                                 │                                       │
│  Limitations:                   │  Limitations:                         │
│  ❌ Adds latency (retrieval)    │  ❌ Requires training time/cost       │
│  ❌ Prompt length increases     │  ❌ Not for frequently-changing data  │
│  ❌ Hallucination still possible│  ❌ Can't easily update after training│
│                                 │                                       │
│  Used when: "I need the model   │  Used when: "I need the model to      │
│  to know what's in my database" │  BEHAVE differently in my domain"     │
└─────────────────────────────────┴───────────────────────────────────────┘
```

### What Is Parameter-Efficient Fine-Tuning (PEFT)?

Full fine-tuning updates all 3.8 billion parameters. The optimizer needs to store:
- Parameter values: 3.8B × 4 bytes (FP32) = 15.2 GB
- Gradients: another 15.2 GB
- Optimizer states (Adam): another 30.4 GB
- Total: ~60 GB → doesn't fit on a 48 GB GPU!

**PEFT** solves this by modifying only a tiny fraction of parameters. LoRA is the most popular PEFT technique.

### How LoRA Works

```
Standard Transformer Attention layer (simplified):
  Output = W × Input
  W is a weight matrix, e.g., 4096 × 4096 = 16.7M parameters per layer

LoRA adds two small matrices alongside W:
  Output = W × Input + (B × A) × Input × (α/r)

  where:
    A: r × 4096  (r rows, original dimension columns)
    B: 4096 × r  (original dimension rows, r columns)
    r = rank (how many dimensions of "change" we allow)
    α = alpha (scaling factor for the adaptation)

With r=16 (as used in this workshop):
  A: 16 × 4096   = 65,536 parameters
  B: 4096 × 16   = 65,536 parameters
  Per layer: 131,072 parameters (vs 16,777,216 for the full weight)
  Reduction: 128× fewer parameters per layer

Total trainable parameters with LoRA r=16:
  4 target modules (q, k, v, o) × 32 layers × 131,072 = ~16.8M
  (In practice ~4.7M because not all modules are in all layers)

  Full fine-tuning: 3,800,000,000 parameters
  LoRA fine-tuning:       4,700,000 parameters
  Savings: 99.88% fewer trainable parameters!
```

---

## Full Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EKS Cluster – default namespace                                              │
│                                                                               │
│  PHASE 1: Training (runs once, ~20-40 minutes)                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GPU Node (g6e.2xlarge) – NVIDIA L40S 48 GB                          │   │
│  │                                                                       │   │
│  │  Job: anyvc-lora-finetune (Kubernetes batch/v1 Job)                  │   │
│  │  Image: 763104351884.dkr.ecr.<REGION>.amazonaws.com/                 │   │
│  │         pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2       │   │
│  │  ServiceAccount: model-storage-sa                                    │   │
│  │                                                                       │   │
│  │  Volumes:                                                             │   │
│  │    /models  ← PVC (S3 CSI driver, read-only S3 bucket mount)        │   │
│  │    /scripts ← ConfigMap (training script + dataset)                  │   │
│  │    /output  ← emptyDir 5Gi (stores adapter before S3 upload)        │   │
│  │                                                                       │   │
│  │  Resources:                                                           │   │
│  │    requests: { cpu: 3, memory: 24Gi, nvidia.com/gpu: 1 }            │   │
│  │    limits:   { cpu: 4, memory: 28Gi, nvidia.com/gpu: 1 }            │   │
│  │                                                                       │   │
│  │  Output: adapter_model.safetensors → s3://…/anyvc-startup-lora/     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  PHASE 2: Serving LoRA Adapter (after training completes)                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GPU Node (g6e.2xlarge) – NVIDIA L40S 48 GB                          │   │
│  │                                                                       │   │
│  │  Deployment: anyvc (vLLM with LoRA enabled)                          │   │
│  │  Volume: s3-bucket-pvc (CSI S3 driver, mounts at /s3)               │   │
│  │                                                                       │   │
│  │  vLLM args:                                                           │   │
│  │    --enable-lora                                                      │   │
│  │    --max-lora-rank=16                                                 │   │
│  │    --lora-modules={name, path, base_model_name}                      │   │
│  │                                                                       │   │
│  │  API: POST /v1/chat/completions                                       │   │
│  │    model: "ministral"            ← uses base model                   │   │
│  │    model: "anyvc-startup-expert" ← uses LoRA adapter                 │   │
│  │  (Both served simultaneously from the same vLLM instance!)           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  Amazon S3: s3://genai-models-<ACCOUNT_ID>/anyvc-startup-lora/               │
│    adapter_model.safetensors   LoRA weight deltas (~50-100 MB)               │
│    adapter_config.json         LoRA configuration (rank, alpha, modules)     │
│    tokenizer.*                 Tokenizer copy for reference                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Configuration (lora-training-job.yaml): Every Parameter

```yaml
Job:
  name: anyvc-lora-finetune
  backoffLimit: 0       # Don't retry if it fails (training failure needs debugging)
  ttlSecondsAfterFinished: 3600  # Auto-delete pod 1hr after completion

Container image:
  763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:
  2.8.0-gpu-py312-cu129-ubuntu22.04-ec2
  # AWS PyTorch training DLC: pre-configured with PyTorch 2.8, CUDA 12.9
  # Optimised for training (vs the vLLM image used for inference)
  # Includes: NCCL, APEX, cuDNN, compatible compiler toolchain

Environment variables (hyperparameters):

HF_MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512"
  # HuggingFace Hub model ID. Used by from_pretrained() to download
  # the model architecture definition and tokenizer config.
  # The actual weights are loaded from /models (local S3 mount),
  # but from_pretrained() needs the Hub ID to get config files
  # that aren't in the local SafeTensors file.

MODEL_PATH = "/models/Ministral-3-8B-Instruct-2512"
  # Path to local model weights (mounted from S3 via PVC).
  # SafeTensors file is read directly from here.

DATASET_PATH = "/scripts/anyvc-startup-dataset.jsonl"
  # Path to the training dataset (mounted from ConfigMap).
  # JSONL format: one JSON object per line.
  # Each object: {"messages": [{"role":"system"...},{"role":"user"...},...]}

OUTPUT_DIR = "/output/anyvc-startup-lora"
  # Where to save the trained adapter files.
  # Saved to emptyDir first, then uploaded to S3.

S3_OUTPUT_PATH = "s3://${S3_BUCKET_NAME}/anyvc-startup-lora/"
  # S3 destination for the trained adapter.
  # boto3 uploads each file in OUTPUT_DIR to this path.

NUM_EPOCHS = "5"
  # Number of complete passes through the training dataset.
  # Epoch 1: model sees all examples once
  # Epoch 5: model has seen each example 5 times
  # More epochs = better learning but risk of overfitting.
  # 5 is a reasonable value for small domain datasets.

BATCH_SIZE = "2"
  # Number of training examples processed per GPU step.
  # 2 = process 2 examples simultaneously (memory allows ~2-4 for 512-token seqs)
  # Higher = more GPU efficiency but more VRAM needed.

GRADIENT_ACCUMULATION = "2"
  # Accumulate gradients over N steps before updating weights.
  # Effective batch size = BATCH_SIZE × GRADIENT_ACCUMULATION = 2 × 2 = 4
  # This simulates a batch size of 4 while only holding 2 in VRAM.
  # Useful when you want larger effective batches but can't fit them in memory.

LEARNING_RATE = "2e-4"
  # Step size for gradient descent. 0.0002.
  # Too high: training instability, loss diverges.
  # Too low: training converges very slowly or gets stuck.
  # 2e-4 is the standard starting point for LoRA fine-tuning.

MAX_SEQ_LENGTH = "512"
  # Maximum tokens per training example.
  # Examples longer than 512 tokens are truncated.
  # Shorter examples are padded to 512.
  # 512 is sufficient for most VC advisor conversations.

LORA_RANK = "16"
  # The "r" in LoRA. Dimensionality of the adaptation matrices.
  # r=4:  Minimal adaptation, fewest params, fastest training
  # r=8:  Light adaptation, good for simple style changes
  # r=16: Moderate adaptation, good for domain knowledge (used here)
  # r=32: Heavy adaptation, for complex domain changes
  # r=64: Very heavy, approaches full fine-tuning quality but larger file

LORA_ALPHA = "32"
  # Scaling factor applied to the LoRA output.
  # Effective learning rate for LoRA = (alpha/rank) × base_lr
  # alpha=32, rank=16 → scale=2 → LoRA updates 2× the base learning rate.
  # Common practice: set alpha = 2 × rank (keeps scale constant as rank changes).
```

---

## Training Script: 8-Step Pipeline (train_lora.py)

```
[Step 1/8] Validate model files
  os.listdir(/models/Ministral-3-8B-Instruct-2512)
  Confirms: consolidated.safetensors exists (10.4 GB)
  Why: Fail fast if the PVC mount is wrong, before wasting time.

[Step 2/8] Load dataset
  File: anyvc-startup-dataset.jsonl (240 KB, ~hundreds of conversations)
  Format: {"messages": [system, user, assistant, user, assistant, ...]}
  Domain: VC advisor evaluating startup pitches, market analysis,
          investment thesis development, pitch deck critique

[Step 3/8] Load tokenizer
  Primary: MistralCommonBackend (mode="finetuning")
           → Native Mistral tokenizer class for training
  Fallback: AutoTokenizer (trust_remote_code=True)
  
  Important settings:
    pad_token = eos_token  (Mistral has no dedicated padding token)
    padding_side = "right" (pad after sequence, not before)
    These are required for SFTTrainer to batch correctly.

[Step 4/8] Load model (FP8 → BF16)
  Mistral3ForConditionalGeneration.from_pretrained(
    quantization_config = FineGrainedFP8Config(dequantize=True)
    # The model weights are stored as FP8 in the SafeTensors file.
    # dequantize=True: load FP8, immediately convert to BF16 for training.
    # FP8 inference is fast; BF16 training is more numerically stable.
    
    device_map = "auto"
    # PyTorch automatically distributes model layers across available GPUs.
    # On single GPU: all layers go to GPU 0.
    
    attn_implementation = "eager"
    # Use standard (eager) attention, not FlashAttention.
    # FlashAttention is optimised for inference; training requires gradients
    # which FlashAttention v2 supports but with more complex setup.
  )
  
  gradient_checkpointing = True
  # Saves intermediate activations during forward pass, recomputes them
  # during backward pass instead of storing them all.
  # Tradeoff: 30% slower training → 50% less VRAM for activations.
  # Without this: training OOMs on 48 GB GPU with long sequences.
  
  Freeze all base parameters: param.requires_grad = False
  # After LoRA is attached, ONLY the LoRA A and B matrices are trained.
  # Base weights are frozen (not updated, not stored in optimizer).

[Step 5/8] Attach LoRA adapters
  LoraConfig:
    r = 16                            (rank)
    lora_alpha = 32                   (scaling)
    target_modules = [q_proj, k_proj, v_proj, o_proj]
    # WHY these 4 modules:
    #   q_proj: How the model formulates questions (what to attend to)
    #   k_proj: What keys each token offers for others to attend to
    #   v_proj: The actual information a token provides when attended to
    #   o_proj: How attended information is projected to output
    #   These 4 control the model's attention patterns = domain focus
    
    lora_dropout = 0.05               (5% dropout for regularisation)
    bias = "none"                     (don't train bias terms)
    task_type = "CAUSAL_LM"          (autoregressive language model)
  
  Trainable parameters: ~4.7M / 3.8B = 0.12% of total

[Step 6/8] Format dataset
  Apply chat template to each example:
  tokenizer.apply_chat_template(messages) 
  → "<s>[INST] You are AnyVC... [/INST] ... </s>"
  This wraps the conversation in the format the model was trained to expect.

[Step 7/8] Train with SFTTrainer (TRL library)
  SFTTrainer = Supervised Fine-tuning Trainer
  Handles: data collation, loss computation, gradient updates, checkpointing
  
  TrainingArguments:
    bf16 = True                    BFloat16 training (matches GPU capability)
    optim = "adamw_torch"          AdamW optimiser (standard for transformers)
    save_strategy = "epoch"        Save checkpoint after each epoch
    report_to = "none"             Don't send metrics to W&B or HuggingFace Hub
    seed = 42                      Reproducible training

[Step 8/8] Save adapter + upload to S3
  model.save_pretrained(/output/anyvc-startup-lora)
  → adapter_model.safetensors  (the trained A and B matrices)
  → adapter_config.json        (rank=16, alpha=32, target_modules)
  
  Key tensor key fix:
  "model.model.language_model.X" → "model.language_model.model.X"
  (Mistral-3 wraps its language model in an extra layer; vLLM expects
   a different key naming convention when loading LoRA adapters.)
  
  boto3 upload: each file → s3://${S3_BUCKET_NAME}/anyvc-startup-lora/
```

---

## Serving with LoRA (vllm-with-lora.yaml)

### S3 CSI Driver Mount

```yaml
PersistentVolume: s3-bucket-pv
  driver: s3.csi.aws.com           # AWS S3 CSI Driver (must be installed)
  bucketName: ${S3_BUCKET_NAME}    # Mounts entire bucket as a filesystem
  region: ${AWS_REGION}
  accessModes: [ReadOnlyMany]      # Multiple pods can mount read-only

PersistentVolumeClaim: s3-bucket-pvc
  → Mounted at /s3 inside the vLLM pod
  → LoRA adapter at /s3/anyvc-startup-lora/adapter_model.safetensors
```

### vLLM LoRA Arguments

```
--enable-lora
  Activates vLLM's LoRA serving subsystem.
  Multiple adapters can be loaded simultaneously.
  Each adapter is loaded on-demand per request.

--max-lora-rank=16
  Maximum rank of any loaded LoRA adapter.
  Must be ≥ the trained adapter's rank (r=16).
  This pre-allocates GPU memory for LoRA weight matrices.
  Setting too high wastes GPU memory; too low prevents loading.

--lora-modules={"name":"anyvc-startup-expert",
                "path":"/s3/anyvc-startup-lora",
                "base_model_name":"ministral"}

  name: "anyvc-startup-expert"
    The model name clients use to request this adapter.
    POST /v1/chat/completions { "model": "anyvc-startup-expert" }
    Distinct from the base model name "ministral".

  path: "/s3/anyvc-startup-lora"
    Where to find the adapter files (mounted from S3 via CSI driver).
    vLLM reads adapter_model.safetensors and adapter_config.json.

  base_model_name: "ministral"
    Links this adapter to the base model.
    Must match --served-model-name.
    Ensures the adapter is applied to the correct base model weights.
```

---

## Memory Profile During Training

```
NVIDIA L40S (48 GB VRAM) during LoRA training:

┌─────────────────────────────────────────────────────────────┐
│  Base model weights (FP8 → BF16):         ~7.5 GB          │
│  (3.8B parameters × 2 bytes BFloat16)                       │
│                                                              │
│  LoRA adapter weights (trainable):        ~0.1 GB           │
│  (4.7M parameters × 4 bytes FP32)                           │
│                                                              │
│  Adam optimizer states:                   ~0.2 GB           │
│  (2 states per trainable param × FP32)                       │
│                                                              │
│  Activation memory (gradient ckpt ON):   ~10–12 GB          │
│  (intermediate tensors for 2-seq × 512-token batch)          │
│  (without gradient_checkpointing: ~20–25 GB → OOM)          │
│                                                              │
│  CUDA runtime + NCCL + framework:        ~2–3 GB            │
│                                                              │
│  Total: ~20–23 GB  ✅ Fits in 48 GB with room to spare       │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"
export AWS_REGION="us-east-1"

# Create ConfigMap with training scripts and dataset
kubectl create configmap anyvc-training-scripts \
  --from-file=600-finetuning/train_lora.py \
  --from-file=600-finetuning/anyvc-startup-dataset.jsonl

# Submit the training job
envsubst < 600-finetuning/lora-training-job.yaml | kubectl apply -f -

# Watch training progress (live log streaming)
kubectl logs -f job/anyvc-lora-finetune

# Check S3 for completed adapter (~20-40 min)
aws s3 ls s3://${S3_BUCKET_NAME}/anyvc-startup-lora/

# Deploy vLLM with LoRA serving
envsubst < 600-finetuning/vllm-with-lora.yaml | kubectl apply -f -

# Wait for deployment
kubectl rollout status deployment/anyvc

# List available models (should show both base + LoRA)
kubectl port-forward svc/vllm-serve-svc 8000:8000 &
curl http://localhost:8000/v1/models

# Compare: base model vs LoRA-adapted model
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ministral","messages":[
    {"role":"user","content":"Evaluate this startup: AI tool for legal research"}
  ]}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"anyvc-startup-expert","messages":[
    {"role":"user","content":"Evaluate this startup: AI tool for legal research"}
  ]}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
# The LoRA-adapted model should respond with more VC-specific framework
# (market sizing, TAM/SAM/SOM, competitive moat, revenue model analysis)
```
