# Module 600 – LoRA Fine-tuning Architecture

> Fine-tunes Ministral-3-8B-Instruct-2512 with Low-Rank Adaptation (LoRA) to create a domain-specific "AnyVC Startup Advisor" model. Uses PEFT + TRL on a single NVIDIA L40S GPU, saves the adapter to S3, then loads it dynamically in vLLM via the `--enable-lora` flag.

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        EKS Cluster – default namespace                        │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GPU Node Pool (g6e.2xlarge)  |  karpenter.sh/nodepool=gpu           │   │
│  │                                                                       │   │
│  │  Phase 1: Training Job                                                │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Job: anyvc-lora-finetune  (batch/v1)                          │  │   │
│  │  │  backoffLimit: 0  |  ttlSecondsAfterFinished: 3600             │  │   │
│  │  │  serviceAccount: model-storage-sa                               │  │   │
│  │  │                                                                  │  │   │
│  │  │  Container: trainer                                             │  │   │
│  │  │  image: 763104351884.dkr.ecr.<REGION>.amazonaws.com/           │  │   │
│  │  │         pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2  │  │   │
│  │  │                                                                  │  │   │
│  │  │  Volumes:                                                        │  │   │
│  │  │    /models  ← PVC (mistral-model-pvc, ReadOnlyMany)             │  │   │
│  │  │    /scripts ← ConfigMap (anyvc-training-scripts, ReadOnly)      │  │   │
│  │  │    /output  ← emptyDir (sizeLimit: 5Gi)                         │  │   │
│  │  │                                                                  │  │   │
│  │  │  Resources:                                                      │  │   │
│  │  │    requests: { cpu: "3", memory: "24Gi", nvidia.com/gpu: "1" } │  │   │
│  │  │    limits:   { cpu: "4", memory: "28Gi", nvidia.com/gpu: "1" } │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  Phase 2: Inference with LoRA                                         │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Deployment: anyvc (vLLM with LoRA)                            │  │   │
│  │  │  image: vllm:0.21.0-gpu-py312-cu130-ubuntu22.04-ec2-v1.0-soci  │  │   │
│  │  │  --enable-lora                                                   │  │   │
│  │  │  --max-lora-rank=16                                             │  │   │
│  │  │  --lora-modules={"name":"anyvc-startup-expert",...}             │  │   │
│  │  │  Volume: s3-bucket-pvc (ReadOnlyMany, CSI S3 driver)           │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Amazon S3: s3://genai-models-<ACCOUNT_ID>/                          │   │
│  │    Ministral-3-8B-Instruct-2512/  (base model, read-only)            │   │
│  │    anyvc-startup-lora/            (LoRA adapter, written by job)     │   │
│  │      ├── adapter_model.safetensors                                   │   │
│  │      ├── adapter_config.json                                         │   │
│  │      └── tokenizer.*                                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## LoRA Training Configuration

### What is LoRA?

```
Standard Fine-tuning:
  Update ALL model weights (3.8B parameters × FP32 = ~15 GB optimizer state)
  Impractical for large models on a single GPU.

LoRA (Low-Rank Adaptation):
  Freeze all base model weights.
  Insert trainable low-rank matrices A and B into attention layers.
  ΔW = B × A  where  A ∈ R^(r×d),  B ∈ R^(d×r),  r << d

  Trainable parameters:
    r=16 rank × 4 target modules (q,k,v,o) × 2 layers ≈ ~4.7M parameters
    vs 3.8B base model parameters → only 0.12% of weights trained
```

### LoRA Hyperparameters (train_lora.py)

```
LORA_RANK=16
    Rank of the decomposition matrices A and B.
    Higher rank = more capacity to learn domain-specific patterns.
    Higher rank = more trainable parameters and VRAM usage.
    r=16 is a good balance for domain adaptation tasks.

LORA_ALPHA=32
    Scaling factor: effective_lr = (lora_alpha / lora_rank) × learning_rate
    lora_alpha=32, lora_rank=16 → scale factor = 2.0
    Higher alpha amplifies LoRA weight contributions.

target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    Attention projection layers to inject LoRA adapters into.
    q_proj: Query projection
    k_proj: Key projection
    v_proj: Value projection
    o_proj: Output projection
    These attention matrices govern the model's focus patterns.

lora_dropout=0.05
    Dropout rate on LoRA layer activations during training.
    Regularization to prevent overfitting on small datasets.

task_type="CAUSAL_LM"
    Causal language modeling (autoregressive, left-to-right).
    Matches the Ministral instruction-following task format.
```

---

## Training Pipeline (train_lora.py – 8 Steps)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  [1/8] Validate model files                                                   │
│       os.listdir(MODEL_PATH=/models/Ministral-3-8B-Instruct-2512)            │
│       Confirms model weights are mounted via PVC                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  [2/8] Load dataset                                                           │
│       File: /scripts/anyvc-startup-dataset.jsonl                             │
│       Format: JSONL with "messages" field (chat format)                       │
│       Content: VC advisor domain conversations                                │
│       Size: ~240 KB (anyvc-startup-dataset.jsonl)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  [3/8] Load tokenizer                                                         │
│       Primary:  MistralCommonBackend (mode="finetuning")                     │
│       Fallback: AutoTokenizer (trust_remote_code=True)                        │
│       pad_token = eos_token  (Mistral has no dedicated pad token)            │
│       padding_side = "right"                                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  [4/8] Load model (FP8 → BF16)                                               │
│       Mistral3ForConditionalGeneration.from_pretrained(                       │
│         HF_MODEL_ID="mistralai/Ministral-3-8B-Instruct-2512",                │
│         quantization_config=FineGrainedFP8Config(dequantize=True),           │
│         device_map="auto",       # Auto-shard across GPU(s)                  │
│         attn_implementation="eager"  # No FlashAttention for training        │
│       )                                                                        │
│       gradient_checkpointing: enabled (use_reentrant=False)                  │
│       All base params frozen (param.requires_grad = False)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│  [5/8] Attach LoRA adapters                                                   │
│       LoraConfig(r=16, lora_alpha=32, dropout=0.05)                          │
│       target_modules: [q_proj, k_proj, v_proj, o_proj]                       │
│       model = get_peft_model(model, lora_config)                             │
│       Trainable: ~4.7M / 3.8B parameters (0.12%)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  [6/8] Format dataset                                                         │
│       Apply chat template: tokenizer.apply_chat_template(messages)           │
│       Fallback: [INST] ... [/INST] format                                    │
│       Output: {"text": "<s>[INST]...[/INST] ...</s>"}                        │
├──────────────────────────────────────────────────────────────────────────────┤
│  [7/8] Train (SFTTrainer from TRL)                                            │
│       TrainingArguments:                                                      │
│         num_train_epochs:              5                                       │
│         per_device_train_batch_size:   2                                      │
│         gradient_accumulation_steps:   2 → effective batch = 4               │
│         learning_rate:                 2e-4                                    │
│         warmup_steps:                  2                                      │
│         max_seq_length:                512 tokens                             │
│         bf16:                          True (BF16 training)                   │
│         optim:                         adamw_torch                            │
│         save_strategy:                 epoch                                  │
│         seed:                          42                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  [8/8] Save adapter + upload to S3                                            │
│       model.save_pretrained(/output/anyvc-startup-lora)                      │
│       Key fix: remap state dict keys                                          │
│         "model.model.language_model.*" →                                      │
│         "model.language_model.model.*"                                        │
│       Upload via boto3 to:                                                    │
│         s3://${S3_BUCKET_NAME}/anyvc-startup-lora/                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Hyperparameters (lora-training-job.yaml)

```
Environment Variables (Kubernetes Job):
  HF_MODEL_ID       = mistralai/Ministral-3-8B-Instruct-2512  (HuggingFace ID)
  MODEL_PATH        = /models/Ministral-3-8B-Instruct-2512    (PVC mount)
  DATASET_PATH      = /scripts/anyvc-startup-dataset.jsonl    (ConfigMap)
  OUTPUT_DIR        = /output/anyvc-startup-lora              (emptyDir)
  S3_OUTPUT_PATH    = s3://${S3_BUCKET_NAME}/anyvc-startup-lora/
  NUM_EPOCHS        = 5
  BATCH_SIZE        = 2
  GRADIENT_ACCUMULATION = 2     → effective batch size = 4
  LEARNING_RATE     = 2e-4
  MAX_SEQ_LENGTH    = 512
  LORA_RANK         = 16
  LORA_ALPHA        = 32
```

---

## Python Dependencies (installed in-pod)

```
transformers==5.11.0    # Mistral3ForConditionalGeneration, FineGrainedFP8Config
datasets==5.0.0         # HuggingFace Datasets (JSONL loading, train/test split)
peft==0.19.1            # LoraConfig, get_peft_model
trl==1.5.1              # SFTTrainer (Supervised Fine-tuning Trainer)
accelerate==1.13.0      # PyTorch distributed training abstraction
sentencepiece==0.2.1    # Tokenizer (Mistral fallback)
mistral-common==1.11.3  # MistralCommonBackend tokenizer
boto3                   # S3 upload of adapter files
```

---

## vLLM Inference with LoRA (vllm-with-lora.yaml)

### S3 CSI Driver Setup

```
PersistentVolume: s3-bucket-pv
  driver: s3.csi.aws.com              # AWS S3 CSI Driver
  bucketName: ${S3_BUCKET_NAME}
  region: ${AWS_REGION}
  capacity: 30Gi
  accessModes: [ReadOnlyMany]

PersistentVolumeClaim: s3-bucket-pvc
  → Mounts entire S3 bucket at /s3 inside the vLLM pod
  → LoRA adapter path: /s3/anyvc-startup-lora/
```

### vLLM LoRA Configuration

```
--enable-lora
    Activates the vLLM LoRA serving feature.
    Allows dynamic per-request LoRA adapter selection.

--max-lora-rank=16
    Maximum allowed LoRA rank for loaded adapters.
    Must be ≥ the trained adapter's rank (r=16).

--lora-modules={"name":"anyvc-startup-expert",
                "path":"/s3/anyvc-startup-lora",
                "base_model_name":"ministral"}
    name:            Alias used in API requests as model name
    path:            Where vLLM reads the adapter files
    base_model_name: Must match the --served-model-name

--max-model-len=2048
    Reduced from 8192 to 2048 for LoRA variant.
    Startup advisor queries are short (typically < 1024 tokens).
    Saves KV cache memory for more concurrent sessions.
```

### LoRA API Usage

```bash
# Use base Ministral model
curl -X POST http://vllm-serve-svc:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ministral",
    "messages": [{"role": "user", "content": "Explain transformers"}]
  }'

# Use fine-tuned AnyVC adapter (same endpoint, different model name)
curl -X POST http://vllm-serve-svc:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anyvc-startup-expert",
    "messages": [{"role": "user", "content": "Evaluate my seed-stage startup pitch"}]
  }'
```

---

## Dataset: AnyVC Startup Advisor

```
File: anyvc-startup-dataset.jsonl
Size: ~240 KB
Format: JSONL, each line:
  {
    "messages": [
      {"role": "system",    "content": "You are AnyVC, an expert VC advisor..."},
      {"role": "user",      "content": "Here is my startup pitch: ..."},
      {"role": "assistant", "content": "Analysis: [structured VC feedback]..."}
    ]
  }

Domain: Venture Capital startup evaluation
  - Pitch deck evaluation
  - Market sizing analysis
  - Business model critique
  - Competitive landscape review
  - Investment thesis framing
```

---

## Memory Profile During Training

```
NVIDIA L40S (48 GB VRAM):
┌─────────────────────────────────────────────────────────┐
│  Base model weights (FP8 dequantized to BF16):          │
│    3.8B × 2 bytes (BF16) ≈ 7.5 GB                       │
│                                                          │
│  Gradient checkpointing buffers:    ~2 GB               │
│    (use_reentrant=False for FSDP-compatible checkpoints) │
│                                                          │
│  LoRA adapter weights (trainable):                       │
│    4.7M × 4 bytes (FP32 optimizer) ≈ 0.1 GB             │
│                                                          │
│  Activation buffers (batch_size=2, seq=512):             │
│    Intermediate activations ≈ 8–12 GB                    │
│                                                          │
│  Total estimated: ~20–22 GB (within 28 GB limit)         │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"
export AWS_REGION="us-east-1"

# Create ConfigMap with training scripts
kubectl create configmap anyvc-training-scripts \
  --from-file=train_lora.py \
  --from-file=anyvc-startup-dataset.jsonl

# Submit training job
envsubst < lora-training-job.yaml | kubectl apply -f -

# Watch training progress
kubectl logs -f job/anyvc-lora-finetune

# Training completes → adapter uploaded to S3
aws s3 ls s3://${S3_BUCKET_NAME}/anyvc-startup-lora/

# Deploy vLLM with LoRA (requires S3 CSI driver)
envsubst < vllm-with-lora.yaml | kubectl apply -f -

# Test LoRA adapter
kubectl port-forward svc/vllm-serve-svc 8000:8000 &
curl http://localhost:8000/v1/models  # Should show both models
```
