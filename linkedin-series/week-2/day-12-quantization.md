# Day 12 — Quantization

> **Hook:** FP32 → FP16 → INT8 → INT4: what actually changes?

---

## The Post

A 70B parameter model won't fit in 48 GB of VRAM.

Not as FP32. Not even as FP16.

But quantize it to INT4 and suddenly it fits — comfortably.

Quantization is one of the highest-leverage tools in LLM inference. Here's exactly how it works.

---

## What Quantization Is

```
  THE CORE IDEA: USE FEWER BITS PER NUMBER
  ═════════════════════════════════════════

  Model weights are just numbers. Big floating-point numbers.

  Original weight (FP32): 0.347291946411...
                           ████████████████████████████████
                           32 bits of precision

  After quantization (INT8): 44  (mapped from original range)
                              ████████
                              8 bits

  After quantization (INT4): 6   (coarser mapping)
                              ████
                              4 bits

  The trade-off: precision ↓, but model still mostly works
  because neural networks are surprisingly tolerant of noise.
```

---

## Precision Formats Compared

```
  NUMERIC PRECISION FORMATS
  ══════════════════════════════════════════════════════════════════

  ┌──────────┬───────┬────────────────────────────┬───────────────┐
  │ Format   │ Bits  │ Range / Precision           │ Use Case      │
  ├──────────┼───────┼────────────────────────────┼───────────────┤
  │ FP32     │  32   │ ±3.4×10^38, 7 sig digits   │ Training      │
  │ BF16     │  16   │ ±3.4×10^38, 3 sig digits   │ Training/inf  │
  │ FP16     │  16   │ ±65,504, 4 sig digits       │ Inference     │
  │ FP8      │   8   │ ±448, 2 sig digits          │ Fast inf      │
  │ INT8     │   8   │ -128 to 127 (integers)      │ Inference     │
  │ INT4     │   4   │ -8 to 7 (integers)          │ Compressed    │
  │ INT2     │   2   │ -2 to 1 (integers)          │ Experimental  │
  └──────────┴───────┴────────────────────────────┴───────────────┘

  BF16 vs FP16:
  - Same 16 bits, different split
  - BF16: 8 exponent bits (same as FP32) — handles large values
  - FP16: 5 exponent bits — can overflow on large values
  - BF16 preferred for LLMs (Ministral-3-8B uses BF16 natively)
```

---

## Memory Impact of Quantization

```
  MODEL SIZE vs QUANTIZATION — MINISTRAL-3-8B (3.8B PARAMS)
  ═══════════════════════════════════════════════════════════

  ┌──────────┬────────────────┬─────────────┬──────────────────┐
  │ Format   │ Bytes/param    │ Model Size  │ Fits on L40S?    │
  ├──────────┼────────────────┼─────────────┼──────────────────┤
  │ FP32     │  4 bytes       │  15.2 GB    │ ✅ Yes (tight)   │
  │ BF16     │  2 bytes       │   7.6 GB    │ ✅ Yes (good)    │
  │ FP8      │  1 byte        │   3.8 GB    │ ✅ Yes (lots)    │
  │ INT8     │  1 byte        │   3.8 GB    │ ✅ Yes (lots)    │
  │ INT4     │  0.5 bytes     │   1.9 GB    │ ✅ Yes (tons)    │
  └──────────┴────────────────┴─────────────┴──────────────────┘

  Now for a 70B model:
  ┌──────────┬────────────────┬─────────────┬──────────────────┐
  │ Format   │ Bytes/param    │ Model Size  │ Fits on L40S?    │
  ├──────────┼────────────────┼─────────────┼──────────────────┤
  │ FP32     │  4 bytes       │  280 GB     │ ❌ No (6 GPUs)  │
  │ BF16     │  2 bytes       │  140 GB     │ ❌ No (3 GPUs)  │
  │ FP8      │  1 byte        │   70 GB     │ ❌ No (2 GPUs)  │
  │ INT8     │  1 byte        │   70 GB     │ ❌ No (2 GPUs)  │
  │ INT4     │  0.5 bytes     │   35 GB     │ ✅ Yes! (1 GPU) │
  └──────────┴────────────────┴─────────────┴──────────────────┘

  INT4 quantization lets you run Llama-70B on a single L40S GPU.
  That's the difference between $8/hr and $50/hr infrastructure.
```

---

## How Quantization Actually Works

```
  THE QUANTIZATION PROCESS
  ═════════════════════════

  Step 1: Find the range of a weight tensor
  ┌──────────────────────────────────────────────────────────┐
  │  Original FP32 weights: [-2.3, 0.1, 1.7, -0.9, 3.2, ...│
  │  Min: -2.3     Max: 3.2                                  │
  └──────────────────────────────────────────────────────────┘

  Step 2: Map to integer range (INT8 example: -128 to 127)
  ┌──────────────────────────────────────────────────────────┐
  │  Scale factor: (3.2 - (-2.3)) / (127 - (-128))         │
  │              = 5.5 / 255 = 0.02157                      │
  │                                                          │
  │  -2.3 → round(-2.3 / 0.02157) = round(-106.6) = -107   │
  │   0.1 → round(0.1  / 0.02157) = round(4.6)   = 5       │
  │   1.7 → round(1.7  / 0.02157) = round(78.8)  = 79      │
  │  -0.9 → round(-0.9 / 0.02157) = round(-41.7) = -42     │
  │   3.2 → round(3.2  / 0.02157) = round(148.4) = 127(cap)│
  └──────────────────────────────────────────────────────────┘

  Step 3: At inference time — dequantize on the fly
  ┌──────────────────────────────────────────────────────────┐
  │  Load INT8 weight (1 byte)                               │
  │  Multiply by scale factor (stored in FP16)              │
  │  Use reconstructed FP16 for compute                     │
  │  GPU does this in hardware — very fast                  │
  └──────────────────────────────────────────────────────────┘
```

---

## Quality vs Compression Trade-off

```
  ACCURACY IMPACT OF QUANTIZATION
  ════════════════════════════════

  Benchmark: Llama-3 8B on MMLU (higher = better)

  ┌──────────┬───────────┬────────────────────────────────────┐
  │ Format   │ MMLU Score│ Quality Loss                       │
  ├──────────┼───────────┼────────────────────────────────────┤
  │ BF16     │   68.4%   │ Baseline (full precision)         │
  │ FP8      │   68.2%   │ -0.2%  ← nearly imperceptible    │
  │ INT8     │   67.9%   │ -0.5%  ← barely noticeable       │
  │ INT4(AWQ)│   67.1%   │ -1.3%  ← acceptable for most     │
  │ INT4(GPTQ│   66.8%   │ -1.6%  ← acceptable for most     │
  │ INT2     │   58.0%   │ -10.4% ← significant degradation │
  └──────────┴───────────┴────────────────────────────────────┘

  Practical takeaway:
  FP8 and INT8 are nearly free — take them without thinking.
  INT4 costs ~1-2% accuracy — usually worth it for the hardware savings.
  INT2 is for research, not production.
```

---

## Quantization Methods

```
  THE MAIN APPROACHES
  ════════════════════

  POST-TRAINING QUANTIZATION (PTQ) — No retraining needed
  ════════════════════════════════

  AWQ (Activation-Aware Weight Quantization)
  ┌──────────────────────────────────────────────────────────┐
  │  Identifies which weights are most important (salient)  │
  │  Protects those weights with higher precision           │
  │  Quantizes less-important weights more aggressively     │
  │  Result: better quality than naive INT4                 │
  └──────────────────────────────────────────────────────────┘

  GPTQ (Generative Pre-Training Quantization)
  ┌──────────────────────────────────────────────────────────┐
  │  Layer-by-layer quantization using Hessian information  │
  │  Minimizes quantization error per layer                 │
  │  Slightly slower to quantize but good quality           │
  └──────────────────────────────────────────────────────────┘

  QUANTIZATION-AWARE TRAINING (QAT) — Retraining required
  ══════════════════════════════════

  ┌──────────────────────────────────────────────────────────┐
  │  Simulates quantization during fine-tuning              │
  │  Model "learns" to be robust to lower precision        │
  │  Best quality at INT4/INT8                             │
  │  Costs: requires training compute + data               │
  │  FP8 models from Mistral use a form of this            │
  └──────────────────────────────────────────────────────────┘

  Dynamic vs Static quantization:
  ┌──────────────────────────────────────────────────────────┐
  │  Static:  Scale factors computed once, stored with model │
  │           Faster at inference (no runtime calibration)  │
  │  Dynamic: Scale factors computed per batch at runtime   │
  │           More accurate, slightly slower               │
  └──────────────────────────────────────────────────────────┘
```

---

## FP8 — The Sweet Spot in 2024+

```
  WHY FP8 IS THE PREFERRED FORMAT NOW
  ═════════════════════════════════════

  NVIDIA H100, L40S, A100 all support FP8 in hardware (Hopper arch)

  FP8 combines the best of both worlds:
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  INT8 advantages:  ✓ Half the memory of FP16/BF16      │
  │                    ✓ 2× memory bandwidth efficiency     │
  │                    ✓ Tensor Core support                │
  │                                                          │
  │  FP precision advantages: ✓ Handles wide dynamic range │
  │                           ✓ No overflow issues          │
  │                           ✓ < 0.2% quality loss        │
  │                                                          │
  │  Result: FP8 is often the best default for production  │
  │          without any quality compromise                 │
  └──────────────────────────────────────────────────────────┘

  Ministral-3-8B supports FP8 natively.
  Our workshop used BF16 (full precision) to show baseline behavior.
  Adding FP8 is a one-line config change in vLLM:
  --dtype fp8
```

---

## Workshop Connection

```
  QUANTIZATION IN THE WORKSHOP
  ═════════════════════════════

  Our model: Ministral-3-8B-Instruct-2512

  Deployed format: BF16 (consolidated.safetensors, 10.4 GB)

  vLLM deployment (Module 100):
  --dtype bfloat16          # full precision baseline

  To enable FP8 (one line change):
  --dtype fp8               # half the memory, same quality

  Impact on our L40S (48 GB):
  ┌──────────────────────────────────────────────────┐
  │  BF16:  7.6 GB weights → ~30 GB for KV cache    │
  │  FP8:   3.8 GB weights → ~37 GB for KV cache    │
  │                          +23% more KV cache     │
  │                          = ~23% more concurrent │
  │                            requests at same VRAM│
  └──────────────────────────────────────────────────┘

  Fine-tuning (Module 600):
  - LoRA training done in BF16 for stability
  - Adapter saved separately (~50-200 MB)
  - Base model can remain FP8; adapter runs in BF16
```

---

## Key Takeaway

> Quantization trades a tiny amount of model accuracy for major hardware savings.  
> FP8 and INT8 are nearly free — always consider them for production.  
> INT4 cuts VRAM in half again with ~1-2% accuracy cost.  
> The right question isn't "can I run this model?" — it's "at what precision makes business sense?"

---

*30-Day Series: LLM Inference Is Everything | Day 12 of 30*
*← [Day 11](./day-11-why-long-context-is-expensive.md) | Next → [Day 13](./day-13-model-size-vs-inference-cost.md)*
