# Day 18 — FlashAttention

> **Hook:** Why attention optimization matters for inference.

---

## The Post

Attention is the heart of every transformer. It's also the most expensive operation.

For years, attention implementations wrote a massive O(n²) matrix to GPU VRAM and read it back — forcing huge, slow memory transfers for every layer.

FlashAttention rewrote the algorithm to never materialize that matrix. The result: 2-4× faster attention, dramatically less memory, and long context that actually becomes viable.

---

## Standard Attention — The Memory Problem

```
  STANDARD ATTENTION: WHAT ACTUALLY HAPPENS IN VRAM
  ════════════════════════════════════════════════════

  Attention formula: Attention(Q, K, V) = softmax(QK^T / √d) × V

  Step-by-step GPU operations:

  1. Compute S = Q × K^T
     ┌───────────────────────────────────────────────────────┐
     │  Q: [seq_len × head_dim]   K^T: [head_dim × seq_len] │
     │  S: [seq_len × seq_len]  ← WRITTEN TO VRAM          │
     │  Size: n² elements                                   │
     │  For 8K context, 1 head: 8192 × 8192 × 2 bytes = 128 MB │
     │  For 32 heads: 4 GB just for attention scores!       │
     └───────────────────────────────────────────────────────┘

  2. Compute P = softmax(S / √d)
     ┌───────────────────────────────────────────────────────┐
     │  READ S back from VRAM                               │
     │  Apply softmax (needs max + sum over entire row)     │
     │  P: [seq_len × seq_len]  ← WRITTEN TO VRAM          │
     └───────────────────────────────────────────────────────┘

  3. Compute O = P × V
     ┌───────────────────────────────────────────────────────┐
     │  READ P back from VRAM                               │
     │  O: [seq_len × head_dim]  ← WRITTEN TO VRAM         │
     └───────────────────────────────────────────────────────┘

  Total VRAM reads/writes: O(n²) — memory bandwidth dominated
  For long contexts: this is the performance killer
```

---

## The Memory Hierarchy That Matters

```
  GPU MEMORY HIERARCHY (key to understanding FlashAttention)
  ═══════════════════════════════════════════════════════════

  ┌────────────────────────────────────────────────────────────┐
  │  SRAM (on-chip, inside SM)   ~20 MB total, ~20 TB/s       │
  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Lightning fast, tiny           │
  └────────────────────────┬───────────────────────────────────┘
                           │  100× slower to cross this boundary
  ┌────────────────────────▼───────────────────────────────────┐
  │  HBM VRAM (off-chip)         48 GB, ~864 GB/s (L40S)      │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░  Large but slow                 │
  └────────────────────────────────────────────────────────────┘

  Standard attention LIVES in HBM — repeatedly reading/writing
  the n×n matrix across that 100× slower boundary.

  FlashAttention STAYS in SRAM — by tiling the computation
  into chunks that fit in the fast on-chip memory.
```

---

## FlashAttention — The Tiling Approach

```
  FLASHATTENTION: TILE AND FUSE
  ══════════════════════════════

  Core idea: Never write the full n×n attention matrix to HBM.
  Process attention in tiles that fit in SRAM.

  ┌─────────────────────────────────────────────────────────────┐
  │                                                              │
  │  Q matrix tiled into row blocks:   Q = [Q₁ | Q₂ | Q₃ ...]  │
  │  K,V tiled into column blocks:     K = [K₁ | K₂ | K₃ ...]  │
  │                                                              │
  │  For each tile Qᵢ:                                          │
  │  ┌──────────────────────────────────────────────────────┐  │
  │  │  Load Qᵢ into SRAM                                   │  │
  │  │  For each tile Kⱼ, Vⱼ:                               │  │
  │  │    Load Kⱼ, Vⱼ into SRAM                            │  │
  │  │    Compute Sᵢⱼ = Qᵢ × Kⱼ^T   (in SRAM)             │  │
  │  │    Track running max/sum for online softmax          │  │
  │  │    Compute partial output Oᵢⱼ += softmax(Sᵢⱼ) × Vⱼ │  │
  │  │  End for                                             │  │
  │  │  Write final Oᵢ to HBM (ONCE)                       │  │
  │  └──────────────────────────────────────────────────────┘  │
  │                                                              │
  │  n×n matrix NEVER written to HBM.                          │
  │  Only Q, K, V (read once) and O (written once).            │
  │                                                              │
  └─────────────────────────────────────────────────────────────┘
```

---

## Standard vs Flash: Memory Access Comparison

```
  MEMORY ACCESS: STANDARD vs FLASH ATTENTION
  ════════════════════════════════════════════

  Sequence length: n  |  Head dimension: d  |  Batch: B  |  Heads: H

  ┌─────────────────────────┬───────────────┬─────────────────────┐
  │ Operation               │ Standard      │ FlashAttention      │
  ├─────────────────────────┼───────────────┼─────────────────────┤
  │ HBM reads               │ O(Bnd + Bn²)  │ O(Bnd)              │
  │ HBM writes              │ O(Bn² + Bnd)  │ O(Bnd)              │
  │ SRAM peak usage         │ O(Bn² / H)    │ O(Bnd / H)          │
  │ Forward pass time       │ Θ(Bn²d)       │ Θ(Bn²d) same FLOPS │
  │ but IO-bound:           │ slow          │ fast                │
  └─────────────────────────┴───────────────┴─────────────────────┘

  Same number of FLOPS — radically different memory access pattern.
  Speed difference comes entirely from memory bandwidth savings.

  Concrete numbers (8K context, Ministral-3-8B, 1 layer):
  ┌────────────────────────┬───────────────┬──────────────────────┐
  │ Metric                 │ Standard      │ FlashAttention 2     │
  ├────────────────────────┼───────────────┼──────────────────────┤
  │ VRAM for attn scores   │ ~128 MB/layer │ ~0 MB (never stored) │
  │ HBM bandwidth used     │ ~16 GB/layer  │ ~0.5 GB/layer        │
  │ Attention time         │ ~45ms/layer   │ ~12ms/layer          │
  │ Max context (48GB)     │ ~16K tokens   │ ~128K+ tokens        │
  └────────────────────────┴───────────────┴──────────────────────┘
```

---

## FlashAttention Versions

```
  EVOLUTION OF FLASHATTENTION
  ════════════════════════════

  FlashAttention v1 (2022 — Dao et al.)
  ┌──────────────────────────────────────────────────────────┐
  │  First IO-aware attention implementation                 │
  │  Tiled computation in SRAM                              │
  │  2-4× faster than standard attention                    │
  │  Supports: A100, H100, RTX series                       │
  └──────────────────────────────────────────────────────────┘

  FlashAttention v2 (2023)
  ┌──────────────────────────────────────────────────────────┐
  │  Better parallelism across attention heads               │
  │  Improved work partitioning (Q outer loop, K/V inner)   │
  │  ~2× faster than FA1                                   │
  │  Supports causal masking efficiently (triangle mask)    │
  │  DEFAULT in vLLM, HuggingFace, most frameworks         │
  └──────────────────────────────────────────────────────────┘

  FlashAttention v3 (2024)
  ┌──────────────────────────────────────────────────────────┐
  │  Designed specifically for H100 Hopper architecture     │
  │  Uses TMA (Tensor Memory Accelerator) hardware units    │
  │  Overlaps GEMM + softmax computation                    │
  │  ~1.5-2× faster than FA2 on H100                       │
  │  FP8 support for quantized inference                    │
  └──────────────────────────────────────────────────────────┘

  For our L40S in the workshop: FlashAttention v2 applies
```

---

## Why FlashAttention Enables Long Context

```
  THE LONG CONTEXT ENABLER
  ═════════════════════════

  Standard attention VRAM limit:
  ┌──────────────────────────────────────────────────────────┐
  │  48 GB VRAM (L40S)                                       │
  │  - Model weights:  7 GB                                  │
  │  - Activation buffers: 3 GB                              │
  │  - Available for attention scores: ~38 GB                │
  │                                                          │
  │  Attention score matrix size:                            │
  │  n² × heads × bytes = n² × 32 × 2 = 64n² bytes          │
  │  Max n: √(38GB / 64) ≈ 24,000 tokens                    │
  │  (and this is PER LAYER — in practice much less)        │
  └──────────────────────────────────────────────────────────┘

  With FlashAttention:
  ┌──────────────────────────────────────────────────────────┐
  │  Attention score matrix: NEVER stored in HBM             │
  │  SRAM usage per tile: ~few MB                           │
  │  Max context: limited by KV cache, not attention matrix  │
  │                                                          │
  │  With PagedAttention + FlashAttention:                   │
  │  128K+ context tokens become practical                  │
  │  Without either: ~8K practical limit on 48 GB           │
  └──────────────────────────────────────────────────────────┘
```

---

## Workshop Connection

```
  FLASHATTENTION IN OUR vLLM DEPLOYMENT
  ═══════════════════════════════════════

  vLLM automatically selects FlashAttention based on GPU:
  - L40S (Ampere/Ada Lovelace): FlashAttention 2 enabled by default
  - H100 (Hopper): FlashAttention 3 if available

  Config (Module 100):
  # FlashAttention is on by default in vLLM 0.21.0+
  # No flag needed — selected automatically
  --dtype bfloat16        # FA2 works best with BF16/FP16

  Chunked prefill (prevents prefill stalls):
  --enable-chunked-prefill  # breaks large prefills into chunks
                             # FA2 processes each chunk in SRAM
                             # keeps TPOT stable during long prompts

  Impact on our benchmarks (Module 300):
  ┌──────────────────────────────────────────────────────────┐
  │  Without FA2: Prefill time scales quadratically with n  │
  │  With FA2:    Prefill time scales near-linearly with n  │
  │               (memory IO bottleneck removed)           │
  └──────────────────────────────────────────────────────────┘
```

---

## Key Takeaway

> FlashAttention doesn't change WHAT is computed — it changes WHERE.  
> By keeping the n×n attention matrix in fast SRAM instead of slow HBM,  
> it achieves 2-4× speedup for free, with zero change to model quality.  
> It's the reason long context is practically viable in modern inference.

---

*30-Day Series: LLM Inference Is Everything | Day 18 of 30*
*← [Day 17](./day-17-speculative-decoding.md) | Next → [Day 19](./day-19-prefix-caching.md)*
