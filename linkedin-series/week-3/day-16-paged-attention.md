# Day 16 — PagedAttention

> **Hook:** Why managing KV cache efficiently changes inference economics.

---

## The Post

Before PagedAttention, GPU memory was the #1 reason inference servers couldn't scale.

Not compute. Not model quality. **Memory fragmentation.**

The vLLM team looked at how operating systems manage memory and applied the same idea to the KV cache. The result was a 24× improvement in memory efficiency.

Here's exactly what they did — and why it matters.

---

## The Problem: KV Cache Fragmentation

```
  THE MEMORY WASTE PROBLEM (pre-PagedAttention)
  ══════════════════════════════════════════════

  Traditional approach: Reserve contiguous memory per sequence

  When a request arrives asking for max 2048 tokens:

  GPU VRAM
  ┌─────────────────────────────────────────────────────────────┐
  │                                                              │
  │  Request A  [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │
  │             used: 200  reserved but EMPTY: 1848 tokens     │
  │                                                              │
  │  Request B  [███████████████████████████░░░░░░░░░░░░░░░░]  │
  │             used: 1500  reserved but EMPTY: 548 tokens     │
  │                                                              │
  │  Request C  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │
  │             used: 50   reserved but EMPTY: 1998 tokens     │
  │                                                              │
  │  Wasted (empty reserved space): ~70% of KV cache VRAM      │
  │  Free VRAM: tiny — new requests get REJECTED               │
  │                                                              │
  └─────────────────────────────────────────────────────────────┘

  Internal fragmentation: Memory reserved but not yet used
  External fragmentation: Freed memory too small for new requests
```

---

## The Solution: Virtual Memory for KV Cache

```
  OPERATING SYSTEM ANALOGY
  ═════════════════════════

  How your OS manages RAM:
  ┌───────────────────────────────────────────────────────────┐
  │  Physical RAM divided into fixed-size PAGES (4 KB each)  │
  │  Each process gets a VIRTUAL address space               │
  │  Page table maps virtual → physical addresses            │
  │  Pages allocated on demand, not upfront                  │
  │  Non-contiguous physical pages → contiguous virtual view │
  └───────────────────────────────────────────────────────────┘

  PagedAttention applies the same idea to KV cache:
  ┌───────────────────────────────────────────────────────────┐
  │  GPU VRAM divided into fixed-size KV BLOCKS (e.g.16 tok) │
  │  Each sequence gets a LOGICAL block table               │
  │  Block table maps logical → physical KV blocks           │
  │  Blocks allocated on demand as tokens are generated      │
  │  Non-contiguous physical blocks → contiguous logical view│
  └───────────────────────────────────────────────────────────┘
```

---

## PagedAttention Memory Layout

```
  PHYSICAL KV CACHE BLOCKS IN GPU VRAM
  ══════════════════════════════════════

  Physical block pool (each block = 16 tokens × all layers × K+V):
  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  │ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │ P8 │ P9 │P10 │P11 │
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
    A    B    A    C    A    B    FREE  D    C    FREE  B    D

  Sequence A logical blocks: [L0→P0, L1→P2, L2→P4]
  Sequence B logical blocks: [L0→P1, L1→P5, L2→P10]
  Sequence C logical blocks: [L0→P3, L1→P8]
  Sequence D logical blocks: [L0→P7, L1→P11]
  Free blocks: P6, P9

  What this means:
  ┌──────────────────────────────────────────────────────────────┐
  │  ✓ Sequences A, B, C, D share the SAME physical VRAM pool  │
  │  ✓ No upfront reservation — blocks allocated one at a time  │
  │  ✓ When sequence finishes: its blocks instantly freed       │
  │  ✓ Fragmentation near-zero (only last block partially used) │
  │  ✓ Large sequences and small sequences coexist efficiently  │
  └──────────────────────────────────────────────────────────────┘
```

---

## Before vs After: Concurrency Numbers

```
  CONCURRENCY COMPARISON — Ministral-3-8B on L40S (48 GB)
  ═════════════════════════════════════════════════════════

  Scenario: Requests with average 512-token output, 8K max context

  WITHOUT PagedAttention:
  ┌──────────────────────────────────────────────────────────────┐
  │  Must reserve 8192 tokens × 128 KB/tok = 1 GB per sequence  │
  │  Available KV VRAM: ~30 GB                                   │
  │  Max concurrent sequences: 30 GB / 1 GB = ~30               │
  │  Actual use: only 6% of reserved memory (avg 512 used)      │
  │  EFFECTIVE concurrent sequences: ~2 (utilization adjusted)  │
  └──────────────────────────────────────────────────────────────┘

  WITH PagedAttention:
  ┌──────────────────────────────────────────────────────────────┐
  │  Allocate only used blocks: 512 tokens × 128 KB = 64 MB avg │
  │  Available KV VRAM: ~30 GB                                   │
  │  Max concurrent sequences: 30 GB / 64 MB = ~468             │
  │  vLLM default cap: 256 (max-num-seqs)                       │
  │  Real improvement: 24× vs naive reservation                 │
  └──────────────────────────────────────────────────────────────┘

  24× more concurrent requests. Same GPU. Same model.
  This is the PagedAttention headline number.
```

---

## Block Table Operations

```
  HOW BLOCK ALLOCATION WORKS DURING DECODE
  ══════════════════════════════════════════

  Sequence A starts (prompt = 20 tokens):
  Step 1: Allocate block P0 (holds tokens 0-15)
          Allocate block P4 (holds tokens 16-19 + room for more)

  Block table A: [L0 → P0, L1 → P4]

  Decode continues, fills block P4:
  Step 2: P4 now full (tokens 16-31)
          Allocate new block P2 (tokens 32+)

  Block table A: [L0 → P0, L1 → P4, L2 → P2]

  Sequence A finishes at token 45:
  Step 3: Free blocks P0, P4, P2 → return to free pool

  ┌─────────────────────────────────────────────────────────────┐
  │  Block size choice matters:                                  │
  │  Small blocks (4 tok): Fine-grained, low fragmentation      │
  │  Large blocks (32 tok): Less overhead, slightly more waste  │
  │  vLLM default: 16 tokens per block                         │
  │                                                             │
  │  Block table lookups add ~1% overhead to attention         │
  │  vs the 24× concurrency improvement — clearly worth it     │
  └─────────────────────────────────────────────────────────────┘
```

---

## Copy-on-Write: Enabling Beam Search & Parallel Sampling

```
  SHARING BLOCKS ACROSS SEQUENCES
  ════════════════════════════════

  Beam search generates multiple candidate sequences from same prefix:

  Prompt: "The best approach to machine learning is"

  Beam 1: [Prompt blocks: P0, P1] → [Beam 1 unique: P5]
  Beam 2: [Prompt blocks: P0, P1] → [Beam 2 unique: P6]
  Beam 3: [Prompt blocks: P0, P1] → [Beam 3 unique: P7]

  P0 and P1 SHARED between all 3 beams (ref count = 3)
  No duplication! Each beam only needs 1 unique block.

  Copy-on-Write:
  ┌──────────────────────────────────────────────────────────────┐
  │  When beam needs to write to a shared block:                │
  │  → Allocate NEW block, copy contents, update block table   │
  │  → Original block stays shared by other beams              │
  │  → Same mechanism as Linux fork() copy-on-write            │
  │                                                             │
  │  Memory savings for beam search (width=4): ~3× less VRAM  │
  └──────────────────────────────────────────────────────────────┘
```

---

## Inference Engine Full Picture

> Inspired by reference: *Inference Engine block diagram*

```
  INFERENCE ENGINE — WHERE PAGEDATTENTION LIVES
  ═══════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────┐
  │  INFERENCE SERVER              │  INFERENCE ENGINE                  │
  │                                │                                     │
  │  ┌──────────────────────────┐  │  ┌─────────────┐  ┌────────────┐ │
  │  │ API Gateway & Load Bal.  │  │  │  Attention  │  │  KV Cache  │ │
  │  └──────────────────────────┘  │  │ Computation │◀▶│  System    │ │
  │                                │  └─────────────┘  │            │ │
  │  ┌──────────────┐  ┌────────┐  │                   │ PagedAtten │ │
  │  │  Request     │  │  P/D   │  │  ┌─────────────┐  │ (THIS DAY)│ │
  │  │  Scheduler   │  │ Disagg │──┼─▶│ Transformer │  └────────────┘ │
  │  └──────────────┘  └────────┘  │  │   Layers    │                 │
  │                                │  └─────────────┘                 │
  │  ┌──────────────────────────┐  │                                   │
  │  │  Continuous Batching     │  │  ┌────────────────────────────┐  │
  │  │  (Day 15)                │  │  │ Optimization Techniques    │  │
  │  └──────────────────────────┘  │  │ Kernel Fusion | Quant | TP │  │
  │                                │  └────────────────────────────┘  │
  │  ┌──────────────────────────┐  │                                   │
  │  │   Response Queue         │  │  ┌────────────────────────────┐  │
  │  └──────────────────────────┘  │  │ Token Generation & Sampling│  │
  │                                │  └────────────────────────────┘  │
  └────────────────────────────────┴───────────────────────────────────┘
                    │                              │
                    └──────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  HARDWARE LAYER             │
                    │  GPU Pool (L40S / H100)     │
                    │  Memory Hierarchy           │
                    │  Network Infrastructure     │
                    └─────────────────────────────┘
```

---

## Workshop Connection

```
  PAGEDATTENTION IN OUR WORKSHOP (Module 100)
  ════════════════════════════════════════════

  vLLM uses PagedAttention by default. No configuration needed.

  Visible effects in our benchmarks (Module 300):
  ┌──────────────────────────────────────────────────────────────┐
  │  Without vLLM (naive server): max 5-10 concurrent requests  │
  │  With vLLM PagedAttention:   max 200+ concurrent requests   │
  │                                                              │
  │  DCGM Exporter metric: DCGM_FI_DEV_FB_USED                 │
  │  With PagedAttention: VRAM fills gradually as tokens grow   │
  │  Without: VRAM reserved upfront, hits limit at 30 requests  │
  └──────────────────────────────────────────────────────────────┘

  LMCache (Module 400) extends PagedAttention:
  → When GPU block pool full: evict cold blocks to CPU RAM
  → When CPU RAM full: evict to Valkey (ElastiCache)
  → PagedAttention blocks are the unit of transfer between tiers
```

---

## Key Takeaway

> PagedAttention treats GPU memory like an OS treats RAM — pages, not reservations.  
> The result: near-zero fragmentation, 24× more concurrent requests, copy-on-write sharing.  
> It's the reason vLLM became the industry standard for inference serving.  
> Memory efficiency IS throughput efficiency.

---

*30-Day Series: LLM Inference Is Everything | Day 16 of 30*
*← [Day 15](./day-15-continuous-batching.md) | Next → [Day 17](./day-17-speculative-decoding.md)*
