# Day 13 — Model Size vs Inference Cost

> **Hook:** Why a smaller model can beat a bigger model in production.

---

## The Post

"Use the biggest model you can afford."

This is the most expensive piece of advice in AI product development.

The right question isn't "what's the most capable model?" It's "what's the most capable model *for this task, at this cost point, at this latency target*?"

Here's the math.

---

## The Naive Assumption vs Reality

```
  NAIVE ASSUMPTION
  ═════════════════

  Model Quality:   70B > 13B > 7B > 3B
  Therefore:       Always use the biggest affordable model

  REALITY
  ════════

  Model Quality at Task X:   70B > 13B > 7B > 3B  ← maybe
  Latency at task X:         70B << 13B << 7B << 3B
  Cost per request:          70B >> 13B >> 7B >> 3B
  Concurrency at budget:     70B << 13B << 7B << 3B
  Fine-tuning benefit:       70B < 13B < 7B < 3B  ← smaller wins

  For MOST production tasks, a well-tuned smaller model beats
  a generic larger model on all dimensions except raw capability.
```

---

## Cost Comparison — Same Hardware

```
  INFERENCE COST: MODEL SIZE ON L40S (48 GB VRAM)
  ═════════════════════════════════════════════════

  All models in BF16, single g6e.2xlarge ($1.00/hr estimate)

  ┌──────────────────┬────────┬──────────┬──────────┬──────────┐
  │ Model            │ Size   │ VRAM used│ TPOT     │ Reqs/hr  │
  ├──────────────────┼────────┼──────────┼──────────┼──────────┤
  │ Ministral-3-8B   │  7.6GB │   ~42GB  │  ~13ms   │ ~15,000  │
  │ Llama-3-8B       │  16GB  │   ~44GB  │  ~20ms   │ ~10,000  │
  │ Llama-3-13B      │  26GB  │   ~46GB  │  ~35ms   │  ~5,000  │
  │ Llama-3-70B(INT4)│  35GB  │   ~47GB  │  ~60ms   │  ~2,500  │
  │ Llama-3-70B(BF16)│  140GB │ NEEDS 4× │  —       │ —        │
  └──────────────────┴────────┴──────────┴──────────┴──────────┘

  Cost per 1M output tokens (at $1/hr, 100 output tokens avg):
  ┌──────────────────┬────────────────────────────────────────┐
  │ Ministral-3-8B   │  $0.07 / 1M tokens                   │
  │ Llama-3-8B       │  $0.10 / 1M tokens                   │
  │ Llama-3-13B      │  $0.20 / 1M tokens                   │
  │ Llama-3-70B INT4 │  $0.40 / 1M tokens                   │
  └──────────────────┴────────────────────────────────────────┘

  Ministral-3-8B at 6× lower cost — is it 6× worse? Almost never.
```

---

## The Task-Fit Matrix

```
  WHICH MODEL SIZE ACTUALLY WINS?
  ════════════════════════════════

  Task Type                    Best Choice    Why
  ────────────────────────────────────────────────────────────────
  Simple Q&A / FAQ bot         3-8B (tuned)  Fast, cheap, accurate
  Customer support triage      3-8B (tuned)  Task is narrow, tunable
  Code completion (snippets)   7-13B         Good at code patterns
  Document summarization       7-13B         Structure matters
  Complex reasoning             70B+          Multi-step logic needed
  Creative writing              70B+          Diversity + coherence
  RAG over enterprise docs     7-13B (tuned) Grounding ≫ raw size
  Classification / extraction  3-8B (tuned)  Structured output task

  Pattern:
  ┌──────────────────────────────────────────────────────────┐
  │  Narrow, well-defined tasks → small fine-tuned models   │
  │  Open-ended, creative tasks → large general models      │
  │  RAG-grounded tasks → medium models + good retrieval    │
  └──────────────────────────────────────────────────────────┘
```

---

## Fine-Tuned Small vs Generic Large

```
  THE BENCHMARK NOBODY SHOWS YOU
  ═══════════════════════════════

  Task: Customer support intent classification
  Dataset: 50K labeled examples from your domain

  ┌─────────────────────────────────┬──────────┬─────────────┐
  │ Model                           │ Accuracy │ Cost/1M tok │
  ├─────────────────────────────────┼──────────┼─────────────┤
  │ GPT-4 (zero-shot)               │  84.2%   │  $30.00     │
  │ Llama-3-70B (zero-shot)         │  81.7%   │   $0.40     │
  │ Llama-3-8B (zero-shot)          │  74.3%   │   $0.10     │
  │ Ministral-3-8B (fine-tuned LoRA)│  91.4%   │   $0.07     │
  └─────────────────────────────────┴──────────┴─────────────┘

  Fine-tuned Ministral-3-8B:
  ✓ More accurate than GPT-4
  ✓ 428× cheaper than GPT-4
  ✓ Runs entirely on your own infrastructure (no API dependency)
  ✓ Data stays private

  This is why Module 600 (LoRA fine-tuning) exists in our workshop.
```

---

## The Model Selection Framework

```
  HOW TO CHOOSE THE RIGHT MODEL SIZE
  ════════════════════════════════════

  Step 1: Define your task precisely
  ┌──────────────────────────────────────────────────────┐
  │  What are the inputs? What are the outputs?         │
  │  Is it classification, generation, or extraction?  │
  │  Is it narrow-domain or general-purpose?           │
  └──────────────────────────────────────────────────────┘
            │
            ▼
  Step 2: Set your latency budget
  ┌──────────────────────────────────────────────────────┐
  │  TTFT target: e.g., < 300ms                         │
  │  TPOT target: e.g., < 40ms                         │
  │  This alone eliminates many large model options     │
  └──────────────────────────────────────────────────────┘
            │
            ▼
  Step 3: Benchmark smallest viable model first
  ┌──────────────────────────────────────────────────────┐
  │  Start with 7B → does it meet quality bar?          │
  │  If no: fine-tune it → does it meet quality bar?   │
  │  If no: try 13B → does it meet quality bar?        │
  │  If no: fine-tune 13B → ...                        │
  │  Go to 70B only if smaller models fail after tuning │
  └──────────────────────────────────────────────────────┘
            │
            ▼
  Step 4: Calculate total cost of ownership
  ┌──────────────────────────────────────────────────────┐
  │  GPU hours × instance cost × expected load          │
  │  Fine-tuning cost (one-time) vs inference savings   │
  │  Breakeven: usually within days to weeks            │
  └──────────────────────────────────────────────────────┘
```

---

## Latency vs Quality Frontier

```
  THE PARETO FRONTIER FOR MODEL SELECTION
  ═════════════════════════════════════════

  Quality
  (task accuracy)
      │
  100%│                                      ● 70B BF16
      │                                 ●  70B INT4
      │                           ● 13B fine-tuned
      │                      ● 13B BF16
      │               ● 7B fine-tuned ←─── sweet spot zone
      │          ● 7B BF16
      │    ● 3B fine-tuned
      │ ● 3B BF16
      └────────────────────────────────────────────────────▶
           Fast                                         Slow
           Cheap                                     Expensive

  Goal: Find the model on the frontier closest to your
        quality threshold. Everything to the right is
        paying for capability you don't need.
```

---

## Workshop Connection

```
  THIS IS WHY WE CHOSE MINISTRAL-3-8B
  ═════════════════════════════════════

  Workshop goals:
  - Demonstrate full inference pipeline end-to-end
  - Fit comfortably in a single L40S (48 GB VRAM)
  - Fast enough to benchmark meaningfully
  - Quality sufficient for demo use cases

  Ministral-3-8B checks all boxes:
  ┌──────────────────────────────────────────────────────┐
  │  7.6 GB BF16 → fits single L40S with room for KV   │
  │  ~13ms TPOT → fast streaming experience            │
  │  131k vocab (Tekken) → efficient tokenization       │
  │  8K context → covers most real-world use cases     │
  │  LoRA fine-tunable → Module 600 demonstration      │
  │  FP8 capable → production quantization path        │
  └──────────────────────────────────────────────────────┘

  For production deployment of a real use case:
  Fine-tune Ministral-3-8B on your domain → Module 600
  Deploy with LMCache → Module 400
  Scale with Ray Serve → Module 800
  Result: GPT-4-level task accuracy at 1% of GPT-4 cost.
```

---

## Key Takeaway

> Bigger is not always better in production.  
> The right model is the smallest one that meets your quality bar after fine-tuning.  
> A 7B model fine-tuned on your task almost always beats a 70B model without it.  
> Start small. Measure. Scale up only when the numbers demand it.

---

*30-Day Series: LLM Inference Is Everything | Day 13 of 30*
*← [Day 12](./day-12-quantization.md) | Next → [Day 14](./day-14-gpu-utilization-vs-good-inference.md)*
