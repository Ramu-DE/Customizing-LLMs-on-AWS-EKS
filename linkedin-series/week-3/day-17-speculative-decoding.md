# Day 17 — Speculative Decoding

> **Hook:** Can a small model help a big model run faster?

---

## The Post

The fundamental constraint of LLM decode is sequential: one token at a time, no parallelism.

Speculative decoding breaks that constraint — without changing the model at all.

The idea: use a tiny draft model to guess several tokens ahead, then verify them all in parallel with the big model.

If the guesses are right, you get multiple tokens for the cost of one step.

---

## The Core Insight

```
  THE PARALLELISM OPPORTUNITY
  ════════════════════════════

  Normal decode (sequential — no way around this):

  Big model step 1: generate token 1    (20ms)
  Big model step 2: generate token 2    (20ms)
  Big model step 3: generate token 3    (20ms)
  Big model step 4: generate token 4    (20ms)
  Big model step 5: generate token 5    (20ms)
  Total: 100ms for 5 tokens

  Speculative decode:

  Draft model generates 5 tokens:      (3ms — tiny model, fast)
  "The", "quick", "brown", "fox", "jumped"

  Big model VERIFIES all 5 in ONE parallel pass:  (20ms)
  "The" ✓  "quick" ✓  "brown" ✓  "fox" ✓  "jumped" ✓
  Total: 23ms for 5 tokens → 4.3× speedup!

  ┌────────────────────────────────────────────────────────────┐
  │  Key insight: Verification is PARALLEL (like prefill)     │
  │  The big model checks all draft tokens in one forward pass │
  │  Cost = 1 big model step + 1 small model step            │
  │  Gain = up to N accepted tokens                          │
  └────────────────────────────────────────────────────────────┘
```

---

## The Full Speculative Decoding Loop

```
  SPECULATIVE DECODING — STEP BY STEP
  ═════════════════════════════════════

  Setup:
  - Target model (big): Llama-3-70B  (slow, high quality)
  - Draft model (small): Llama-3-8B  (fast, good enough for guessing)
  - Speculation length γ = 4 (draft 4 tokens ahead)

  ──────────────────────────────────────────────────────────────

  Step 1: DRAFT PHASE
  ┌──────────────────────────────────────────────────────────────┐
  │  Context: "The capital of France is"                         │
  │                                                               │
  │  Draft model generates γ=4 tokens auto-regressively:        │
  │  x̃₁ = "Paris"    (prob 0.95 in draft)                      │
  │  x̃₂ = ","        (prob 0.88)                              │
  │  x̃₃ = "which"    (prob 0.72)                              │
  │  x̃₄ = "is"       (prob 0.81)                              │
  │                                                               │
  │  Draft model time: ~4ms (4× tiny forward passes)           │
  └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
  Step 2: VERIFICATION PHASE
  ┌──────────────────────────────────────────────────────────────┐
  │  Target model processes: context + all 4 draft tokens       │
  │  IN ONE PARALLEL FORWARD PASS (like prefill)               │
  │                                                               │
  │  Target model outputs at each position:                     │
  │  p(x₁|context)    → "Paris" has prob 0.97  ✓ ACCEPT       │
  │  p(x₂|..Paris)    → "," has prob 0.91      ✓ ACCEPT       │
  │  p(x₃|..Paris,)   → "which" has prob 0.68  ✓ ACCEPT       │
  │  p(x₄|..which)    → "is" has prob 0.43     ✗ REJECT       │
  │                                                               │
  │  Target model time: ~20ms (1 big forward pass)             │
  └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
  Step 3: ACCEPT/REJECT
  ┌──────────────────────────────────────────────────────────────┐
  │  Token 1 "Paris":  accepted (0.97 ≥ 0.95) ✓              │
  │  Token 2 ",":      accepted (0.91 ≥ 0.88) ✓              │
  │  Token 3 "which":  accepted (0.68 ≥ 0.72) BORDERLINE...  │
  │                    stochastic accept: coin flip → ✓       │
  │  Token 4 "is":     rejected (0.43 < draft's 0.81)        │
  │                    sample from corrected distribution     │
  │                    → output "the" instead               │
  │                                                              │
  │  Result: 3 tokens accepted + 1 corrected = 4 total tokens  │
  │  Crucially: OUTPUT IS IDENTICAL to what big model alone    │
  │  would have produced. Zero quality loss.                   │
  └──────────────────────────────────────────────────────────────┘

  Total time: 4ms + 20ms = 24ms for 4 tokens
  Without speculative: 4 × 20ms = 80ms for 4 tokens
  Speedup: 3.3× on this example
```

---

## The Acceptance Rate Is Everything

```
  WHAT DETERMINES SPEEDUP
  ═════════════════════════

  α = token acceptance rate (how often draft matches target)

  Theoretical speedup = γ × α / (1 + γ × (cost_draft / cost_target))

  ┌────────────────────────┬─────────────┬────────────────────────┐
  │  Acceptance Rate (α)   │ Speedup     │ When this happens      │
  ├────────────────────────┼─────────────┼────────────────────────┤
  │  0.5  (50%)            │  ~1.3×      │ Very different models  │
  │  0.7  (70%)            │  ~2.0×      │ Moderate alignment     │
  │  0.8  (80%)            │  ~2.5×      │ Good draft/target pair │
  │  0.9  (90%)            │  ~3.2×      │ Closely related models │
  │  0.95 (95%)            │  ~3.8×      │ Near-ideal             │
  └────────────────────────┴─────────────┴────────────────────────┘

  Best draft/target pairs:
  ┌────────────────────────────────────────────────────────────────┐
  │  Llama-3-70B  + Llama-3-8B  → α ≈ 0.85, speedup ~2.5×       │
  │  Llama-3-70B  + Llama-3-1B  → α ≈ 0.75, speedup ~2.0×       │
  │  GPT-4        + GPT-4o-mini → α ≈ 0.80 (industry estimate)  │
  │  Code tasks                  → α often higher (predictable)  │
  │  Creative tasks              → α often lower (diverse output) │
  └────────────────────────────────────────────────────────────────┘

  Code and repetitive text = high acceptance = big speedup
  Creative writing = low acceptance = smaller speedup
```

---

## Variants of Speculative Decoding

```
  THE SPECULATIVE DECODING FAMILY
  ═════════════════════════════════

  ① Classic Speculative Decoding
  ┌──────────────────────────────────────────────────────────┐
  │  Separate draft model (different architecture)          │
  │  Best quality, requires serving 2 models               │
  │  Memory: +draft model VRAM                             │
  └──────────────────────────────────────────────────────────┘

  ② Self-Speculative Decoding (Medusa / EAGLE)
  ┌──────────────────────────────────────────────────────────┐
  │  Draft heads added to the SAME model                    │
  │  Extra prediction heads at the final layer              │
  │  No separate draft model → half the memory overhead    │
  │  EAGLE-2: state-of-the-art, ~3× speedup at α≈0.85     │
  └──────────────────────────────────────────────────────────┘

  ③ Draft via N-gram / Retrieval
  ┌──────────────────────────────────────────────────────────┐
  │  Match recent context to n-gram patterns in prompt      │
  │  No model needed for drafting                          │
  │  Works best for tasks with repetition                  │
  │  vLLM --speculative-model "[ngram]"                    │
  └──────────────────────────────────────────────────────────┘

  ④ Lookahead Decoding
  ┌──────────────────────────────────────────────────────────┐
  │  Parallel Jacobi decoding without a draft model         │
  │  Generate multiple positions simultaneously             │
  │  Verify + correct in one pass                          │
  └──────────────────────────────────────────────────────────┘
```

---

## Speculative Decoding Does NOT Hurt Quality

```
  THE GUARANTEE: DISTRIBUTION-PRESERVING
  ════════════════════════════════════════

  Standard speculative decoding uses a rejection sampling scheme
  that PROVES the output distribution matches the target model exactly.

  Proof sketch:
  - Accept token x̃ₜ if rand() < min(1, p_target(x̃ₜ) / p_draft(x̃ₜ))
  - On rejection, sample from adjusted distribution:
    p_corrected(x) ∝ max(0, p_target(x) - p_draft(x))
  - Mathematical result: marginal distribution = p_target

  ┌────────────────────────────────────────────────────────────┐
  │  This means:                                               │
  │  ✓ Zero quality degradation (provably identical outputs)  │
  │  ✓ No hallucination increase                             │
  │  ✓ Same sampling temperature behavior                    │
  │  ✓ Can be enabled/disabled transparently               │
  │                                                            │
  │  Speculative decoding is a FREE speedup.                  │
  │  If acceptance rate is high enough, there is no downside. │
  └────────────────────────────────────────────────────────────┘
```

---

## Workshop Connection

```
  SPECULATIVE DECODING IN vLLM (Our Stack)
  ══════════════════════════════════════════

  Enable in vLLM deployment (Module 100):

  # vllm-deployment.yml
  args:
    - --model ministral-3b              # target model
    - --speculative-model ministral-1b  # draft model (smaller)
    - --num-speculative-tokens 4        # γ = 4 draft tokens

  Or use n-gram drafting (no second model needed):
    - --speculative-model "[ngram]"
    - --ngram-prompt-lookup-max 4
    - --num-speculative-tokens 4

  When to enable for our workshop:
  ┌──────────────────────────────────────────────────────────────┐
  │  Good fit: RAG responses (repetitive grounded text) → 700   │
  │  Good fit: Code generation tasks → predictable patterns    │
  │  Poor fit: High-temp creative generation → low acceptance  │
  │  Poor fit: Very short responses → overhead not worth it    │
  └──────────────────────────────────────────────────────────────┘
```

---

## Key Takeaway

> Speculative decoding uses a fast small model to guess ahead, then verifies in parallel.  
> With good draft/target alignment: 2-4× speedup with zero quality loss — provably.  
> It's the only way to get parallel compute benefits during the inherently sequential decode phase.  
> vLLM supports it natively. Enable it, measure acceptance rate, keep if > 0.7.

---

*30-Day Series: LLM Inference Is Everything | Day 17 of 30*
*← [Day 16](./day-16-paged-attention.md) | Next → [Day 18](./day-18-flash-attention.md)*
