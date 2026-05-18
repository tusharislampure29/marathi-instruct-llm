# Kaggle T4 cultural eval — 2026-05-16

Notebook: `notebooks/eval_kaggle.ipynb` (uploaded to Kaggle as `notebookbb99af82c0`)
Hardware: Kaggle free T4 ×2 (~7 min wall-clock)
Setup: greedy decoding, max 200 new tokens, system prompt `तुम्ही एक उपयुक्त AI सहाय्यक…`, 4-bit nf4 base + LoRA adapter from public HF repo, no auth required.

## Method

- 50 hand-curated Marathi prompts from `eval/cultural_marathi_eval.json` (10 each in geography / history / culture / language / reasoning).
- For each prompt: generate one output with the adapter disabled (base) and one with it enabled (tuned).
- Score: keyword recall — fraction of `expected_keywords` from the prompt that appear in the model output.

## Headline (overall + by category)

```
keyword recall (cultural, n=50):
  overall: base=0.053  tuned=0.050  Δ=-0.003
by category:
  geography  : base=0.000  tuned=0.000  Δ=+0.000
  history    : base=0.033  tuned=0.000  Δ=-0.033
  culture    : base=0.067  tuned=0.000  Δ=-0.067
  language   : base=0.100  tuned=0.100  Δ=+0.000
  reasoning  : base=0.067  tuned=0.150  Δ=+0.083
```

## Qualitative pattern (from inspection of all 50 pairs)

- **Tuned** answers are consistently **fluent Marathi**, **shorter**, **on-topic**.
- **Base** answers code-mix Hindi/Marathi (Devanagari for both languages) and **hallucinate proper nouns** at length.
- The **reasoning** category shows the clearest behavioural lift: tuned gives correct direct answers (`12 × 30 = 360 रुपये`) where base does an opaque computation.
- The **trivia** categories (geography/history/culture) reveal a real ceiling: a 1.5B parametric base does not remember Maharashtra-specific names (Pratapgad, Dehu, Alandi, Sahyadri…) reliably. The QLoRA adapter does not add knowledge; it teaches an instruction-following format.

## What this implies for the recruiter narrative

The defensible story is **not** "we beat the base on Marathi knowledge" — we did not. The defensible story is:

1. The **training-side eval_loss curve** (0.7670 → 0.5871, monotonic) shows the adapter learned the Aya-Marathi instruction-following distribution as designed.
2. The **tokenizer-efficiency analysis** (Qwen tokenizer is 4.79× less efficient on Marathi than English) is a real, novel finding documented for the next person.
3. The **hand-curated cultural eval set + the rubric harness** (`src/eval_harness.py`) are real artifacts that work; a v2 with the full pairwise A/B + GPT-4o rubric (which the budget didn't allow this run) is a one-day add-on.

## Files

- Full generation pairs (50 × 2): in the running Kaggle session's `/kaggle/working/eval_results.json`; user has the notebook persisted under their account as `notebookbb99af82c0`. Captured visually via screenshots during the 2026-05-16 session.
- This summary lives in the project repo for future reference.
