# LinkedIn post

The single primary draft below is the merged A+B hybrid — recruiter-friendly hook on top, technical depth in the middle, honest limitations at the bottom. Drafts A and B (preserved as alternates further down) are kept in case you want a terser variant later.

---

## Primary draft (recommended) — merged hook + technical depth

> 83 million people speak Marathi. It's the third most-spoken language in India. I couldn't find a single open-source, purpose-built, instruction-tuned LLM for it.
>
> So I built one this week, on a free Colab T4.
>
> Adapter: huggingface.co/tusharislampure29/qwen2.5-1.5b-marathi-instruct
> Code:    github.com/tusharislampure29/marathi-instruct-llm
> Dataset: huggingface.co/datasets/tusharislampure29/marathi-instruct-30k
>
> The fix is a QLoRA adapter on Qwen2.5-1.5B-Instruct. 27,000 Marathi instruction pairs from CohereForAI's Aya collection, Devanagari-script filtered, length filtered, MinHash-deduped at Jaccard 0.85. Three epochs, around seven hours of T4 time split across two Colab sessions, with auto-resume to Hugging Face Hub after one session disconnected at step 1333.
>
> What changed, measured three different ways so I couldn't fool myself:
>
> ▸ **Held-out 500-prompt test set:** ROUGE-L F1 **+40 %**, chrF **+41 %**, sacreBLEU **+276 %** over the base model.
>
> ▸ **GPT-4o pairwise judge on 100 prompts (A/B order randomised to remove position bias):** tuned wins **71**, base wins 19, ties 10. The tuned model is preferred **3.7× more often than the reverse**.
>
> ▸ **50-prompt hand-curated Marathi cultural eval**, each response scored 1-5 by GPT-4o on fluency, factuality, cultural_accuracy, and instruction_following. Biggest lift is on Marathi-language reasoning: instruction_following 1.1 → 1.7 in that category. Concretely, the tuned model returns `१२ × ३० = ३६० रुपये` in Marathi where the base mixes Hindi and hallucinates.
>
> Three things I had to learn the hard way that you won't find in a tutorial:
>
> **1. Qwen2.5's tokenizer is 4.79× less efficient on Marathi than English.** A 1024-token context fits 161 Marathi words versus 882 English. 98.8 % of Marathi words get fragmented into two or more tokens. Worst case in the corpus: _गोलंदाजांपैकी_ splits into 14 tokens. This is the real bottleneck for Marathi LLMs, not raw model capacity, and it's the v2 hill I want to die on.
>
> **2. Stock TRL + Qwen2.5 + T4 + fp16 = a bf16-grad-scaler crash the docs don't warn you about.** Qwen ships bf16 weights, Turing GPUs (T4 / 2080) have no native bf16 hardware, and `_amp_foreach_non_finite_check_and_unscale_cuda` can't unscale a mixed bf16/fp16 grad pile. The fix is Unsloth — drop-in TRL replacement, native fp16 dispatch, ~2× faster. The hours I lost on this before I gave up on stock TRL are in the repo's "did not work" log.
>
> **3. The most useful TrainerCallback I wrote was the one that pushes the LoRA adapter to Hugging Face Hub on every `save_steps`.** That callback was the difference between "Colab disconnected at step 1333" being a five-minute resume vs. a wasted day. The Phase 2 eval taught me the same lesson for evaluation runs: my first 6.5-hour eval generation was reaped before I could download the working files. The recovery was switching to Kaggle's Save & Run All (committed runs persist working dirs, interactive sessions don't).
>
> Honest about what this model **is not**:
>
> It's 1.5 billion parameters. On Maharashtra long-tail trivia — Shivaji-era history, exact district counts, which saint came from which village — both base and tuned score near the GPT-4o rubric floor. That's a parametric-knowledge ceiling and no amount of LoRA fine-tuning can fix it. For anything that requires guaranteed factual correctness in Marathi, this adapter needs to be paired with retrieval. Where it _is_ a clean win over the base: Marathi reasoning, conversational fluency, instruction-following, register, and not falling back into Hindi or English mid-response.
>
> Two charts and one tokenizer chart attached:
> – Training: eval_loss 0.767 → 0.587 across 12 mid-training checkpoints, monotonic. No overfit.
> – Eval: ROUGE / chrF / BLEU side-by-side and the 71/10/19 pairwise split.
> – Tokenizer: the 4.79× Marathi-vs-English efficiency gap that motivates v2.
>
> If you work on Indic NLP at AI4Bharat, Sarvam, Krutrim, Reliance Jio AI, or anywhere with a real Indic-LLM gap to close — I'd love to talk. Apache 2.0. The full eval JSONs (every per-sample generation, every rubric score) and a "did not work" engineering log are in the repo so the numbers are reproducible end-to-end.
>
> #MachineLearning #NLP #LLM #IndicNLP #Marathi #OpenSource #QLoRA #Qwen

## Image attachments (in order)

LinkedIn lets you attach up to 9 images per post. Attach these 3 — they carry the post even for someone who only skims:

1. `eval/tokenizer_analysis/efficiency.png` — the 4.79× Marathi-vs-English chart. This is the single most distinctive artifact in the project and the visual hook.
2. `eval/charts/training_loss.png` — the eval_loss curve. Sells the "I actually trained this, monotonic descent, no overfit" point in one glance.
3. `eval/charts/eval_comparison.png` — base vs tuned on ROUGE / chrF / BLEU + the 71/10/19 pairwise outcome. Sells the headline result.

## Posting checklist

- [x] All numbers cross-checked with `eval/results/full_eval_2026-05-17.md` (source of truth).
- [x] HF model card updated with same numbers (`docs/HF_MODEL_CARD.md` → pushed to Hub).
- [x] README.md updated with same numbers (→ pushed to GitHub).
- [ ] Three attachments ready: `tokenizer_analysis/efficiency.png`, `charts/training_loss.png`, `charts/eval_comparison.png`.
- [ ] Paste primary draft into LinkedIn composer.
- [ ] Drag-drop the three images into the composer (in the order above so the tokenizer chart is the cover/thumbnail).
- [ ] Re-read the post once on the preview to make sure links render as link previews and Marathi text shows correctly.
- [ ] No spammy tagging. Tag only people / orgs there's a real connection with.
- [ ] **User clicks Post.** Claude does not click Post.

---

## Alternate Draft A — punchy / shorter (for if you want a terser variant)

> **83 million Marathi speakers. Roughly zero open-source instruction-tuned LLMs purpose-built for them.**
>
> I trained one on a free Colab T4 this week — and shipped a real eval.
>
> 🔗 Model:   huggingface.co/tusharislampure29/qwen2.5-1.5b-marathi-instruct
> 🔗 Code:    github.com/tusharislampure29/marathi-instruct-llm
> 🔗 Dataset: huggingface.co/datasets/tusharislampure29/marathi-instruct-30k
>
> Results vs Qwen2.5-1.5B-Instruct on 500 held-out prompts:
> • ROUGE-L F1 +40 %, chrF +41 %, sacreBLEU +276 %
> • GPT-4o pairwise (n=100, position-randomised): tuned 71 %, base 19 %, ties 10 % — preferred 3.7× more often
> • Cultural-set rubric: biggest lift on Marathi reasoning (instruction_following 1.1 → 1.7)
> • Training eval_loss: 0.767 → 0.587 across 3 epochs, monotonic
>
> Three findings the tutorials don't tell you:
> 1. Qwen2.5's BPE is 4.79× less efficient on Marathi than English — 1024 tokens = 161 Marathi words vs 882 English. The v2 hill to die on.
> 2. Stock TRL + Qwen2.5 + T4 + fp16 crashes with `_amp_foreach_non_finite_check_and_unscale_cuda`. Unsloth fixes it.
> 3. A TrainerCallback that pushes the LoRA adapter to HF Hub on every save_steps is the difference between "Colab disconnected" and "lost a day".
>
> What it can't do: store Maharashtra long-tail trivia. 1.5B parametric ceiling. Pair with retrieval if you care about factual correctness.
>
> Apache 2.0. Full eval JSONs and engineering log in the repo. If you work on Indic NLP, let's talk.
>
> #MachineLearning #NLP #LLM #IndicNLP #Marathi #OpenSource

---

## Alternate Draft B — engineering-log style (for technical / peer audiences)

> Shipped a Marathi instruction-tuned LLM on a free Colab T4 this week. Open-source, Apache 2.0.
>
> 🔗 huggingface.co/tusharislampure29/qwen2.5-1.5b-marathi-instruct
> 🔗 github.com/tusharislampure29/marathi-instruct-llm
>
> Stack: Qwen2.5-1.5B-Instruct + QLoRA (nf4, r=16, α=32, target = all attn + MLP projs) + Unsloth + TRL SFTTrainer on 27k filtered + MinHash-deduped Marathi instruction pairs from Aya. Effective batch 32, lr 2e-4 cosine, 3 epochs, ~7 hours of T4 (split across two sessions with HF Hub auto-resume on Colab disconnect at step 1333).
>
> Three eval layers (picking one would have been dishonest):
> • Auto-metrics on a 500-prompt test split: ROUGE-L F1 0.0314 → 0.0441 (+40 %), chrF 23.64 → 33.43 (+41 %), sacreBLEU 5.13 → 19.30 (+276 %).
> • GPT-4o pairwise on 100 prompts (position-randomised): tuned 71, base 19, ties 10. Cleanest behavioural signal because the judge sees full outputs head-to-head, not n-gram overlap with one reference.
> • GPT-4o 4-axis rubric on 50 hand-curated cultural prompts: biggest lift is the Marathi-reasoning category (instruction_following 1.1 → 1.7). Long-tail trivia stays near the floor for both columns — that's a 1.5B parametric ceiling, not a fine-tuning one. The rubric kept me from over-claiming.
>
> Things I had to learn the hard way (all in the repo):
> • Stock TRL on Qwen2.5 on Turing GPUs crashes with `_amp_foreach_non_finite_check_and_unscale_cuda`. Unsloth's native fp16 dispatch is the fix.
> • Qwen2.5's BPE is 4.79× worse on Marathi than English. 1024 tokens = 161 Marathi words.
> • My first 6.5-hour Kaggle eval run was reaped before I could download working files (Quick Save doesn't persist working dirs). Save & Run All / Commit on Kaggle servers does, and survives idle-session timeout.
> • Unsloth's HF stats endpoint times out at 120 s on Colab; monkey-patch `time_limited_stats_check` and move on.
> • A TrainerCallback that pushes the LoRA adapter to HF Hub every save_steps mine kicked in at step 1333; the second session picked up the latest checkpoint live from Hub.
>
> Hiring for Indic-NLP / multilingual-LLM / efficient-finetuning roles? Code + eval JSONs + decisions log all in the repo.
>
> #LLM #NLP #IndicNLP #Marathi #QLoRA #Unsloth #Qwen
