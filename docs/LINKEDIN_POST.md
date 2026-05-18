# LinkedIn post drafts

Pick one. Both are aligned with the final HF model card + README numbers — keep them aligned if any number changes.

---

## Draft A — punchy / hook-first (recommended for recruiter reach)

> **83 million Marathi speakers. Roughly zero open-source instruction-tuned LLMs purpose-built for them.**
>
> I trained one on a free Colab T4 this week — and shipped a real eval.
>
> 🔗 Model:  huggingface.co/tusharislampure29/qwen2.5-1.5b-marathi-instruct
> 🔗 Code:   github.com/tusharislampure29/marathi-instruct-llm
> 🔗 Dataset: huggingface.co/datasets/tusharislampure29/marathi-instruct-30k
>
> **What it is:** a QLoRA LoRA adapter for `Qwen/Qwen2.5-1.5B-Instruct`, trained on 27k curated Marathi instruction pairs filtered + MinHash-deduped from CohereForAI Aya. 3 epochs, ~7 hours on a free Colab T4 (split across two sessions, auto-resumed from HF Hub after a Colab disconnect at step 1333).
>
> **Results vs the base Qwen2.5-1.5B-Instruct:**
> • Held-out test set (n=500): **ROUGE-L +40 %, chrF +41 %, BLEU +276 %**
> • **GPT-4o pairwise judge (n=100, position-randomized): tuned wins 71 %, base 19 %, ties 10 %** — preferred 3.7× more often than the reverse
> • Hand-curated 50-prompt Marathi cultural eval (4-axis rubric, GPT-4o judge): biggest behavioural lift is on Marathi-language reasoning — `instruction_following` 1.1 → 1.7 — where the tuned model gives `१२ × ३० = ३६० रुपये` while the base code-mixes Hindi and hallucinates.
> • Training eval_loss: **0.767 → 0.587 (−23.5 %), monotonic across all 12 checkpoints**. No overfitting; the model was still improving at the end of 3 epochs.
>
> **Three things I learned that aren't in any tutorial:**
>
> 1. **Qwen2.5's tokenizer is 4.79× less efficient on Marathi than English.** A 1024-token context fits **161 Marathi words vs 882 English words**. 98.8 % of Marathi words get fragmented into 2+ tokens. Worst case: _गोलंदाजांपैकी_ → 14 tokens. The single strongest argument for shipping a Marathi-extended tokenizer in v2.
>
> 2. **Free Colab T4 + Qwen2.5 + stock TRL = `_amp_foreach_non_finite_check_and_unscale_cuda` bf16 crash.** Qwen2.5 ships bf16 weights, T4 has no native bf16 hardware, and the fp16 grad scaler can't unscale a mixed bf16/fp16 grad pile. The fix is Unsloth — drop-in TRL replacement, native fp16 dispatch, 2× faster.
>
> 3. **Translated benchmarks miss what Marathi speakers actually care about.** So I hand-curated a 50-prompt cultural eval (geography, history including Shivaji Maharaj, warkari tradition, idioms, reasoning in Marathi), scored every response on a 4-axis rubric via GPT-4o, and ran a position-randomized pairwise A/B. The 71 % win-rate is the honest signal — and the rubric kept me honest about where the 1.5B base hits a parametric-knowledge ceiling on long-tail trivia.
>
> **The "did not work" log** is the most useful part of the repo: bf16-on-Turing, eval generations lost when a Kaggle interactive session got reaped (recovered via Save & Run All committed runs), Unsloth's HF stats check timing out at 120s. The kind of thing you only find out by actually shipping.
>
> Apache 2.0. Tokenizer-efficiency chart, full eval JSONs, training notebook, decisions log — all in the repo.
>
> If you work on Indic NLP at AI4Bharat, Sarvam, Krutrim, Reliance Jio AI, or anywhere else — there's a real gap to close and a 1.5B model on a free GPU is the entry point.
>
> #MachineLearning #NLP #LLM #IndicNLP #Marathi #OpenSource #QLoRA #Qwen

---

## Draft B — engineering-log style (for technical / peer audiences)

> Shipped a Marathi instruction-tuned LLM on a free Colab T4 this week. Open-source, Apache 2.0.
>
> 🔗 huggingface.co/tusharislampure29/qwen2.5-1.5b-marathi-instruct
> 🔗 github.com/tusharislampure29/marathi-instruct-llm
>
> **Stack:** Qwen2.5-1.5B-Instruct + QLoRA (nf4, r=16, α=32, target = all attn + MLP projs) + Unsloth + TRL SFTTrainer on 27k filtered + MinHash-deduped Marathi instruction pairs from Aya. Effective batch 32 (4 × grad-accum 8), lr 2e-4 cosine, 3 epochs, ~7 hours on a free T4 (split across two sessions with HF Hub auto-resume on Colab disconnect).
>
> **Three eval layers — because picking one would have been dishonest:**
>
> **1. Auto-metrics on a held-out 500-prompt test split (greedy, max 256 tokens):**
> &nbsp;&nbsp; ROUGE-L F1: 0.0314 → **0.0441** (+40 %)
> &nbsp;&nbsp; chrF: 23.64 → **33.43** (+41 %)
> &nbsp;&nbsp; sacreBLEU: 5.13 → **19.30** (+276 %)
>
> **2. GPT-4o pairwise A/B on 100 prompts (position-randomized to neutralise position bias):**
> &nbsp;&nbsp; Tuned wins **71**, base wins 19, ties 10. **3.7× preference ratio.** Cleanest behavioural signal in the release because the judge sees full outputs head-to-head, not n-gram overlap with one reference.
>
> **3. 4-axis GPT-4o rubric on a 50-prompt hand-curated Marathi cultural eval (geography / history / culture / language / reasoning), each scored 1–5 on fluency, factuality, cultural_accuracy, instruction_following:**
> &nbsp;&nbsp; Strongest lift is **reasoning**: instruction_following 1.1 → 1.7, fluency 1.2 → 1.7. Long-tail trivia (Shivaji-era history, district counts, saints' villages) sits near the rubric floor for both columns — a 1.5B parametric ceiling, not a fine-tuning one. The rubric kept me from over-claiming.
>
> **Things I had to learn the hard way (all written up in the repo):**
> • Stock TRL on Qwen2.5 on Turing GPUs (T4/2080) crashes with a bf16-grad-scaler error because Qwen ships bf16 and T4 has no native bf16 hardware. Fix: Unsloth's native fp16 dispatch.
> • Qwen2.5's BPE is 4.79× less efficient on Marathi than English (`src/tokenizer_analysis.py`). 1024 tokens = 161 Marathi words. This is the v2 hill to die on.
> • My first 6.5-hour eval generation run on Kaggle was reaped before I could download `/kaggle/working/` — Quick Save doesn't persist working-dir files. Re-ran as Save & Run All (Commit) on Kaggle servers; that mode does persist, and committed runs survive idle-session timeout. Lesson: never depend on a live Kaggle session for >40 min of unsaved output.
> • Unsloth's HF stats endpoint check times out at 120 s on Colab; monkey-patch `time_limited_stats_check` and move on.
> • A `TrainerCallback` that pushes the LoRA adapter to HF Hub every `save_steps` is the difference between "Colab disconnected at hour 10" being a recoverable annoyance vs. lost work. Mine kicked in at step 1333; the second session picked up the latest checkpoint live from Hub.
>
> If you're hiring for Indic-NLP / multilingual LLM / efficient-finetuning roles, I'd love to chat — code + eval JSONs + decisions log all in the repo.
>
> #LLM #NLP #IndicNLP #Marathi #QLoRA #Unsloth #Qwen

---

## Suggested visual

Attach `eval/tokenizer_analysis/efficiency.png` — the 4.79× tokenizer-efficiency chart. Single most legible artifact in the project; works as a thumbnail even for readers who scroll past.

## Posting checklist before publishing

- [x] All numbers cross-checked with `eval/results/full_eval_2026-05-17.md` (final source of truth).
- [x] HF model card updated with same numbers (`docs/HF_MODEL_CARD.md`).
- [x] README.md updated with same numbers.
- [ ] HF model card pushed to the public adapter repo on the Hub.
- [ ] GitHub repo `tusharislampure29/marathi-instruct-llm` pushed and README renders correctly.
- [ ] Attach `eval/tokenizer_analysis/efficiency.png` as the cover image.
- [ ] Pick draft A (recruiter-skim) or draft B (peer-engineer) — A is the safer default.
- [ ] Tag relevant people / orgs only if there's a real connection. **No spammy tagging.**
- [ ] User clicks Post in Chrome — Claude does NOT post automatically.
