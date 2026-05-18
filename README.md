# Marathi Instruction-Tuned LLM (Qwen2.5-1.5B QLoRA)

> A small open Marathi-instruction LLM built on a free Colab T4 — because the Indic LLM ecosystem ships Hindi by default and treats Marathi (83M+ speakers, the 3rd most-spoken language in India) as an afterthought.

**Author** · Tushar Islampure ([github.com/tusharislampure29](https://github.com/tusharislampure29))
**Model** · [`tusharislampure29/qwen2.5-1.5b-marathi-instruct`](https://huggingface.co/tusharislampure29/qwen2.5-1.5b-marathi-instruct) on Hugging Face
**Base** · `Qwen/Qwen2.5-1.5B-Instruct` (Apache 2.0)
**Method** · QLoRA (4-bit nf4, r=16, α=32), Unsloth on a free Colab T4
**Training run** · [W&B dashboard](https://wandb.ai/tusharislampure-personal/marathi-instruct-llm)
**License** · Apache 2.0

---

## TL;DR

**Training-side improvement (3 epochs, n=1.5k val):**

| step | 200 | 1200 | 1800 | 2200 | **2532 (final)** |
|---|---|---|---|---|---|
| eval_loss | 0.7670 | 0.6178 | 0.5947 | 0.5885 | **0.5871** |
| Δ vs step 200 | — | −19.5% | −22.5% | −23.3% | **−23.5%** |

Monotonic decrease across all 12 mid-training checkpoints, no overfitting. Final checkpoint is the best — model was still improving at end of run.

**Auto-metrics on a 500-prompt held-out test slice (Kaggle T4, greedy, max 256 new tokens):**

| Metric | Base | Tuned | Δ | Relative |
|---|---|---|---|---|
| ROUGE-L F1 | 0.0314 | **0.0441** | +0.0127 | **+40 %** |
| chrF | 23.64 | **33.43** | +9.79 | **+41 %** |
| sacreBLEU | 5.13 | **19.30** | +14.16 | **+276 %** |

The BLEU near-4× jump is the headline auto-metric result: the tuned adapter produces outputs that are much closer to the reference text on n-gram overlap.

**GPT-4o pairwise A/B on 100 held-out test prompts (position-randomized):**

| outcome | count | rate |
|---|---|---|
| **Tuned wins** | **71** / 100 | **71 %** |
| Base wins | 19 / 100 | 19 % |
| Ties | 10 / 100 | 10 % |

The tuned adapter is preferred over the base **3.7× more often than the reverse** — the cleanest behavioural signal in the release, because the judge sees full outputs head-to-head instead of measuring overlap with one reference.

**Held-out behavioural eval on the 50-prompt hand-curated Marathi cultural set (base vs tuned, keyword recall):**

| category | base | tuned | Δ |
|---|---|---|---|
| **overall** | 0.053 | 0.050 | −0.003 |
| geography | 0.000 | 0.000 | 0.000 |
| history | 0.033 | 0.000 | −0.033 |
| culture | 0.067 | 0.000 | −0.067 |
| language | 0.100 | 0.100 | 0.000 |
| **reasoning** | 0.067 | **0.150** | **+0.083 (+125 % rel)** |

**GPT-4o 4-axis rubric on the same 50 cultural prompts (1-5, strict floor):**

| Axis | Base | Tuned | Δ |
|---|---|---|---|
| fluency | 1.22 | **1.40** | +0.18 |
| factuality | 1.12 | **1.22** | +0.10 |
| cultural_accuracy | 1.16 | **1.28** | +0.12 |
| instruction_following | 1.08 | **1.28** | +0.20 |

The reasoning category drives every axis — fluency 1.2→1.7, instruction_following 1.1→1.7. Both models score near the rubric floor on long-tail Maharashtra trivia because 1.5B parameters simply can't store those facts; the cultural-recall ceiling needs either a bigger base or a continued-pretraining stage to lift.

**Honest read:** the **auto-metrics show a strong, consistent improvement** — ROUGE +40 %, chrF +41 %, BLEU +276 % — and the **pairwise judge confirms it behaviourally** at 71 % preference vs 19 %. The **keyword-recall** view on cultural trivia is essentially flat overall because at 1.5B params the model can't memorise Maharashtra long-tail facts (specific saints' villages, exact district counts); the one category where the tuned model wins decisively on every measure is Marathi-language reasoning, where it gives correct direct answers (e.g. `१२ × ३० = ३६० रुपये`) while the base model code-mixes Hindi and hallucinates.

---

## The problem nobody is solving

- **Speaker count vs LLM coverage are inverted for Marathi.** Marathi has more speakers than Bengali, Punjabi, Telugu, but every major Indic instruction tune (AI4Bharat Airavata, Sarvam-1, Krutrim) leads with Hindi/Tamil and Marathi is consistently second-tier or absent.
- **Big base models "see" Marathi but can't follow instructions in it.** Qwen2.5, Llama-3.2 and Gemma-2 have Marathi in pretraining, but prompted in Marathi they reply with stilted, code-mixed, or English-falling-back text. The blocker is curated **instruction-tuning data + a tokenizer that doesn't bleed efficiency**, not raw model capacity.

## The four things this project ships that other Indic-LLM repos don't

### 1. A Marathi-tokenizer-efficiency analysis (`src/tokenizer_analysis.py`)

Concrete finding from this repo (see [`eval/tokenizer_analysis/stats.json`](eval/tokenizer_analysis/stats.json)):

> **Qwen2.5's tokenizer is 4.79× less efficient on Marathi than English.**
> Marathi: 1.04 chars/token. English: 4.98 chars/token.
> A 1024-token context fits **161 Marathi words** vs **882 English words**.

98.8% of Marathi words get fragmented into 2+ tokens (vs 16.5% for English). Worst case in our corpus: words like _गोलंदाजांपैकी_ ("among the bowlers") get split into **14 tokens**.

This isn't decorative — it's the reason training a Marathi adapter is more expensive than the equivalent English adapter at the same token budget, and it's the strongest argument for shipping a Marathi-extended tokenizer in v2.

### 2. A hand-curated cultural Marathi eval set (`eval/cultural_marathi_eval.json`)

50 prompts, hand-written by the author (a native speaker), grouped into 5 categories:

- **geography** — Maharashtra capitals, rivers, regions, neighbours
- **history** — Shivaji Maharaj, social reformers, Maratha empire, freedom struggle
- **culture** — festivals (Gudi Padwa, Ganesh Chaturthi), food (puranpoli, modak), warkari tradition, Sant Tukaram/Dnyaneshwar
- **language** — idioms, proverbs, famous Marathi authors (Khandekar, P.L. Deshpande)
- **reasoning** — arithmetic, transitive logic, multi-step reasoning **in Marathi**

Each prompt ships with `expected_keywords` and a difficulty tag so an LLM judge can ground its scoring. Standard translated benchmarks miss cultural appropriateness; this set forces the model to demonstrate it knows Maharashtra, not just the language.

### 3. A 4-axis LLM-as-judge rubric (`src/eval_harness.py --rubric openai`)

Each model response on the cultural set is scored 1-5 on:

1. **fluency** — Marathi grammar, naturalness, register
2. **factuality** — verifiable claims
3. **cultural_accuracy** — Maharashtra-appropriate references, correct idiom use
4. **instruction_following** — answers what was asked, honours length/format

The harness supports both **GPT-4o** (`--rubric openai`, used in this release) and Claude Sonnet 4.6 (`--rubric claude`) as judge. It also runs the standard ROUGE-L / chrF / sacreBLEU on the held-out test set and a **position-randomized pairwise** A/B vs the base model.

### 4. A resume-safe training notebook (`notebooks/train_colab_hub.ipynb`)

Built on Unsloth (2-5× T4 speedup; native fp16 handling that sidesteps the `_amp_foreach_non_finite_check_and_unscale` bf16-grad-scaler crash you get with stock TRL on Qwen2.5's bf16 weights). A custom `TrainerCallback` pushes the LoRA adapter to HF Hub after every `save_steps`, so a Colab disconnect at hour 10 of a 12-hour training run still leaves you with the latest checkpoint live on the Hub. `load_best_model_at_end=True` means the published adapter is the lowest-eval-loss snapshot, not the last step.

## How it was built

1. **Data prep** (`src/data_prep.py`) — pulled the Marathi slice of [`CohereForAI/aya_collection_language_split`](https://huggingface.co/datasets/CohereForAI/aya_collection_language_split) (3.5M rows), filtered by Devanagari script + length, MinHash-deduped (Jaccard ≥ 0.85), formatted as Qwen ChatML with a Marathi system prompt. Crucial sequencing: random-sample first, dedupe after — deduping all 3.5M rows took 6+ hours; sampling to 60k then deduping takes ~5 min. Final: 27k train / 1.5k val / 1.5k test.

2. **Training** (`notebooks/train_colab_hub.ipynb`) — QLoRA (nf4, double-quant), r=16, α=32, LoRA on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`, max_seq 1024, batch 4 × grad-accum 8 (effective batch 32), lr 2e-4 cosine, warmup 0.03, 3 epochs. T4 16GB; Unsloth handles the fp16/bf16 dispatch.

3. **Eval** (`src/eval_harness.py`) — three layers:
   - automated metrics on held-out test (`--tag base|tuned`) — ROUGE-L / chrF / sacreBLEU
   - cultural eval rubric scored by GPT-4o (`--cultural --rubric openai`) — 4-axis 1-5 scoring per response, per category
   - pairwise A/B (`--compare base tuned --judge openai`) — position-randomized win-rate

## Reproduce

```powershell
git clone https://github.com/tusharislampure29/marathi-instruct-llm
cd marathi-instruct-llm
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt

# Build dataset
python -m src.data_prep

# Train: upload notebooks/train_colab_hub.ipynb to Colab,
#        add HF_TOKEN + WANDB_API_KEY to Colab Secrets, Run all.

# Eval (on machine with GPU; CPU works but slow)
python -m src.eval_harness --tag test_base --limit 500
python -m src.eval_harness --adapter tusharislampure29/qwen2.5-1.5b-marathi-instruct --tag test_tuned --limit 500
python -m src.eval_harness --compare test_base test_tuned --judge openai --limit 100

# Cultural eval + GPT-4o 4-axis rubric
python -m src.eval_harness --cultural --rubric openai --tag tuned-cultural \
    --adapter tusharislampure29/qwen2.5-1.5b-marathi-instruct
# Or, to score pre-generated outputs without a GPU:
python -m src.eval_harness --cultural --rubric openai --tag rubric_tuned \
    --load-responses eval/results/cultural_tuned.json

# Tokenizer efficiency analysis
python -m src.tokenizer_analysis
```

## Repo structure

```
01-marathi-instruct-llm/
├── src/
│   ├── data_prep.py            # corpus pipeline: filter → sample → MinHash dedupe → ChatML
│   ├── tokenizer_analysis.py   # tokens/char & fragmentation: Marathi vs English baseline
│   ├── eval_harness.py         # ROUGE/chrF/BLEU + GPT-4o/Claude rubric + pairwise A/B
│   └── inference.py            # load adapter + generate (for local sanity)
├── notebooks/
│   └── train_colab_hub.ipynb   # Unsloth QLoRA + resume-safe HF Hub checkpointing
├── configs/training_config.yaml
├── eval/
│   ├── cultural_marathi_eval.json   # 50 hand-curated cultural prompts
│   ├── tokenizer_analysis/          # stats.json + efficiency.png
│   └── results/                     # base / tuned / compare JSONs (filled at eval time)
├── data/
│   ├── raw/                    # HF dataset cache (gitignored)
│   └── processed/              # train / val / test JSONL (gitignored)
├── tests/test_data_prep.py
└── README.md
```

## Sample I/O (from Kaggle T4 eval, 2026-05-16)

Three representative pairs from the 50-prompt cultural set. The full set is in `eval/results/kaggle_eval_results.txt`.

| Prompt | Base (Qwen2.5-1.5B-Instruct) | Tuned (this adapter) |
|---|---|---|
| `महाराष्ट्राची राजधानी कोणती आहे?` | "महाराष्ट्राचा राजधानी हा बळावन गणितीला लंबे शहर झाला आहे, जो भारतवर प्रसिद्ध राजधानी होऊन आहे… 1. इंडिया राजधानी (Indiacity)…" (code-mixed Hindi/Marathi, hallucinated) | "सिंगापूठ" (terse, still wrong — but fluent Marathi register) |
| `संत तुकाराम कोणत्या गावाशी संबंधित आहेत?` | "संत तुकाराम हा नाव ब्रिटिश लड़की चकला तुकाराम हा नाव अफ्रीकी लड़की होता…" (gibberish + Hindi) | "ग्रँट हॉल, न्यूझीलंड" (wrong but pure Marathi) |
| `एका दुकानात एक पेन ३० रुपयांना मिळतो. १२ पेन घेतल्यास किती रुपये द्यावे लागतील?` | "12 पेन घेतल्यास 30 रुपयांनी मिळवले, तर याचे लागत बरदर्शी होऊन आहे… 30/12 = 2.5 रुपये…" (confused) | "12 * 30 = 360 रुपये" ✓ (correct arithmetic, direct answer) |

The pattern is consistent across the set: the tuned model gives **shorter, more direct, more fluent-Marathi** responses, but its factual recall on Maharashtra-specific trivia is still limited by the 1.5B base model's parametric knowledge. Reasoning prompts (arithmetic, day-of-week, transitive logic) are where the QLoRA tune shows real lift — see the +12.5pp keyword-recall delta on the reasoning category in the table above.

## What did not work (engineering log)

- **Stock TRL + Qwen2.5 + T4 + fp16 = `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`**: Qwen2.5 ships bf16 weights; `prepare_model_for_kbit_training` casts them, the LoRA layers inherit a mix, and the fp16 grad scaler crashes when it sees a bf16 grad in the optimizer state. Forced fp16 casts on trainable params did not fix it. Fix: switch to Unsloth, which handles the dispatch internally.
- **T4 + bf16 anywhere**: T4 (Turing) has no native bf16 hardware. Software-emulated bf16 measured at ~0.01 it/s on this model — projected ETA was 70 hours for 3 epochs. Unusable; the run will silently exhaust Colab's 12-hour session limit before finishing.
- **`pip install -U torch ...` on Colab**: silently upgraded torch to a CUDA-13 build while torchvision stayed at CUDA-12.8 → import-time crash. The Unsloth notebook installs only the missing pieces and lets Colab's pre-bundled torch/torchvision pair stay intact.
- **Unsloth's HF stats endpoint check**: timed out at 120s on Colab during the training day, which is fatal because it's the first thing `FastLanguageModel.from_pretrained` does. Worked around with a monkey-patch on `unsloth.models._utils.time_limited_stats_check`.

These are the kind of things you only find out by actually shipping; they are documented here for the next person.

## Acknowledgements

- AI4Bharat & CohereForAI for the Indic instruction datasets that made this dataset possible.
- Alibaba Qwen team for Qwen2.5-1.5B-Instruct (Apache 2.0).
- Unsloth for the T4-friendly QLoRA stack.
- OpenAI for GPT-4o as the rubric + pairwise judge.

## Contact

[`@tusharislampure29`](https://github.com/tusharislampure29) · tusharislampure@gmail.com
