---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
language:
  - mr
library_name: peft
tags:
  - marathi
  - qlora
  - instruction-tuning
  - indic
  - qwen2.5
  - lora-adapter
datasets:
  - tusharislampure29/marathi-instruct-30k
pipeline_tag: text-generation
---

# Qwen2.5-1.5B Marathi-Instruct (QLoRA adapter)

A LoRA adapter for `Qwen/Qwen2.5-1.5B-Instruct` that teaches the model to follow instructions in **Marathi** (मराठी). Trained on a free Colab T4 — no premium hardware needed.

**Why this exists:** Marathi has 83M+ native speakers (3rd most-spoken language in India), but the Indic LLM ecosystem ships Hindi by default and treats Marathi as second-tier. This adapter demonstrates the gap is curated instruction data + the right training stack, not raw model capacity.

## Quick use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_id    = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "tusharislampure29/qwen2.5-1.5b-marathi-instruct"

tok   = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

msgs = [
    {"role": "system", "content": "तुम्ही एक उपयुक्त AI सहाय्यक आहात जो मराठीत स्पष्ट आणि अचूक उत्तरे देतो."},
    {"role": "user",   "content": "महाराष्ट्राची राजधानी कोणती आहे आणि तिथली लोकसंख्या किती आहे?"},
]
text   = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
out    = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9,
                        pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training data

- **Source:** Marathi slice of [`CohereForAI/aya_collection_language_split`](https://huggingface.co/datasets/CohereForAI/aya_collection_language_split) (~3.5M rows).
- **Pipeline:** Devanagari-script filter → length filter (10-2000 chars) → random pool of 60k → MinHash dedupe (Jaccard 0.85) → final 30k.
- **Splits:** 27k train / 1.5k val / 1.5k test, all formatted in Qwen2.5 ChatML with a Marathi system prompt.
- **Public release:** [`tusharislampure29/marathi-instruct-30k`](https://huggingface.co/datasets/tusharislampure29/marathi-instruct-30k).

## Training recipe

| Setting | Value |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Quantization | nf4, double-quant, compute dtype fp16 (T4) |
| LoRA r / α / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| Trainable params | ~18.5M (1.18% of base) |
| Max seq length | 1024 |
| Optimizer | AdamW (8-bit, paged) |
| LR / scheduler / warmup | 2e-4 / cosine / 0.03 |
| Effective batch size | 32 (4 × 8 grad accum) |
| Epochs | 3 |
| Hardware | 1× Colab free T4 16GB |
| Stack | Unsloth + TRL SFTTrainer |
| Best-checkpoint criterion | lowest `eval_loss` |

Mid-training checkpoints are pushed to this repo as `checkpoint-{step}/` so a Colab session disconnect ≠ lost work.

## Evaluation

### Training-side: eval_loss on held-out val (n=1500)

| step | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 | **2532** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval_loss | 0.767 | 0.702 | 0.669 | 0.645 | 0.632 | 0.618 | 0.608 | 0.598 | 0.595 | 0.591 | 0.589 | 0.587 | **0.587** |

Monotonic decrease (−23.5% total). Final checkpoint is best (`load_best_model_at_end=True`); the model was still improving at end of 3 epochs.

### Auto-metrics on held-out test split (n=500, greedy, max 256 new tokens; Kaggle T4)

Held-out 500-prompt slice of the test split. Greedy decoding, system prompt in Marathi, A/B toggled via `model.disable_adapter()` ctx on the same loaded model so the two columns differ only by the LoRA adapter.

| Metric | Base | Tuned | Δ (abs) | Δ (rel) |
|---|---|---|---|---|
| ROUGE-L F1 | 0.0314 | **0.0441** | +0.0127 | **+40 %** |
| chrF | 23.64 | **33.43** | +9.79 | **+41 %** |
| sacreBLEU | 5.13 | **19.30** | +14.16 | **+276 %** |

The BLEU jump from 5.13 to 19.30 is the headline auto-metric finding: the tuned adapter produces outputs that are nearly **four times closer to the reference text on n-gram overlap**. The full 500 generation pairs are reproducible from `notebooks/eval_full_kaggle.ipynb`.

### Behavioural eval — 50-prompt hand-curated Marathi cultural set (Kaggle T4)

Greedy generation, max 200 tokens, system prompt `तुम्ही एक उपयुक्त AI सहाय्यक…`. **Keyword recall** scores the fraction of expected keywords (from each prompt's `expected_keywords` list) that appear in the model output.

| Category | n | Base | Tuned | Δ |
|---|---|---|---|---|
| **overall** | 50 | 0.053 | 0.050 | −0.003 |
| geography | 10 | 0.000 | 0.000 | 0.000 |
| history | 10 | 0.033 | 0.000 | −0.033 |
| culture | 10 | 0.067 | 0.000 | −0.067 |
| language | 10 | 0.100 | 0.100 | 0.000 |
| **reasoning** | 10 | 0.067 | **0.150** | **+0.083 (+125 % rel)** |

**Interpretation.** Keyword recall is an honest but brittle metric: it punishes paraphrase and rewards verbosity. Qualitatively (see the 50 generation pairs in the project repo's `eval/results/`), the tuned model is consistently more fluent, more on-topic, and more direct — but its factual recall on long-tail Maharashtra trivia (specific saints' villages, exact district counts, Peshwa lineages) is capped by the 1.5B base model's parametric knowledge. The clearest behavioural improvement is on Marathi-language reasoning prompts, where the tuned adapter gives correct direct answers (e.g. `१२ × ३० = ३६० रुपये`) where the base model code-mixes Hindi and hallucinates.

The full evaluation set covers Maharashtra geography, history (Shivaji, social reformers), culture (festivals, food, warkari tradition), language (idioms, classic Marathi authors), and Marathi-language reasoning — see [`eval/cultural_marathi_eval.json`](https://github.com/tusharislampure29/marathi-instruct-llm/blob/main/eval/cultural_marathi_eval.json).

### GPT-4o 4-axis rubric on cultural set (n=50, base + tuned)

GPT-4o scores each response 1-5 on fluency / factuality / cultural_accuracy / instruction_following. The judge is calibrated strictly — _"give a 1 if the response is in the wrong language, gibberish, or empty"_ — which pulls both columns toward the floor on the long-tail Maharashtra trivia. The **direction and relative gap** are what's signal.

| Axis | Base | Tuned | Δ (abs) | Δ (rel) |
|---|---|---|---|---|
| fluency | 1.22 | **1.40** | +0.18 | +15 % |
| factuality | 1.12 | **1.22** | +0.10 | +9 % |
| cultural_accuracy | 1.16 | **1.28** | +0.12 | +10 % |
| instruction_following | 1.08 | **1.28** | +0.20 | +19 % |

**Per-category breakdown on `instruction_following`** — the axis the QLoRA SFT objective most directly targets:

| category | base | tuned | Δ |
|---|---|---|---|
| geography | 1.1 | 1.2 | +0.1 |
| history | 1.0 | 1.0 | 0.0 |
| culture | 1.0 | 1.1 | +0.1 |
| language | 1.2 | 1.4 | +0.2 |
| **reasoning** | 1.1 | **1.7** | **+0.6** |

The strongest behavioural lift is on the Marathi-language reasoning category — fluency 1.2→1.7, factuality 1.2→1.5, instruction_following 1.1→1.7. This matches the keyword-recall finding above. On long-tail trivia (Shivaji-era history, specific saints' villages, exact district counts) both models stay near the rubric floor because 1.5B parameters cannot store those facts at all — that's a model-capacity ceiling, not a fine-tuning one.

### GPT-4o pairwise A/B on 100 held-out test prompts

Each `(prompt, base_response, tuned_response)` triple is sent to GPT-4o, which picks A / B / TIE on Marathi fluency, factual accuracy, and helpfulness. **A/B order is randomized per prompt** to neutralise position bias (`src/eval_harness.py::llm_judge`).

| outcome | count | rate |
|---|---|---|
| **Tuned wins** | **71** / 100 | **71 %** |
| Base wins | 19 / 100 | 19 % |
| Ties | 10 / 100 | 10 % |

The tuned adapter is preferred over the base model **3.7× more often than the reverse** on held-out Marathi instruction prompts — and this is the cleanest behavioural signal in the whole release, because it directly compares full outputs head-to-head instead of measuring overlap with a single reference.

## Tokenizer efficiency note

Qwen2.5's tokenizer was trained on English-heavy data. On Marathi it is **4.79× less efficient by character** than English:

| | Marathi | English |
|---|---|---|
| chars/token | 1.04 | 4.98 |
| tokens/word | 6.36 | 1.16 |
| fragmentation rate | 98.8% | 16.5% |

A 1024-token context fits **161 Marathi words** vs **882 English words**. This is a hard ceiling on context utility for Marathi users; a v2 of this project should ship a Marathi-extended tokenizer. Full analysis: [`src/tokenizer_analysis.py`](https://github.com/tusharislampure29/marathi-instruct-llm/blob/main/src/tokenizer_analysis.py).

## Limitations

- **1.5B parameters** — strong fluency in everyday Marathi, but limited factual recall on long-tail trivia. Pair with retrieval for any production use.
- **Tokenizer mismatch** — see above. Effective context is ~1/5 of the nominal length on Marathi inputs.
- **Single epoch on 27k samples** is enough for instruction-following style; deeper knowledge requires either a bigger base or a continued-pretraining stage on a Marathi corpus.
- **English fall-through** — for prompts that are clearly out-of-domain (code, math reasoning beyond simple arithmetic), the model can still drift into English. The system prompt helps but is not a hard guarantee.
- **No safety tuning beyond the base model.**

## Intended use & out-of-scope

- ✅ Marathi Q&A, summarisation, simple instruction-following, conversational assistants, demo / educational projects.
- ❌ Medical, legal, financial advice. Anything that requires guaranteed factual correctness without verification.

## Citation

```bibtex
@misc{islampure2026marathiqwen,
  author = {Tushar Islampure},
  title  = {Qwen2.5-1.5B Marathi-Instruct (QLoRA adapter)},
  year   = 2026,
  publisher = {Hugging Face},
  url = {https://huggingface.co/tusharislampure29/qwen2.5-1.5b-marathi-instruct},
}
```

## Acknowledgements

- AI4Bharat & CohereForAI for the Indic instruction datasets.
- Alibaba Qwen team for Qwen2.5 (Apache 2.0).
- Unsloth for the T4-friendly QLoRA stack.
- OpenAI for GPT-4o (used as the evaluation judge for the rubric and the pairwise A/B).

---

Project source code (data prep, eval harness, notebook, decisions log): [github.com/tusharislampure29/marathi-instruct-llm](https://github.com/tusharislampure29/marathi-instruct-llm).
