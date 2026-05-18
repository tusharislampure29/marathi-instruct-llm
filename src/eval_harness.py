"""
Evaluate a (base model, adapter) pair for Marathi instruction-following.

Modes:
  1. Held-out test set (data/processed/test.jsonl):
        - ROUGE-L, chrF, sacreBLEU vs reference
        - Pairwise LLM-judge win-rate (--compare base tuned --judge claude|openai)
  2. Hand-curated cultural eval set (eval/cultural_marathi_eval.json):
        - Per-response rubric scoring on 4 axes
          (fluency, factuality, cultural-accuracy, instruction-following; 1-5)
        - Judge provider: claude (Sonnet 4.6) or openai (gpt-4o)
        - Per-category breakdown (geography / history / culture / language / reasoning)
        - --load-responses lets the rubric score pre-generated outputs (e.g. from a Kaggle
          notebook run) without re-running model generation on the local machine.

Run base on held-out test:
    python -m src.eval_harness --model Qwen/Qwen2.5-1.5B-Instruct --tag base

Run adapter on cultural eval (rubric scored by GPT-4o):
    python -m src.eval_harness \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --adapter tusharislampure29/qwen2.5-1.5b-marathi-instruct \\
        --tag tuned-cultural --cultural --rubric openai

Pairwise compare on test set with GPT-4o:
    python -m src.eval_harness --compare base tuned --judge openai

Score pre-generated cultural responses (no GPU needed):
    python -m src.eval_harness --cultural --rubric openai \\
        --tag rubric_tuned --load-responses eval/results/cultural_tuned.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
CULTURAL_EVAL = PROJECT_ROOT / "eval" / "cultural_marathi_eval.json"

MARATHI_SYSTEM = "तुम्ही एक उपयुक्त AI सहाय्यक आहात जो मराठीत स्पष्ट आणि अचूक उत्तरे देतो."


def load_cultural() -> list[dict]:
    if not CULTURAL_EVAL.exists():
        raise FileNotFoundError(f"{CULTURAL_EVAL} not found.")
    return json.loads(CULTURAL_EVAL.read_text(encoding="utf-8"))["prompts"]


def load_test(limit: int | None = None) -> list[dict]:
    path = PROCESSED_DIR / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run data_prep first.")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def generate_responses(model_id: str, adapter_id: str | None, prompts: list[str],
                       max_new_tokens: int = 256) -> list[str]:
    """Lazily import torch/transformers so the harness can import without them."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[gen] loading {model_id}" + (f" + adapter {adapter_id}" if adapter_id else ""))
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
    )
    if adapter_id:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_id)

    model.eval()
    outputs: list[str] = []
    for p in prompts:
        messages = [
            {"role": "system", "content": MARATHI_SYSTEM},
            {"role": "user", "content": p},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(gen.strip())
    return outputs


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    from rouge_score import rouge_scorer
    import sacrebleu

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_f = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    return {
        "rougeL_f1": sum(rouge_l_f) / len(rouge_l_f) if rouge_l_f else 0.0,
        "chrF": chrf,
        "bleu": bleu,
        "n": len(predictions),
    }


def run_single(model_id: str, adapter_id: str | None, tag: str, limit: int | None) -> Path:
    rows = load_test(limit)
    prompts = [r["instruction"] for r in rows]
    refs = [r["response"] for r in rows]
    preds = generate_responses(model_id, adapter_id, prompts)
    metrics = compute_metrics(preds, refs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{tag}.json"
    payload = {
        "model_id": model_id,
        "adapter_id": adapter_id,
        "tag": tag,
        "metrics": metrics,
        "samples": [
            {"instruction": p, "reference": r, "prediction": pr}
            for p, r, pr in zip(prompts[:20], refs[:20], preds[:20])
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{tag}] {metrics}")
    print(f"  → {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


JUDGE_PROMPT = """You are an expert Marathi linguist. Compare two responses (A and B) to the same prompt and pick the better one based on: (1) Marathi fluency and grammar, (2) accuracy, (3) helpfulness.

Prompt: {prompt}

Response A:
{a}

Response B:
{b}

Reply with EXACTLY one line: "A", "B", or "TIE". No explanation."""


def llm_judge(prompts: list[str], a_responses: list[str], b_responses: list[str],
              provider: str) -> dict:
    """Returns {wins_a, wins_b, ties, total}. Randomizes A/B order to remove position bias."""
    import random
    rng = random.Random(0)

    if provider == "claude":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        def ask(prompt_text: str) -> str:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return resp.content[0].text.strip().upper()

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        def ask(prompt_text: str) -> str:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return resp.choices[0].message.content.strip().upper()
    else:
        raise ValueError(f"unknown judge provider: {provider}")

    wins_a = wins_b = ties = 0
    for p, a, b in zip(prompts, a_responses, b_responses):
        # Randomize A/B order so the judge cannot favor a fixed position.
        if rng.random() < 0.5:
            verdict = ask(JUDGE_PROMPT.format(prompt=p, a=a, b=b))
            mapped = {"A": "A", "B": "B"}.get(verdict, "TIE")
        else:
            verdict = ask(JUDGE_PROMPT.format(prompt=p, a=b, b=a))
            mapped = {"A": "B", "B": "A"}.get(verdict, "TIE")
        if mapped == "A":
            wins_a += 1
        elif mapped == "B":
            wins_b += 1
        else:
            ties += 1
        time.sleep(0.2)
    total = wins_a + wins_b + ties
    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties, "total": total,
            "win_rate_a": wins_a / total if total else 0.0,
            "win_rate_b": wins_b / total if total else 0.0}


def run_compare(tag_a: str, tag_b: str, judge: str, limit: int | None) -> None:
    a_path = RESULTS_DIR / f"{tag_a}.json"
    b_path = RESULTS_DIR / f"{tag_b}.json"
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))

    rows = load_test(limit or 100)  # judge calls cost money; cap at 100 by default
    prompts = [r["instruction"] for r in rows]

    a_preds = [s["prediction"] for s in a["samples"]][:len(prompts)]
    b_preds = [s["prediction"] for s in b["samples"]][:len(prompts)]
    n = min(len(a_preds), len(b_preds), len(prompts))
    prompts, a_preds, b_preds = prompts[:n], a_preds[:n], b_preds[:n]

    print(f"[judge:{judge}] comparing {tag_a} vs {tag_b} on {n} samples")
    result = llm_judge(prompts, a_preds, b_preds, judge)
    result["a_tag"] = tag_a
    result["b_tag"] = tag_b
    result["judge"] = judge
    out = RESULTS_DIR / f"compare_{tag_a}_vs_{tag_b}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {result}")
    print(f"  → {out.relative_to(PROJECT_ROOT)}")


RUBRIC_PROMPT = """You are a strict Marathi linguistics evaluator. Score the assistant's response on four axes.

Each axis is 1-5 (1=very poor, 5=excellent). Be honest — give a 1 if the response is in the wrong language, gibberish, or empty.

Axes:
  1. fluency             — Marathi grammar, naturalness, register
  2. factuality          — claims that are verifiable as true (use general knowledge)
  3. cultural_accuracy   — appropriateness to Marathi/Maharashtra context, idiom use
  4. instruction_following — answers what was asked, follows length/format constraints

Prompt (Marathi):
{prompt}

Expected answer should mention any of these keywords: {keywords}

Assistant's response:
{response}

Reply with ONE LINE of valid JSON: {{"fluency": int, "factuality": int, "cultural_accuracy": int, "instruction_following": int}}"""


def _rubric_client(provider: str):
    """Build the provider client. Caller must set the appropriate env var."""
    if provider == "claude":
        from anthropic import Anthropic
        return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise ValueError(f"unknown rubric provider: {provider}")


def _rubric_ask(provider: str, client, user_text: str) -> str:
    """Send one rubric prompt, return the raw text reply."""
    if provider == "claude":
        return client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": user_text}],
        ).content[0].text.strip()
    if provider == "openai":
        return client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            messages=[{"role": "user", "content": user_text}],
        ).choices[0].message.content.strip()
    raise ValueError(f"unknown rubric provider: {provider}")


def _score_one(provider: str, client, prompt: str, keywords: list[str], response: str) -> dict:
    """Returns 4-axis rubric scores for one response; degrades gracefully on non-JSON reply."""
    user_text = RUBRIC_PROMPT.format(
        prompt=prompt,
        keywords=", ".join(keywords) or "(open-ended)",
        response=response,
    )
    text = _rubric_ask(provider, client, user_text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {"fluency": 0, "factuality": 0, "cultural_accuracy": 0,
                "instruction_following": 0, "_raw": text, "_error": "no json"}
    try:
        d = json.loads(text[start:end + 1])
        for k in ("fluency", "factuality", "cultural_accuracy", "instruction_following"):
            d[k] = max(1, min(5, int(d.get(k, 0))))
        return d
    except Exception as e:
        return {"fluency": 0, "factuality": 0, "cultural_accuracy": 0,
                "instruction_following": 0, "_raw": text, "_error": str(e)}


def run_cultural(model_id: str, adapter_id: str | None, tag: str,
                 rubric: str | None, limit: int | None,
                 load_responses: str | None = None) -> Path:
    """Generate on the hand-curated cultural set; optionally score each response with a rubric.

    If `load_responses` is given, skip model generation and load `(prompt, response)` pairs
    from that JSON file (must follow the `run_cultural` output schema). Lets the rubric run
    on a CPU-only box against outputs produced by a Kaggle/Colab notebook.
    """
    items = load_cultural()
    if limit:
        items = items[:limit]

    if load_responses:
        loaded = json.loads(Path(load_responses).read_text(encoding="utf-8"))
        # Build a {id -> response} map from the loaded file so we can align by item id.
        resp_by_id = {s["id"]: s["response"] for s in loaded["samples"]}
        missing = [i["id"] for i in items if i["id"] not in resp_by_id]
        if missing:
            raise ValueError(f"--load-responses file is missing {len(missing)} ids "
                             f"(e.g. {missing[:3]}); regenerate it on the full cultural set.")
        preds = [resp_by_id[i["id"]] for i in items]
    else:
        prompts = [i["prompt"] for i in items]
        preds = generate_responses(model_id, adapter_id, prompts)

    out = {
        "model_id": model_id,
        "adapter_id": adapter_id,
        "tag": tag,
        "rubric": rubric,
        "loaded_from": load_responses,
        "n": len(items),
        "samples": [],
    }

    if rubric in ("claude", "openai"):
        client = _rubric_client(rubric)
        per_cat: dict[str, list[dict]] = {}
        for item, response in zip(items, preds):
            score = _score_one(rubric, client, item["prompt"],
                               item.get("expected_keywords", []), response)
            sample = {
                "id": item["id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "prompt": item["prompt"],
                "expected_keywords": item.get("expected_keywords", []),
                "response": response,
                "scores": score,
            }
            out["samples"].append(sample)
            per_cat.setdefault(item["category"], []).append(score)
            time.sleep(0.2)
        out["per_category_mean"] = {
            cat: {
                axis: round(sum(s.get(axis, 0) for s in scores) / len(scores), 2)
                for axis in ("fluency", "factuality", "cultural_accuracy", "instruction_following")
            }
            for cat, scores in per_cat.items()
        }
        all_scores = [s for cat in per_cat.values() for s in cat]
        out["overall_mean"] = {
            axis: round(sum(s.get(axis, 0) for s in all_scores) / len(all_scores), 2)
            for axis in ("fluency", "factuality", "cultural_accuracy", "instruction_following")
        }
    else:
        for item, response in zip(items, preds):
            out["samples"].append({
                "id": item["id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "prompt": item["prompt"],
                "expected_keywords": item.get("expected_keywords", []),
                "response": response,
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{tag}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{tag}] wrote {len(items)} samples → {out_path.relative_to(PROJECT_ROOT)}")
    if "overall_mean" in out:
        print(f"  overall rubric: {out['overall_mean']}")
        print(f"  per-category : {out['per_category_mean']}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--tag", default=None, help="Output filename tag (e.g. base, tuned).")
    p.add_argument("--limit", type=int, default=None, help="Cap test samples (for fast iter).")
    p.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"),
                   help="Run LLM-judge head-to-head between two prior runs.")
    p.add_argument("--judge", choices=["claude", "openai"], default="claude")
    p.add_argument("--cultural", action="store_true",
                   help="Use the hand-curated cultural eval set instead of test.jsonl.")
    p.add_argument("--rubric", choices=["claude", "openai"], default=None,
                   help="With --cultural: score each response on a 4-axis rubric via the chosen provider.")
    p.add_argument("--load-responses", default=None,
                   help="With --cultural: load pre-generated responses from this JSON file "
                        "(skips model generation, runs rubric only).")
    args = p.parse_args()

    if args.compare:
        run_compare(args.compare[0], args.compare[1], args.judge, args.limit)
    elif args.cultural:
        if not args.tag:
            p.error("--tag is required with --cultural")
        run_cultural(args.model, args.adapter, args.tag, args.rubric, args.limit,
                     load_responses=args.load_responses)
    else:
        if not args.tag:
            p.error("--tag is required when not using --compare")
        run_single(args.model, args.adapter, args.tag, args.limit)


if __name__ == "__main__":
    main()
