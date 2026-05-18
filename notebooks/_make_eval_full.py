"""Build notebooks/eval_full_kaggle.py and .ipynb with first N test rows embedded.

Run from project root with the project venv:
    py -3.12 notebooks/_make_eval_full.py
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_JSONL = ROOT / "data" / "processed" / "test.jsonl"
CULTURAL_JSON = ROOT / "eval" / "cultural_marathi_eval.json"
OUT_PY = ROOT / "notebooks" / "eval_full_kaggle.py"
OUT_IPYNB = ROOT / "notebooks" / "eval_full_kaggle.ipynb"

TEST_LIMIT = 500

# ---- load and shrink test rows ----
rows: list[dict] = []
with TEST_JSONL.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        # only ship the fields the notebook needs
        rows.append({"instruction": r["instruction"], "response": r["response"]})
        if len(rows) >= TEST_LIMIT:
            break

test_json = json.dumps(rows, ensure_ascii=False)
test_b64 = base64.b64encode(gzip.compress(test_json.encode("utf-8"), 9)).decode("ascii")
print(f"test rows: {len(rows)}  raw={len(test_json)/1024:.0f} KB  gzip+b64={len(test_b64)/1024:.0f} KB")

# ---- load cultural eval ----
cultural_obj = json.loads(CULTURAL_JSON.read_text(encoding="utf-8"))
cultural_json = json.dumps(cultural_obj, ensure_ascii=False, separators=(",", ":"))
print(f"cultural prompts: {len(cultural_obj['prompts'])}  raw={len(cultural_json)/1024:.0f} KB")

# ---- the runtime script (jinja-ish via str.format would clash with json braces; use placeholders) ----
SCRIPT_TEMPLATE = r'''# === Full eval: held-out test (n=__TEST_LIMIT__, ROUGE/chrF/BLEU) + cultural (n=50, base+tuned) ===
# Loads public Qwen/Qwen2.5-1.5B-Instruct + public LoRA adapter, generates base vs tuned outputs
# greedy on both the embedded test slice and the inline cultural eval, computes auto-metrics,
# saves 4 JSON files matching the local src/eval_harness.py schemas so the local rubric/pairwise
# passes work against them without any further setup.
#
# DURABILITY: every partial+final save is also uploaded to HF Hub
# (tusharislampure29/qwen2.5-1.5b-marathi-instruct, eval_outputs/ subdir). Even if the Kaggle
# session is reaped mid-run, the latest partial files survive on HF.
#
# Outputs (under /kaggle/working/):
#   test_base.json, test_tuned.json          - schema: {model_id, adapter_id, tag, metrics, samples}
#   cultural_base.json, cultural_tuned.json  - schema: {model_id, adapter_id, tag, rubric, n, samples}
#   eval_outputs.zip                         - bundle of all four for one-click download

import base64, gzip, json, os, subprocess, time
from collections import defaultdict

# 0. HF_TOKEN for durable uploads (substituted from local env at build time)
HF_TOKEN = "__HF_TOKEN_PLACEHOLDER__"
HF_REPO  = "tusharislampure29/qwen2.5-1.5b-marathi-instruct"
HF_SUBDIR = "eval_outputs"

# 1. install
for pkg in ["transformers", "peft", "bitsandbytes", "accelerate", "rouge_score", "sacrebleu", "huggingface_hub"]:
    subprocess.run(["pip", "install", "-q", "-U", pkg], capture_output=True)

from huggingface_hub import HfApi
_hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN and not HF_TOKEN.startswith("__") else None
def hf_push(local_path, repo_path):
    if _hf_api is None:
        print(f"  [hf_push] skipped (no token): {local_path}")
        return
    try:
        _hf_api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{HF_SUBDIR}/{repo_path}",
            repo_id=HF_REPO, repo_type="model",
            commit_message=f"eval: {repo_path}",
        )
        print(f"  [hf_push] OK: {repo_path}")
    except Exception as e:
        print(f"  [hf_push] FAIL {repo_path}: {e}")

import torch
torch.manual_seed(42)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

CFG = {
    "base_id":          "Qwen/Qwen2.5-1.5B-Instruct",
    "adapter_id":       "tusharislampure29/qwen2.5-1.5b-marathi-instruct",
    "system_prompt":    "तुम्ही एक उपयुक्त AI सहाय्यक आहात जो मराठीत स्पष्ट आणि अचूक उत्तरे देतो.",
    "max_new_cultural": 200,
    "max_new_test":     256,
    "out_dir":          "/kaggle/working",
}

# 2. inline test slice (first __TEST_LIMIT__ rows of held-out test.jsonl, gzip+b64)
TEST_B64 = "__TEST_B64__"
TEST_ROWS = json.loads(gzip.decompress(base64.b64decode(TEST_B64)).decode("utf-8"))
print(f"test rows embedded: {len(TEST_ROWS)}")

# 3. inline cultural eval (50 prompts)
CULTURAL = json.loads(r"""__CULTURAL_JSON__""")
print(f"cultural prompts: {len(CULTURAL['prompts'])}")

# 4. load base + attach LoRA adapter (both public, no token needed)
# GPU-aware: bnb 4-bit needs Turing or newer (T4 = 7.5, P100 = 6.0). On Pascal we fall back to fp16,
# which still fits the 1.5B model with plenty of headroom in P100's 16 GB.
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
_gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
_compute_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
_use_4bit = _compute_cap[0] >= 7  # Turing (T4) is 7.5, Ampere/Hopper are higher
print(f"GPU: {_gpu_name}  compute_capability: {_compute_cap}  use_4bit: {_use_4bit}")

tok = AutoTokenizer.from_pretrained(CFG["base_id"])
if _use_4bit:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    _base = AutoModelForCausalLM.from_pretrained(CFG["base_id"], quantization_config=bnb,
                                                  device_map="auto", torch_dtype=torch.float16)
else:
    _base = AutoModelForCausalLM.from_pretrained(CFG["base_id"], device_map="auto",
                                                  torch_dtype=torch.float16)
model = PeftModel.from_pretrained(_base, CFG["adapter_id"])
model.eval()
print("model + adapter loaded")

# 5. greedy generate (one prompt, one mode)
def gen(prompt: str, max_new: int, use_adapter: bool) -> str:
    msgs = [{"role": "system", "content": CFG["system_prompt"]},
            {"role": "user",   "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=max_new, do_sample=False, temperature=1.0, top_p=1.0,
                  pad_token_id=tok.eos_token_id)
    if use_adapter:
        with torch.no_grad():
            out = model.generate(**inputs, **kwargs)
    else:
        with model.disable_adapter(), torch.no_grad():
            out = model.generate(**inputs, **kwargs)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  wrote {path} ({os.path.getsize(path)/1024:.0f} KB)")

# 6. cultural first (faster - if test crashes we still have this)
from tqdm.auto import tqdm
print("\n--- cultural generation (n=50 x 2) ---")
t0 = time.time()
base_cult, tuned_cult = [], []
for item in tqdm(CULTURAL["prompts"]):
    common = {"id": item["id"], "category": item["category"], "difficulty": item["difficulty"],
              "prompt": item["prompt"], "expected_keywords": item.get("expected_keywords", [])}
    base_cult.append({**common, "response": gen(item["prompt"], CFG["max_new_cultural"], False)})
    tuned_cult.append({**common, "response": gen(item["prompt"], CFG["max_new_cultural"], True)})
print(f"cultural done in {(time.time()-t0)/60:.1f} min")

write_json(f"{CFG['out_dir']}/cultural_base.json", {
    "model_id": CFG["base_id"], "adapter_id": None, "tag": "cultural_base",
    "rubric": None, "n": len(base_cult), "samples": base_cult})
write_json(f"{CFG['out_dir']}/cultural_tuned.json", {
    "model_id": CFG["base_id"], "adapter_id": CFG["adapter_id"], "tag": "cultural_tuned",
    "rubric": None, "n": len(tuned_cult), "samples": tuned_cult})
hf_push(f"{CFG['out_dir']}/cultural_base.json",  "cultural_base.json")
hf_push(f"{CFG['out_dir']}/cultural_tuned.json", "cultural_tuned.json")

# 7. test set (long pass - ~80 min for n=500)
print(f"\n--- test generation (n={len(TEST_ROWS)} x 2) ---")
t0 = time.time()
base_test, tuned_test = [], []
for row in tqdm(TEST_ROWS):
    instr, ref = row["instruction"], row["response"]
    b = gen(instr, CFG["max_new_test"], False)
    t = gen(instr, CFG["max_new_test"], True)
    base_test.append({"instruction": instr, "reference": ref, "prediction": b})
    tuned_test.append({"instruction": instr, "reference": ref, "prediction": t})
    # incremental save every 100 in case of OOM/timeout + durable push to HF Hub
    if (len(base_test) % 100) == 0:
        write_json(f"{CFG['out_dir']}/test_base.partial.json", {
            "model_id": CFG["base_id"], "adapter_id": None, "tag": "test_base_partial",
            "metrics": {"n": len(base_test)}, "samples": base_test})
        write_json(f"{CFG['out_dir']}/test_tuned.partial.json", {
            "model_id": CFG["base_id"], "adapter_id": CFG["adapter_id"], "tag": "test_tuned_partial",
            "metrics": {"n": len(tuned_test)}, "samples": tuned_test})
        hf_push(f"{CFG['out_dir']}/test_base.partial.json",  f"test_base.partial_{len(base_test):04d}.json")
        hf_push(f"{CFG['out_dir']}/test_tuned.partial.json", f"test_tuned.partial_{len(tuned_test):04d}.json")
print(f"test done in {(time.time()-t0)/60:.1f} min")

# 8. auto-metrics
from rouge_score import rouge_scorer
import sacrebleu
def compute_metrics(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return {"rougeL_f1": (sum(rouge_l)/len(rouge_l)) if rouge_l else 0.0,
            "chrF": sacrebleu.corpus_chrf(preds, [refs]).score,
            "bleu": sacrebleu.corpus_bleu(preds, [refs]).score,
            "n":    len(preds)}

refs    = [s["reference"]  for s in base_test]
m_base  = compute_metrics([s["prediction"] for s in base_test],  refs)
m_tuned = compute_metrics([s["prediction"] for s in tuned_test], refs)

write_json(f"{CFG['out_dir']}/test_base.json", {
    "model_id": CFG["base_id"], "adapter_id": None, "tag": "test_base",
    "metrics": m_base, "samples": base_test})
write_json(f"{CFG['out_dir']}/test_tuned.json", {
    "model_id": CFG["base_id"], "adapter_id": CFG["adapter_id"], "tag": "test_tuned",
    "metrics": m_tuned, "samples": tuned_test})
hf_push(f"{CFG['out_dir']}/test_base.json",  "test_base.json")
hf_push(f"{CFG['out_dir']}/test_tuned.json", "test_tuned.json")

# 9. keyword recall on cultural (sanity check)
def kw_recall_by_cat(rows):
    by_cat, overall = defaultdict(list), []
    for r in rows:
        kws = r.get("expected_keywords") or []
        if not kws: continue
        hits = sum(1 for k in kws if k in r["response"]) / len(kws)
        by_cat[r["category"]].append(hits); overall.append(hits)
    return {"overall": (sum(overall)/len(overall)) if overall else 0.0,
            "by_category": {c: sum(v)/len(v) for c, v in by_cat.items()}}
kb = kw_recall_by_cat(base_cult); kt = kw_recall_by_cat(tuned_cult)

# 10. bundle
import zipfile
zip_path = f"{CFG['out_dir']}/eval_outputs.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in ["test_base.json", "test_tuned.json", "cultural_base.json", "cultural_tuned.json"]:
        z.write(f"{CFG['out_dir']}/{fn}", arcname=fn)
print(f"\nbundle: {zip_path} ({os.path.getsize(zip_path)/1024:.0f} KB)")
hf_push(zip_path, "eval_outputs.zip")

# 11. headline
print("\n=== HEADLINE ===")
print(f"test set (n={m_base['n']}):")
print(f"  rougeL_f1: base={m_base['rougeL_f1']:.4f}  tuned={m_tuned['rougeL_f1']:.4f}  delta={m_tuned['rougeL_f1']-m_base['rougeL_f1']:+.4f}")
print(f"  chrF     : base={m_base['chrF']:6.2f}   tuned={m_tuned['chrF']:6.2f}   delta={m_tuned['chrF']-m_base['chrF']:+.2f}")
print(f"  bleu     : base={m_base['bleu']:6.2f}   tuned={m_tuned['bleu']:6.2f}   delta={m_tuned['bleu']-m_base['bleu']:+.2f}")
print(f"\ncultural keyword recall (n=50):")
print(f"  overall: base={kb['overall']:.3f}  tuned={kt['overall']:.3f}  delta={kt['overall']-kb['overall']:+.3f}")
for c in kt["by_category"]:
    b, t = kb["by_category"][c], kt["by_category"][c]
    print(f"  {c:12s}: base={b:.3f}  tuned={t:.3f}  delta={t-b:+.3f}")

print("\n=== DONE ===")
'''

script = (SCRIPT_TEMPLATE
          .replace("__TEST_LIMIT__", str(TEST_LIMIT))
          .replace("__TEST_B64__", test_b64)
          .replace("__CULTURAL_JSON__", cultural_json))

OUT_PY.write_text(script, encoding="utf-8")
print(f"wrote {OUT_PY.relative_to(ROOT)} ({OUT_PY.stat().st_size/1024:.0f} KB)")

# ---- build ipynb ----
nb = {
    "cells": [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": script.splitlines(keepends=True),
    }],
    "metadata": {
        "kernelspec":     {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info":  {"name": "python"},
        "accelerator":    "GPU",
        "kaggle":         {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                           "dockerImageVersionId": 30761, "isInternetEnabled": True,
                           "language": "python", "sourceType": "notebook"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT_IPYNB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUT_IPYNB.relative_to(ROOT)} ({OUT_IPYNB.stat().st_size/1024:.0f} KB)")
