"""
One-shot prep for Colab via HF Hub:
1. Create dataset repo + push train.jsonl/val.jsonl
2. Create model repo (target for the trained adapter)
3. Build a Colab-ready notebook that loads data from the dataset repo, embeds
   the training config inline, and pushes the adapter to the model repo
4. Push that notebook to the model repo so Colab can open it from URL

Run: python scripts/push_to_hub.py  (HF_TOKEN must be set in the env)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "configs" / "training_config.yaml"

USERNAME = "tusharislampure29"
DATASET_REPO = f"{USERNAME}/marathi-instruct-30k"
MODEL_REPO = f"{USERNAME}/qwen2.5-1.5b-marathi-instruct"


def build_notebook(cfg: dict) -> dict:
    """ipynb JSON with config baked in and data loaded from HF Hub."""
    cfg_literal = json.dumps(cfg, ensure_ascii=False, indent=2)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Marathi QLoRA Fine-Tune — Qwen2.5-1.5B-Instruct\n",
                "\n",
                "Author: Tushar Islampure (github.com/tusharislampure29)\n",
                "\n",
                "Loads data from `tusharislampure29/marathi-instruct-30k` on HF Hub and pushes the trained adapter to `tusharislampure29/qwen2.5-1.5b-marathi-instruct`.\n",
                "\n",
                "Before Run all: Runtime > Change runtime type > T4 GPU. Add HF_TOKEN (write) and WANDB_API_KEY to Colab Secrets (key icon, left sidebar).\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 1. install training stack\n",
                "%pip install -q -U torch transformers datasets accelerate peft trl bitsandbytes wandb huggingface_hub sentencepiece einops\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 2. auth + config (embedded inline so this notebook is self-contained)\n",
                "import os, json, torch\n",
                "from google.colab import userdata\n",
                "\n",
                "os.environ['HF_TOKEN']        = userdata.get('HF_TOKEN')\n",
                "os.environ['WANDB_API_KEY']   = userdata.get('WANDB_API_KEY')\n",
                "\n",
                f"cfg = {cfg_literal}\n",
                "\n",
                "print('GPU :', torch.cuda.get_device_name(0))\n",
                "print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')\n",
                "print('base:', cfg['model']['base_id'])\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 3. tokenizer + 4-bit base model\n",
                "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
                "\n",
                "tok = AutoTokenizer.from_pretrained(cfg['model']['base_id'])\n",
                "if tok.pad_token is None:\n",
                "    tok.pad_token = tok.eos_token\n",
                "\n",
                "bnb = BitsAndBytesConfig(\n",
                "    load_in_4bit=cfg['quantization']['load_in_4bit'],\n",
                "    bnb_4bit_quant_type=cfg['quantization']['bnb_4bit_quant_type'],\n",
                "    bnb_4bit_use_double_quant=cfg['quantization']['bnb_4bit_use_double_quant'],\n",
                "    bnb_4bit_compute_dtype=getattr(torch, cfg['quantization']['bnb_4bit_compute_dtype']),\n",
                ")\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    cfg['model']['base_id'], quantization_config=bnb, device_map='auto', torch_dtype=torch.bfloat16,\n",
                ")\n",
                "model.config.use_cache = False\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 4. LoRA\n",
                "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
                "model = prepare_model_for_kbit_training(model)\n",
                "lora_cfg = LoraConfig(\n",
                "    r=cfg['lora']['r'], lora_alpha=cfg['lora']['alpha'],\n",
                "    lora_dropout=cfg['lora']['dropout'], bias=cfg['lora']['bias'],\n",
                "    task_type=cfg['lora']['task_type'], target_modules=cfg['lora']['target_modules'],\n",
                ")\n",
                "model = get_peft_model(model, lora_cfg)\n",
                "model.print_trainable_parameters()\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 5. data — pulled from HF Hub (private dataset)\n",
                "from datasets import load_dataset\n",
                f"DATASET = '{DATASET_REPO}'\n",
                "train_ds = load_dataset(DATASET, data_files='train.jsonl', split='train', token=os.environ['HF_TOKEN'])\n",
                "val_ds   = load_dataset(DATASET, data_files='val.jsonl',   split='train', token=os.environ['HF_TOKEN'])\n",
                "print(f'train: {len(train_ds)}  val: {len(val_ds)}')\n",
                "print('sample:\\n', train_ds[0]['text'][:600])\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 6. SFT training\n",
                "from trl import SFTConfig, SFTTrainer\n",
                "import wandb\n",
                "\n",
                "wandb.init(project='marathi-instruct-llm', name='qwen2.5-1.5b-qlora-v1')\n",
                "\n",
                "tcfg = cfg['training']\n",
                "args = SFTConfig(\n",
                "    output_dir='/content/out',\n",
                "    num_train_epochs=tcfg['num_train_epochs'],\n",
                "    per_device_train_batch_size=tcfg['per_device_train_batch_size'],\n",
                "    gradient_accumulation_steps=tcfg['gradient_accumulation_steps'],\n",
                "    learning_rate=tcfg['learning_rate'],\n",
                "    lr_scheduler_type=tcfg['lr_scheduler_type'],\n",
                "    warmup_ratio=tcfg['warmup_ratio'],\n",
                "    weight_decay=tcfg['weight_decay'],\n",
                "    max_length=tcfg['max_seq_length'],\n",
                "    bf16=tcfg['bf16'],\n",
                "    gradient_checkpointing=tcfg['gradient_checkpointing'],\n",
                "    logging_steps=tcfg['logging_steps'],\n",
                "    eval_strategy='steps',\n",
                "    eval_steps=tcfg['eval_steps'],\n",
                "    save_steps=tcfg['save_steps'],\n",
                "    save_total_limit=tcfg['save_total_limit'],\n",
                "    report_to='wandb',\n",
                "    dataset_text_field=cfg['data']['text_field'],\n",
                ")\n",
                "trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok)\n",
                "trainer.train()\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 7. push adapter to HF Hub\n",
                "from huggingface_hub import login\n",
                "login(token=os.environ['HF_TOKEN'])\n",
                f"REPO = '{MODEL_REPO}'\n",
                "trainer.model.push_to_hub(REPO, private=False)\n",
                "tok.push_to_hub(REPO)\n",
                "print(f'pushed adapter to https://huggingface.co/{REPO}')\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 8. sanity check — Marathi generation\n",
                "model.eval()\n",
                "prompts = [\n",
                "    'महाराष्ट्राची राजधानी कोणती आहे आणि तिथली लोकसंख्या किती आहे?',\n",
                "    'महात्मा गांधींबद्दल तीन वाक्यांत माहिती द्या.',\n",
                "    'एक चांगला सॉफ्टवेअर इंजिनिअर होण्यासाठी कोणते गुण आवश्यक आहेत?',\n",
                "]\n",
                "for p in prompts:\n",
                "    msgs = [\n",
                "        {'role': 'system', 'content': 'तुम्ही एक उपयुक्त AI सहाय्यक आहात जो मराठीत स्पष्ट आणि अचूक उत्तरे देतो.'},\n",
                "        {'role': 'user',   'content': p},\n",
                "    ]\n",
                "    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)\n",
                "    inputs = tok(text, return_tensors='pt').to(model.device)\n",
                "    with torch.no_grad():\n",
                "        out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)\n",
                "    print('Q:', p)\n",
                "    print('A:', tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip())\n",
                "    print('---')\n",
            ],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    print(f"create dataset repo {DATASET_REPO} (private)")
    api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)

    print(f"upload train.jsonl ({(DATA_DIR / 'train.jsonl').stat().st_size/1e6:.1f} MB)")
    api.upload_file(
        path_or_fileobj=str(DATA_DIR / "train.jsonl"),
        path_in_repo="train.jsonl",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print("upload val.jsonl")
    api.upload_file(
        path_or_fileobj=str(DATA_DIR / "val.jsonl"),
        path_in_repo="val.jsonl",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print("upload test.jsonl")
    api.upload_file(
        path_or_fileobj=str(DATA_DIR / "test.jsonl"),
        path_in_repo="test.jsonl",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )

    print(f"\ncreate model repo {MODEL_REPO} (public — adapter target)")
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, exist_ok=True)

    nb = build_notebook(cfg)
    nb_path = ROOT / "notebooks" / "train_colab_hub.ipynb"
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nbuilt {nb_path.relative_to(ROOT)} ({nb_path.stat().st_size/1024:.1f} KB)")

    print(f"upload notebook to {MODEL_REPO}")
    api.upload_file(
        path_or_fileobj=str(nb_path),
        path_in_repo="train_colab.ipynb",
        repo_id=MODEL_REPO,
        repo_type="model",
    )

    raw_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/train_colab.ipynb"
    colab_url = f"https://colab.research.google.com/#fileId=https%3A//huggingface.co/{MODEL_REPO}/resolve/main/train_colab.ipynb"

    print("\n=== ALL UPLOADED ===")
    print(f"dataset: https://huggingface.co/datasets/{DATASET_REPO}")
    print(f"model:   https://huggingface.co/{MODEL_REPO}")
    print(f"raw notebook: {raw_url}")
    print(f"colab open:   {colab_url}")


if __name__ == "__main__":
    main()
