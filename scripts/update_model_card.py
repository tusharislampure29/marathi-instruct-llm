"""Push docs/HF_MODEL_CARD.md to the public adapter repo as README.md.

Run after every meaningful change to docs/HF_MODEL_CARD.md (eval numbers,
methodology, limitations). The Hub keeps full git history; the previous card
is always recoverable.

Run:  HF_TOKEN must be set; then `py -3.12 scripts/update_model_card.py`
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
LOCAL_CARD = ROOT / "docs" / "HF_MODEL_CARD.md"
REPO_ID = "tusharislampure29/qwen2.5-1.5b-marathi-instruct"

COMMIT_MESSAGE = (
    "Update model card with final eval: GPT-4o pairwise 71% win-rate, "
    "4-axis rubric (reasoning instr-follow 1.1->1.7), auto-metrics on n=500 "
    "test split (ROUGE-L +40%, chrF +41%, sacreBLEU +276%)"
)


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var not set")

    api = HfApi(token=token)
    who = api.whoami()
    print(f"auth as: {who['name']}")
    print(f"local card: {LOCAL_CARD.relative_to(ROOT)} ({LOCAL_CARD.stat().st_size} bytes)")

    api.upload_file(
        path_or_fileobj=str(LOCAL_CARD),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=COMMIT_MESSAGE,
    )
    print(f"pushed to https://huggingface.co/{REPO_ID}")

    remote_path = hf_hub_download(repo_id=REPO_ID, filename="README.md", repo_type="model")
    head = Path(remote_path).read_text(encoding="utf-8")[:600]
    print("--- remote head ---")
    print(head)


if __name__ == "__main__":
    main()
