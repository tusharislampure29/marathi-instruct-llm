"""
Marathi instruction-data preparation pipeline.

Pulls Marathi slices from public instruction datasets, filters by script and
length, dedupes with MinHash, formats as ChatML, and writes train/val/test
JSONL files to data/processed/.

Marathi shares the Devanagari script with Hindi and Sanskrit, so script-only
detection cannot distinguish languages. We trust the source dataset's language
metadata (mr / mar_Deva splits) and use Devanagari-presence as a safety net
that rejects English residue from failed translations.

Run: python -m src.data_prep
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

# Windows consoles default to cp1252 and crash on Devanagari / arrow chars.
# Reconfigure stdout to UTF-8 so Marathi text and symbols print cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Devanagari Unicode block: U+0900–U+097F (covers Marathi, Hindi, Sanskrit, Nepali).
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
# Other Indic scripts we want to reject if they dominate a response (means
# the source labelling was wrong).
OTHER_INDIC_RE = re.compile(r"[ঀ-৿਀-૿଀-௿ఀ-౿ಀ-೿ഀ-ൿ]")

MIN_CHARS = 10
MAX_CHARS = 2000
MIN_DEVANAGARI_RATIO = 0.50  # at least half the letters must be Devanagari

DEDUPE_JACCARD = 0.85
MINHASH_PERM = 128

VAL_FRAC = 0.05
TEST_FRAC = 0.05
SEED = 42


# Marathi-containing instruction datasets.
# Aya marathi (~1M+ rows) is plenty on its own for a 1.5B QLoRA fine-tune.
# Bactrian-X (mr) was tried and removed because it ships an old-style loading
# script that the current `datasets` library refuses to execute for security.
DATASET_SOURCES = [
    {
        "name": "CohereForAI/aya_collection_language_split",
        "config": "marathi",
        "split": "train",
        "instruction_field": "inputs",
        "response_field": "targets",
    },
]


@dataclass
class Sample:
    instruction: str
    response: str
    source: str

    def chatml_text(self) -> str:
        # Qwen2.5 ChatML format. System prompt in Marathi to bias generation.
        return (
            "<|im_start|>system\n"
            "तुम्ही एक उपयुक्त AI सहाय्यक आहात जो मराठीत स्पष्ट आणि अचूक उत्तरे देतो.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{self.instruction.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n{self.response.strip()}<|im_end|>"
        )


def is_mostly_devanagari(text: str) -> bool:
    if not text:
        return False
    deva_chars = len(DEVANAGARI_RE.findall(text))
    other_indic = len(OTHER_INDIC_RE.findall(text))
    if other_indic > deva_chars:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters == 0:
        return False
    return (deva_chars / letters) >= MIN_DEVANAGARI_RATIO


def is_valid(sample: Sample) -> bool:
    full = sample.instruction + " " + sample.response
    if not (MIN_CHARS <= len(full) <= MAX_CHARS * 2):
        return False
    if not is_mostly_devanagari(sample.response):
        return False
    return True


def load_source(spec: dict) -> Iterator[Sample]:
    """Yield Sample objects from one source. Logs and yields nothing on failure."""
    try:
        if spec.get("config"):
            ds = load_dataset(spec["name"], spec["config"], split=spec["split"], cache_dir=str(RAW_DIR))
        else:
            ds = load_dataset(spec["name"], split=spec["split"], cache_dir=str(RAW_DIR))
    except Exception as e:
        print(f"[WARN] Could not load {spec['name']} ({spec.get('config')}): {e}")
        return

    instr_f = spec["instruction_field"]
    resp_f = spec["response_field"]
    extra_f = spec.get("extra_input_field")
    lang_f = spec.get("lang_field")
    lang_v = spec.get("lang_value")

    for row in ds:
        if lang_f and row.get(lang_f) != lang_v:
            continue
        instr = row.get(instr_f)
        resp = row.get(resp_f)
        if not isinstance(instr, str) or not isinstance(resp, str):
            continue
        if extra_f:
            extra = row.get(extra_f)
            if isinstance(extra, str) and extra.strip():
                instr = f"{instr.strip()}\n\n{extra.strip()}"
        yield Sample(instruction=instr, response=resp, source=spec["name"])


def shingles(text: str, k: int = 5) -> set[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def dedupe(samples: list[Sample]) -> list[Sample]:
    """MinHash LSH dedupe. Returns kept samples (first occurrence wins)."""
    lsh = MinHashLSH(threshold=DEDUPE_JACCARD, num_perm=MINHASH_PERM)
    kept: list[Sample] = []
    for i, s in enumerate(tqdm(samples, desc="dedupe")):
        m = MinHash(num_perm=MINHASH_PERM)
        for sh in shingles(s.instruction + " " + s.response):
            m.update(sh.encode("utf-8"))
        key = f"s{i}"
        if lsh.query(m):
            continue
        lsh.insert(key, m)
        kept.append(s)
    return kept


def split_and_write(samples: list[Sample]) -> dict[str, int]:
    rng = random.Random(SEED)
    rng.shuffle(samples)
    n = len(samples)
    n_test = int(n * TEST_FRAC)
    n_val = int(n * VAL_FRAC)
    test = samples[:n_test]
    val = samples[n_test : n_test + n_val]
    train = samples[n_test + n_val :]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = PROCESSED_DIR / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for s in split_data:
                f.write(json.dumps({**asdict(s), "text": s.chatml_text()}, ensure_ascii=False) + "\n")
        counts[split_name] = len(split_data)
        print(f"  wrote {len(split_data):>6} samples -> {path.relative_to(PROJECT_ROOT)}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-source", type=int, default=None,
                        help="Cap rows scanned per source (for fast iteration).")
    parser.add_argument("--target-size", type=int, default=30000,
                        help="Final dataset size after dedupe. ~30k is the right "
                             "size for QLoRA on Qwen2.5-1.5B / one Colab T4 run. "
                             "Pass 0 to keep everything.")
    parser.add_argument("--predupe-pool", type=int, default=60000,
                        help="Random pool size before dedupe. We oversample 2x the target "
                             "so dedupe can drop near-duplicates and we still have enough. "
                             "Aya marathi has ~3.5M rows; deduping all of them takes hours, "
                             "so we sample first then dedupe.")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_samples: list[Sample] = []
    per_source_counts: dict[str, int] = {}
    for spec in DATASET_SOURCES:
        source_kept = 0
        for i, s in enumerate(load_source(spec)):
            if args.limit_per_source and i >= args.limit_per_source:
                break
            if is_valid(s):
                all_samples.append(s)
                source_kept += 1
        per_source_counts[spec["name"]] = source_kept
        print(f"[{spec['name']}] kept {source_kept} after language+length filter")

    total_after_filter = len(all_samples)
    print(f"\nTotal after filter: {total_after_filter}")

    # Subsample BEFORE dedupe — deduping millions of rows with MinHash takes hours
    # on 3.5M Aya rows. Sampling to ~2x target keeps dedupe minutes-fast while
    # still letting MinHash drop near-duplicates from the pool.
    rng = random.Random(SEED)
    if args.predupe_pool and len(all_samples) > args.predupe_pool:
        rng.shuffle(all_samples)
        all_samples = all_samples[: args.predupe_pool]
        print(f"Pre-dedupe random pool: {len(all_samples)}")

    deduped = dedupe(all_samples)
    print(f"After dedupe       : {len(deduped)}")

    if args.target_size and len(deduped) > args.target_size:
        rng.shuffle(deduped)
        deduped = deduped[: args.target_size]
        print(f"Final target size  : {len(deduped)}")

    counts = split_and_write(deduped)

    summary = {
        "per_source_after_filter": per_source_counts,
        "total_after_filter": total_after_filter,
        "predupe_pool_size": len(all_samples),
        "after_dedupe": len(deduped),
        "final_size": sum(counts.values()),
        "splits": counts,
    }
    (PROCESSED_DIR / "stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nStats -> {(PROCESSED_DIR / 'stats.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
