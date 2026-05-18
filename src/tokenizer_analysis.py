"""
Tokenizer efficiency analysis: how badly does Qwen2.5's English-leaning BPE
tokenizer fragment Marathi?

Outputs:
  eval/tokenizer_analysis/stats.json    — raw numbers
  eval/tokenizer_analysis/efficiency.png — bar-chart comparison

Run: python -m src.tokenizer_analysis
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSONL = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
OUT_DIR = PROJECT_ROOT / "eval" / "tokenizer_analysis"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SAMPLE_SIZE = 1000
SEED = 42

# A fixed reference set of 100 short English instruction-style sentences.
# Hand-picked to roughly mirror the genre of the Marathi instruction data
# (short questions / explanations / how-tos). Length range ~30-200 chars.
ENGLISH_REFERENCE = [
    "Write a short paragraph about the importance of clean drinking water.",
    "Explain the difference between machine learning and deep learning.",
    "Summarise the main causes of inflation in two sentences.",
    "Give three reasons why exercising daily is good for health.",
    "What is the capital of France and how many people live there?",
    "Describe the role of mitochondria inside a cell.",
    "List four common features of a relational database.",
    "Translate the following sentence into formal English: I gotta go now.",
    "Suggest a healthy breakfast for someone who works out in the morning.",
    "Explain in plain English what a neural network does.",
    "Give a recipe for tomato soup that takes under thirty minutes.",
    "What are three good practices for writing clean code?",
    "Outline a study plan for someone preparing for a maths exam.",
    "Describe the plot of Romeo and Juliet in three short sentences.",
    "How does photosynthesis convert sunlight into chemical energy?",
    "Write a polite email asking your manager for a day off.",
    "What is the difference between weather and climate?",
    "List five renewable sources of energy.",
    "Suggest a name for a small bookstore in a quiet neighbourhood.",
    "Explain how compound interest grows money over time.",
    "What are the three branches of government in a democracy?",
    "Describe how to brew a cup of green tea.",
    "Give a one-sentence summary of the theory of relativity.",
    "What does an operating system actually do?",
    "Write a short bedtime story about a friendly robot.",
    "List the seven continents in order of size.",
    "How do you politely decline a job offer over email?",
    "Explain what a stack data structure is using a simple analogy.",
    "Give three tips for taking better mobile phone photographs.",
    "Why is regular sleep important for cognitive performance?",
    "Describe the difference between an array and a linked list.",
    "What is the role of the World Health Organisation?",
    "Suggest three simple exercises a beginner can do at home.",
    "Explain why the sky appears blue during the day.",
    "List four key skills required to become a software engineer.",
    "Summarise the main events of World War II in two sentences.",
    "How does a vaccine train the immune system?",
    "Describe the function of a router in a home network.",
    "What does the acronym HTTP stand for and what does it do?",
    "Write a thank-you note to a teacher who helped you a lot.",
    "Explain the basic idea behind blockchain in plain English.",
    "Give three reasons why someone might learn a second language.",
    "What is the difference between a virus and a bacterium?",
    "Outline a one-week plan to start a daily reading habit.",
    "Describe what cloud computing is and why companies use it.",
    "List the steps involved in writing a research paper.",
    "Explain how a search engine ranks web pages.",
    "What are three common symptoms of dehydration?",
    "Give a short biography of Marie Curie in three sentences.",
    "How does a refrigerator keep food cold?",
    "Describe the importance of biodiversity in an ecosystem.",
    "What is recursion in programming?",
    "Explain the rules of a simple board game like tic-tac-toe.",
    "Write a friendly birthday message for a colleague.",
    "What is the boiling point of water at sea level?",
    "List the planets of the solar system in order from the sun.",
    "How do you change a flat tyre on a car?",
    "Describe the colour wheel and primary colours.",
    "Explain what an algorithm is using everyday vocabulary.",
    "What are three good habits for personal finance?",
    "Summarise the plot of The Great Gatsby in two sentences.",
    "How does a microwave oven heat food?",
    "Describe the structure of an atom in simple terms.",
    "List five common signs of a phishing email.",
    "Write a short paragraph about your favourite season.",
    "What is the difference between HTML, CSS and JavaScript?",
    "Explain how plants reproduce through pollination.",
    "Describe how a credit card works in two short sentences.",
    "List three benefits of using version control like git.",
    "How do you politely correct someone in a meeting?",
    "What is the role of the heart in the circulatory system?",
    "Explain the concept of supply and demand briefly.",
    "Suggest three indoor activities for a rainy day.",
    "Describe the water cycle in three steps.",
    "What is a binary search and when would you use it?",
    "Give a one-sentence summary of Hamlet.",
    "How do solar panels convert sunlight into electricity?",
    "Describe what an API does using a restaurant analogy.",
    "What are three common cybersecurity threats?",
    "Explain how Wi-Fi differs from cellular data.",
    "List four characteristics of a good leader.",
    "Describe how vaccines were developed for COVID-19 in plain English.",
    "What is the difference between fiction and non-fiction?",
    "Give a brief introduction to the topic of climate change.",
    "Explain how a search index speeds up database lookups.",
    "List three reasons why drinking water is important.",
    "Describe what a smart contract is on a blockchain.",
    "What is the role of chlorophyll in plants?",
    "Write a short product description for a wireless mouse.",
    "How does a digital camera capture an image?",
    "List four common ways to reduce household waste.",
    "Describe the parts of an essay introduction.",
    "Explain what dark matter is in two sentences.",
    "What is the function of red blood cells in the body?",
    "Give three ways to politely end a long phone call.",
    "Describe how a hashmap stores and looks up values.",
    "What is the difference between renewable and non-renewable energy?",
    "Explain the basic structure of DNA in everyday language.",
    "List five Indian states by population.",
    "Describe the lifecycle of a butterfly in four stages.",
    "Write a polite reminder email for a missed deadline.",
    "What is the difference between weather forecasting and astrology?",
]


@dataclass
class Stats:
    name: str
    n_samples: int
    total_chars: int
    total_words: int
    total_tokens: int
    chars_per_token: float
    tokens_per_char: float
    tokens_per_word: float
    fragmentation_rate: float  # fraction of words that get split into >1 token
    sample_fragmentation: list  # 5 example (word, token_count) showing worst cases


WORD_RE = re.compile(r"[ऀ-ॿ]+|[A-Za-z0-9]+")


def words_of(text: str) -> list[str]:
    return WORD_RE.findall(text)


def analyse(name: str, texts: list[str], tokenizer) -> Stats:
    total_chars = 0
    total_words = 0
    total_tokens = 0
    fragmented_words = 0
    word_examples = []
    for t in texts:
        total_chars += len(t)
        ws = words_of(t)
        total_words += len(ws)
        total_tokens += len(tokenizer.encode(t, add_special_tokens=False))
        for w in ws:
            n = len(tokenizer.encode(w, add_special_tokens=False))
            if n > 1:
                fragmented_words += 1
                if len(word_examples) < 200:
                    word_examples.append((w, n))
    word_examples.sort(key=lambda x: -x[1])
    return Stats(
        name=name,
        n_samples=len(texts),
        total_chars=total_chars,
        total_words=total_words,
        total_tokens=total_tokens,
        chars_per_token=total_chars / max(total_tokens, 1),
        tokens_per_char=total_tokens / max(total_chars, 1),
        tokens_per_word=total_tokens / max(total_words, 1),
        fragmentation_rate=fragmented_words / max(total_words, 1),
        sample_fragmentation=[(w, n) for w, n in word_examples[:5]],
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    if not TRAIN_JSONL.exists():
        raise SystemExit(f"Marathi train file not found: {TRAIN_JSONL}")

    print(f"Loading Marathi samples from {TRAIN_JSONL.relative_to(PROJECT_ROOT)}")
    marathi_texts: list[str] = []
    with TRAIN_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # use instruction+response (excluding ChatML tags) for a fair char count
            marathi_texts.append(f"{row['instruction']} {row['response']}")

    rng = random.Random(SEED)
    rng.shuffle(marathi_texts)
    marathi_texts = marathi_texts[:SAMPLE_SIZE]

    print(f"Marathi samples: {len(marathi_texts)}")
    print(f"English samples: {len(ENGLISH_REFERENCE)}")

    marathi = analyse("marathi", marathi_texts, tok)
    english = analyse("english", ENGLISH_REFERENCE, tok)

    ratio = {
        "tokens_per_char": marathi.tokens_per_char / english.tokens_per_char,
        "tokens_per_word": marathi.tokens_per_word / english.tokens_per_word,
        "chars_per_token": marathi.chars_per_token / english.chars_per_token,
        "fragmentation_rate": (
            marathi.fragmentation_rate / english.fragmentation_rate
            if english.fragmentation_rate > 0 else float("inf")
        ),
    }

    summary = {
        "tokenizer": MODEL_ID,
        "marathi": asdict(marathi),
        "english": asdict(english),
        "marathi_to_english_ratio": ratio,
        "headline": (
            f"Qwen2.5's tokenizer uses {ratio['tokens_per_char']:.2f}x more tokens "
            f"per character on Marathi than English "
            f"({marathi.chars_per_token:.2f} chars/token vs "
            f"{english.chars_per_token:.2f}). "
            f"A 1024-token context fits ~{int(marathi.tokens_per_word ** -1 * 1024)} Marathi words "
            f"vs ~{int(english.tokens_per_word ** -1 * 1024)} English words."
        ),
    }
    stats_path = OUT_DIR / "stats.json"
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {stats_path.relative_to(PROJECT_ROOT)}")
    print("\n" + summary["headline"])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["tokens/char", "tokens/word", "fragmentation %"]
        marathi_vals = [
            marathi.tokens_per_char,
            marathi.tokens_per_word,
            marathi.fragmentation_rate * 100,
        ]
        english_vals = [
            english.tokens_per_char,
            english.tokens_per_word,
            english.fragmentation_rate * 100,
        ]
        x = range(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([i - width / 2 for i in x], english_vals, width, label="English")
        ax.bar([i + width / 2 for i in x], marathi_vals, width, label="Marathi")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("ratio / percent")
        ax.set_title(f"Qwen2.5 tokenizer: Marathi vs English\n(English/Marathi ratio per metric)")
        for i, (e, m) in enumerate(zip(english_vals, marathi_vals)):
            ax.text(i - width / 2, e, f"{e:.2f}", ha="center", va="bottom", fontsize=9)
            ax.text(i + width / 2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
        ax.legend()
        fig.tight_layout()
        chart_path = OUT_DIR / "efficiency.png"
        fig.savefig(chart_path, dpi=130)
        print(f"Wrote {chart_path.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"[warn] could not render chart: {e}")


if __name__ == "__main__":
    main()
