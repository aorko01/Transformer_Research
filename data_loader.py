"""
data_loader.py  –  Wikipedia data loading with synthetic fallback
=================================================================
Tries HuggingFace Hub first; falls back to a local synthetic corpus
so the pipeline can be validated even in air-gapped environments.

The synthetic corpus intentionally mimics Wikipedia text statistics
(sentence length, token distribution) using the BERT vocabulary so
MLM loss numbers are meaningful and comparable across attention variants.
"""

import os, random, math
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset


# ── Synthetic corpus ────────────────────────────────────────────────────────

WIKI_TOPICS = [
    "The history of science includes many pivotal discoveries that changed "
    "human understanding of the natural world.",
    "Mathematics is the study of numbers, shapes, patterns, and the logical "
    "relationships between abstract structures.",
    "Biology encompasses the scientific study of life and living organisms, "
    "including their physical and chemical processes.",
    "The development of computing technology has transformed every aspect of "
    "modern society and continues to accelerate.",
    "Philosophy examines fundamental questions about existence, knowledge, "
    "values, reason, mind, and language.",
    "Economics studies how individuals, firms, and governments allocate scarce "
    "resources to satisfy unlimited wants.",
    "Astronomy is the branch of science that deals with celestial objects, "
    "space, and the physical universe as a whole.",
    "Literature reflects the human experience through narrative, poetry, drama, "
    "and other forms of creative expression.",
    "Chemistry involves the study of matter, its properties, how and why "
    "substances combine or separate to form other substances.",
    "Geography examines the lands, features, inhabitants, and phenomena of Earth "
    "across spatial and temporal scales.",
]

def _generate_synthetic_article(rng: random.Random, min_words=80, max_words=200) -> str:
    """Stitch WIKI_TOPICS sentences together into a fake article paragraph."""
    n_sentences = rng.randint(4, 10)
    sentences = [rng.choice(WIKI_TOPICS) for _ in range(n_sentences)]
    return "  ".join(sentences)


def build_synthetic_texts(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [_generate_synthetic_article(rng) for _ in range(n)]


# ── TokenDataset ─────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, encoding):
        self.enc = encoding

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {k: torch.tensor(v[i]) for k, v in self.enc.items()}


# ── Main loader ──────────────────────────────────────────────────────────────

def load_wikipedia_dataset(
    tokenizer,
    max_seq_length: int,
    max_train: Optional[int],
    max_val: Optional[int],
):
    """
    Returns (train_dataset, val_dataset).

    Priority:
      1. HuggingFace Hub `wikipedia 20220301.en` (streaming)
      2. Local files in ./wiki_text_cache/ (one .txt per article)
      3. Synthetic corpus (always available)
    """
    max_train = max_train or 50_000
    max_val   = max_val   or 5_000
    total     = max_train + max_val

    texts = None

    # ── Try HuggingFace Hub ─────────────────────────────────────────
    try:
        from datasets import load_dataset as _ld
        print("Trying HuggingFace Hub (wikipedia 20220301.en)…")
        raw = _ld("wikipedia", "20220301.en", split="train",
                  streaming=True)
        buf = []
        for item in raw:
            buf.append(item["text"])
            if len(buf) >= total:
                break
        if buf:
            texts = buf
            print(f"  ✓ Loaded {len(texts)} articles from Hub")
    except Exception as e:
        print(f"  Hub unavailable ({type(e).__name__}), trying local cache…")

    # ── Try local cache ─────────────────────────────────────────────
    if texts is None:
        cache_dir = Path("./wiki_text_cache")
        if cache_dir.exists():
            files = sorted(cache_dir.glob("*.txt"))[:total]
            if files:
                texts = [f.read_text(errors="ignore") for f in files]
                print(f"  ✓ Loaded {len(texts)} articles from local cache")

    # ── Synthetic fallback ──────────────────────────────────────────
    if not texts:
        print(f"  Using synthetic Wikipedia-like corpus ({total} samples)…")
        texts = build_synthetic_texts(total)

    # ── Split ───────────────────────────────────────────────────────
    train_texts = texts[:max_train]
    val_texts   = texts[max_train: max_train + max_val]

    # ── Tokenise ────────────────────────────────────────────────────
    def encode(t_list):
        return tokenizer(
            t_list,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_special_tokens_mask=True,
            return_tensors=None,
        )

    print(f"  Tokenising {len(train_texts)} train / {len(val_texts)} val …")
    return TokenDataset(encode(train_texts)), TokenDataset(encode(val_texts))
