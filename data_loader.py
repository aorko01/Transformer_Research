"""
data_loader.py
==============
Loads the English Wikipedia dataset for BERT pretraining using the
Hugging Face datasets library.

The datasets library automatically caches downloaded datasets, so no
manual cache handling is required.
"""

from typing import Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class TokenDataset(Dataset):
    def __init__(self, encoding):
        self.enc = encoding

    def __len__(self):
        return len(self.enc["input_ids"])

    #returns the input id , attention mask and special attention mask for every input id/sentence
    def __getitem__(self, idx):
        sample = {}

        for k, v in self.enc.items():
            sample[k] = torch.tensor(v[idx])

        return sample


def load_wikipedia_dataset(
    tokenizer,
    max_seq_length: int,
    max_train: Optional[int],
    max_val: Optional[int],
):
    """
    Returns:
        train_dataset, val_dataset
    """

    max_train = max_train or 50_000
    max_val = max_val or 5_000
    total = max_train + max_val

    print("Loading English Wikipedia...")

    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
    )

    if len(dataset) < total:
        raise RuntimeError(
            f"Dataset contains only {len(dataset)} articles; "
            f"requested {total}."
        )

    texts = dataset["text"][:total]

    train_texts = texts[:max_train]
    val_texts = texts[max_train:]

    def encode(texts):
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_special_tokens_mask=True,
            return_tensors=None,
        )

    print(f"Tokenizing {len(train_texts)} training articles...")
    train_encoding = encode(train_texts)

    print(f"Tokenizing {len(val_texts)} validation articles...")
    val_encoding = encode(val_texts)

    print("Done.")

    return (
        TokenDataset(train_encoding),
        TokenDataset(val_encoding),
    )