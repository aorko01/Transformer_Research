"""
data_loader.py
==============
Streams the English Wikipedia dataset for BERT pretraining using the
Hugging Face datasets library — no full corpus is materialized in RAM.
"""
from typing import Optional
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset


class StreamingTokenDataset(IterableDataset):     # NEW: replaces eager TokenDataset
    """Tokenizes articles on the fly as they're pulled from the HF stream."""
    def __init__(self, hf_stream, tokenizer, max_seq_length):
        self.hf_stream = hf_stream
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __iter__(self):
        for example in self.hf_stream:
            text = example.get("text")
            if not text:
                continue
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_special_tokens_mask=True,
                return_tensors=None,
            )
            yield {k: torch.tensor(v) for k, v in enc.items()}


def load_wikipedia_dataset(
    tokenizer,
    max_seq_length: int,
    max_train: Optional[int],
    max_val: Optional[int],
):
    """
    Returns:
        train_dataset, val_dataset   (both streaming IterableDatasets)
    """
    max_val = max_val or 10_000
    print("Streaming English Wikipedia (no full download)...")
    dataset = load_dataset(
        "wikimedia/wikipedia",   # official Parquet mirror — no loading script
        "20231101.en",           # latest stable English dump
        split="train",
        streaming=True,           # NEW: stream instead of eager-loading the full split
        trust_remote_code=True,
    )

    val_stream   = dataset.take(max_val)            # NEW: first N articles → val
    train_stream = dataset.skip(max_val)            # NEW: remainder → train
    if max_train is not None:
        train_stream = train_stream.take(max_train)  # NEW: optional cap, else full corpus
    train_stream = train_stream.shuffle(buffer_size=10_000, seed=42)  # NEW: streaming shuffle buffer

    train_ds = StreamingTokenDataset(train_stream, tokenizer, max_seq_length)
    val_ds   = StreamingTokenDataset(val_stream, tokenizer, max_seq_length)

    print("Streaming dataset ready (tokenization happens per-batch during training).")
    return train_ds, val_ds