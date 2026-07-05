"""
data_loader.py
==============
Downloads the full English Wikipedia dataset to disk once (cached across
runs), then reads articles from that local, memory-mapped copy while
tokenizing on the fly — disk reads are far cheaper than streaming the
dataset over the network on every step.
"""
from typing import Optional
import torch
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset


class DiskTokenDataset(IterableDataset):
    """Tokenizes articles on the fly as they're read from the local, disk-backed dataset.

    Long articles are split into consecutive max_seq_length chunks instead of
    truncating everything past the first chunk. The final leftover chunk of an
    article is dropped if it has fewer than max_seq_length/2 tokens, otherwise
    it's padded up to max_seq_length.

    Tokenization is CPU-bound, so this is meant to run inside DataLoader worker
    processes (num_workers > 0): each worker shards the underlying disk-backed
    dataset so articles are tokenized in parallel with the GPU running the
    previous batch, instead of blocking the main process between steps.
    """
    def __init__(self, hf_source, tokenizer, max_seq_length):
        self.hf_source = hf_source
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __iter__(self):
        half_len = self.max_seq_length / 2
        pad_id = self.tokenizer.pad_token_id

        worker_info = get_worker_info()
        source = self.hf_source
        if worker_info is not None:
            # give each worker process a disjoint slice, else every worker
            # would iterate the full dataset and duplicate all the data
            source = source.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        for example in source:
            text = example.get("text")
            if not text:
                continue

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                return_overflowing_tokens=True,   # chunk the article instead of truncating it
                return_special_tokens_mask=True,
                padding=False,
                return_tensors=None,
            )
            enc.pop("overflow_to_sample_mapping", None)

            n_chunks = len(enc["input_ids"])
            for i in range(n_chunks):
                chunk = {k: v[i] for k, v in enc.items()}
                seq_len = len(chunk["input_ids"])

                if i == n_chunks - 1 and seq_len < self.max_seq_length:
                    if seq_len < half_len:
                        continue  # leftover too small to be useful: drop it
                    pad_len = self.max_seq_length - seq_len
                    chunk["input_ids"]           = chunk["input_ids"] + [pad_id] * pad_len
                    chunk["attention_mask"]      = chunk["attention_mask"] + [0] * pad_len
                    chunk["special_tokens_mask"] = chunk["special_tokens_mask"] + [1] * pad_len
                    if "token_type_ids" in chunk:
                        chunk["token_type_ids"] = chunk["token_type_ids"] + [0] * pad_len

                yield {k: torch.tensor(v) for k, v in chunk.items()}


def load_wikipedia_dataset(
    tokenizer,
    max_seq_length: int,
    max_train: Optional[int],
    max_val: Optional[int],
    data_dir: Optional[str] = None,
):
    """
    Downloads the full English Wikipedia dataset to `data_dir` (skipped on
    later runs if already cached there), then wraps the local, memory-mapped
    copy in disk-backed datasets that tokenize per-batch during training.

    Returns:
        train_dataset, val_dataset   (both IterableDatasets backed by local disk)
    """
    max_val = max_val or 10_000
    print(f"Downloading (or reusing cached) English Wikipedia to disk at "
          f"{data_dir or '~/.cache/huggingface/datasets'} ...")
    dataset = load_dataset(
        "wikimedia/wikipedia",   # official Parquet mirror — no loading script
        "20231101.en",           # latest stable English dump
        split="train",
        cache_dir=data_dir,      # full download to local disk (memory-mapped Arrow cache)
        trust_remote_code=True,
    )
    print(f"Full dataset on disk: {len(dataset)} articles. Reading from local disk copy...")

    val_hf = dataset.select(range(max_val))
    if max_train is not None:
        train_hf = dataset.select(range(max_val, min(max_val + max_train, len(dataset))))
    else:
        train_hf = dataset.select(range(max_val, len(dataset)))
    train_hf = train_hf.shuffle(seed=42)   # shuffles an index array only; content stays memory-mapped on disk

    train_ds = DiskTokenDataset(train_hf, tokenizer, max_seq_length)
    val_ds   = DiskTokenDataset(val_hf, tokenizer, max_seq_length)

    print("Dataset ready (tokenization happens per-batch while reading from disk).")
    return train_ds, val_ds
