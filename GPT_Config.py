from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


@dataclass
class TrainingConfig:
    max_steps: int = 50000
    lr: float = 6e-4
    min_lr: float = 6e-5          # cosine decay floor
    warmup_steps: int = 500       # linear warmup steps before cosine decay kicks in
    Batch: int = 4
    Sequence_length: int = 1024
    Total_batches: int = 524288   # target tokens per optimizer step (drives grad accumulation)
    eval_interval: int = 250      # run validation every N steps
    eval_iters: int = 100         # number of random batches averaged per validation pass
    data_dir: str = "data/tinystories"
    out_dir: str = "out"
    seed: int = 1337