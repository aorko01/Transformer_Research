from dataclasses import dataclass

@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
@dataclass
class TrainingConfig:
    max_steps:int=50000
    lr:float=6e-4
    Batch:int=4
    Sequence_length:int=1024
    Total_batches:int=524288
    
