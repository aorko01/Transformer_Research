# GPT

A minimal, from-scratch GPT-2 style transformer (nanoGPT-inspired) with a basic training loop.

## Structure

```
.
├── model/
│   ├── GPT.py             # Full model: embeddings, stacked blocks, weight tying, init
│   ├── attention_block.py # Transformer block (pre-norm attn + mlp, residual connections)
│   ├── attention.py       # Multi-head causal self-attention
│   ├── mlp.py             # Feed-forward (4x expansion, GELU)
│   └── layer_norm.py      # LayerNorm wrapper (supports bias=False)
├── GPT_Config.py           # ModelConfig (model size) and TrainingConfig (training hparams)
├── Dataloader.py           # Sequential batch loader over tokenized text (tiktoken, gpt2 encoding)
├── training_utils.py       # loss calculation + AdamW optimizer setup (decay/no-decay param groups)
├── train.py                # Training loop entrypoint
├── data/
│   └── shakespeare/
│       └── input.txt       # training corpus (add your own text file here)
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Put a plain text file at `data/shakespeare/input.txt`. It gets tokenized with the GPT-2 BPE (`tiktoken`).

## Train

```bash
python3 train.py
```

Model/training hyperparameters (layers, heads, embedding size, batch size, learning rate, steps, etc.) live in `GPT_Config.py`.

## Contributing notes

- Model code lives under `model/`, one component per file. `Attention_Block` composes `Attention` + `MLP` with the pre-norm + residual pattern; don't remove the residual adds (`x + ...`) — training was verified to blow up without them.
- Data loading, config, and training-loop utilities are kept flat at the project root, outside `model/`, since they're training-specific rather than part of the model definition.