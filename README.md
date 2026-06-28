# Transformer_Research

## Environment setup

From the repository root, create and activate a Python virtual environment, then install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run training

Run the default training script:

```bash
python train.py --epochs 3 --batch_size 32 --output_dir ./run_vanilla
```

To use a custom attention implementation:

```bash
python train.py \
  --custom_attention custom_attention_template.MyCustomAttention \
  --output_dir ./run_custom
```

If you want to resume from a checkpoint later, use:

```bash
python resume_training.py \
  --checkpoint ./run_vanilla/checkpoint_epoch_2.pt \
  --epochs 5 \
  --custom_attention custom_attention_template.MyCustomAttention
```
