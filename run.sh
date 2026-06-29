python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python train.py --epochs 3 --batch_size 32 --output_dir ./run_vanilla
