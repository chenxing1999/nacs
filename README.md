# Quick start

1. Installation:

```bash
# Create virtual env with your favorite environment manager
# Here I use venv
python -m venv env

# Activate environment
source env/bin/activate

# Install required packages
pip install -e '.[dev]'
```

2. Run step 1 -- Coreset selection

```bash
python scripts/train_choose_selflc_v5.py --arch dcnv2 --dataset criteo --batch_size 8192 --data_size 0.01 --n_split 3
```

3. Run step 2 -- Denoise

```bash
python scripts/denoise.py --arch dcnv2 --dataset criteo --data_path outputs/dcnv2-criteo-0.01-v2-ablation
```

4. Run retrain

```bash
python scripts/train_subset.py --arch dcnv2 \
     --dataset criteo \
     --subset_path outputs/dcnv2-criteo-0.01-v2-ablation/hyperparam-test.pth \
     --loss selflc \
     --batch_size 8192
```

Note: The weight decay is different between data size.
Go to file `src/const.py` for weight decay search results.

# Acks
