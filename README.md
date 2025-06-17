# Quick start

## Setup

1. Create a virtual environment

```shell
# Create virtual env with your favorite environment manager
# Here I use venv
python -m venv env

# Activate environment
source env/bin/activate

# Install required packages
pip install -e '.[dev]'
```

2. Download data and run preprocess:

```shell
# Download Criteo dataset
wget https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz -O dataset/criteo/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
tar xvf dataset/criteo/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz -C dataset/criteo/

# Get data splits
zstd --decompress bin/splits.zst.zst -o bin/splits.zst
zstd --decompress bin/splits.zst -o dataset/criteo/splits.bin

# Run preprocess
python scripts/preprocess.py criteo
```

## Run main algorithm

1. Run step 1 -- Coreset selection

```shell
python scripts/train_choose_selflc_v5.py --arch dcnv2 --dataset criteo --batch_size 8192 --data_size 0.01 --n_split 3
```

2. Run step 2 -- Denoise

```shell
python scripts/denoise.py --arch dcnv2 --dataset criteo --data_path outputs/dcnv2-criteo-0.01-v2-ablation
```

3. Run retrain

```shell
python scripts/train_subset.py --arch dcnv2 \
     --dataset criteo \
     --subset_path outputs/dcnv2-criteo-0.01-v2-ablation/hyperparam-test.pth \
     --loss selflc \
     --batch_size 8192
```

Note: The weight decay is different between data size.
Go to file `src/const.py` for weight decay search results.

# Acknowledgement

This code is based on:

- https://github.com/YuYang0901/CREST
- https://github.com/chenxing1999/recsys-benchmark
