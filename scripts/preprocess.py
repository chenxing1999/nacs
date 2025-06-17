import argparse

from src.datasets.avazu import AvazuDataset
from src.datasets.criteo_fm import CriteoDataset

parser = argparse.ArgumentParser()
parser.add_argument("dataset")


args = parser.parse_args()

if args.dataset == "criteo":
    dataset = CriteoDataset(
        train_test_info="dataset/criteo/splits.bin",
        dataset_name="train",
        dataset_path="dataset/criteo/train.txt",
        cache_path="dataset/criteo/criteo-fm",
    )
else:
    dataset = AvazuDataset(
        train_test_info="dataset/avazu/splits.bin",
        dataset_name="train",
        dataset_path="dataset/ctr/avazu/train",
        cache_path="dataset/ctr/avazu/avazu-fm",
    )
