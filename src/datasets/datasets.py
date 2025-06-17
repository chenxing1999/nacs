from src.const import AVAZU_DATASET_CONFIG, CRITEO_DATASET_CONFIG

from .avazu import AvazuDataset
from .criteo_fm import CriteoDataset


def get_dataset(args, split="train", train_transform=True):
    assert split in ["train", "val", "test", "debug", "old"]
    if args.dataset in ["criteo", "avazu"]:
        return get_ctr_dataset(args, split, train_transform)
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")


def get_ctr_dataset(args, split=True, train_transform=True):

    dataset_name = split
    train = split == "train"
    if train:
        dataset_name = "train"
    if args.debug and train:
        dataset_name = "debug"

    if args.dataset == "criteo":
        return CriteoDataset(**CRITEO_DATASET_CONFIG, dataset_name=dataset_name)
    elif args.dataset == "avazu":
        return AvazuDataset(**AVAZU_DATASET_CONFIG, dataset_name=dataset_name)
    else:
        raise ValueError()
