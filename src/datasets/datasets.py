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
        return CriteoDataset(
            train_test_info="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/criteo/criteo-common-split/train_test_val_info.bin",
            dataset_name=dataset_name,
            dataset_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/criteo/train.txt",
            cache_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/criteo/criteo-fm",
        )
    elif args.dataset == "avazu":
        return AvazuDataset(
            train_test_info="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/avazu/train_test_info.bin",
            dataset_name=dataset_name,
            dataset_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/avazu/train",
            cache_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/avazu/avazu-fm",
        )
    else:
        raise ValueError()
