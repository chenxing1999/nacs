import os
import torchvision
import torchvision.transforms as transforms
from .criteo_fm import CriteoDataset
from .avazu import AvazuDataset

def get_dataset(args, split="train", train_transform=True):
    train = split == "train"
    assert split in ["train", "val", "test", "debug", "old"]
    if args.dataset in ["criteo", "avazu"]:
        return get_ctr_dataset(args, split, train_transform)

    if args.dataset in ['cifar10', 'cifar100']:
        if args.dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif args.dataset == 'cifar100':
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        else:
            raise NotImplementedError

        if train and train_transform:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        dataset = torchvision.datasets.__dict__[args.dataset.upper()](
            root=args.data_dir, train=train, 
            transform=transform, download=True)
    elif args.dataset == 'tinyimagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        
        dirname = 'tiny-imagenet-200' 
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])      

        if train:
            data_dir = os.path.join(args.data_dir, f'{dirname}/train')
        else:
            data_dir = os.path.join(args.data_dir, f'{dirname}/val')

        dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)

    else:
        raise NotImplementedError(f'Unknown dataset: {args.dataset}')

    return dataset


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
