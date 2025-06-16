from torch.utils.data import Dataset

from src.datasets.datasets import get_dataset


class IndexedDataset(Dataset):
    def __init__(self, args, train=True, train_transform=False, split=None, start=0):
        super().__init__()
        self.start = start

        assert isinstance(train, bool)
        if split is None:
            if train:
                split = "train"
            else:
                split = "val"
        self.dataset = get_dataset(args, split, train_transform=train_transform)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index + self.start

    def __getitems__(self, indices):
        results = self.dataset.__getitems__(indices)
        return [(data, target, idx) for (data, target), idx in zip(results, indices)]

    def __len__(self):
        return len(self.dataset)

    def clean(self):
        # self._cachers = []
        pass

    def cache(self):
        pass
