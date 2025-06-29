import shutil
import struct
from collections import defaultdict
from pathlib import Path
from typing import List, Literal

import lmdb
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm


def get_offsets(field_dims: List[int]) -> torch.Tensor:
    field_dims_tensor = torch.tensor(field_dims)
    field_dims_tensor = torch.cat(
        [torch.tensor([0], dtype=torch.long), field_dims_tensor]
    )
    offsets = torch.cumsum(field_dims_tensor[:-1], 0).unsqueeze(0)

    return offsets


class AvazuDataset(Dataset):
    """
    Avazu Click-Through Rate Prediction Dataset

    Dataset preparation
        Remove the infrequent features (appearing in less than threshold instances)
        and treat them as a single feature

    :param train_test_info
    :param dataset_name

    :param dataset_path: avazu train path
    :param cache_path: lmdb cache path
    :param rebuild_cache: If True, lmdb cache is refreshed
    :param min_threshold: infrequent feature threshold

    Reference
        https://www.kaggle.com/c/avazu-ctr-prediction

    Note: Copied from pytorch-fm and modified.
    """

    def __init__(
        self,
        train_test_info: str,
        dataset_name: Literal["train", "val", "test"],
        dataset_path=None,
        cache_path=".avazu",
        rebuild_cache=False,
        min_threshold=2,
    ):
        self.NUM_FEATS = 22
        self.min_threshold = min_threshold

        train_test_info = torch.load(train_test_info)
        self._preprocess_timestamp = train_test_info.get("metadata", {}).get(
            "preprocess_timestamp", False
        )
        logger.debug(f"preprocess timestamp: {self._preprocess_timestamp}")
        assert not self._preprocess_timestamp, "Not supported"

        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError("create cache: failed: dataset_path is None")
            self.__build_cache(
                dataset_path,
                cache_path,
                train_test_info,
            )
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)

        # Hook my dataset
        self._line_in_dataset = list(train_test_info[dataset_name])
        self._line_in_dataset.sort()

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"] - 1
            self.field_dims = np.frombuffer(
                txn.get(b"field_dims"), dtype=np.uint32
            ).astype(np.int64)

        self._offsets = get_offsets(self.field_dims.tolist())[0].numpy()

    def __getitem__(self, index):
        index = self._line_in_dataset[index]
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack(">I", index)), dtype=np.uint32
            ).astype(dtype=np.int64)
        return np_array[1:], np_array[0]

    def __getitems__(self, indices: List[int]):
        indices = [self._line_in_dataset[idx] for idx in indices]
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            keys = [struct.pack(">I", index) for index in indices]
            res = cursor.getmulti(keys)

            # Use numpy read instead of torch to support old version torch
            array = np.stack([np.frombuffer(arr, dtype=np.uint32) for key, arr in res])
            array = array.astype(np.int64)

        return [(arr[1:], arr[0]) for arr in array]

    def __len__(self):
        return len(self._line_in_dataset)

    def __build_cache(self, path, cache_path, train_test_info=None):
        if train_test_info is None:
            feat_mapper, defaults = self.__get_feat_mapper(path)
        else:
            feat_mapper = train_test_info["feat_mapper"]
            defaults = train_test_info["defaults"]

        num_feats = self.NUM_FEATS
        if self._preprocess_timestamp:
            num_feats = num_feats + 3

        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(num_feats, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b"field_dims", field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description("Create avazu dataset cache: counting features")
            for line in pbar:
                values = line.rstrip("\n").split(",")
                if len(values) != self.NUM_FEATS + 2:
                    continue
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i + 1]] += 1
        feat_mapper = {
            i: {feat for feat, c in cnt.items() if c >= self.min_threshold}
            for i, cnt in feat_cnts.items()
        }
        feat_mapper = {
            i: {feat: idx for idx, feat in enumerate(cnt)}
            for i, cnt in feat_mapper.items()
        }
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description("Create avazu dataset cache: setup lmdb")
            for line in pbar:
                values = line.rstrip("\n").split(",")
                if len(values) != self.NUM_FEATS + 2:
                    continue

                extra_feats = []

                n_extra = len(extra_feats)
                np_array = np.zeros(self.NUM_FEATS + 1 + n_extra, dtype=np.uint32)
                np_array[0] = int(values[1])
                for i in range(1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i + 1], defaults[i])

                for i, feat in enumerate(extra_feats):
                    idx = self.NUM_FEATS + 1 + i
                    np_array[idx] = feat_mapper[idx].get(feat, defaults[idx])

                buffer.append((struct.pack(">I", item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer

    def describe(self):
        logger.info("Avazu TorchFM dataset")
        logger.info(f"sum field dims: {sum(self.field_dims)}")
        logger.info(f"length field dims: {len(self.field_dims)}")
        return

    def pop_info(self):
        return {}
