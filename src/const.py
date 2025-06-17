CRITEO_DATASET_CONFIG = dict(
    train_test_info="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/criteo/criteo-common-split/train_test_val_info.bin",
    dataset_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/criteo/train.txt",
    cache_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/criteo/criteo-fm",
)


AVAZU_DATASET_CONFIG = dict(
    train_test_info="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/avazu/train_test_info.bin",
    dataset_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/avazu/train",
    cache_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/avazu/avazu-fm",
)


"""
Weight Decay note
       |0.5  |1   |5
Criteo |5e-4 |5e-4|1e-5
Avazu  |5e-4 |5e-4|1e-5
"""
