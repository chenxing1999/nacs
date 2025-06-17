CRITEO_DATASET_CONFIG = dict(
    train_test_info="dataset/criteo/splits.bin",
    dataset_path="dataset/criteo/train.txt",
    cache_path="dataset/criteo/criteo-fm",
)


AVAZU_DATASET_CONFIG = dict(
    train_test_info="dataset/avazu/splits.bin",
    dataset_path="dataset/avazu/train",
    cache_path="dataset/avazu/avazu-fm",
)


"""
Weight Decay note
       |0.5  |1   |5
Criteo |5e-4 |5e-4|1e-5
Avazu  |5e-4 |5e-4|1e-5
"""
