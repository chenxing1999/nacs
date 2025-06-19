"""
This file is actually not used in the final version, but I keep it here
as I think it is useful for future works on submodular optimization.
"""

import math
import time
from typing import List, Set, Tuple

import numpy as np
import torch

# Keep old code for checking algorithm correctness if necessary
try:
    from submodlib.functions.facilityLocation import FacilityLocationFunction

    from utils.submodular import faciliy_location_order as faciliy_location_order_cpp
except ImportError:
    pass


device = "cuda"


@torch.jit.script
def naive_greed_optimize(
    sims: torch.Tensor,
    num_per_class: int,
    N: int,
    # device="cuda",
) -> Tuple[List[int], List[float]]:

    selected_set: List[int] = list()
    gains: List[float] = list()

    device = torch.device("cuda")
    cur_values = torch.ones((1, N), device=device) * float("-inf")
    cur_val_sum: float = 0.0
    for _ in range(num_per_class):

        # Compute gain: Shape N
        # gain[i] = Score increase with adding i to selected_set
        gain = torch.maximum(sims, cur_values).sum(dim=1) - cur_val_sum
        gain[selected_set] = float("-inf")
        idx = int(torch.argmax(gain).item())

        selected_set.append(idx)
        gains.append(gain[idx].item())

        cur_values = torch.maximum(sims[idx], cur_values)
        cur_val_sum = float(cur_values.sum().item())

    return selected_set, gains


def lazier_torch(
    sims: torch.Tensor,
    num_per_class: int,
    N: int,
) -> Tuple[List[int], List[float]]:
    """
    Lazier Than Lazy Greedy paper algorithm
    """
    results: Set[int] = set()
    gains = []
    eps = 1e-3
    s = int(N * math.log(1 / eps) / num_per_class)
    device = torch.device("cuda")

    cur_values = torch.ones((1, N), device=device) * float("-inf")
    cur_val_sum: float = 0.0

    for i in range(num_per_class):

        # Sample a subset_R from V \ results ~ O(s)
        subset_r = []
        for item in torch.randperm(N).tolist():
            if item not in results:
                subset_r.append(item)
            if len(subset_r) == s:
                break

        subset_r = torch.tensor(subset_r, device=device)

        # Calculate Gain(alpha | results)
        gain = torch.maximum(sims[subset_r], cur_values).sum(dim=1) - cur_val_sum
        idx = int(torch.argmax(gain).item())
        true_idx = int(subset_r[idx].item())

        results.add(true_idx)
        gains.append(gain[idx].item())

        cur_values = torch.maximum(sims[true_idx], cur_values)
        cur_val_sum = float(cur_values.sum().item())
    return list(results), gains


def faciliy_location_order(
    c: int,
    X: torch.Tensor,
    y: torch.Tensor,
    metric,
    num_per_class,
    weights=None,
    mode="sparse",
    num_n=128,
    optimizer_function=None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Args:
        X: Shape N x Num Classes
            X: ground set V_p

    Returns:

    Note:
        N is Random subset size

    """
    if optimizer_function is None:
        optimizer_function = lazier_torch

    class_indices = torch.where(y == c)[0]
    if metric != "sims":
        X = X[class_indices]
    else:
        X = X[class_indices]
        X = X[:, class_indices]

    N = X.shape[0]

    # TODO: fix monkeypatch below
    if hasattr(optimizer_function, "set_start_set"):
        optimizer_function.set_start_set(class_indices)

    if mode == "dense":
        pass

    start = time.time()

    # Get Similarity matrix
    # dists = (X.unsqueeze(0) - X.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
    if metric == "euclidean":
        dists = torch.cdist(X, X)
        gamma = 1.0 / X.shape[1]
        sims = torch.exp(-gamma * dists)
    elif metric == "identity":
        dists = torch.cdist(X, X)
        const = dists.max()
        sims = const - dists
    elif metric == "sims":
        sims = X
    else:
        raise ValueError()

    S_time = time.time() - start

    # ---Begin of Greed (line 24 - 30 in original algorithm)
    start = time.time()
    # selected_set = []
    # gains = []
    # cur_values = torch.ones((1, N), device=device) * float("-inf")
    # cur_val_sum = 0
    # for _ in range(num_per_class):
    #
    #     # Compute gain: Shape N
    #     # gain[i] = Score increase with adding i to selected_set
    #     gain = torch.zeros(N, device=device)
    #     gain = torch.maximum(sims, cur_values).sum(dim=1) - cur_val_sum
    #     gain[selected_set] = float("-inf")
    #     idx = torch.argmax(gain).item()
    #     # print(f"Choose {idx} with gain: {gain[idx]}")
    #     selected_set.append(idx)
    #     gains.append(gain[idx].item())
    #     cur_values = torch.maximum(sims[idx], cur_values)
    #     cur_val_sum = cur_values.sum()
    #     #print(cur_val_sum)

    # selected_set, gains = naive_greed_optimize(sims, num_per_class, N)
    selected_set, gains = optimizer_function(sims, num_per_class, N)

    greed_time = time.time() - start
    # Lines 35 to 46
    sz = torch.zeros(len(selected_set), device=device, dtype=torch.float32)
    max_values = torch.max(sims[:, selected_set], dim=1)
    if weights is None:
        weights = torch.ones(N, device=device)

    mask = max_values.values > 0
    indices = max_values.indices[mask]
    # arange = torch.arange(N, device=device)
    sz.index_add_(0, indices, weights[mask])
    sz[torch.where(sz == 0)] = 1

    return (
        class_indices[selected_set].cpu().numpy(),
        sz.cpu().numpy(),
        greed_time,
        S_time,
    )


def get_orders_and_weights(
    B,
    X: torch.Tensor,
    metric,
    y=None,
    weights=None,
    equal_num=False,
    outdir=".",
    mode="sparse",
    num_n=128,
    optimizer_function=None,
    verbose=False,
):
    """
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist
    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    """
    N = X.shape[0]
    if y is None:
        raise NotImplementedError()
    classes, class_nums = torch.unique(y, return_counts=True)
    classes = classes.cpu().int().tolist()
    class_nums = class_nums.cpu()

    # classes = classes.astype(np.int32).tolist()
    C = len(classes)  # number of classes

    num_per_class: torch.Tensor
    if equal_num:
        # class_nums = [sum(y == c) for c in classes]
        target_num = int(math.ceil(B / C))
        num_per_class = target_num * torch.ones(len(classes), dtype=torch.int32)
        minority = class_nums < target_num
        n_minority = sum(minority)
        if n_minority > 0:
            extra = sum([max(0, target_num - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(math.ceil(extra / n_minority))
    else:
        # num_per_class = torch.from_numpy(
        #     np.ceil(np.divide([sum(y == i) for i in classes], N) * B)
        # ).int()
        # _, count = torch.unique(y, True, False, True)
        # count = count.int()
        num_per_class = (class_nums * B // N).cpu()

    if verbose:
        print(f"Greedy: selecting {num_per_class} elements")

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(
        *map(
            lambda c: faciliy_location_order(
                c[1],
                X,
                y,
                metric,
                num_per_class[c[0]],
                weights,
                mode,
                num_n,
                optimizer_function,
            ),
            enumerate(classes),
        )
    )
    if verbose:
        print(
            "time (sec) for computing facility location:"
            f" {greedy_times} similarity time {similarity_times}",
        )

    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        y_np = y.cpu().numpy()
        class_ratios = np.divide([np.sum(y_np == i) for i in classes], N)
        props = np.rint(class_ratios / np.min(class_ratios))  # TODO

    order_mg_all = np.array(order_mg_all, dtype=object)
    cluster_sizes_all = np.array(cluster_sizes_all, dtype=object)

    # len(order_mg_all[c]) = Number of sample in subset belong to class c
    tmp = np.max([len(order_mg_all[c]) / props[c] for c, _ in enumerate(classes)])
    for i in range(int(np.rint(tmp))):
        for c, _ in enumerate(classes):
            ndx = slice(
                i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c]))
            )
            order_mg = np.append(order_mg, order_mg_all[c][ndx])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])

    order_mg = np.array(order_mg, dtype=np.int32)

    weights_mg = np.array(
        weights_mg, dtype=np.float32
    )  # / sum(weights_mg) TODO: removed division!
    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []  # order_mg_all[rows_selector, cluster_order].flatten(order='F')
    weights_sz = (
        []
    )  # cluster_sizes_all[rows_selector, cluster_order].flatten(order='F')
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals


def faciliy_location_order_new(
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]

    if mode == "dense":
        num_n = None

    start = time.time()
    obj = FacilityLocationFunction(
        n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n
    )
    S_time = time.time() - start

    start = time.time()
    greedyList = obj.maximize(
        budget=num_per_class,
        optimizer="LazyGreedy",
        # optimizer="NaiveGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
        # verbose=True,
    )
    order = list(map(lambda x: x[0], greedyList))
    sz = list(map(lambda x: x[1], greedyList))
    greedy_time = time.time() - start

    S = obj.sijs
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(num_per_class, dtype=np.float64)

    # for i in range(N):
    #     if np.max(S[i, order]) <= 0:
    #         continue
    #     if weights is None:
    #         sz[np.argmax(S[i, order])] += 1
    #     else:
    #         sz[np.argmax(S[i, order])] += weights[i]
    # sz[np.where(sz == 0)] = 1
    indices = np.argmax(S[:, order], 1)

    max_values = np.take_along_axis(S[:, order], indices, 1).todense()
    indices = np.asarray(indices).squeeze()
    max_values = np.asarray(max_values).squeeze()

    mask = np.where(max_values > 0)
    if weights is None:
        np.add.at(sz, indices[mask], 1)
    else:
        # weights = np.ones(N)
        # raise NotImplementedError()
        np.add.at(sz, indices[mask], weights[mask])

    sz[np.where(sz == 0)] = 1

    return class_indices[order], sz, greedy_time, S_time


if __name__ == "__main__":
    num_classes = 2
    N = 1024
    X = torch.rand((N, num_classes), device=device)
    y = torch.zeros(size=(N,), device=device, dtype=torch.int)
    weights = torch.rand(N, device=device)
    metric = "euclidean"

    start = time.time()
    result = faciliy_location_order(
        0,
        X,
        y,
        metric,
        256,
        weights,
    )
    print(result)
    print("Torch time:", time.time() - start)

    print()
    print("-" * 10)
    print("cpp")
    # obj = FacilityLocationFunction(
    #     n=len(X),
    #     mode="dense",
    #     data=X,
    #     metric=metric,
    #     #num_neighbors=N,
    # )
    # greedyList = obj.maximize(
    #     budget=3,
    #     optimizer="NaiveGreedy",
    #     stopIfZeroGain=False,
    #     stopIfNegativeGain=False,
    #     verbose=True,
    # )
    # print(greedyList)
    X = X.cpu().numpy()
    y = y.cpu().numpy()
    weights = weights.cpu().numpy()
    start = time.time()
    result = faciliy_location_order_cpp(0, X, y, metric, 256, weights, "sparse")
    print(result)
    print("CPP time:", time.time() - start)

    print()
    start = time.time()
    result = faciliy_location_order_new(0, X, y, metric, 256, weights, "sparse")
    print(result)
    print("New time:", time.time() - start)
