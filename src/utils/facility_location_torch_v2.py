import torch
import random
import math
import time

device = "cuda"


def faciliy_location_order(X: torch.Tensor, num_per_class, mask=None):
    """
    Batched submodular optimization algorithm

    Args:
        X: model's gradient (n_batch, batch_size, hidden_dim)
        num_per_class: Number of sample choosed for subset

    Returns:
        best subset (n_batch, num_per_class)

    """

    start = time.time()
    # n_batches, batch_size, batch_size
    dists = torch.cdist(X, X)
    n_batch, batch_size, _ = X.shape
    # const = dists.max(dim=1, keepdims=True)
    const = 1
    sims = const - dists

    eps = 1e-3
    s = int(batch_size * math.log(1 / eps) / num_per_class)


    candidate_weight = torch.ones((n_batch, batch_size), device=device)
    if mask is not None:
        candidate_weight[mask] = 0
    n_cand = batch_size

    gain_results = torch.zeros((n_batch, num_per_class), device=device)

    arange_idx = torch.arange(n_batch, device=device)
    cur_val_sum = torch.zeros((n_batch, 1), device=device)
    cur_values = torch.ones(
        (n_batch, 1, batch_size), device=device
    ) * float("-inf")

    results = torch.zeros((n_batch, num_per_class), device=device, dtype=torch.long)

    for i in range(num_per_class):
        # Sample a subset_R from V / results
        subsets = torch.multinomial(
            candidate_weight, num_samples=s, replacement=False
        )

        subset_sims = sims[arange_idx.unsqueeze(1), subsets]
        gain = torch.maximum(subset_sims, cur_values).sum(dim=2) - cur_val_sum

        gain_values, indices = torch.max(gain, dim=1)
        true_idx = subsets[arange_idx, indices]

        candidate_weight[arange_idx, true_idx] = 0
        gain_results[:, i] = gain_values
        results[:, i] = true_idx

        cur_values = torch.maximum(
            sims[arange_idx, true_idx].unsqueeze(1),
            cur_values,
        )

        cur_val_sum = cur_values.sum(dim=-1)


    # print(time.time() - start)
    return results
