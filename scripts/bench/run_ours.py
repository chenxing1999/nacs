import os
import sys
import time

import torch

CUR_FOLDER = os.path.dirname(__file__)
sys.path.append(CUR_FOLDER)

from ours.naive_greed_batch import lazier_torch, naive_greed_batch


# This function here is to confirm the algo is correct
# by compare the result with submodlib
def get_euclid_sims(X):
    dists = torch.cdist(X, X)
    gamma = 1.0 / X.shape[-1]
    sims = torch.exp(-gamma * dists)
    return sims


generator = torch.Generator(device="cuda")
generator.manual_seed(42)
num_per_class = 100
N = 10000
X = torch.rand(size=(N, 100), generator=generator, device="cuda")

eps = 1e-4
run_lazy = False

n_batch = 1
X = X.unsqueeze(0).repeat((n_batch, 1, 1))

# X = X.cpu().numpy()

start = time.time()

n_runs = 5
for _ in range(n_runs):
    # sims = 1 - torch.cdist(X, X)
    sims = get_euclid_sims(X)
    # res, gain = naive_greed_optimize(sims, num_per_class, len(X))

    if run_lazy:
        res, gain = lazier_torch(X, num_per_class, None, sims, eps)
    else:
        res, gain = naive_greed_batch(X, num_per_class, None, sims=sims)


S_time = time.time() - start
print(S_time / n_runs / n_batch)
