import time

import torch
from submodlib.functions.facilityLocation import FacilityLocationFunction

generator = torch.Generator(device="cuda")
generator.manual_seed(42)
num_per_class = 200
N = 10000
eps = 1e-4
X = torch.rand(size=(N, 100), generator=generator, device="cuda")


use_cuda_kernel = False


# X = X.cpu().numpy()
def get_euclid_sims(X):
    dists = torch.cdist(X, X)
    gamma = 1.0 / X.shape[-1]
    sims = torch.exp(-gamma * dists)
    return sims


start = time.time()

n_runs = 5
for _ in range(n_runs):

    if use_cuda_kernel:
        sims = get_euclid_sims(X)
        sims = sims.cpu().numpy()
        obj = FacilityLocationFunction(
            n=len(X),
            mode="dense",
            data=X,
            metric="euclidean",
            num_neighbors=None,
            sijs=sims,
            separate_rep=False,
        )
    else:
        obj = FacilityLocationFunction(
            n=len(X),
            mode="dense",
            data=X,
            metric="euclidean",
            num_neighbors=None,
        )

    greedyList = obj.maximize(
        budget=num_per_class,
        # optimizer="LazyGreedy",
        optimizer="StochasticGreedy",
        epsilon=eps,
        # optimizer="NaiveGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
        # verbose=True,
    )

S_time = time.time() - start
print(S_time / n_runs)
