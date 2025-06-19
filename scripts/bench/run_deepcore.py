import os
import sys

import numpy as np

CUR_FOLDER = os.path.dirname(__file__)
sys.path.append(CUR_FOLDER)
import time
from argparse import Namespace

import torch
from deepcore.submodular_function import FacilityLocation
from deepcore.submodular_optimizer import NaiveGreedy, StochasticGreedy


def get_euclid_sims(X):
    dists = np.expand_dims(X, 0) - np.expand_dims(X, 1)
    dists = np.linalg.norm(dists, 2, -1)
    gamma = 1.0 / X.shape[1]
    sims = np.exp(-gamma * dists)
    return sims


args = Namespace(print_freq=1000000)


optimizer_name = "lazier"


generator = torch.Generator(device="cuda")
generator.manual_seed(42)
num_per_class = 100
N = 50000
X = torch.rand(size=(N, 100), generator=generator, device="cuda")

X = X.cpu().numpy()

start = time.time()
n_runs = 5
for _ in range(n_runs):
    sims = get_euclid_sims(X)
    index = np.arange(N)
    submod_function = FacilityLocation(
        index=index,
        similarity_matrix=sims,
    )
    if optimizer_name == "naive":
        optimizer = NaiveGreedy(
            args,
            index,
            num_per_class,
            [],
        )
    else:
        optimizer = StochasticGreedy(
            args,
            index,
            num_per_class,
            [],
            1e-3,
        )
    class_result = optimizer.select(
        gain_function=submod_function.calc_gain,
        update_state=submod_function.update_state,
    )

print((time.time() - start) / n_runs)
