import math
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.datasets import IndexedDataset
from src.models import get_model
from src.utils import facility_location_torch_v2, get_args
from src.utils.selflc import ProSelfLC

start_time = time.time()


# Use CUDA if available and set random seed for reproducibility
# args = get_args(argv=["--dataset", "criteo", "--debug"])
# args = get_args(argv=[
#     "--dataset", "avazu",
#     "--batch_size", "8192",
#     "--arch", "dcnv2",
# ])
args = get_args()
args.seed = 42
data_size = 0.05

run_name = f"{args.arch}-{args.dataset}-{data_size}-v2-ablation"
print("Run name", run_name, "initialized")
writer = SummaryWriter(f"logs/{run_name}")
if torch.cuda.is_available():
    device = args.device = "cuda"

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
else:
    args.device = "cpu"
    torch.manual_seed(args.seed)


split = "train"
if args.dataset in ["criteo_cl", "avazu_cl"]:
    split = "old"

train_dataset = IndexedDataset(args, train=True, train_transform=True, split=split)
train_loader = DataLoader(
    train_dataset,
    args.batch_size,
    num_workers=4,
    shuffle=True,
)

targets = torch.zeros(len(train_dataset), dtype=torch.long)
for x, y, idx in train_loader:
    targets[idx] = y

neg_location = torch.where(targets == 0)[0]
pos_location = torch.where(targets == 1)[0]

val_loader = torch.utils.data.DataLoader(
    IndexedDataset(args, train=False),
    batch_size=8192,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

model = get_model(train_dataset, args.arch)
model = model.to(device)

if args.arch == "deepfm" and args.dataset == "avazu" and data_size == 0.05:
    optimizer = torch.optim.Adam(model.parameters(), 0.01, weight_decay=1e-5)
elif args.arch == "deepfm":
    optimizer = torch.optim.Adam(model.parameters(), 0.01, weight_decay=5e-5)
elif args.dataset == "criteo_cl":
    # optimizer = torch.optim.Adam(model.parameters(), 0.005, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), 0.005, weight_decay=5e-5)
    optimizer = torch.optim.Adam(model.parameters(), 0.01, weight_decay=1e-5)
else:
    # DCNv2 bigger model --> Easier to overfit
    optimizer = torch.optim.Adam(model.parameters(), 0.01, weight_decay=1e-5)

# optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-6)
# optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-7)
# optimizer = torch.optim.SGD(model.parameters(), 0.01, weight_decay=1e-5)


criterion = torch.nn.BCEWithLogitsLoss()


step = 0
sum_loss = 0
output_dir = "outputs/data_logs/"
os.makedirs(output_dir, exist_ok=True)


def generate_random_set(
    random_subset_size,
    equal_num,
    neg_location,
    pos_location,
) -> torch.Tensor:
    """Generate a random set with defined ratio of positive and negative element"""

    if equal_num:
        pos_ratio, neg_ratio = 1, 1
    else:
        neg_length = len(neg_location)
        pos_length = len(pos_location)

        total = pos_length + neg_length
        pos_ratio = 2 * pos_length / total
        neg_ratio = 2 * neg_length / total

    neg_ratio, pos_ratio = neg_ratio / 2, pos_ratio / 2

    # Split the train dataset into `batches` random parts
    indices = []
    class_indices = neg_location
    size = int(random_subset_size * neg_ratio)
    # new_indices = torch.randperm(len(class_indices))
    # new_indices = torch.multinomial(neg_weight, size)
    # new_indices = class_indices[new_indices[:size]]
    new_indices = random.choices(class_indices, k=size)
    indices.append(torch.tensor(new_indices))

    class_indices = pos_location
    size = int(random_subset_size * pos_ratio)
    # new_indices = torch.randperm(len(class_indices))
    # new_indices = torch.multinomial(pos_weight, size)
    # new_indices = class_indices[new_indices[:size]]
    new_indices = random.choices(class_indices, k=size)
    indices.append(torch.tensor(new_indices))
    # indices.append(new_indices)

    indices = torch.concat(indices)
    return indices


class CustomLazier:

    def __init__(
        self,
        random_set_indices: torch.Tensor,
        cur_subset: torch.Tensor,
    ):
        # Check if any element from random belong to cur_subset
        # mask = random_set_indices.unsqueeze(0) == cur_subset.unsqueeze(1)
        # location = torch.where(mask)

        self.random_set_indices = random_set_indices
        self.cur_subset = cur_subset

        # add all item from location[1] to self.results

        self.start = 0

    def set_start_set(self, class_indices):
        mask = torch.isin(
            self.random_set_indices[class_indices], self.cur_subset, assume_unique=True
        )
        location = torch.nonzero(mask, as_tuple=True)[0]
        self.start_results = set(location.cpu().tolist())

    def __call__(self, sims, num_per_class, N) -> Tuple[List[int], List[float]]:

        # results: Set[int] = self.results.copy()
        results = set()
        start_results = self.start_results

        gains = []
        eps = 1e-3
        s = int(N * math.log(1 / eps) / num_per_class)
        device = torch.device("cuda")

        if len(start_results) > 0:
            cur_set = torch.tensor(list(start_results), dtype=torch.long)
            cur_values: torch.Tensor = torch.max(sims[cur_set], dim=0).values
            cur_val_sum: float = cur_values.sum().item()
        else:
            cur_values = torch.ones(N, device=device) * float("-inf")
            cur_val_sum = 0

        for i in range(num_per_class):

            # Sample a subset_R from V \ results ~ O(s)
            subset_r = []
            for item in torch.randperm(N).tolist():
                if item not in results and item not in start_results:
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


def find_top_grad(
    train_dataset,
    subset_size,
    model,
    batches=100,
    random_batch_size=16384,
    equal_num=True,
    final_subset: Optional[torch.Tensor] = None,
):
    model.eval()
    # preds = torch.zeros(
    #     (len(train_dataset), 2),
    #     device=device,
    # )

    assert equal_num, "Speed-up version not support equal_num=False"
    assert random_batch_size * batches % 2 == 0

    n_batch = 8  # number of batch compute together
    (batches - 1) // n_batch + 1

    start_gen_time = time.time()
    total_data = random_batch_size * batches // 2

    # Sampling positive so that in each batch there is no repeated
    # batch_size = (random_batch_size // 2) * n_batch
    all_positives = []
    n1 = len(pos_location) // (random_batch_size // 2) // n_batch
    assert n1 > 0
    length = (random_batch_size // 2) * n_batch * n1
    n2 = (total_data - 1) // length + 1
    for _ in range(n2):
        pos = torch.randperm(len(pos_location))[:length]
        all_positives.append(pos_location[pos])
    all_positives = torch.concat(all_positives)[:total_data]
    all_positives = all_positives.reshape(batches, -1)

    neg_loc_tmp = torch.randperm(len(neg_location))[:total_data]
    all_negatives = neg_location[neg_loc_tmp].reshape(batches, -1)

    print("total", time.time() - start_gen_time)

    # (batches, random_batch_size)
    full_batch_idx = torch.concat([all_positives, all_negatives], dim=-1).flatten()

    # import pdb; pdb.set_trace()
    # sanity check
    subset_dataset = Subset(train_dataset, full_batch_idx)

    loader = DataLoader(
        subset_dataset,
        random_batch_size * n_batch,
        pin_memory=True,
        num_workers=4,
    )

    num_step = total_step + 1
    for data, target, index in tqdm.tqdm(loader, ascii=True):
        data = data.to(device)
        target = target.to(device)
        index = index.to(device)

        with torch.no_grad():
            logit = model(data)

        y_pred = torch.sigmoid(logit)
        eps = 1e-6
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        tmp_pred = y_pred

        tmp_pred.requires_grad_(True)
        loss = criterion_selflc(tmp_pred, target, total_step + 1)
        # loss = tce_loss(tmp_pred, targets[tmp_index], drop_rate)
        gradient = torch.autograd.grad(loss, tmp_pred)[0]
        # gradient = tmp_pred - identity_matrix[targets[tmp_index]]

        mom = (momentum * 0.9 + gradient * 0.1) / (1 - 0.9**num_step)
        var = (grad_var * 0.999 + gradient * gradient * 0.001) / (1 - 0.999**num_step)
        gradient = mom / (1e-6 + var.sqrt())

        # start choose coreset
        pos_gradient = gradient.view(-1, random_batch_size // 2)[::2]
        pos_index = index.view(-1, random_batch_size // 2)[::2]
        n_batch = pos_gradient.shape[0]

        pos_gradient = pos_gradient.unsqueeze(-1)

        # Get list of item already selected
        mask = torch.isin(pos_index, final_subset)
        pos_subset = facility_location_torch_v2.faciliy_location_order(
            pos_gradient, subset_size // 2 // batches, mask
        )

        pos_subset = pos_index[
            torch.arange(n_batch).unsqueeze(-1), pos_subset
        ].flatten()

        # Repeat for negative
        neg_gradient = gradient.view(-1, random_batch_size // 2)[1::2]
        neg_index = index.view(-1, random_batch_size // 2)[1::2]
        n_batch = neg_gradient.shape[0]

        neg_gradient = neg_gradient.unsqueeze(-1)

        mask = torch.isin(neg_index, final_subset)
        neg_subset = facility_location_torch_v2.faciliy_location_order(
            neg_gradient, subset_size // 2 // batches, mask
        )
        neg_subset = neg_index[
            torch.arange(n_batch).unsqueeze(-1), neg_subset
        ].flatten()

        final_subset = torch.concat([final_subset, pos_subset, neg_subset])

    final_weight = None
    # return torch.unique(torch.concat(final_subset)).cpu(), final_weight
    # Note: Final Weight is currently implement wrong. Please dont use
    return final_subset, final_weight


# k = int(data_size * len(train_dataset) / 0.9)
# k = int(data_size * len(train_dataset) / 0.9)
k = int(data_size * len(train_dataset) / 0.9)
if args.debug:
    batches = 33
    batch_size = int(len(train_dataset) / batches)
else:
    if args.dataset == "criteo_cl":
        # batches = 99 * int(data_size / 0.01 * 0.7 / 0.8)
        # batches = 80
        batches = 99 * int(data_size / 0.02)
    if data_size in [0.01, 0.05, 0.1]:
        batches = 99 * int(data_size / 0.01)
        # batches = 40 * int(data_size / 0.01)
    else:
        batches = 48
    # batches = 1000
    batch_size = 16384 * 2
    # batch_size = 16384 * 2
    # batch_size = 16384 * 3

n_splits = 3
if data_size == 0.05:
    n_splits = 5

n_splits = args.n_splits
print(f"k={k}")


def train_epoch_simple(train_loader, model, optimizer):
    device = "cuda"
    model.train()
    sum_loss = 0
    step = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    for data, target, _ in train_loader:
        data = data.to(device)
        target = target.to(device)

        y_pred = model(data)
        loss = criterion(y_pred, target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        step += 1
    return step, sum_loss


def train_epoch_simple2(
    train_loader,
    model,
    optimizer: torch.optim.Adam,
    momentum,
    grad_var,
    criterion: ProSelfLC,
):
    device = "cuda"
    model.train()
    sum_loss = 0
    step = 0
    # criterion = torch.nn.BCEWithLogitsLoss()
    for data, target, _ in train_loader:
        data = data.to(device)
        target = target.to(device)

        y_pred = torch.sigmoid(model(data))
        eps = 1e-6
        y_pred = torch.clamp(y_pred, eps, 1 - eps)

        loss = criterion(
            y_pred,
            target.float(),
            total_step + step,
        )

        # import pdb; pdb.set_trace()

        with torch.no_grad():
            gradients = torch.autograd.grad(loss, y_pred, retain_graph=True)[0]
            gradients = gradients.sum()
            momentum = 0.9 * momentum + 0.1 * gradients
            grad_var = 0.999 * grad_var + 0.001 * (gradients * gradients)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        step += 1
    return step, sum_loss, momentum, grad_var


def train_epoch_weighted(train_loader, model, optimizer, weight):
    device = "cuda"
    model.train()
    sum_loss = 0
    step = 0
    # criterion = torch.nn.BCEWithLogitsLoss()
    for data, target, idx in train_loader:
        data = data.to(device)
        target = target.to(device)

        y_pred = model(data)
        loss = F.binary_cross_entropy_with_logits(
            y_pred,
            target.float(),
            weight[idx],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        step += 1
    return step, sum_loss


final_subset = torch.tensor([], device="cuda", dtype=torch.long)
start = 0
total_step = 0

equal_num = True
# equal_num = False

intervals = 1
if args.arch == "dcnv2" and args.dataset == "criteo_cl":
    intervals = 1

grad_var = torch.tensor(0, device=device)
momentum = torch.tensor(0, device=device)
criterion_selflc = ProSelfLC(
    (intervals * sum(range(1, n_splits + 1))) * len(train_loader),
    # 27 * len(train_loader),
    # int(27 * len(train_loader) * 0.01 // 3),
    16,
    0.5,
)

final_subset, weight = find_top_grad(
    train_dataset,
    int(k // n_splits),
    # k // 5,
    model,
    batches // n_splits,
    batch_size,
    equal_num,
    final_subset,
)


print("Dataset length:", len(train_dataset))
best_auc = 0

# TODO: Find subset
dataset = Subset(train_dataset, final_subset.cpu())
train_loader = DataLoader(dataset, 8192, True, num_workers=4)

count = 1


n_epoches = intervals * n_splits + 1

# for epoch in range(100):
for epoch in range(100):
    print(f"Epoch: {epoch:02d}")
    # Train model
    step, sum_loss, momentum, grad_var = train_epoch_simple2(
        train_loader,
        model,
        optimizer,
        momentum,
        grad_var,
        criterion_selflc,
    )
    # step, sum_loss = train_epoch_simple(train_loader, model, optimizer)
    total_step += step

    # Find extra set
    # if epoch in [2, 3, 6] and count < n_splits:
    if (epoch + 1) % intervals == 0 and count < n_splits:
        if count == n_splits - 1:
            k_size = k - len(final_subset)
        else:
            k_size = k // n_splits
        final_subset, weight = find_top_grad(
            train_dataset,
            k_size,
            # k * 2 // 5,
            model,
            batches // n_splits,
            batch_size,
            equal_num,
            final_subset,
        )
        dataset = Subset(train_dataset, final_subset.cpu())
        train_loader = DataLoader(dataset, 8192, True, num_workers=4)
        count += 1
        # torch.save(final_subset, "subset_crest.pth")

    # Validation
    model.eval()

    device = "cuda"
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    criterion = criterion.to(device)

    log_loss = 0.0
    all_y_true = []
    all_y_pred = []

    if epoch > n_epoches - 3:
        for idx, batch in enumerate(val_loader):
            inputs, labels, _ = batch
            all_y_true.extend(labels.tolist())

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            log_loss += criterion(outputs, labels.float()).item()

            outputs = torch.sigmoid(outputs)
            all_y_pred.extend(outputs.cpu().tolist())
        auc = roc_auc_score(all_y_true, all_y_pred)
        log_loss = log_loss / len(all_y_pred)
    else:
        auc = 0.5
        log_loss = 1.0

    improve = False
    if auc > best_auc:
        from pathlib import Path

        folder = f"outputs/{run_name}/"
        os.makedirs(folder, exist_ok=True)
        folder = Path(folder)
        torch.save(final_subset, folder / "subset.pth")
        torch.save(model.state_dict(), folder / "model.pth")

        improve = True

    best_auc = max(auc, best_auc)
    train_loss = sum_loss / step

    print(
        f"{epoch=}"
        f"-- {train_loss=:.4f}"
        f"-- {auc=:.4f} -- {log_loss=:.4f} -- {best_auc=:.4f}\n"
    )

    writer.add_scalar("train/loss", train_loss, total_step)
    writer.add_scalar("val/loss", log_loss, total_step)
    writer.add_scalar("val/auc", auc, total_step)
    writer.add_scalar("best_auc", auc, total_step)

    if not improve and epoch >= n_epoches:
        break


print(time.time() - start_time, "seconds")
