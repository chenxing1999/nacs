import argparse
from pathlib import Path

import torch
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets import IndexedDataset
from src.models import get_model
from src.utils import set_seed


# Params
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--arch", default="dcnv2")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_path")
    return parser.parse_args()


# args = Namespace(dataset="criteo", arch="dcnv2", debug=False)
args = parse_args()
set_seed(42)
# steps = 32
steps = 10
folder = Path(args.data_path)
subset_path = folder / "subset.pth"
# subset_path = folder / "easy_crest_mc_sub_tmp.pth"
model_path = folder / "model.pth"
output_path = folder / "hyperparam-test.pth"


# Main
train_dataset = IndexedDataset(args, split="train")

subset = torch.load(subset_path)
state = torch.load(model_path)

model = get_model(train_dataset, args.arch)
model.load_state_dict(state)

device = "cuda"


# Monte Carlo Dropout?
def enable_dropout(p):
    if isinstance(p, torch.nn.Dropout):
        p.train()
    return p


model.eval()
model.to(device)
model.apply(enable_dropout)


subset_dataset = Subset(train_dataset, subset.cpu())
subset_loader = DataLoader(
    subset_dataset, batch_size=8192 * 4, num_workers=4, pin_memory=True
)
loss_sum = torch.zeros(len(train_dataset), device=device)
loss_sq = torch.zeros(len(train_dataset), device=device)

# Calculate Monte Carlo approximation
for _ in tqdm.tqdm(range(steps), ascii=True):
    for item, target, idx in subset_loader:
        item = item.to(device)
        target = target.to(device)

        with torch.no_grad():
            logits = model(item)
            loss = F.binary_cross_entropy_with_logits(
                logits, target.float(), reduction="none"
            )
        loss_sum[idx] += loss
        loss_sq[idx] += loss * loss

mean = loss_sum[subset] / steps
var = loss_sq[subset] / steps - mean * mean

c = 1.96 * (var / steps).sqrt()
upper_loss = mean + c
# torch.save(upper_loss, "/tmp/upperloss_avazu.pth")
# upper_loss = torch.load("/tmp/upperloss_avazu.pth")

sorted_upper = torch.sort(upper_loss)
v, i = sorted_upper
values = v.cpu().numpy()
idx = i.cpu().numpy()


crest_subset = torch.load(subset_path).to("cuda")


# Get non-noisy subset
print("Convert to non noisy")
easiness = torch.from_numpy(idx).to("cuda")
end_hard = int(len(idx) * 0.9)
noisy_indices = easiness[end_hard:]
not_noisy = easiness[:end_hard]
mask = torch.isin(crest_subset, noisy_indices, assume_unique=True)

percent = len(not_noisy) / len(train_dataset) * 100
print("Total data:", len(not_noisy), f"{percent:.2f}% of training data")


torch.save(crest_subset[~mask], output_path)
