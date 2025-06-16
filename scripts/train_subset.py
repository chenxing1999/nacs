import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets import IndexedDataset
from src.models import get_model
from src.utils import get_args, set_seed
from src.utils.ctr_utils import evaluate_model
from src.utils.dataset_utils import (
    get_count,
    get_oov_tokens,
)
from src.utils.losses import TCE_Loss, rce_loss
from src.utils.selflc import ProSelfLC

# Use CUDA if available and set random seed for reproducibility
# args = get_args(argv=["--dataset", "criteo", "--debug"])
args = get_args()
args.seed = 42

# run_name = input("Run name: ")
run_name = "subset"
writer = SummaryWriter(f"logs/{run_name}")
set_seed(args.seed)
if torch.cuda.is_available():
    device = args.device = "cuda"
else:
    device = args.device = "cpu"

train_dataset = IndexedDataset(args, train_transform=True)
# subset_path = "outputs/debugs_main/easy_crest_v7.pth"
# subset_path = "outputs/dcnv2-criteo-0.01-v2-debug/easy_crest_mc_sub.pth"
# subset_path = "outputs/deepfm-criteo-0.01-v2-debug/easy_crest_mc_sub_tmp.pth"
subset_path = "outputs/dcnv2-avazu-0.01-v2-debug/easy_crest_mc_sub.pth"
subset_path = "outputs/dcnv2-avazu-0.01-v2-efficiency/easy_crest_mc_sub.pth"
subset_path = args.subset_path
# subset_path = "outputs/kcenter/avazu_0.05.pth"

# subset_path = "outputs/deepfm-avazu-0.01-v2-debug/easy_crest_mc_sub.pth"

# subset_path = "outputs/dcnv2-criteo-0.005-v2/easy_crest_mc_sub.pth"
# subset_path = "outputs/dcnv2-criteo-0.05-v2/easy_crest_mc_sub_tmp.pth"
# subset_path = "outputs/dcnv2-criteo-0.01-v2-debug/easy_crest_mc_sub_tmp.pth"
# subset_path = "outputs/deepfm-criteo-0.05-v2/easy_crest-full.pth"
# subset_path = "outputs/debugs_main/subset.pth"
subset_data = torch.load(subset_path, map_location="cpu")


def convert(item):
    if isinstance(item, torch.Tensor):
        return item.item()
    return item


subset_data = list(map(convert, subset_data))

# ori code. Kcenter has some weird bug
# subset_dataset = Subset(train_dataset, subset_data)

# subset_dataset = Subset(train_dataset, subset_data)
subset_dataset = train_dataset
train_loader = DataLoader(
    subset_dataset,
    args.batch_size,
    num_workers=4,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    IndexedDataset(args, split="val"),
    batch_size=4096,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)


model = get_model(train_dataset, args.arch)
model = model.to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    0.001,
    # weight_decay=5e-4,
    weight_decay=1e-6,
)
print(optimizer)
# optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-6)
# optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-7)
# optimizer = torch.optim.SGD(model.parameters(), 0.01, weight_decay=1e-5)


criterion_selflc = ProSelfLC(
    # 4 * len(train_loader),
    # 20 * len(train_loader),
    # 25 * len(train_loader),
    25 * len(train_loader),  # used for Criteo DFM 1%
    16,
    0.5,
    False,
)

criterion_tce = TCE_Loss(10 * len(train_loader))

total_step = 0
step = 0
sum_loss = 0
output_dir = "outputs/data_logs/"
os.makedirs(output_dir, exist_ok=True)

if args.dataset == "criteo":
    pos_weight = 1 / (27277461 / 9395032)
elif args.dataset == "avazu":
    pos_weight = 5491354 / 26851819
else:
    raise ValueError()

pw = torch.tensor(pos_weight)
# pos_weight = None
# pos_weight = 1


def train_epoch_simple(train_loader, model, optimizer):
    device = "cuda"
    model.train()
    sum_loss = 0
    step = 0
    # pw = torch.tensor(5491354 / 26851819)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    for data, target, _ in train_loader:
        data = data.to(device)
        target = target.to(device)

        # data = random_mask(data, oov_tokens, 0.01)

        y_pred = model(data)

        if args.loss == "selflc":
            y_pred = torch.sigmoid(y_pred)
            eps = 1e-6
            y_pred = torch.clamp(y_pred, eps, 1 - eps)
            loss = criterion_selflc(
                y_pred,
                target.float(),
                total_step + step,
                pos_weight=pos_weight,
            )
        elif args.loss == "rce":
            # loss = criterion(y_pred, target.float())
            loss = rce_loss(y_pred, target.float(), pos_weight=pw)
        elif args.loss == "tce":
            loss = criterion_tce(
                y_pred, target.float(), total_step + step, pos_weight=pw
            )
        elif args.loss == "bce":
            loss = F.binary_cross_entropy_with_logits(y_pred, target.float())
        else:
            raise ValueError()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        step += 1
    return step, sum_loss


best_auc = 0
best_log_loss = 100
max_patience = 1

feat_count = get_count(train_loader)
non_oov_tokens = torch.where(feat_count > 0)[0]
oov_tokens = get_oov_tokens(train_loader.dataset)

folder = f"outputs/{run_name}/"
os.makedirs(folder, exist_ok=True)
folder = Path(folder)
auc_model_path = folder / "best_auc.pth"
loss_model_path = folder / "best_logloss.pth"

generator = torch.Generator()
generator.manual_seed(10)
patience = max_patience

for epoch in range(20):
    print(f"Epoch: {epoch:02d}")
    # Train model
    step, sum_loss = train_epoch_simple(train_loader, model, optimizer)
    # step, sum_loss = train_epoch_simple(train_loader, model, optimizer)
    total_step += step
    # Validation
    model.eval()
    auc, log_loss = evaluate_model(
        model,
        val_loader,
        generator,
        device,
        non_oov_tokens,
        oov_tokens,
    )
    improve = False

    if auc > best_auc:
        torch.save(model.state_dict(), auc_model_path)
        improve = True

    if log_loss < best_log_loss:
        torch.save(model.state_dict(), loss_model_path)
        improve = True

    best_auc = max(auc, best_auc)
    best_log_loss = min(log_loss, best_log_loss)
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

    if not improve:
        patience -= 1
    else:
        patience = max_patience

    if patience < 0:
        print("Model val performance not improve.... Stop for full evaluation")
        break


best_state = torch.load(auc_model_path)
model.load_state_dict(best_state)
model.eval()


val_loader = torch.utils.data.DataLoader(
    IndexedDataset(args, split="test"),
    batch_size=4096,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

auc, log_loss = evaluate_model(model, val_loader, None, device)

print(f"Final AUC {auc:.4f} - LogLoss: {log_loss:.4f}")

print("Evaluate best Logloss model")
best_state = torch.load(loss_model_path)
model.load_state_dict(best_state)
model.eval()

auc, log_loss = evaluate_model(model, val_loader, None, device)

print(f"Final AUC {auc:.4f} - LogLoss: {log_loss:.4f}")
