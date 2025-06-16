import torch
from sklearn.metrics import roc_auc_score

from src.utils.dataset_utils import convert_to_oov


def evaluate_model(
    model,
    val_loader,
    generator=None,
    device="cuda",
    non_oov_tokens=None,
    oov_tokens=None,
):

    remove_oov = False
    if non_oov_tokens is not None:
        assert oov_tokens is not None
        remove_oov = True

    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    criterion = criterion.to(device)

    log_loss = 0.0
    all_y_true = []
    all_y_pred = []
    for idx, batch in enumerate(val_loader):
        inputs, labels, _ = batch
        all_y_true.extend(labels.tolist())

        inputs = inputs.to(device)
        if remove_oov:
            inputs = convert_to_oov(inputs, oov_tokens, non_oov_tokens, model.offsets)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        log_loss += criterion(outputs, labels.float()).item()

        outputs = torch.sigmoid(outputs)
        all_y_pred.extend(outputs.cpu().tolist())

    if generator:
        tmp = torch.randperm(len(all_y_true), generator=generator)[:10].tolist()
        print("Sample pred", [all_y_pred[idx] for idx in tmp])
        print("Sample true", [all_y_true[idx] for idx in tmp])

    auc = roc_auc_score(all_y_true, all_y_pred)
    log_loss = log_loss / len(all_y_pred)
    return auc, log_loss
