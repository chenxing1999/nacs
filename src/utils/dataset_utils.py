import torch
from torch.utils.data import ConcatDataset, Subset

from src.datasets import IndexedDataset

device = "cuda"


def _get_field_dims(dataset):
    if isinstance(dataset, ConcatDataset):
        dataset = dataset.datasets[0]

    if isinstance(dataset, Subset):
        field_dims = dataset.dataset.dataset.field_dims
    elif isinstance(dataset, IndexedDataset):
        field_dims = dataset.dataset.field_dims
    else:
        field_dims = dataset.field_dims
    return field_dims


def get_count(loader):
    field_dims = _get_field_dims(loader.dataset)

    field_dims_tensor = torch.tensor(field_dims)
    n_fields = torch.sum(field_dims_tensor).item()

    device = "cuda"
    field_dims_tensor = torch.cat(
        [torch.tensor([0], dtype=torch.long), field_dims_tensor]
    )
    offsets = torch.cumsum(field_dims_tensor[:-1], 0).unsqueeze(0)
    offsets = offsets.to(device)

    count = torch.zeros(n_fields, device=device)
    for data, target, idx in loader:
        data = data.to(device) + offsets

        # torch.index_add(count, 0, data, torch.tensor(1, device=device))
        count[data] += 1

    return count


def get_oov_tokens(dataset):
    field_dims = _get_field_dims(dataset)
    field_dims_tensor = torch.tensor(field_dims)
    # field_dims_tensor = torch.cat(
    #     [torch.tensor([0], dtype=torch.long), field_dims_tensor]
    # )
    # offsets = torch.cumsum(field_dims_tensor, 0)

    # oov = offsets[1:].to(device)
    oov = field_dims_tensor - 1
    return oov.to(device)


def convert_to_oov(
    data,
    oov_tokens: torch.Tensor,
    non_oov,
    offsets,
):
    tmp = data + offsets
    mask = ~torch.isin(tmp, non_oov)

    oov_tokens = oov_tokens.repeat((data.shape[0], 1))
    data[mask] = oov_tokens[mask]
    return data


def random_mask(data, oov_tokens, p=0.01):
    mask = torch.rand(data.shape, device=device)
    mask = mask < p

    oov_tokens = oov_tokens.repeat((data.shape[0], 1))
    data[mask] = oov_tokens[mask]
    return data
