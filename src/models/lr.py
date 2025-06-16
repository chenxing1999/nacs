import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Optional, Dict


class LogisticRegression(nn.Module):
    def __init__(
        self,
        field_dims: List[int],
        num_factor: int,
        hidden_sizes: List[int],
        p_dropout: float = 0.1,
        use_batchnorm=False,
        embedding_config: Optional[Dict] = None,
        empty_embedding=False,
    ):
        """
        Args:
            field_dims: List of field dimension for each features
            num_factor: Low-level embedding vector dimension
            hidden_sizes: MLP layers' hidden sizes
            p_dropout: Dropout rate per MLP layer
            embedding_config
        """

        super().__init__()
        num_inputs = sum(field_dims)

        if not empty_embedding:
            # self.embedding = get_embedding(
            #     embedding_config,
            #     field_dims,
            #     num_factor,
            #     mode=None,
            #     field_name="deepfm",
            # )
            self.embedding = nn.Embedding(num_inputs, num_factor)
            nn.init.xavier_uniform_(self.embedding.weight)

        # self.fc = nn.EmbeddingBag(num_inputs, 1, mode="sum")
        self.fc = nn.Embedding(num_inputs, 1)
        deep_branch_inp = num_factor * len(field_dims)
        self.linear_layer = nn.Linear(deep_branch_inp, 1)

        field_dims_tensor = torch.tensor(field_dims)
        field_dims_tensor = torch.cat(
            [torch.tensor([0], dtype=torch.long), field_dims_tensor]
        )
        offsets = torch.cumsum(field_dims_tensor[:-1], 0).unsqueeze(0)
        self.register_buffer("offsets", offsets)


    def forward(self, x):
        x = x + self.offsets
        emb = self.embedding(x)
        b, num_field, size = emb.shape
        emb = emb.reshape((b, num_field * size))
        scores = self.linear_layer(emb)
        scores = scores.squeeze(-1)

        return scores
