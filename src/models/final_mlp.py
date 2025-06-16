# =========================================================================
# Copyright (C) 2023. FuxiCTR Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from typing import List

import torch
from torch import nn


class FinalMLP(nn.Module):
    def __init__(
        self,
        field_dims: List[int],
        num_factor: int,
        hidden_size1: List[int] = None,
        hidden_size2: List[int] = None,
        num_heads=200,
        p_dropout=0.5,
    ):
        super().__init__()
        if hidden_size1 is None:
            hidden_size1 = [100, 100, 100]

        if hidden_size2 is None:
            hidden_size2 = [100, 100, 100]

        num_inputs = sum(field_dims)
        self.embedding = nn.Embedding(num_inputs, num_factor)
        nn.init.xavier_uniform_(self.embedding.weight)

        deep_branch_inp = num_factor * len(field_dims)

        layers: List[nn.Module] = []
        for size in hidden_size1:
            layers.append(nn.Linear(deep_branch_inp, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))

            deep_branch_inp = size
        self.mlp1 = nn.Sequential(*layers)

        deep_branch_inp = num_factor * len(field_dims)

        layers: List[nn.Module] = []
        for size in hidden_size2:
            layers.append(nn.Linear(deep_branch_inp, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))

            deep_branch_inp = size
        self.mlp2 = nn.Sequential(*layers)

        self.fusion_module = InteractionAggregation(
            hidden_size1[-1],
            hidden_size2[-1],
            num_heads=num_heads,
        )

        field_dims_tensor = torch.tensor(field_dims)
        field_dims_tensor = torch.cat(
            [torch.tensor([0], dtype=torch.long), field_dims_tensor]
        )
        offsets = torch.cumsum(field_dims_tensor[:-1], 0).unsqueeze(0)
        self.register_buffer("offsets", offsets)

    def forward(self, x):
        """
        Inputs: [X,y]
        """
        x = x + self.offsets
        emb: torch.Tensor = self.embedding(x)

        flat_emb = emb.flatten(1)

        feat1, feat2 = flat_emb, flat_emb
        y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        return y_pred.squeeze(-1)


class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert (
            x_dim % num_heads == 0 and y_dim % num_heads == 0
        ), "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(
            torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, output_dim)
        )
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(
            torch.matmul(
                head_x.unsqueeze(2), self.w_xy.view(self.num_heads, self.head_x_dim, -1)
            ).view(-1, self.num_heads, self.output_dim, self.head_y_dim),
            head_y.unsqueeze(-1),
        ).squeeze(-1)
        output += xy.sum(dim=1)
        return output


if __name__ == "__main__":
    field_dims = [1, 2, 3]

    model = FinalMLP(field_dims, 10, num_heads=50)

    x = torch.tensor([[0, 1, 1], [0, 1, 1]])

    print(model(x))
