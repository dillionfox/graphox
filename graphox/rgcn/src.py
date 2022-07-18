"""Simple implementation of Ricci Graph Convolutional Network (RGCN)
Copyright (C) 2022 Dillion Fox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

from abc import ABC

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool


class CurvatureValues(object):
    def __init__(self, num_nodes, ricci_filename):
        self.num_nodes = num_nodes
        self.ricci_filename = ricci_filename
        self.w_mul = None
        self.compute_w_mul()

    def compute_w_mul(self):
        df = pd.read_csv(self.ricci_filename, header=None, names=[0, 1, 2])
        w_mul = np.array(df.sort_values(by=[0, 1])[2].tolist())
        w_mul += np.float(abs(min(w_mul)))
        w_mul = torch.from_numpy(w_mul)
        w_mul.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.w_mul = w_mul


class CurvatureGraph(object):
    def __init__(self, G, curvature_values: torch.tensor, num_classes=2, device=None):
        self.G = G
        self.num_classes = num_classes
        self.curvature_values = curvature_values
        self.device = device

    def call(self):
        w_mul = self.compute_convolution_weights(self.G.edge_index, self.curvature_values)
        model = CurvatureGraphNN(self.G.num_features, self.num_classes, w_mul, d_hidden=64, p=0.5, device=self.device)
        return model

    def compute_convolution_weights(self, edge_index, edge_weight):
        print('1', edge_index.get_device())
        print('2', edge_weight.get_device())
        edge_weight.to(self.device)
        deg_inv_sqrt = scatter_add(edge_weight, edge_index[0], dim=0).pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        w_mul = deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]
        return torch.tensor(w_mul.clone().detach(), dtype=torch.float, device=self.device)


class CurvatureGraphNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, w_mul, d_hidden=64, p=0.0, device=None):
        super(CurvatureGraphNN, self).__init__()
        self.conv1 = MessagePassingConvLayer(num_features, d_hidden, w_mul, p=p, device=device)
        self.conv2 = MessagePassingConvLayer(d_hidden, d_hidden, w_mul, p=p, device=device)
        self.lin = Linear(d_hidden, num_classes, device=device)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        # x = torch.nn.functional.dropout(data.x, p=0.6, training=self.training)

        # 1. Obtain node embeddings
        print(data)
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = self.conv2(x, data.edge_index)
        # x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, d_hidden]

        # 3. Apply a final classifier
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.lin(x)
        x = torch.nn.functional.log_softmax(x, dim=1)

        return x


class MessagePassingConvLayer(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels, weight_agg, p=0.6, device=None):
        super(MessagePassingConvLayer, self).__init__(aggr='add')
        self.dropout = p
        self.out_channels = out_channels
        self.weight_agg = weight_agg
        self.lin = torch.nn.Linear(in_channels, out_channels, device=device)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        weight_agg = torch.nn.functional.dropout(self.weight_agg, p=self.dropout, training=self.training)
        return weight_agg.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
