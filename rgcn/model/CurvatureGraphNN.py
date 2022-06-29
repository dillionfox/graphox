from abc import ABC

import numpy as np
import pandas as pd
import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add


class CurvatureGraphNN(torch.nn.Module):
    def __init__(self, d_input, d_output, w_mul, d_hidden=64, p=0.0):
        super(CurvatureGraphNN, self).__init__()
        self.conv1 = MessagePassingConvLayer(d_input, d_hidden, w_mul, p=p)
        self.conv2 = MessagePassingConvLayer(d_hidden, d_output, w_mul, p=p)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index,
        x = torch.nn.functional.dropout(data.x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)


class MessagePassingConvLayer(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels, weight_agg, p=0.6):
        super(MessagePassingConvLayer, self).__init__(aggr='add')
        self.dropout = p
        self.out_channels = out_channels
        self.weight_agg = weight_agg
        self.lin = Linear(in_channels, out_channels)

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


def compute_convolution_weights(edge_index, edge_weight=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    deg_inv_sqrt = scatter_add(edge_weight, edge_index[0], dim=0).pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    w_mul = deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]
    return torch.tensor(w_mul.clone().detach(), dtype=torch.float)


def fetch_curvature_results(filename, num_nodes):
    df1 = pd.read_csv(filename, sep=' ', header=None, names=[0, 1, 2])
    df = pd.concat([df1, df1.rename(columns={0: 1, 1: 0})]).reset_index(drop=True).drop_duplicates()
    w_mul = np.concatenate([df.sort_values(by=[0, 1])[2].tolist(), [1 for i in range(num_nodes)]]).astype(np.float16)
    w_mul += np.float(abs(min(w_mul)))
    return torch.from_numpy(w_mul)


def call(data, arg, d_input, d_output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = r'./data/Ricci/graph_' + arg.dataset + '.edge_list'
    curvature_values = fetch_curvature_results(filename, data.num_nodes)
    data.edge_index = add_self_loops(data.edge_index, num_nodes=data.x.size(0))[0]
    w_mul = compute_convolution_weights(data.edge_index, curvature_values).to(device)
    model = CurvatureGraphNN(d_input, d_output, w_mul, d_hidden=64, p=0.5)
    data = data.to(device)
    model.to(device).reset_parameters()
    return data, model
