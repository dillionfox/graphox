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
        self.w_mul = torch.from_numpy(w_mul)


class CurvatureGraph(object):
    def __init__(self, G, curvature_values: CurvatureValues, num_classes=2):
        self.G = G
        self.num_classes = num_classes
        self.curvature_values = curvature_values

    def call(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        w_mul = self.compute_convolution_weights(self.G.edge_index, self.curvature_values)
        model = CurvatureGraphNN(self.G.num_features, self.num_classes, w_mul, d_hidden=64, p=0.5)
        model.to(device).reset_parameters()
        return device, model

    @staticmethod
    def compute_convolution_weights(edge_index, edge_weight):
        print('>>>>>:::::', edge_index.shape, edge_weight.shape)
        deg_inv_sqrt = scatter_add(edge_weight, edge_index[0], dim=0).pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        w_mul = deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]
        return torch.tensor(w_mul.clone().detach(), dtype=torch.float)


class CurvatureGraphNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, w_mul, d_hidden=64, p=0.0):
        super(CurvatureGraphNN, self).__init__()
        self.conv1 = MessagePassingConvLayer(num_features, d_hidden, w_mul, p=p)
        self.conv2 = MessagePassingConvLayer(d_hidden, d_hidden, w_mul, p=p)
        self.lin = Linear(d_hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x = torch.nn.functional.dropout(data.x, p=0.6, training=self.training)

        # 1. Obtain node embeddings
        x = self.conv1(x, data.edge_index)
        x = x.relu()
        x = self.conv2(x, data.edge_index)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, d_hidden]

        # 3. Apply a final classifier
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.lin(x)

        return x


class MessagePassingConvLayer(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels, weight_agg, p=0.6):
        super(MessagePassingConvLayer, self).__init__(aggr='add')
        self.dropout = p
        self.out_channels = out_channels
        self.weight_agg = weight_agg
        self.lin = torch.nn.Linear(in_channels, out_channels)

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
