from abc import ABC

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn import global_mean_pool


class CurvatureGraph(object):
    def __init__(self, G, curvature_values):
        self.G = G
        self.curvature_values = curvature_values
        self.ricci_filename = '/Users/dfox/code/graphox/data/immotion_edge_curvatures_formatted.csv'

    @staticmethod
    def compute_convolution_weights(edge_index, edge_weight):
        deg_inv_sqrt = scatter_add(edge_weight, edge_index[0], dim=0).pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        w_mul = deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]
        return torch.tensor(w_mul.clone().detach(), dtype=torch.float)

    def fetch_curvature_results(self, num_nodes):
        df = pd.read_csv(self.ricci_filename, sep=' ', header=None, names=[0, 1, 2])
        w_mul = np.concatenate([df.sort_values(by=[0, 1])[2].tolist(), [1 for i in range(num_nodes)]]).astype(
            np.float16)
        w_mul += np.float(abs(min(w_mul)))
        return torch.from_numpy(w_mul)

    def call(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        curvature_values = self.fetch_curvature_results(self.G.num_nodes)
        self.G.edge_index = add_self_loops(self.G.edge_index, num_nodes=self.G.x.size(0))[0]
        w_mul = self.compute_convolution_weights(self.G.edge_index, curvature_values).to(device)
        model = CurvatureGraphNN(self.G.num_features, self.G.num_classes, w_mul, d_hidden=64, p=0.5)
        data = self.G.to(device)
        model.to(device).reset_parameters()
        return data, model


class CurvatureGraphNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, w_mul, d_hidden=64, p=0.0):
        super(CurvatureGraphNN, self).__init__()
        self.conv1 = MessagePassingConvLayer(num_features, d_hidden, w_mul, p=p)
        self.conv2 = MessagePassingConvLayer(d_hidden, num_classes, w_mul, p=p)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self):
        # 1. Obtain node embeddings
        x = self.conv1(self.data.x, self.data.edge_index)
        x = x.relu()
        x = self.conv2(x, self.data.edge_index)
        x = x.relu()
        x = self.conv3(x, self.data.edge_index)

        # 2. Readout layer
        x = global_mean_pool(x)  # [batch_size, d_hidden]

        # 3. Apply a final classifier
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

    def forward_bu(self, data):
        # Dropout 1
        x = torch.nn.functional.dropout(data.x, p=0.6, training=self.training)

        # Obtain node embeddings
        x = self.conv1(x, data.edge_index)
        x = torch.nn.functional.relu(x)

        # Dropout 2
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)

        # Convolutional layer 2
        x = self.conv2(x, data.edge_index)

        # Return softmax
        return torch.nn.functional.log_softmax(x, dim=1)


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


