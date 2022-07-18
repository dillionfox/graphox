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

from datetime import datetime

import torch
from graphox.rgcn.data.immotion.immotion_dataset import ImMotionDataset
from graphox.rgcn.src import CurvatureGraph, CurvatureValues, CurvatureGraphNN
from torch_geometric.loader import DataLoader
from torch.utils.data.dataloader import default_collate

from typing import Any


def train(dataset: DataLoader,
          model: CurvatureGraphNN,
          optimizer: Any,
          device) -> tuple:
    model.train()
    for data in dataset:
        out = model(data)
        pred = out.max(1)[1].to(device='cpu').long()
        y = data.y.to(device='cpu').long()
        # loss = torch.nn.functional.nll_loss(pred, y)
        loss = torch.nn.NLLLoss(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model, optimizer


def test(dataset: DataLoader,
         model: CurvatureGraphNN) -> float:
    # Building out metrics with ignite. Work in progress.
    correct = 0
    for data in dataset:  # Iterate in batches over the training/test dataset.
        out = model(data)
        pred = out.max(1)[1]
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(dataset.dataset)  # Derive ratio of correct predictions.


def rgcn_trainer(data_path: str,
                 num_trials: int,
                 ricci_filename: str) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Slurp up pyg graphs into pyg Dataset
    data_raw = ImMotionDataset(data_path)

    # Split into test/train
    train_dataset = data_raw[:658]
    test_dataset = data_raw[658:]

    # Convert test/train sets to Data Loaders
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Structure edge curvatures in CurvatureValues instance
    curvature_values = CurvatureValues(data_raw[0].num_nodes, ricci_filename=ricci_filename).w_mul

    # Train 'num_trials' models
    print('Trial, Epoch, Train acc, Test acc, Time')
    for i in range(num_trials):

        # Instantiate CurvatureGraph object with graph topology and edge curvatures
        sample_graph = data_raw[0]
        sample_graph.to(device)
        curvature_values.to(device)
        curvature_graph_obj = CurvatureGraph(sample_graph, curvature_values, device=device)

        # Construct RGCN model
        model = curvature_graph_obj.call()

        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train model and compute metrics
        for epoch in range(1000):
            t_initial = datetime.now()
            model, optimizer = train(train_data, model, optimizer, device)
            train_acc = test(train_data, model)
            test_acc = test(test_data, model)
            t_final = datetime.now()
            print(i, epoch, train_acc, test_acc, t_final - t_initial)


if __name__ == '__main__':
    import sys
    import os

    number_of_trials = 2
    graphs_path = sys.argv[1]
    ricci_path = sys.argv[2]

    if not os.path.exists(graphs_path):
        print('invalid path')

    rgcn_trainer(graphs_path, number_of_trials, ricci_path)
