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
import random

import torch
from graphox.dataloader.immotion_dataset import ImMotionDataset
from graphox.rgcn.src import CurvatureGraph, CurvatureValues, CurvatureGraphNN
from torch_geometric.loader import DataLoader
from torch.utils.data.dataloader import default_collate

from typing import Any


criterion = torch.nn.CrossEntropyLoss()


def train(dataset: DataLoader,
          model: CurvatureGraphNN,
          optimizer: Any,
          device) -> tuple:
    model.train()
    for data in dataset:
        out = model(data)
        y = data.y
        try:
            loss = torch.nn.functional.nll_loss(out.cpu(), y.cpu().long())
        except Exception as e:
            print('Data:', data)
            print('y:', y)
            exit()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model, optimizer, loss


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
                 config: dict,
                 ricci_filename: str) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Slurp up pyg graphs into pyg Dataset
    data_raw = ImMotionDataset(data_path)

    number_patients = data_raw.len()
    patient_indices = list(range(number_patients))

    train_fraction = 0.9
    number_train_points = int(number_patients * train_fraction)

    train_indices = random.sample(patient_indices, number_train_points)
    test_indices = list(set(patient_indices) - set(train_indices))

    # Split into test/train
    train_dataset = data_raw[train_indices]
    test_dataset = data_raw[test_indices]

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
        curvature_graph_obj = CurvatureGraph(sample_graph, curvature_values, device=device)

        # Construct RGCN model
        model = curvature_graph_obj.call()

        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'],
                                     )

        # Train model and compute metrics
        for epoch in range(1000):
            t_initial = datetime.now()
            model, optimizer, loss = train(train_data, model, optimizer, device)
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
