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

import os
import random
from pathlib import Path

import click
import torch
from torch_geometric.loader import DataLoader

from graphox.dataloader.immotion_dataset import ImMotionDataset
from graphox.rgcn.src import CurvatureGraph, CurvatureValues


def train_rgcn(config):
    data_raw = ImMotionDataset(config['graphs_path'])
    curvature_values = CurvatureValues(data_raw[0].num_nodes,
                                       ricci_filename=config['curvature_file']).w_mul

    # Instantiate CurvatureGraph object with graph topology and edge curvatures
    sample_graph = data_raw[0]
    curvature_graph_obj = CurvatureGraph(sample_graph, curvature_values,
                                         d_hidden=config['d_hidden'], p=config['p'])
    net = curvature_graph_obj.call()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)

    # Construct RGCN model
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config["lr"],
        weight_decay=config['weight_decay'],
        momentum=config['momentum'])

    number_patients = data_raw.len()
    patient_indices = list(range(number_patients))

    train_fraction = 0.8
    number_train_points = int(number_patients * train_fraction)

    train_indices = random.sample(patient_indices, number_train_points)
    test_indices = list(set(patient_indices) - set(train_indices))

    # Split into test/train
    train_dataset = data_raw[train_indices]
    test_dataset = data_raw[test_indices]

    # Convert test/train sets to Data Loaders
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for epoch in range(config['epochs']):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader):
            inputs, labels = data.x, data.y
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)
            loss = criterion(outputs.cpu(), labels.cpu().long())
            loss.backward()
            optimizer.step()

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data.x, data.y
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs.cpu(), labels.cpu().long())
                val_loss += loss.cpu().numpy()
                val_steps += 1

        print("loss =", val_loss / val_steps, "accuracy =", correct / total)
    print("Finished Training")


@click.command()
@click.option('--path', default=os.getcwd(), help="Path to graphox directory. Default is cwd.")
@click.option('--epochs', default=100, help="Number of epochs")
@click.option('--lr', default=0.05, help="Learning rate for optimizer")
@click.option('--weight_decay', default=0.025, help="Weight decay for optimizer")
@click.option('--momentum', default=0.1, help="Momentum for optimizer")
@click.option('--d_hidden', default=32, help="Dimension for hidden layers")
@click.option('--p', default=0.6, help="Dropout rate in hidden layers")
def main(
        path,
        epochs,
        lr,
        weight_decay,
        momentum,
        d_hidden,
        p):

    graphox_path = Path(path)
    graphs_path = graphox_path.joinpath('graphox/output/pt_graphs')
    curvature_path = graphox_path.joinpath('graphox/output/pt_edge_curvatures.csv')

    config = {
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "d_hidden": d_hidden,
        "p": p,
        "graphs_path": graphs_path,
        "curvature_file": curvature_path,
    }

    train_rgcn(config)


if __name__ == '__main__':
    main()
