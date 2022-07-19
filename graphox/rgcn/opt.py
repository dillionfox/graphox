from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from datetime import datetime
import random

import torch
from graphox.rgcn.data.immotion.immotion_dataset import ImMotionDataset
from graphox.rgcn.src import CurvatureGraph, CurvatureValues, CurvatureGraphNN
from graphox.rgcn.rgcn import train, test
from torch_geometric.loader import DataLoader
from torch.utils.data.dataloader import default_collate

from typing import Any


def train_rgcn(config: dict,
               num_trials: int = 10) -> None:

    pt_graphs_path = '/home/dfox/code/graphox/output/pt_graphs'
    edge_curvatures_file_path = '/home/dfox/code/graphox/output/pt_edge_curvatures.csv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Slurp up pyg graphs into pyg Dataset
    data_raw = ImMotionDataset(pt_graphs_path)

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
    curvature_values = CurvatureValues(data_raw[0].num_nodes, ricci_filename=edge_curvatures_file_path).w_mul

    # Train 'num_trials' models
    loss = None
    train_acc = None
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
        for epoch in range(100):
            t_initial = datetime.now()
            model, optimizer, loss = train(train_data, model, optimizer, device)
            train_acc = test(train_data, model)
            test_acc = test(test_data, model)
            t_final = datetime.now()

        with tune.checkpoint_dir(i) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=loss, accuracy=train_acc)


def main():

    config = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.uniform(0, 1e-1),
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        partial(train_rgcn, num_trials=10),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    main()
