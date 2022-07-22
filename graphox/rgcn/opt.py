import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch_geometric.loader import DataLoader

from graphox.dataloader.immotion_dataset import ImMotionDataset
from graphox.rgcn.src import CurvatureGraph, CurvatureValues


def train_rgcn(config, checkpoint_dir=None):
    ##

    data_dir = '/home/dfox/code/graphox/output/pt_graphs'
    edge_curvatures_file_path = '/home/dfox/code/graphox/output/pt_edge_curvatures.csv'

    train_dataset = ImMotionDataset(data_dir, subset='train')
    test_dataset = ImMotionDataset(data_dir, subset='test')
    curvature_values = CurvatureValues(train_dataset[0].num_nodes,
                                       ricci_filename=edge_curvatures_file_path).w_mul

    # Instantiate CurvatureGraph object with graph topology and edge curvatures
    sample_graph = train_dataset[0]
    curvature_graph_obj = CurvatureGraph(sample_graph, curvature_values,
                                         d_hidden=config['d_hidden'], p=config['p'])
    net = curvature_graph_obj.call()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

    # Construct RGCN model
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=config["lr"],
        weight_decay=config['weight_decay'],
        momentum=config['momentum'])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

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

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


def main(num_samples=576, max_num_epochs=10, gpus_per_trial=1):
    config = {
        "lr": tune.choice([0.001, 0.005, 0.01, 0.05]),  # 4
        "weight_decay": tune.choice([0.01, 0.05]),  # 2
        "momentum": tune.choice([0, 0.5]),  # 2
        "d_hidden": tune.choice([32, 64, 128]),  # 3
        "p": tune.choice([0.4, 0.8]),  # 2
        "version": tune.choice(['v0', 'v1', 'v2', 'v3', 'v4', 'v5']),  # 6
        "epochs": 30,
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=3,
        reduction_factor=2)
    result = tune.run(
        tune.with_parameters(train_rgcn),
        resources_per_trial={"cpu": 12, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    main()
