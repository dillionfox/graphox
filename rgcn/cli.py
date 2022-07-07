from datetime import datetime

import torch
from graphox.rgcn.data.immotion.immotion_dataset import ImMotionDataset
from graphox.rgcn.rgcn import CurvatureGraph, CurvatureValues
from torch_geometric.loader import DataLoader


def train(dataset, model, optimizer):
    model.train()
    for data in dataset:
        pred = model(data)
        loss = torch.nn.functional.nll_loss(pred, data.y.type(torch.LongTensor))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model, optimizer


def test(dataset, model):
    correct = 0
    for data in dataset:  # Iterate in batches over the training/test dataset.
        out = model(data)
        pred = out.max(1)[1]
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(dataset.dataset)  # Derive ratio of correct predictions.


def main(data_path, num_trials, ricci_filename='/Users/dfox/code/graphox/data/immotion_edge_curvatures_formatted.csv'):
    data_raw = ImMotionDataset(data_path)
    train_dataset = data_raw[:650]
    test_dataset = data_raw[650:]
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)
    curvature_values = CurvatureValues(data_raw[0].num_nodes, ricci_filename=ricci_filename).w_mul
    print('Trial, Epoch, Train acc, Test acc, Time')
    for i in range(num_trials):
        curvature_graph_obj = CurvatureGraph(data_raw[0], curvature_values)
        device, model = curvature_graph_obj.call()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(20):
            t_initial = datetime.now()
            model, optimizer = train(train_data, model, optimizer)
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

    main(graphs_path, number_of_trials, ricci_path)
