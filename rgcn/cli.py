import numpy as np
import torch
from torch_geometric.loader import DataLoader

from graphox.rgcn.rgcn import CurvatureGraph, CurvatureValues
from graphox.rgcn.data.immotion.immotion_dataset import ImMotionDataset


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
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(dataset.dataset)  # Derive ratio of correct predictions.


def main(num_trials):
    data_raw = ImMotionDataset('/Users/dfox/code/graphox/notebooks/data/full')
    train_dataset = data_raw[:650]
    test_dataset = data_raw[650:]
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)
    curvature_values = CurvatureValues(data_raw[0].num_nodes).w_mul
    for i in range(num_trials):
        curvature_graph_obj = CurvatureGraph(data_raw[0], curvature_values)
        device, model = curvature_graph_obj.call()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(20):
            model, optimizer = train(train_data, model, optimizer)
            train_acc = test(train_data, model)
            test_acc = test(test_data, model)
            print('Epoch: {}, Train acc: {}, Test acc: {}'.format(epoch, train_acc, test_acc))


if __name__ == '__main__':
    number_of_trials = 2
    main(number_of_trials)
