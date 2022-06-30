import argparse

import numpy as np
import torch

from rgcn import CurvatureGraph
from utils import load_data


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    pred = model(data)
    loss = torch.nn.functional.nll_loss(pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def val(data, model):
    model.eval()
    logits = model(data)
    accuracies = [logits[mask].max(1)[1].eq(data.y[mask]).sum().item() / mask.sum().item() for _, mask in
                  data('train_mask', 'val_mask', 'test_mask')]
    accuracies.append(torch.nn.functional.nll_loss(model(data)[data.val_mask], data.y[data.val_mask]))
    return accuracies


def main(num_trials, d_input):
    accuracy_list = []
    for i in range(num_trials):
        data = torch.load('/Users/dfox/code/graphox/data/G_annotated_masked.pt')
        curvature_graph_obj = CurvatureGraph(data, 'STRING', d_input, 2)
        data, model = curvature_graph_obj.call()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
        best_val_acc = test_acc = 0.0
        best_val_loss = np.inf
        for epoch in range(200):
            train(data, model, optimizer)
            train_acc, val_acc, tmp_test_acc, val_loss = val(data, model)
            if val_acc >= best_val_acc or val_loss <= best_val_loss:
                if val_acc >= best_val_acc:
                    test_acc = tmp_test_acc
                best_val_acc = np.max((val_acc, best_val_acc))
                best_val_loss = np.min((float(val_loss), best_val_loss))
        print('Experiment: {}, Test: {}'.format(i + 1, test_acc))
        accuracy_list.append(test_acc * 100)
    print('Experiments: {}, Mean: {}, std: {}\n'.format(num_trials, np.mean(accuracy_list),
                                                        np.std(accuracy_list)))


if __name__ == '__main__':
    main(2, 823)
