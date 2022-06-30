import argparse

import numpy as np
import torch

from model import CurvatureGraphNN
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


def main(args, d_input, d_output):
    accuracy_list = []
    for i in range(args.num_trials):
        data = load_data('./data', args.dataset)
        data, model = CurvatureGraphNN.call(data, args, d_input, d_output)
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
    print('Experiments: {}, Mean: {}, std: {}\n'.format(args.num_trials, np.mean(accuracy_list),
                                                        np.std(accuracy_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CGNN')
    parser.add_argument('--dataset', type=str, help="Name of the datasets", required=True)
    parser.add_argument('--num_trials', type=int, help="The number of the repeating experiments", default=50)
    args = parser.parse_args()

    datasets_config = {
        'Cora': {'d_input': 1433,
                 'd_output': 7},
        'Citeseer': {'d_input': 3703,
                     'd_output': 6},
        'PubMed': {'d_input': 500,
                   'd_output': 3},
        'CS': {'d_input': 6805,
               'd_output': 15},
        'Physics': {'d_input': 8415,
                    'd_output': 5},
        'Computers': {'d_input': 767,
                      'd_output': 10},
        'Photo': {'d_input': 745,
                  'd_output': 8},
        'WikiCS': {'d_input': 300,
                   'd_output': 10},
    }

    main(args, datasets_config[args.dataset]['d_input'], datasets_config[args.dataset]['d_output'])
