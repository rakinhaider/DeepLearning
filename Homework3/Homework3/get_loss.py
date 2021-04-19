import numpy as np
from models import MulticlassClassification
from basic_main import BasicMain
import torch
import os
import pandas as pd
import sys


def load_log(bm):
    path = os.path.join("logs", bm.title, "log.pt")
    log = torch.load(path)
    return log


def get_best_test_loss(log, criterion):
    early_stop_index = get_best_index(log, criterion)
    print(early_stop_index)
    return log['Test Loss'][early_stop_index]


def get_best_test_accuracy(log, criterion):
    early_stop_index = get_best_index(log, criterion)
    print(early_stop_index)
    return log['Test Acc'][early_stop_index]


def get_best_index(log, criterion):
    if criterion.endswith('Loss'):
        print(np.argmin(log[criterion]))
        print(np.min(log[criterion]))
        return np.argmin(log[criterion])
    elif criterion.endswith('Acc'):
        print(np.argmax(log[criterion]))
        print(np.max(log[criterion]))
        return np.argmax(log[criterion])
    else:
        return -1


if __name__ == "__main__":
    sys.argv = ['main.py', '--ptb',
                '--student', 'template',
                '--data', '../Data/PTB',
                '--random-seed', '0',
                '--batch-size', '20',
                '--truncate', '80',
                '--num-workers', '0',
                '--num-internals', '128',
                '--num-graph-layers', '0',
                '--num-perms', '0',
                '--learning-rate', '0.01',
                '--weight-decay', '5e-5',
                '--clip', 'inf',
                '--num-epochs', '100',
                '--device', 'cpu',
                ]
    bm = BasicMain()
    print(bm.title)
    log = load_log(bm)
    print(log)
    for l in log:
        if len(log[l]):
            print(l, log[l])

    if bm.cora:
        criterion = 'Validate Acc'
    elif bm.ptb:
        criterion = 'Validate Loss'
    else:
        pass
    print(get_best_test_loss(log, criterion))
    print(get_best_test_accuracy(log, criterion))