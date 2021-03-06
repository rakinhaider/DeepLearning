from basic_main import BasicMain
import torch
import os
import pandas as pd
import sys


def get_best_criteria(bm, criterion="Train Loss"):
    path = os.path.join("logs", bm.title, "log.pt")
    log = torch.load(path)
    df = pd.DataFrame(log)
    return df.loc[100][criterion]


if __name__ == "__main__":
    for batch_size in [100, 500, 3000, 5000]:
        for learning_rate in [0.001, 0.0001, 0.00001]:
            sys.argv = ['', '--batch-size', str(batch_size),
                        '--learning-rate', str(learning_rate)]
            bm = BasicMain()
            # print(bm.title)
            c = get_best_criteria(bm)
            print(batch_size, learning_rate, c, sep = '\t\t')
        print()

    criterion = 'Train Acc'
    print(criterion)
    for batch_size in [100, 500, 3000, 5000]:
        for learning_rate in [0.001, 0.0001, 0.00001]:
            sys.argv = ['', '--batch-size', str(batch_size),
                        '--learning-rate', str(learning_rate)]
            bm = BasicMain()
            c = get_best_criteria(bm, criterion=criterion)
            print(batch_size, learning_rate, c, sep='\t\t')
        print()

    criterion = 'Test Acc'
    print(criterion)
    for batch_size in [100, 500, 3000, 5000]:
        for learning_rate in [0.001, 0.0001, 0.00001]:
            sys.argv = ['', '--batch-size', str(batch_size),
                        '--learning-rate', str(learning_rate)]
            bm = BasicMain()
            c = get_best_criteria(bm, criterion=criterion)
            print(batch_size, learning_rate, c, sep='\t\t')
        print()
