from basic_main import BasicMain
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

def visualize_cnn(bm):
    path = os.path.join('logs',
                        bm.title,
                        'log.pt')
    log = torch.load(path)
    print(path)
    df = pd.DataFrame(log)
    df = df[1:]

    plt.plot(range(1, bm.num_epochs+1), df['Train Acc'], label='Train')
    plt.plot(range(1, bm.num_epochs+1), df['Test Acc'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    fname = 'cnn_s_acc.pdf' if bm.shuffle else 'cnn_acc.pdf'
    plt.savefig('images/' + fname, format='pdf')


if __name__ == "__main__":
    bm = BasicMain()
    visualize_cnn(bm)