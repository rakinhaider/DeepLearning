import wget
from wget import bar_adaptive
import pandas as pd
import os

if __name__ == "__main__":
    f = open('dataset_urls.tsv', 'r')
    for line in f:
        splits = line.split()
        name = splits[0]
        dir = splits[1]
        subfolder = splits[2]
        dir_path = os.path.join('../Data/', subfolder)
        os.makedirs(dir_path, exist_ok=True)
        if os.path.exists(dir_path + name):
            os.remove(dir_path + name)
        filename = wget.download(dir+name, out= dir_path + name,
                                 bar=bar_adaptive)
