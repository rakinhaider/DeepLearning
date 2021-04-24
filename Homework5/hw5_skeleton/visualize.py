import matplotlib.pyplot as plt
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg-str')
    parser.add_argument('--kernel-sigma')
    args = parser.parse_args()

    dir_name = '_'.join(['MMD', str(args.reg_str), str(args.kernel_sigma)])
    file_path = os.path.join('logs', dir_name, 'output')
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('[Epoch'):
                splits = line.split()
                # print(splits[5], splits[12], splits[15])
                results.append([float(splits[5]), float(splits[12]), float(splits[15])])

    # Loss plot
    plt.plot(range(len(results)), [t[0] for t in results], label='Training Loss')
    plt.legend()
    plot_path = os.path.join('image', dir_name, 'loss.pdf')
    plt.savefig(plot_path, format='pdf')
    plt.show()

    # Accuracy Plot
    plt.plot(range(len(results)), [t[1] for t in results], label='Source Accuracy')
    plt.plot(range(len(results)), [t[2] for t in results], label='Target Accuracy')
    plt.legend()
    plot_path = os.path.join('image', dir_name, 'acc.pdf')
    plt.savefig(plot_path, format='pdf')
    plt.show()