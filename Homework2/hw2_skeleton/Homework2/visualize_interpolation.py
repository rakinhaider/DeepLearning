from basic_main import BasicMain
import torch
import math
import os
import sys
import argparse
from models import MNISTClassification
import matplotlib.pyplot as plt


class VisualizeInterpolation(BasicMain):
    def __init__(
            self,
            /,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----

        Returns
        -------
        """

        super(VisualizeInterpolation, self).__init__()

        # Prepare datasets.
        if self.debug:
            temp = self.num_samples
            self.num_samples = 200
        self.load_datasets()
        if self.debug:
            self.num_samples = temp

        # Prepare model.
        self.prepare_model()

        self.visualize_interpolation()

    def visualize_interpolation(self,
                                /,
                                ) -> None:
        self.shuffle = True
        s_init, s_best = self.get_paths()
        self.plot_interpolation_curve(s_init, s_best)
        self.shuffle = False
        init, best = self.get_paths()
        self.plot_interpolation_curve(init, best)
        plt.title('Linear Interpolation of CNN on MNIST')
        plt.xlabel('Interpolation Parameter \u03B1')
        plt.ylabel('Loss Function L(\u03F4)')
        plt.legend()
        # plt.show()
        plt.savefig('images/interpolation.pdf', format='pdf')
        print(s_init, s_best, init, best)

    def get_paths(self):
        self.generate_title()
        print(self.title)
        init = os.path.join(
            "logs",
            self.title,
            "init.pt",
        )
        best = os.path.join(
            "logs",
            self.title,
            "best.pt",
        )

        return init, best

    def plot_interpolation_curve(self, init, best):
        init_model = torch.load(init)
        best_model = torch.load(best)
        keys = init_model.keys()

        self.model.criterion = MNISTClassification.LOSS

        interpolate_loss = []
        alphas = list(torch.arange(0, 1.8, 0.1))
        for alpha in alphas:
            alpha_model = {}
            for k in keys:
                alpha_model[k] = (1 - alpha) * init_model[k]
                alpha_model[k] += alpha * best_model[k]

            self.model.load_state_dict(alpha_model)
            loss, _ = self.evaluate_minibatch(self.loader_valid_raw)
            interpolate_loss.append(loss)

        # print(alphas, interpolate_loss)
        label = "Shuffled" if self.shuffle else "Raw"
        plt.plot(alphas, interpolate_loss, label=label)


if __name__ == "__main__":
    # sys.argv = ['visualize_interpolation.py', '--cnn', '--debug']
    # print(sys.argv)
    VisualizeInterpolation()
