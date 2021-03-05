from visualize import Visualize
import os
import torch
import argparse
from models import StackCNN, MNISTClassification
from datasets import MNISTDataset
from homework.template.cnn import DualCNN
import sys
from main import Main

class VisualizeInterpolation(Visualize):
    def __init__(
        self,
        /,
    ) -> None:
        # Create training console arguments.
        self.console = argparse.ArgumentParser(
            description="Homework 2 (Visualization)",
        )
        self.console.add_argument(
            "--cnn",
            action="store_true",
            help="Render CNN interpolation.",
        )

        # Parse the command line arguments.
        self.args = self.console.parse_args()
        self.cnn = self.args.cnn

        # Visualize minibatch performance.
        if self.cnn:
            self.visualize_interpolation(
                cnn=True,
                num_samples=-1, random_seed=0,
                normalize=True, shuffle=False,
                kernel=5, stride=1, ginvariant=False,
                amprec=False,
                optim_alg="sgd", wd=0,
                batch_size=300,
                lr=0.01
            )

    def visualize_interpolation(self,
                                /,
                                **shared
                                ) -> None:
        shared['shuffle'] = True
        s_init, s_best = self.get_paths(**shared)
        self.plot_interpolation_curve(s_init, s_best,
                                      **shared)
        shared['shuffle'] = False
        init, best = self.get_paths(**shared)
        self.plot_interpolation_curve(init, best,
                                      **shared
                                      )

    def get_paths(self, **shared):
        init = os.path.join(
            "logs",
            self.get_title(
                **shared,
            ),
            "init.pt",
        )
        best = os.path.join(
            "logs",
            self.get_title(
                **shared,
            ),
            "best.pt",
        )

        return init, best

    def plot_interpolation_curve(self, init, best, **shared):
        init_model = torch.load(init)
        best_model = torch.load(best)
        keys = init_model.keys()

        for alpha in torch.arange(0, 1, 0.1):
            alpha_model = {}
            for k in keys:
                alpha_model[k] = alpha * init_model[k]
                alpha_model[k] += (1-alpha) * best_model[k]

            model = StackCNN(
                torch.Generator('cpu'), torch.Generator('cpu'), None,
                num_input_channels=1, num_output_channels=32,
                num_internal_channels=100,
                conv_kernel=shared['kernel'], conv_stride=shared['stride'],
                pool_kernel=3, pool_stride=1, padding=1,
                num_labels=MNISTDataset.NUM_LABELS, num_internals=100,
                height=MNISTDataset.HEIGHT, width=MNISTDataset.WIDTH,
                criterion=MNISTClassification.ACCURACY,
                dual_cnn=DualCNN,
                amprec=shared['amprec'],
            )

            # print(model)
            model.load_state_dict(alpha_model)
            self.evaluate_minibatch(model, loader)


if __name__ == "__main__":
    print(sys.argv)
    VisualizeInterpolation()
