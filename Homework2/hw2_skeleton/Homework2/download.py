# All imports
import torchvision
import torch


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Download >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


def download_mnist() -> None:
    r"""
    Download MNIST data.

    Args
    ----

    Returns
    -------
    """
    # /
    # ANNOTATE
    # /
    train: torchvision.datasets.MNIST
    test: torchvision.datasets.MNIST

    # Download.
    train = torchvision.datasets.MNIST(
        "../Data/MNIST",
        train=True, download=True,
    )
    test = torchvision.datasets.MNIST(
        "../Data/MNIST",
        train=False, download=True,
    )

    # Get tensors.
    train_images, train_labels = train.data, train.targets
    test_images, test_labels = test.data, test.targets

    # Ensure tensor types.
    train_images = train_images.long()
    train_labels = train_labels.long()
    test_images = test_images.long()
    test_labels = test_labels.long()

    # Compress and integrate data.
    assert (
        torch.all(train_images.to(torch.uint8).long() == train_images)
        and torch.all(test_images.to(torch.uint8).long() == test_images)
        and torch.all(train_labels.to(torch.uint8).long() == train_labels)
        and torch.all(test_labels.to(torch.uint8).long() == test_labels)
    ), "[\033[91mError\033[0m]: MNIST data can not be compressed."
    torch.save(
        (
            (train_images.to(torch.uint8), train_labels.to(torch.uint8)),
            (test_images.to(torch.uint8), test_labels.to(torch.uint8)),
        ),
        "../Data/mnist.ptds",
    )


# Run.
if (__name__ == "__main__"):
    download_mnist()
else:
    pass