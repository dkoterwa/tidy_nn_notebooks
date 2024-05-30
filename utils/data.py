from typing import Tuple
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def load_mnist() -> Tuple[torchvision.datasets.mnist.MNIST, torchvision.datasets.mnist.MNIST]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist/", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist/", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def load_mnist_dataloaders(
    train_dataset: torchvision.datasets.mnist.MNIST,
    test_dataset: torchvision.datasets.mnist.MNIST,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
