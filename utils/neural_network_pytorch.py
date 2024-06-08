import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple
import torch.utils


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_function):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.activation_function = activation_function

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_function(layer(x))
        x = self.layers[-1](x)
        return x


def train_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    labels: torch.Tensor,
    loss_function: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, np.ndarray, np.ndarray]:
    optimizer.zero_grad()
    batch = torch.flatten(batch, start_dim=1)
    output = model(batch)
    train_loss = loss_function(output, labels)
    train_loss.backward()
    optimizer.step()
    _, preds = torch.max(output, 1)
    return train_loss.item(), preds.cpu().numpy(), labels.cpu().numpy()


def test_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    labels: torch.Tensor,
    loss_function: torch.nn.modules.loss._Loss,
) -> Tuple[float, np.ndarray, np.ndarray]:
    batch = torch.flatten(batch, start_dim=1)
    output = model(batch)
    test_loss = loss_function(output, labels)
    _, preds = torch.max(output, 1)
    return test_loss.item(), preds.cpu().numpy(), labels.cpu().numpy()


def training_loop(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: torch.nn.Module,
    loss_function: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> Dict[str, Dict[str, float]]:
    metrics = {f"epoch{i+1}": {} for i in range(num_epochs)}
    progress_bar = tqdm(range(num_epochs), desc="Training progress")

    for epoch in progress_bar:
        model.train()
        train_loss_total = 0
        train_preds = []
        train_labels = []

        for batch, labels in train_dataloader:
            train_loss, preds, labels = train_batch(model, batch, labels, loss_function, optimizer)
            train_loss_total += train_loss
            train_preds.extend(preds)
            train_labels.extend(labels)

        train_loss_avg = train_loss_total / len(train_dataloader)
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="weighted")

        metrics[f"epoch{epoch+1}"]["train_loss"] = np.round(train_loss_avg, 2)
        metrics[f"epoch{epoch+1}"]["train_accuracy"] = np.round(train_accuracy, 2)
        metrics[f"epoch{epoch+1}"]["train_f1"] = np.round(train_f1, 2)

        model.eval()
        test_loss_total = 0
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for batch, labels in test_dataloader:
                test_loss, preds, labels = test_batch(model, batch, labels, loss_function)
                test_loss_total += test_loss
                test_preds.extend(preds)
                test_labels.extend(labels)

        test_loss_avg = test_loss_total / len(test_dataloader)
        test_accuracy = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="weighted")

        metrics[f"epoch{epoch+1}"]["test_loss"] = np.round(test_loss_avg, 2)
        metrics[f"epoch{epoch+1}"]["test_accuracy"] = np.round(test_accuracy, 2)
        metrics[f"epoch{epoch+1}"]["test_f1"] = np.round(test_f1, 2)

        progress_bar.set_description(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss_avg:.2f} | Test Loss: {test_loss_avg:.2f}"
        )

    return metrics


if __name__ == "__main__":
    activation_fn = nn.ReLU()
    model = NeuralNetwork(
        input_size=784, hidden_sizes=[128, 64], output_size=10, activation_function=activation_fn
    )
    print(model)
