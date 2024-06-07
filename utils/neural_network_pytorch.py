import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict
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


def training_loop(
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    model: NeuralNetwork,
    loss_function: torch.nn.modules.loss,
    optimizer: torch.optim,
    num_epochs: int,
) -> Dict[str, float]:
    metrics = {f"epoch{i+1}": {} for i in range(num_epochs)}
    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0
        train_preds = []
        train_labels = []
        for batch, labels in train_dataloader:
            optimizer.zero_grad()
            batch = torch.flatten(batch, start_dim=1)
            output = model(batch)
            train_loss = loss_function(output, labels)
            train_loss.backward()
            optimizer.step()
            train_loss_total += train_loss.item()
            _, preds = torch.max(output, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        metrics[f"epoch{epoch+1}"]["train_accuracy"] = np.round(
            accuracy_score(train_labels, train_preds), 2
        )
        metrics[f"epoch{epoch+1}"]["train_f1"] = np.round(
            f1_score(train_labels, train_preds, average="weighted"), 2
        )

        model.eval()
        test_loss_total = 0
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for batch, labels in test_dataloader:
                batch = torch.flatten(batch, start_dim=1)
                output = model(batch)
                test_loss = loss_function(output, labels)
                test_loss_total += test_loss.item()
                _, preds = torch.max(output, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        metrics[f"epoch{epoch+1}"]["test_accuracy"] = np.round(
            accuracy_score(test_labels, test_preds), 2
        )
        metrics[f"epoch{epoch+1}"]["test_f1"] = np.round(
            f1_score(test_labels, test_preds, average="weighted"), 2
        )
        print(
            f"epoch {epoch+1}/{num_epochs} | train loss: {train_loss_total/len(train_dataloader):.2f} | test loss: {test_loss_total/len(test_dataloader):.2f}"
        )
    return metrics


if __name__ == "__main__":
    activation_fn = nn.ReLU()
    model = NeuralNetwork(
        input_size=784, hidden_sizes=[128, 64], output_size=10, activation_function=activation_fn
    )
    print(model)
