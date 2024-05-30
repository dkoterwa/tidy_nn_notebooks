import torch.nn as nn
import torch


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


def training_loop(train_dataloader, test_dataloader, model, loss_function, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for batch, labels in train_dataloader:
            optimizer.zero_grad()
            batch = torch.flatten(batch, start_dim=1)
            output = model(batch)
            train_loss = loss_function(output, labels)
            train_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch, labels in test_dataloader:
                batch = torch.flatten(batch, start_dim=1)
                output = model(batch)
                test_loss = loss_function(output, labels)
        print(f"epoch {epoch+1}/{num_epochs} | train loss: {train_loss} | test loss: {test_loss}")


if __name__ == "__main__":
    activation_fn = nn.ReLU()
    model = NeuralNetwork(
        input_size=784, hidden_sizes=[128, 64], output_size=10, activation_function=activation_fn
    )
    print(model)
