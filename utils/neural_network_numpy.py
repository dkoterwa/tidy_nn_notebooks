import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation_function):
        self.activation_function = activation_function
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]

    def forward(self, x):
        self.z_values = []
        self.a_values = [x]
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(x, weight) + bias
            x = self.activation_function(z)
            self.z_values.append(z)
            self.a_values.append(x)

        # Output layer without activation function
        z = np.dot(x, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.a_values.append(z)
        return z

    def backward(self, x, y, learning_rate):
        # Forward pass
        output = self.forward(x)

        # Backward pass
        deltas = [output - y]
        print(f"Shape of deltas: {np.shape(deltas)}")
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(deltas[0], self.weights[i + 1].T) * self.activation_function.derivative(
                self.z_values[i]
            )
            deltas.insert(0, delta)

        # Update weights and biases
        batch_size = x.shape[0]
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.a_values[i].T, deltas[i]) / batch_size
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / batch_size


class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Define the neural network
    input_size = 2
    hidden_sizes = [2]
    output_size = 1
    activation_function = ReLU()
    learning_rate = 0.1

    # Initialize the neural network
    nn = NeuralNetwork(input_size, hidden_sizes, output_size, activation_function)

    # Perform a forward pass
    output = nn.forward(X)
    print("Forward pass output:")
    print(output)

    # Perform a backward pass
    nn.backward(X, y, learning_rate)

    # Print updated weights and biases
    print("Updated weights:")
    for weight in nn.weights:
        print(weight)
    print("Updated biases:")
    for bias in nn.biases:
        print(bias)
