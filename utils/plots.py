from matplotlib import pyplot as plt
from typing import Dict, List
from math import cos, sin, atan


def create_key_from_size(hidden_sizes):
    return ", ".join([f"hidden{i+1}: {size}" for i, size in enumerate(hidden_sizes)])


def plot_metric_for_multiple_models(
    metrics: Dict[str, Dict[str, float]], metric_names: List[str], num_training_epochs: int
) -> None:
    assert all(
        metric_name
        in [
            "train_loss",
            "test_loss",
            "train_accuracy",
            "test_accuracy",
            "train_f1",
            "test_f1",
        ]
        for metric_name in metric_names
    ), "Incorrect metric name(s) passed as parameter"

    num_metrics = len(metric_names)
    fig, axs = plt.subplots(ncols=2, figsize=(14, 6))

    for i, metric_name in enumerate(metric_names):
        ax = axs[i] if num_metrics > 1 else axs  # Handle single metric case
        for model_name, model_metrics in metrics.items():
            metric_values = [
                model_metrics[f"epoch{epoch}"][metric_name]
                for epoch in range(1, num_training_epochs + 1)
            ]
            ax.plot(
                range(1, num_training_epochs + 1),
                metric_values,
                label=model_name,
                marker="o",
                markersize=3,
            )
        ax.set_xlabel("Epochs")
        ax.set_title(f"{metric_name} across epochs")
        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
        ax.set_xlim(1, num_training_epochs)
        ax.set_xticks(ticks=range(1, num_training_epochs + 1))
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# taken from https://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network
class Neuron:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = plt.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        plt.gca().add_patch(circle)


class Layer:
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return (
            self.horizontal_distance_between_neurons
            * (self.number_of_neurons_in_widest_layer - number_of_neurons)
            / 2
        )

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = plt.Line2D(
            (neuron1.x - x_adjustment, neuron2.x + x_adjustment),
            (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
        )
        plt.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw(self.neuron_radius)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            plt.text(x_text, self.y, "Hidden Layer 1", fontsize=12)
        elif layerType == -1:
            plt.text(x_text, self.y, "Output Layer", fontsize=12)
        else:
            plt.text(x_text, self.y, "Hidden Layer " + str(layerType + 1), fontsize=12)


class NeuralNetwork:
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self):
        plt.figure(figsize=(16, 10))
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                i = -1
            layer.draw(i)
        plt.axis("scaled")
        plt.axis("off")
        plt.title("Neural Network architecture", fontsize=15)
        plt.show()


class DrawNN:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def draw(self):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer)
        for l in self.neural_network:
            network.add_layer(l)
        network.draw()
