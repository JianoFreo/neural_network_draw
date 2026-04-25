try:
    from ._deps import plt
except ImportError:
    from _deps import plt

try:
    from .layer import Layer
except ImportError:
    from layer import Layer


class NeuralNetwork:
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self):
        plt.figure(figsize=(10, 6))

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                layer.draw(-1)
            else:
                layer.draw(i)

        plt.axis('scaled')
        plt.axis('off')
        plt.title('Neural Network Architecture', fontsize=15)
        plt.show()
