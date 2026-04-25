try:
    from .neuralNetwork import NeuralNetwork
except ImportError:
    from neuralNetwork import NeuralNetwork


class DrawNN:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def draw(self):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer)

        for layer_size in self.neural_network:
            network.add_layer(layer_size)

        network.draw()
