from neural_network_draw import NeuralNetwork


def run():
    network = NeuralNetwork(12)
    network.add_layer(4)
    network.add_layer(8)
    network.add_layer(6)
    network.add_layer(10)
    network.add_layer(3)
    network.draw()


if __name__ == "__main__":
    run()