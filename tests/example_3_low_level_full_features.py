from neural_network_draw import NeuralNetwork


def run():
    network = NeuralNetwork(4, title="Handwritten Digit Classifier (Simplified)")

    network.add_layer(3, labels=["pixel_1", "pixel_2", "pixel_3"])

    network.add_layer(
        4,
        weights=[
            [0.10, -0.20, 0.30, 0.40],
            [0.50, 0.60, -0.70, 0.80],
            [-0.10, 0.20, 0.30, -0.40],
        ],
        biases=[0.10, 0.20, 0.30, 0.40],
        activation="ReLU",
    )

    network.add_layer(
        2,
        weights=[
            [0.20, -0.30],
            [0.10, 0.40],
            [0.50, -0.60],
            [0.20, 0.10],
        ],
        biases=[0.05, -0.05],
        activation="Softmax",
        labels=["Cat", "Dog"],
    )

    network.draw(show_weights=True)


if __name__ == "__main__":
    run()