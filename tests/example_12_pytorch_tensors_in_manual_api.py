"""Tensors work in the original manual APIs too (CPU or CUDA, even with requires_grad)."""
import torch

from neural_network_draw import DrawNN, NeuralNetwork


def run():
    torch.manual_seed(5)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Low-level API: tensors as weights / biases. Layout is [previous_size][layer_size].
    network = NeuralNetwork(4, title=f"Low-level API with tensors ({device})")
    network.add_layer(3, labels=["a", "b", "c"])
    network.add_layer(4, weights=torch.randn(3, 4, device=device, requires_grad=True),
                      biases=torch.randn(4, device=device), activation="ReLU")
    network.add_layer(2, weights=torch.randn(4, 2, device=device),
                      biases=torch.zeros(2, device=device), activation="Softmax",
                      labels=["Yes", "No"])
    network.draw(show_weights=True)

    # High-level API: a list of tensors, None for "no biases" on the input layer.
    DrawNN([3, 4, 2],
           weights=[torch.randn(3, 4), torch.randn(4, 2)],
           biases=[None, torch.randn(4), torch.randn(2)],
           activations=[None, "Tanh", "Sigmoid"],
           title="High-level API with tensors").draw(show_weights=True)


if __name__ == "__main__":
    run()
