"""Before vs after training: the diagram shows the REAL weights of the model.

A tiny network learns XOR. Compare the random initial weights (first window)
with the learned ones (second window).
"""
import torch
import torch.nn as nn

from neural_network_draw import DrawNN


def run():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid())
    labels = dict(input_labels=["x1", "x2"], output_labels=["XOR"])

    DrawNN(model, title="XOR network - BEFORE training", **labels).draw(show_weights=True)

    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(500):
        opt.zero_grad()
        nn.functional.binary_cross_entropy(model(X), y).backward()
        opt.step()

    DrawNN(model, title="XOR network - AFTER training", **labels).draw(show_weights=True)


if __name__ == "__main__":
    run()
