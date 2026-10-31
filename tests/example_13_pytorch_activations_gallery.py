"""Activation modules are shown under their layer; no-op layers are skipped."""
import torch
import torch.nn as nn

from neural_network_draw import DrawNN


def run():
    torch.manual_seed(6)
    model = nn.Sequential(
        nn.Linear(3, 4), nn.LeakyReLU(), nn.Dropout(0.5),   # Dropout: ignored
        nn.Linear(4, 4), nn.ReLU(), nn.Sigmoid(),           # two activations -> "ReLU -> Sigmoid"
        nn.Linear(4, 3), nn.Identity(), nn.GELU(),          # Identity: ignored
        nn.Linear(3, 2), nn.Softmax(dim=1),
    )
    DrawNN(model, title="Activation gallery").draw()


if __name__ == "__main__":
    run()
