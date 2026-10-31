"""Nested modules: blocks inside a model, Sequential inside Sequential, ModuleList."""
import torch
import torch.nn as nn

from neural_network_draw import DrawNN


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(0.1))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Block(5, 7), Block(7, 4))   # nested containers
        self.head = nn.ModuleList([nn.Linear(4, 3), nn.Sigmoid()])


def run():
    torch.manual_seed(2)
    DrawNN(Net(), title="Nested modules (Dropout is ignored)").draw()


if __name__ == "__main__":
    run()
