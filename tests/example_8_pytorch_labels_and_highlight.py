"""A PyTorch model with feature names, class names and a highlighted path."""
import torch
import torch.nn as nn

from neural_network_draw import DrawNN


class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.act1 = nn.Linear(4, 6), nn.ReLU()
        self.fc2, self.act2 = nn.Linear(6, 6), nn.Tanh()
        self.out, self.softmax = nn.Linear(6, 3), nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.softmax(self.out(x))


def run():
    torch.manual_seed(0)
    DrawNN(
        IrisNet(),
        input_labels=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
        output_labels=["Setosa", "Versicolor", "Virginica"],
        highlight_path=[[0], [1, 2], [0], [1]],
        title="Iris classifier (from a PyTorch model)",
    ).draw(show_weights=False)


if __name__ == "__main__":
    run()
