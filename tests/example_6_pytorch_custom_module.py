"""Draw a custom nn.Module (works with the one-liner draw_model too)."""
import torch.nn as nn

from neural_network_draw import DrawNN, draw_model


class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(6, 6)
        self.act2 = nn.Tanh()
        self.out = nn.Linear(6, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.softmax(self.out(x))


def run():
    model = IrisNet()

    # Option 1: structure only
    draw_model(model)

    # Option 2: with labels, weights and a highlighted path.
    # Anything you pass explicitly is added on top of what's read from the model.
    DrawNN(
        model,
        input_labels=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
        output_labels=["Setosa", "Versicolor", "Virginica"],
        highlight_path=[[0], [1, 2], [0], [1]],
    ).draw(show_weights=True)


if __name__ == "__main__":
    run()
