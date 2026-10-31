"""Layers that can't be drawn accurately are reported, never silently drawn wrong."""
import warnings

import torch
import torch.nn as nn

from neural_network_draw import DrawNN, UnsupportedModelError


def run():
    torch.manual_seed(4)
    model = nn.Sequential(
        nn.Linear(3, 5), nn.BatchNorm1d(5), nn.ReLU(), nn.Linear(5, 2),
    )

    # 1) default (strict): a clear error naming the problem
    try:
        DrawNN(model)
    except UnsupportedModelError as e:
        print("strict=True  ->", e, "\n")

    # 2) strict=False: warn, then draw the model WITHOUT the unsupported layer
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dnn = DrawNN(model, strict=False, title="BatchNorm skipped (strict=False)")
    print("strict=False -> warning:", caught[0].message, "\n")
    dnn.draw()


if __name__ == "__main__":
    run()
