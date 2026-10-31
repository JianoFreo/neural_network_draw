import os
import sys

import matplotlib

matplotlib.use("Agg")  # never open a window during tests

# make `import neural_network_draw` work without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
