"""The original list-based APIs and the original example scripts must keep working
exactly as before (and without PyTorch installed)."""
import glob
import os
import runpy
import subprocess
import sys

import pytest

from neural_network_draw import DrawNN, Layer, Neuron, NeuralNetwork

HERE = os.path.dirname(__file__)
EXAMPLES = sorted(glob.glob(os.path.join(HERE, "example_*.py")))


@pytest.mark.parametrize("path", EXAMPLES, ids=[os.path.basename(p) for p in EXAMPLES])
def test_every_example_script_runs(path):
    if "pytorch" in os.path.basename(path):
        pytest.importorskip("torch")  # the PyTorch examples need torch; the others never do
    runpy.run_path(path)["run"]()


def test_original_examples_present():
    names = {os.path.basename(p) for p in EXAMPLES}
    for n in ("example_1_basic_structure.py", "example_2_structure_with_labels.py",
              "example_3_low_level_full_features.py", "example_4_high_level_full_features.py"):
        assert n in names


def test_public_names_unchanged():
    assert all(callable(c) for c in (Neuron, Layer, NeuralNetwork, DrawNN))


def test_low_level_api_with_everything():
    net = NeuralNetwork(4, title="t")
    net.add_layer(3, labels=["a", "b", "c"])
    net.add_layer(4, weights=[[0.1, -0.2, 0.3, 0.4]] * 3, biases=[0.1] * 4, activation="ReLU")
    net.add_layer(2, weights=[[0.2, -0.3]] * 4, biases=[0.05, -0.05], activation="Softmax",
                  labels=["Cat", "Dog"])
    net.set_highlight_path([[0], [1, 2], [0]])
    net.draw(show_weights=True)
    assert [s["size"] for s in net.layer_specs] == [3, 4, 2]
    assert net.layer_specs[1]["weights"] == [[0.1, -0.2, 0.3, 0.4]] * 3  # values untouched
    assert net.layer_specs[1]["activation"] == "ReLU"


def test_high_level_api_with_everything():
    d = DrawNN([3, 4, 2],
               weights=[[[0.1] * 4] * 3, [[0.2] * 2] * 4],
               biases=[None, [0.1] * 4, [0.2] * 2],
               activations=[None, "ReLU", "Softmax"],
               input_labels=["x1", "x2", "x3"], output_labels=["Cat", "Dog"],
               title="Custom", highlight_path=[[0], [1, 2], [0]])
    d.draw(show_weights=True, figsize=(8, 8))
    assert d.title == "Custom" and d.neural_network == [3, 4, 2]


def test_structure_only_and_defaults():
    d = DrawNN([4, 6, 3])
    assert d.title == "Neural Network Architecture"
    assert d.weights is None and d.biases is None and d.activations is None
    d.draw()


def test_numpy_and_tuples_still_fine_in_old_api():
    np = pytest.importorskip("numpy")
    DrawNN((3, 2), weights=[np.ones((3, 2))], biases=[None, np.zeros(2)]).draw(show_weights=True)


def test_library_imports_and_draws_without_torch_installed():
    """Simulate a machine with no PyTorch: `import torch` must fail, library must still work."""
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import matplotlib; matplotlib.use('Agg')\n"
        "from neural_network_draw import DrawNN, NeuralNetwork\n"
        "DrawNN([3, 4, 2], weights=[[[0.1]*4]*3, [[0.2]*2]*4]).draw(show_weights=True)\n"
        "n = NeuralNetwork(3); n.add_layer(3); n.add_layer(2); n.draw()\n"
        "print('ok')\n"
    )
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_short_activation_labels_keep_original_spacing_long_ones_get_room():
    def spacing(activation):
        net = NeuralNetwork(4)
        net.add_layer(3)
        net.add_layer(4, activation=activation)
        net.draw()
        return net.layers[1].horizontal_distance_between_neurons

    assert spacing("ReLU") == 2.0          # unchanged from before
    assert spacing("Softmax") == 2.0
    assert spacing("ReLU → Sigmoid") > 2.0  # stacked activations get extra room
