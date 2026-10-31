"""
Automated tests for the PyTorch support AND for backward compatibility with the
original list-based API. Run from the repo root:

    pip install -e . pytest torch
    pytest tests/test_pytorch_support.py
"""
import matplotlib

matplotlib.use("Agg")  # no window pops up during tests

import matplotlib.pyplot as plt
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from neural_network_draw import DrawNN, NeuralNetwork, draw_model  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def small_model():
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.Softmax(dim=1))


# ---------------------------------------------------------------- extraction
def test_sizes_activations_and_title_from_model():
    d = DrawNN(small_model())
    assert d.neural_network == [3, 4, 2]
    assert d.activations == [None, "ReLU", "Softmax"]
    assert d.title == "Sequential Architecture"


def test_weights_are_transposed_to_in_out_layout():
    model = small_model()
    d = DrawNN(model)
    # torch: [out][in]  ->  library: [in][out]
    assert len(d.weights[0]) == 3 and len(d.weights[0][0]) == 4
    assert d.weights[0][2][1] == pytest.approx(model[0].weight[1, 2].item())
    assert d.biases[0] is None
    assert d.biases[1] == pytest.approx(model[0].bias.tolist())


def test_values_are_plain_python_floats():
    d = DrawNN(small_model())
    assert isinstance(d.weights[0][0][0], float)
    assert isinstance(d.biases[1][0], float)


def test_linear_without_bias():
    d = DrawNN(nn.Sequential(nn.Linear(2, 3, bias=False)))
    assert d.biases == [None, None]


def test_custom_module_and_nested_containers():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(4, 5), nn.Sigmoid(), nn.Dropout(0.1))
            self.head = nn.Linear(5, 2)

        def forward(self, x):
            return self.head(self.body(x))

    d = DrawNN(Net())
    assert d.neural_network == [4, 5, 2]
    assert d.activations == [None, "Sigmoid", None]  # Dropout ignored
    assert d.title == "Net Architecture"


def test_trained_weights_with_grad_and_after_backward():
    model = small_model()  # parameters require grad
    model(torch.randn(5, 3)).sum().backward()
    DrawNN(model).draw(show_weights=True)  # must not raise


def test_user_arguments_override_model_values():
    d = DrawNN(small_model(), title="Mine", activations=[None, "Tanh", "Sigmoid"],
               input_labels=["a", "b", "c"])
    assert d.title == "Mine"
    assert d.activations == [None, "Tanh", "Sigmoid"]
    assert d.input_labels == ["a", "b", "c"]


# ------------------------------------------------------------------- errors
def test_model_without_linear_layers_raises():
    with pytest.raises(ValueError, match="No nn.Linear"):
        DrawNN(nn.Sequential(nn.ReLU()))


def test_non_chained_linear_layers_raise():
    class Branchy(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 3)
            self.b = nn.Linear(5, 2)  # doesn't follow a's output

    with pytest.raises(ValueError, match="feed-forward chain"):
        DrawNN(Branchy())


def test_lazy_linear_must_be_initialised():
    with pytest.raises(ValueError, match="lazy"):
        DrawNN(nn.Sequential(nn.LazyLinear(3)))


# ------------------------------------------------------------ drawing paths
def test_draw_with_model_all_entry_points():
    model = small_model()
    DrawNN(model).draw()
    DrawNN.from_pytorch(model).draw(show_weights=True)
    draw_model(model, show_weights=True, input_labels=["x1", "x2", "x3"],
               output_labels=["Cat", "Dog"])
    net = NeuralNetwork.from_pytorch(model)
    net.set_highlight_path([[0], [1, 2], [0]])
    net.draw(show_weights=True)


def test_neuralnetwork_from_pytorch_matches_model():
    net = NeuralNetwork.from_pytorch(small_model())
    assert [s["size"] for s in net.layer_specs] == [3, 4, 2]
    assert net.layer_specs[1]["activation"] == "ReLU"
    assert net.title == "Sequential Architecture"


def test_from_pytorch_rejects_non_models():
    with pytest.raises(TypeError):
        NeuralNetwork.from_pytorch([3, 4, 2])


# --------------------------------------- raw tensors in the *old* API as well
def test_tensors_accepted_in_low_level_api():
    net = NeuralNetwork(4)
    net.add_layer(3)
    net.add_layer(4, weights=torch.randn(3, 4), biases=torch.zeros(4), activation="ReLU")
    net.draw(show_weights=True)
    assert isinstance(net.layer_specs[1]["weights"], list)


def test_tensors_accepted_in_high_level_api():
    DrawNN([3, 4], weights=[torch.randn(3, 4)], biases=[None, torch.zeros(4)]).draw(
        show_weights=True)


# ------------------------------------------------ original API still works
def test_original_list_api_unchanged():
    DrawNN([4, 6, 3]).draw()
    DrawNN([3, 4, 2],
           weights=[[[0.1] * 4] * 3, [[0.2] * 2] * 4],
           biases=[None, [0.1] * 4, [0.2] * 2],
           activations=[None, "ReLU", "Softmax"],
           input_labels=["x1", "x2", "x3"], output_labels=["Cat", "Dog"],
           highlight_path=[[0], [1, 2], [0]]).draw(show_weights=True)
    assert DrawNN([3, 4, 2]).title == "Neural Network Architecture"


def test_original_low_level_api_unchanged():
    net = NeuralNetwork(4)
    net.add_layer(3, labels=["a", "b", "c"])
    net.add_layer(4, weights=[[0.1, -0.2, 0.3, 0.4]] * 3, biases=[0.1] * 4, activation="ReLU")
    net.draw(show_weights=True)
