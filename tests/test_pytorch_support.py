"""
Tests for the PyTorch integration. Run from the repo root:

    pip install -e . pytest torch
    pytest tests/
"""
import copy
import warnings

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from neural_network_draw import (  # noqa: E402
    DrawNN, NeuralNetwork, UnsupportedModelError, draw_model, extract_architecture,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def small_model():
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.Softmax(dim=1))


# ============================================================ model inspection
class TestInspection:
    def test_sizes_activations_title(self):
        d = DrawNN(small_model())
        assert d.neural_network == [3, 4, 2]
        assert d.activations == [None, "ReLU", "Softmax"]
        assert d.title == "Sequential Architecture"

    def test_issue_example_api(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3), nn.Softmax(dim=1))
        dnn = DrawNN.from_pytorch(model)
        dnn.draw(show_weights=True)
        assert dnn.neural_network == [4, 8, 3]

    @pytest.mark.parametrize("act", [
        nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU, nn.GELU, nn.SiLU, nn.Softplus,
        nn.LogSoftmax, nn.Hardtanh, nn.PReLU, nn.Mish,
    ])
    def test_activation_names(self, act):
        a = act(dim=1) if act is nn.LogSoftmax else act()
        d = DrawNN(nn.Sequential(nn.Linear(2, 3), a))
        assert d.activations == [None, act.__name__]

    def test_softmax_dim_argument(self):
        assert DrawNN(nn.Sequential(nn.Linear(2, 3), nn.Softmax(dim=-1))).activations[-1] == "Softmax"

    def test_two_activations_on_one_layer_are_both_shown(self):
        d = DrawNN(nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Sigmoid()))
        assert d.activations == [None, "ReLU → Sigmoid"]

    def test_activation_subclass_keeps_standard_name(self):
        class MyReLU(nn.ReLU):
            pass
        assert DrawNN(nn.Sequential(nn.Linear(2, 3), MyReLU())).activations[-1] == "ReLU"

    @pytest.mark.parametrize("noop", [nn.Dropout(0.3), nn.Dropout1d(), nn.AlphaDropout(),
                                      nn.Identity(), nn.Flatten()])
    def test_noop_layers_are_ignored(self, noop):
        d = DrawNN(nn.Sequential(nn.Linear(2, 3), noop, nn.ReLU(), nn.Linear(3, 1)))
        assert d.neural_network == [2, 3, 1]
        assert d.activations == [None, "ReLU", None]

    def test_deep_and_wide_configurations(self):
        layers = []
        sizes = [5, 9, 7, 7, 3, 1]
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        d = DrawNN(nn.Sequential(*layers))
        assert d.neural_network == sizes and d.activations == [None] + ["Tanh"] * 5

    def test_single_linear_model(self):
        d = DrawNN(nn.Linear(3, 2))
        assert d.neural_network == [3, 2]
        assert d.title == "Linear Architecture"

    def test_linear_without_bias(self):
        assert DrawNN(nn.Sequential(nn.Linear(2, 3, bias=False))).biases == [None, None]

    def test_layer_reused_in_sequential_is_drawn_twice(self):
        shared = nn.Linear(3, 3)
        d = DrawNN(nn.Sequential(shared, nn.ReLU(), shared))
        assert d.neural_network == [3, 3, 3]
        assert d.weights[0] == d.weights[1]


# ===================================================== nested / custom modules
class TestNestedAndCustom:
    def test_custom_module(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1, self.act, self.fc2 = nn.Linear(4, 5), nn.Sigmoid(), nn.Linear(5, 2)

            def forward(self, x):
                return self.fc2(self.act(self.fc1(x)))

        d = DrawNN(Net())
        assert d.neural_network == [4, 5, 2]
        assert d.activations == [None, "Sigmoid", None]
        assert d.title == "Net Architecture"

    def test_nested_sequentials_and_modules(self):
        class Block(nn.Module):
            def __init__(self, i, o):
                super().__init__()
                self.seq = nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Dropout(0.1))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.b1, self.b2 = Block(4, 6), Block(6, 3)
                self.head = nn.Sequential(nn.Linear(3, 2), nn.Softmax(dim=1))

        d = DrawNN(Net())
        assert d.neural_network == [4, 6, 3, 2]
        assert d.activations == [None, "ReLU", "ReLU", "Softmax"]

    def test_module_list(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(3, 4), nn.Linear(4, 2)])

        assert DrawNN(Net()).neural_network == [3, 4, 2]

    def test_empty_container_is_harmless(self):
        assert DrawNN(nn.Sequential(nn.Linear(2, 2), nn.Sequential())).neural_network == [2, 2]


# ================================================= execution-order tracing
class TestExampleInput:
    class OutOfOrder(nn.Module):
        """Layers are *defined* in the wrong order; forward uses them correctly."""
        def __init__(self):
            super().__init__()
            self.second = nn.Linear(5, 2)
            self.relu = nn.ReLU()
            self.first = nn.Linear(3, 5)

        def forward(self, x):
            return self.second(self.relu(self.first(x)))

    def test_registration_order_is_detected_as_inconsistent(self):
        with pytest.raises(UnsupportedModelError, match="example_input"):
            DrawNN(self.OutOfOrder())

    def test_example_input_uses_real_execution_order(self):
        d = DrawNN(self.OutOfOrder(), example_input=torch.zeros(1, 3))
        assert d.neural_network == [3, 5, 2]
        assert d.activations == [None, "ReLU", None]

    def test_unused_layers_are_not_drawn(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.unused, self.b = nn.Linear(3, 4), nn.Linear(9, 9), nn.Linear(4, 2)

            def forward(self, x):
                return self.b(self.a(x))

        assert DrawNN(Net(), example_input=torch.zeros(1, 3)).neural_network == [3, 4, 2]

    def test_tuple_of_args_accepted(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 2)

            def forward(self, x, scale):
                return self.fc(x) * scale

        assert DrawNN(Net(), example_input=(torch.zeros(1, 3), 2.0)).neural_network == [3, 2]

    def test_tracing_does_not_change_model_state(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(0.5), nn.Linear(4, 2))
        model.train()
        model[2].eval()  # deliberately mixed flags
        flags = [m.training for m in model.modules()]
        before = copy.deepcopy(model.state_dict())  # includes BatchNorm running stats
        with pytest.raises(UnsupportedModelError):  # BatchNorm unsupported ...
            DrawNN(model, example_input=torch.randn(4, 3))
        assert [m.training for m in model.modules()] == flags  # ... but state is restored
        after = model.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)  # running stats untouched
        assert not any(h for m in model.modules() for h in m._forward_hooks.values())  # hooks removed

    def test_tracing_records_no_gradients(self):
        model = small_model()
        DrawNN(model, example_input=torch.randn(2, 3))
        assert all(p.grad is None for p in model.parameters())

    def test_failed_forward_still_cleans_up(self):
        model = small_model()
        with pytest.raises(RuntimeError):
            DrawNN(model, example_input=torch.zeros(1, 99))  # wrong shape
        assert not any(m._forward_hooks for m in model.modules())
        assert model.training


# ===================================================== weights / bias / values
class TestValues:
    def test_weights_transposed_to_in_out_layout(self):
        model = small_model()
        d = DrawNN(model)
        assert len(d.weights) == 2
        assert (len(d.weights[0]), len(d.weights[0][0])) == (3, 4)
        assert (len(d.weights[1]), len(d.weights[1][0])) == (4, 2)
        for i in range(3):
            for j in range(4):
                assert d.weights[0][i][j] == pytest.approx(model[0].weight[j, i].item())

    def test_biases(self):
        model = small_model()
        d = DrawNN(model)
        assert d.biases[0] is None
        assert d.biases[1] == pytest.approx(model[0].bias.tolist())
        assert d.biases[2] == pytest.approx(model[2].bias.tolist())

    def test_known_values(self):
        lin = nn.Linear(2, 2)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor([[1., 2.], [3., 4.]]))  # [out][in]
            lin.bias.copy_(torch.tensor([.5, -.5]))
        d = DrawNN(nn.Sequential(lin))
        assert d.weights[0] == [[1., 3.], [2., 4.]]  # [in][out]
        assert d.biases[1] == [.5, -.5]

    def test_values_are_plain_python_floats(self):
        d = DrawNN(small_model())
        assert type(d.weights[0][0][0]) is float
        assert type(d.biases[1][0]) is float

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
    def test_dtypes(self, dtype):
        d = DrawNN(small_model().to(dtype))
        assert type(d.weights[0][0][0]) is float
        d.draw(show_weights=True)

    def test_extracted_values_are_copies(self):
        model = small_model()
        d = DrawNN(model)
        before = model[0].weight.detach().clone()
        d.weights[0][0][0] = 123.0
        assert torch.equal(model[0].weight, before)


# ============================================================ tensor handling
class TestTensors:
    @pytest.mark.parametrize("device", DEVICES)
    def test_model_on_device(self, device):
        model = small_model().to(device)
        d = DrawNN(model)
        assert d.neural_network == [3, 4, 2]
        assert type(d.weights[0][0][0]) is float
        d.draw(show_weights=True)

    @pytest.mark.parametrize("device", DEVICES)
    def test_example_input_on_device(self, device):
        model = small_model().to(device)
        assert DrawNN(model, example_input=torch.zeros(1, 3, device=device)).neural_network == [3, 4, 2]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensors_in_manual_api(self):
        net = NeuralNetwork(4)
        net.add_layer(3)
        net.add_layer(4, weights=torch.randn(3, 4, device="cuda"),
                      biases=torch.zeros(4, device="cuda"))
        net.draw(show_weights=True)

    def test_tensors_requiring_grad(self):
        w = torch.randn(3, 4, requires_grad=True)
        b = torch.zeros(4, requires_grad=True)
        net = NeuralNetwork(4)
        net.add_layer(3)
        net.add_layer(4, weights=w, biases=b)
        net.draw(show_weights=True)
        assert isinstance(net.layer_specs[1]["weights"], list)
        assert w.requires_grad and b.requires_grad and w.grad is None

    def test_manual_api_accepts_tensors(self):
        net = NeuralNetwork(4)
        net.add_layer(3)
        net.add_layer(4, weights=torch.randn(3, 4), biases=torch.zeros(4), activation="ReLU")
        net.draw(show_weights=True)

    def test_high_level_api_accepts_tensors(self):
        DrawNN([3, 4], weights=[torch.randn(3, 4)], biases=[None, torch.zeros(4)]).draw(
            show_weights=True)

    def test_sizes_may_be_tensor_scalars(self):
        assert DrawNN([torch.tensor(3), torch.tensor(2)]).neural_network == [3, 2]

    def test_scalar_bias_tensor_per_neuron(self):
        net = NeuralNetwork(2)
        net.add_layer(2)
        net.add_layer(2, biases=list(torch.tensor([0.5, 1.5])))  # list of 0-d tensors
        assert net.layer_specs[1]["biases"] == [0.5, 1.5]
        net.draw()


# ======================================================= model is never changed
class TestNoSideEffects:
    def test_inspection_and_drawing_leave_model_untouched(self):
        model = small_model()
        model.train()
        model(torch.randn(5, 3)).sum().backward()
        state = copy.deepcopy(model.state_dict())
        grads = [p.grad.clone() for p in model.parameters()]
        flags = [m.training for m in model.modules()]
        req = [p.requires_grad for p in model.parameters()]

        DrawNN(model).draw(show_weights=True)
        NeuralNetwork.from_pytorch(model).draw()
        draw_model(model)

        assert all(torch.equal(state[k], model.state_dict()[k]) for k in state)
        assert all(torch.equal(g, p.grad) for g, p in zip(grads, model.parameters()))
        assert [m.training for m in model.modules()] == flags
        assert [p.requires_grad for p in model.parameters()] == req

    def test_works_on_model_with_grads_and_after_training_step(self):
        model = small_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(3):
            opt.zero_grad()
            model(torch.randn(4, 3)).sum().backward()
            opt.step()
        DrawNN(model).draw(show_weights=True)

    def test_trained_weights_are_what_is_drawn(self):
        model = nn.Sequential(nn.Linear(2, 2))
        with torch.no_grad():
            model[0].weight.fill_(7.0)
        assert DrawNN(model).weights[0] == [[7.0, 7.0], [7.0, 7.0]]


# =========================================================== unsupported stuff
class TestUnsupported:
    def test_conv_is_not_silently_ignored(self):
        model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Flatten(), nn.Linear(4 * 6 * 6, 2))
        with pytest.raises(UnsupportedModelError, match=r"'0' \(Conv2d"):
            DrawNN(model)

    def test_batchnorm_unsupported(self):
        with pytest.raises(UnsupportedModelError, match="BatchNorm1d"):
            DrawNN(nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Linear(3, 1)))

    def test_layernorm_embedding_rnn_unsupported(self):
        for bad in (nn.LayerNorm(3), nn.Embedding(5, 3), nn.GRU(3, 3)):
            with pytest.raises(UnsupportedModelError):
                DrawNN(nn.Sequential(nn.Linear(2, 3), bad))

    def test_glu_unsupported_because_it_halves_features(self):
        with pytest.raises(UnsupportedModelError, match="GLU"):
            DrawNN(nn.Sequential(nn.Linear(2, 4), nn.GLU(), nn.Linear(2, 1)))

    def test_custom_leaf_module_unsupported(self):
        class Square(nn.Module):
            def forward(self, x):
                return x * x

        with pytest.raises(UnsupportedModelError, match="Square"):
            DrawNN(nn.Sequential(nn.Linear(2, 3), Square()))

    def test_linear_subclass_with_custom_forward_unsupported(self):
        class Weird(nn.Linear):
            def forward(self, x):
                return super().forward(x) * 2

        with pytest.raises(UnsupportedModelError, match="Weird"):
            DrawNN(nn.Sequential(Weird(2, 3)))

    def test_container_with_own_parameters_unsupported(self):
        class Scaled(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 2)
                self.scale = nn.Parameter(torch.ones(1))

            def forward(self, x):
                return self.fc(x) * self.scale

        with pytest.raises(UnsupportedModelError, match="parameters of its own"):
            DrawNN(Scaled())

    def test_activation_before_first_linear_unsupported(self):
        with pytest.raises(UnsupportedModelError, match="network input"):
            DrawNN(nn.Sequential(nn.ReLU(), nn.Linear(2, 3)))

    def test_no_linear_layers(self):
        with pytest.raises(UnsupportedModelError, match="No nn.Linear"):
            DrawNN(nn.Sequential(nn.Dropout()))
        with pytest.raises(UnsupportedModelError):
            DrawNN(nn.Sequential(nn.Conv2d(1, 2, 3)))  # only unsupported layers

    def test_non_chained_linears(self):
        class Branchy(nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b = nn.Linear(4, 3), nn.Linear(5, 2)

        with pytest.raises(UnsupportedModelError, match="feed-forward chain"):
            DrawNN(Branchy())

    def test_lazy_linear_must_be_initialised(self):
        with pytest.raises(UnsupportedModelError, match="lazy"):
            DrawNN(nn.Sequential(nn.LazyLinear(3)))

    def test_lazy_linear_after_forward_pass_works(self):
        model = nn.Sequential(nn.LazyLinear(3))
        model(torch.zeros(1, 4))
        assert DrawNN(model).neural_network == [4, 3]

    def test_error_lists_every_unsupported_module(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.LayerNorm(3), nn.Linear(3, 1))
        with pytest.raises(UnsupportedModelError) as e:
            DrawNN(model)
        assert "BatchNorm1d" in str(e.value) and "LayerNorm" in str(e.value)

    def test_unsupported_error_is_a_valueerror(self):
        assert issubclass(UnsupportedModelError, ValueError)

    def test_strict_false_skips_with_warning(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.ReLU(), nn.Linear(3, 1))
        with pytest.warns(UserWarning, match="BatchNorm1d"):
            d = DrawNN(model, strict=False)
        assert d.neural_network == [2, 3, 1]
        assert d.activations == [None, "ReLU", None]

    def test_strict_false_still_rejects_shape_breaking_models(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.Conv1d(1, 1, 1), nn.Linear(5, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(UnsupportedModelError, match="feed-forward chain"):
                DrawNN(model, strict=False)

    def test_wide_layer_warns(self):
        with pytest.warns(UserWarning, match="may be slow"):
            DrawNN(nn.Sequential(nn.Linear(2, 200)))


# ============================================================ public entry points
class TestEntryPoints:
    def test_all_entry_points_draw(self):
        model = small_model()
        DrawNN(model).draw()
        DrawNN.from_pytorch(model).draw(show_weights=True)
        draw_model(model, show_weights=True, input_labels=["x1", "x2", "x3"],
                   output_labels=["Cat", "Dog"])
        net = NeuralNetwork.from_pytorch(model, input_labels=["a", "b", "c"],
                                         output_labels=["Cat", "Dog"])
        net.set_highlight_path([[0], [1, 2], [0]])
        net.draw(show_weights=True)

    def test_user_arguments_override_model_values(self):
        d = DrawNN(small_model(), title="Mine", activations=[None, "Tanh", "Sigmoid"],
                   input_labels=["a", "b", "c"], output_labels=["x", "y"],
                   highlight_path=[[0], [1], [0]])
        assert (d.title, d.activations) == ("Mine", [None, "Tanh", "Sigmoid"])
        assert d.input_labels == ["a", "b", "c"] and d.highlight_path == [[0], [1], [0]]
        d.draw(show_weights=True)

    def test_user_can_override_weights(self):
        d = DrawNN(small_model(), weights=[[[1.0] * 4] * 3, [[1.0] * 2] * 4])
        assert d.weights[0][0][0] == 1.0

    def test_neuralnetwork_from_pytorch_specs(self):
        net = NeuralNetwork.from_pytorch(small_model(), title="T")
        assert [s["size"] for s in net.layer_specs] == [3, 4, 2]
        assert net.layer_specs[1]["activation"] == "ReLU" and net.title == "T"
        assert net.layer_specs[0]["weights"] is None and net.layer_specs[0]["biases"] is None

    def test_from_pytorch_rejects_non_models(self):
        for cls in (DrawNN, NeuralNetwork):
            with pytest.raises(TypeError):
                cls.from_pytorch([3, 4, 2])

    def test_extract_architecture_is_pure_data(self):
        arch = extract_architecture(small_model())
        assert set(arch) == {"sizes", "weights", "biases", "activations"}
        assert arch["sizes"] == [3, 4, 2]

    def test_rendering_gets_same_result_as_manual_definition(self):
        model = small_model()
        auto = DrawNN(model)
        manual = DrawNN([3, 4, 2], weights=auto.weights, biases=auto.biases,
                        activations=[None, "ReLU", "Softmax"], title=auto.title)
        auto_net, manual_net = (NeuralNetwork.from_pytorch(model),
                                NeuralNetwork(4, title=auto.title))
        for i, size in enumerate([3, 4, 2]):
            manual_net.add_layer(size, weights=auto.weights[i - 1] if i else None,
                                 biases=auto.biases[i], activation=manual.activations[i])
        assert auto_net.layer_specs == manual_net.layer_specs
