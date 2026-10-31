"""Read the architecture (sizes, weights, biases, activations) out of a PyTorch model.

PyTorch is *not* imported here -- modules and tensors are handled by
duck-typing -- so this file is safe to import even if torch is not installed.
"""

import warnings

try:
    from ._convert import to_plain
except ImportError:
    from _convert import to_plain


# Warn (don't fail) when a layer is bigger than this: every neuron and every
# connection is drawn, so very wide layers produce unreadable / slow figures.
LARGE_LAYER_WARNING = 64


def _is_leaf(module):
    return next(iter(module.children()), None) is None


def _is_linear(module):
    """nn.Linear and friends (LazyLinear, NonDynamicallyQuantizableLinear, ...)."""
    weight = getattr(module, "weight", None)
    return (
        type(module).__name__.endswith("Linear")
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and weight is not None
        and getattr(weight, "dim", lambda: 0)() == 2
    )


def _is_activation(module):
    """Any layer from torch.nn.modules.activation (ReLU, Sigmoid, Softmax, GELU, ...)."""
    return (
        type(module).__module__ == "torch.nn.modules.activation"
        and type(module).__name__ != "MultiheadAttention"
    )


def extract_architecture(model):
    """
    Walk a ``torch.nn.Module`` and return everything needed to draw it:

        {
          "sizes":       [in, h1, h2, ..., out],
          "weights":     [W0, W1, ...]   # W_k shaped [size_k][size_k+1] (library convention)
          "biases":      [None, b1, b2, ...],
          "activations": [None, "ReLU", ..., "Softmax"],
        }

    Rules:
      * Every ``nn.Linear`` becomes one layer of neurons (its ``out_features``);
        the first Linear also creates the input layer (its ``in_features``).
      * An activation module (ReLU, Sigmoid, Softmax, ...) is attached to the
        Linear layer right before it. Other modules (Dropout, BatchNorm,
        Flatten, ...) are ignored.
      * Layers are read in registration order (``model.modules()``), which is
        the forward order for ``nn.Sequential`` and for models that define
        their layers in the order they use them.
      * Activations applied with ``F.relu(...)`` inside ``forward`` can't be
        detected; pass ``activations=[...]`` to override.
    """
    sizes = []
    weights = []
    biases = [None]
    activations = [None]

    for module in model.modules():
        if not _is_leaf(module):
            continue

        if _is_linear(module):
            if type(module.weight).__name__ == "UninitializedParameter":
                raise ValueError(
                    "The model contains an uninitialized lazy layer. Run one forward "
                    "pass (e.g. model(torch.zeros(1, n_features))) before drawing it."
                )
            in_f, out_f = int(module.in_features), int(module.out_features)

            if not sizes:
                sizes.append(in_f)
            elif sizes[-1] != in_f:
                raise ValueError(
                    f"Can't draw this model as a simple feed-forward chain: a Linear layer "
                    f"outputs {sizes[-1]} features but the next Linear layer expects {in_f}. "
                    f"(Branching / parallel layers aren't supported.)"
                )
            sizes.append(out_f)

            # torch stores weight as [out][in]; this library wants [in][out]
            weights.append(module.weight.detach().cpu().t().tolist())
            biases.append(to_plain(module.bias) if module.bias is not None else None)
            activations.append(None)

        elif _is_activation(module) and len(sizes) >= 2 and activations[-1] is None:
            activations[-1] = type(module).__name__

    if len(sizes) < 2:
        raise ValueError(
            "No nn.Linear layers were found in this model, so there is nothing to draw. "
            "Only fully-connected (Linear) networks are supported."
        )

    big = [s for s in sizes if s > LARGE_LAYER_WARNING]
    if big:
        warnings.warn(
            f"This model has layers with up to {max(big)} neurons. Every neuron and "
            f"connection gets drawn, so the diagram may be slow and hard to read.",
            stacklevel=3,
        )

    return {
        "sizes": sizes,
        "weights": weights,
        "biases": biases,
        "activations": activations,
    }
