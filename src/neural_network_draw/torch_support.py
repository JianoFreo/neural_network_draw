"""Read the architecture (sizes, weights, biases, activations) out of a PyTorch model.

This is the *inspection* half of the PyTorch integration. It only produces plain
python data (see ``extract_architecture``); drawing is done by the existing
``NeuralNetwork`` / ``DrawNN`` code, which knows nothing about PyTorch.

PyTorch is not imported at module level, so this file can be imported even when
torch isn't installed. Modules and tensors are recognised by duck-typing.
"""

import warnings

try:
    from ._convert import to_plain
except ImportError:
    from _convert import to_plain


# Warn (don't fail) when a layer is bigger than this: every neuron and every
# connection is drawn, so very wide layers produce unreadable / slow figures.
LARGE_LAYER_WARNING = 64

# Only classes that live here are trusted to behave like the standard torch layer.
_TORCH_MODULES_PREFIX = "torch.nn.modules."

# Layers that don't change the shape or values at inference time, so leaving them
# out of the diagram is accurate.
_PASSTHROUGH = {
    "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout",
    "FeatureAlphaDropout", "Identity", "Flatten",
}
# Containers with no children are harmless no-ops too.
_EMPTY_CONTAINERS = {"Sequential", "ModuleList", "ModuleDict"}
# Lives in torch.nn.modules.activation but is not an element-wise activation.
_NOT_ACTIVATIONS = {"MultiheadAttention", "GLU"}  # GLU halves the feature count


class UnsupportedModelError(ValueError):
    """The model contains something this library can't draw accurately."""


def _owner(module):
    """The class that actually defines ``forward`` for this module (honours subclassing)."""
    for cls in type(module).__mro__:
        if "forward" in cls.__dict__:
            return cls
    return type(module)


def _classify(module):
    """Return (kind, name) where kind is 'linear', 'activation', 'passthrough' or None."""
    owner = _owner(module)
    name = owner.__name__
    if not owner.__module__.startswith(_TORCH_MODULES_PREFIX):
        return None, type(module).__name__  # custom forward -> we can't know what it does

    if name == "Linear" and hasattr(module, "in_features") and hasattr(module, "weight"):
        return "linear", name
    if owner.__module__ == "torch.nn.modules.activation" and name not in _NOT_ACTIVATIONS:
        return "activation", name
    if name in _PASSTHROUGH:
        return "passthrough", name
    return None, type(module).__name__


def _walk(module, prefix=""):
    """Leaf modules in registration order as (qualified_name, module).

    Unlike ``Module.modules()`` this does NOT de-duplicate, so a layer that is
    listed twice in a Sequential is correctly drawn twice.
    """
    children = [(n, c) for n, c in module._modules.items() if c is not None]
    if not children:
        yield (prefix or "<model>"), module
        return
    for n, c in children:
        yield from _walk(c, f"{prefix}.{n}" if prefix else n)


def _containers_with_own_parameters(module, prefix=""):
    """Non-leaf modules that own parameters directly: their math isn't visible to us."""
    children = [(n, c) for n, c in module._modules.items() if c is not None]
    if not children:
        return
    if any(True for _ in module.parameters(recurse=False)):
        yield (prefix or "<model>"), module
    for n, c in children:
        yield from _containers_with_own_parameters(c, f"{prefix}.{n}" if prefix else n)


def _trace_execution_order(model, example_input):
    """
    Run ONE forward pass with hooks to learn which leaf modules run, and in what order.

    The model is put in eval mode and run under ``torch.no_grad()`` so nothing is
    learned, no gradients are recorded and BatchNorm running stats don't change;
    every module's original ``training`` flag is restored afterwards. Parameters
    are never touched.
    """
    import torch

    names = {}
    for name, module in _walk(model):
        names.setdefault(id(module), name)
    leaves = {id(m): m for _, m in _walk(model)}

    executed = []
    handles = [m.register_forward_hook(lambda mod, inp, out: executed.append(mod))
               for m in leaves.values()]
    training_flags = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            if isinstance(example_input, (tuple, list)):
                model(*example_input)
            else:
                model(example_input)
    finally:
        for h in handles:
            h.remove()
        for m, flag in training_flags:
            m.training = flag

    return [(names[id(m)], m) for m in executed]


def extract_architecture(model, strict=True, example_input=None):
    """
    Inspect a ``torch.nn.Module`` and return everything needed to draw it:

        {
          "sizes":       [in, h1, h2, ..., out],
          "weights":     [W0, W1, ...]   # W_k shaped [size_k][size_k+1] (library convention)
          "biases":      [None, b1, b2, ...],
          "activations": [None, "ReLU", ..., "Softmax"],
        }

    Supported: ``nn.Linear`` layers (each becomes a layer of neurons; the first also
    creates the input layer), activation modules from ``torch.nn`` (ReLU, Sigmoid,
    Tanh, Softmax, ...; several in a row are shown as "ReLU → Sigmoid"), and layers
    that don't change the data at inference (Dropout*, Identity, Flatten). Nested
    containers and custom modules are fine as long as they are built from those.

    Anything else (Conv, BatchNorm, RNNs, custom operations, parameters owned
    directly by a container, GLU, ...) can't be drawn accurately:
      * strict=True (default): raise ``UnsupportedModelError`` listing them.
      * strict=False: skip them with a warning (the diagram then omits them).

    Layer order:
      * By default layers are read in registration order, which is the forward
        order for ``nn.Sequential`` and for modules that define layers in the order
        they are used.
      * Pass ``example_input`` (a tensor, or a tuple of positional args) to run one
        forward pass in eval mode / no_grad and use the *actual* execution order
        instead. Layers that are defined but never run are left out.

    Activations written as ``F.relu(...)`` inside ``forward`` are invisible to
    inspection; override with ``DrawNN(model, activations=[None, "ReLU", ...])``.

    The model is only read: parameters, gradients and training flags are unchanged,
    and values are detached and copied to CPU python floats.
    """
    unsupported = []  # (qualified_name, class_name, reason)

    for name, module in _containers_with_own_parameters(model):
        unsupported.append((name, type(module).__name__,
                            "has parameters of its own that are used outside its child layers"))

    if example_input is None:
        ordered = list(_walk(model))
    else:
        ordered = _trace_execution_order(model, example_input)

    sizes = []
    weights = []
    biases = [None]
    activations = [None]

    for name, module in ordered:
        kind, cls_name = _classify(module)

        if kind is None:
            if not (type(module).__name__ in _EMPTY_CONTAINERS and not module._modules):
                unsupported.append((name, cls_name, "no way to draw this layer type accurately"))

        elif kind == "linear":
            if type(module.weight).__name__ == "UninitializedParameter":
                raise UnsupportedModelError(
                    "The model contains an uninitialized lazy layer. Run one forward "
                    "pass (e.g. model(torch.zeros(1, n_features))) before drawing it."
                )
            in_f, out_f = int(module.in_features), int(module.out_features)

            if not sizes:
                sizes.append(in_f)
            elif sizes[-1] != in_f:
                if strict and unsupported:
                    break  # the unsupported module is the root cause; report it instead
                raise UnsupportedModelError(
                    f"Can't draw this model as a simple feed-forward chain: layer '{name}' "
                    f"expects {in_f} features but the previous Linear layer outputs "
                    f"{sizes[-1]}. (Branching, skip connections and parallel layers aren't "
                    f"supported. If the layers just aren't defined in forward order, pass "
                    f"example_input=... to use the real execution order.)"
                )
            sizes.append(out_f)

            # torch stores weight as [out][in]; this library wants [in][out]
            weights.append(module.weight.detach().cpu().t().tolist())
            biases.append(to_plain(module.bias) if module.bias is not None else None)
            activations.append(None)

        elif kind == "activation":
            if not sizes:
                unsupported.append((name, cls_name,
                                    "activation applied to the network input (there is no "
                                    "neuron layer before it to attach it to)"))
            elif activations[-1] is None:
                activations[-1] = cls_name
            else:  # e.g. ReLU followed by Sigmoid on the same layer: show both
                activations[-1] = f"{activations[-1]} → {cls_name}"

        # kind == "passthrough": nothing to draw

    if unsupported:
        listing = ", ".join(f"'{n}' ({c}: {why})" for n, c, why in unsupported)
        if strict:
            raise UnsupportedModelError(
                f"This model can't be drawn accurately. Unsupported: {listing}. Only "
                f"fully-connected networks made of nn.Linear layers, activations and no-op "
                f"layers (Dropout, Identity, Flatten) are supported. Pass strict=False to "
                f"skip these with a warning instead (the diagram will then omit them)."
            )
        warnings.warn(f"Skipped unsupported parts of the model, so the diagram does not "
                      f"show them: {listing}.", stacklevel=3)

    if len(sizes) < 2:
        raise UnsupportedModelError(
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
