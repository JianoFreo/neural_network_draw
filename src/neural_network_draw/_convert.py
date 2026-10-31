"""Helpers for accepting PyTorch tensors / NumPy arrays anywhere plain lists work.

Nothing here imports torch or numpy, so the library keeps working (and stays
importable) when PyTorch is not installed. Tensors are detected by duck-typing.
"""


def to_plain(value):
    """Recursively convert tensors / arrays / tuples into plain python lists & floats.

    * ``None`` stays ``None`` (so "no weights" / "no bias" still works)
    * torch tensors (any device, any requires_grad) -> nested lists of floats
    * numpy arrays / numpy scalars -> nested lists of floats
    * lists / tuples are converted element by element, so a list that mixes
      ``None`` and tensors (e.g. ``biases=[None, tensor, tensor]``) works
    * everything else (ints, floats, strings) is returned unchanged
    """
    if value is None:
        return None

    # torch.Tensor: detach from the autograd graph and move off GPU first
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()

    # numpy arrays / numpy scalars
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]

    return value


def is_torch_module(obj):
    """True if ``obj`` looks like a ``torch.nn.Module`` (duck-typed, no torch import)."""
    return (
        hasattr(obj, "modules")
        and hasattr(obj, "_modules")
        and hasattr(obj, "children")
        and hasattr(obj, "named_parameters")
        and callable(getattr(obj, "modules"))
    )
