"""Weight-level patching/ablation utilities.

Unlike activation patching (which intervenes on intermediate activations during
a forward pass), weight patching directly modifies the model's parameters and
observes the effect on outputs.

Key operations:
- ablate_weight_entry: zero out a single weight matrix entry
- ablate_weight_row:   zero out an entire slice along a given dimension
- perturb_weight_entry: add noise to a single entry
- weight_intervention: context manager that temporarily modifies a parameter
- run_with_weight_intervention: convenience wrapper around the above
"""
import contextlib

import torch


def _get_underlying_model(model):
    """Unwrap torch.compile wrapper (if any) to get the original module."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


@contextlib.contextmanager
def weight_intervention(model, param_name, modifier_fn):
    """Context manager that temporarily modifies a weight parameter.

    Args:
        model:        the model (or torch.compile'd wrapper).
        param_name:   full parameter name, e.g. ``"blocks.0.mlp.W_in"``.
        modifier_fn:  callable ``(tensor) -> tensor`` that returns the modified
                      weight. Receives a clone of the original tensor.

    The original weight is restored exactly on exit, even if an exception is
    raised inside the ``with`` block.
    """
    raw_model = _get_underlying_model(model)
    param = dict(raw_model.named_parameters())[param_name]
    original_data = param.data.clone()
    try:
        param.data = modifier_fn(param.data.clone())
        yield model
    finally:
        param.data = original_data


def run_with_weight_intervention(model, tokens, param_name, modifier_fn,
                                 forward_fn=None):
    """Run a forward pass with a temporary weight modification.

    Args:
        model:       the model.
        tokens:      ``[batch, seq_len]`` input token IDs.
        param_name:  weight parameter to modify.
        modifier_fn: callable ``(tensor) -> tensor``.
        forward_fn:  optional callable ``(model, tokens) -> logits``. If None,
                     defaults to ``model(tokens).logits`` for HuggingFace-style
                     interfaces, or ``model(tokens)`` for plain modules.

    Returns:
        ``[batch, seq_len, vocab]`` logits with the modification active.
    """
    if forward_fn is None:
        def forward_fn(m, t):
            out = m(t)
            return out.logits if hasattr(out, "logits") else out

    with weight_intervention(model, param_name, modifier_fn):
        with torch.no_grad():
            logits = forward_fn(model, tokens)
    return logits


# ---------------------------------------------------------------------------
# Modifier helpers — pass these (curried) to weight_intervention's modifier_fn
# ---------------------------------------------------------------------------

def ablate_weight_entry(weight_tensor, indices):
    """Zero out a single entry in a weight matrix.

    Args:
        weight_tensor: the weight tensor (already cloned by the caller).
        indices:       tuple of ints indexing into the tensor.

    Returns:
        modified tensor with ``weight_tensor[indices] = 0``.
    """
    w = weight_tensor.clone()
    w[indices] = 0.0
    return w


def ablate_weight_row(weight_tensor, dim, index):
    """Zero out an entire slice along a given dimension."""
    w = weight_tensor.clone()
    slices = [slice(None)] * w.ndim
    slices[dim] = index
    w[tuple(slices)] = 0.0
    return w


def perturb_weight_entry(weight_tensor, indices, noise_scale=1.0):
    """Add Gaussian noise to a single weight entry."""
    w = weight_tensor.clone()
    w[indices] += torch.randn(1, device=w.device).item() * noise_scale
    return w


def set_weight_entry(weight_tensor, indices, value):
    """Set a single weight entry to a specific value."""
    w = weight_tensor.clone()
    w[indices] = value
    return w


def ablate_weight_entries_batch(weight_tensor, indices_list):
    """Zero out multiple entries at once (e.g. an entire neuron's row)."""
    w = weight_tensor.clone()
    for idx in indices_list:
        w[idx] = 0.0
    return w
