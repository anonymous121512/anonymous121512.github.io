"""End-to-end mini example: ablate one weight, measure KL across prompts,
build a quantile histogram of per-position effect.

This is a self-contained illustration. Plug in your own ``load_model``,
tokeniser, and prompt corpus.

Conceptually equivalent to a single weight's detail page on the live site.
"""
import math
import numpy as np
import torch

from patching import weight_intervention, ablate_weight_entry
from metrics import kl_divergence_all_positions, top_k_affected_tokens


# ---------------------------------------------------------------------------
# 1. Pluggable hooks — replace these with your own loading code.
# ---------------------------------------------------------------------------

def load_model_and_tokeniser():
    """Return (model, tokeniser). Replace with your project-specific loader.

    For the live site we use the public weight-sparse model from
    Gao et al. (2025) (arXiv:2511.13653, model id `csp_yolo1`) and the
    `tinypython_2k` tokeniser bundled with that release.
    """
    raise NotImplementedError("Plug in your own model loader.")


def load_prompts(tokeniser, n=1_000, seq_len=64):
    """Return ``[n, seq_len]`` token IDs, one row per prompt."""
    raise NotImplementedError("Plug in your own corpus.")


# ---------------------------------------------------------------------------
# 2. Forward pass — original vs ablated.
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_kl(model, tokens, param_name, indices):
    """Run model with and without the single-entry ablation.

    Returns a ``[batch, seq_len]`` tensor of KL(orig || ablated) per position.
    """
    orig_logits = model(tokens)
    if hasattr(orig_logits, "logits"):
        orig_logits = orig_logits.logits

    with weight_intervention(model, param_name,
                             lambda w: ablate_weight_entry(w, indices)):
        ablated_logits = model(tokens)
        if hasattr(ablated_logits, "logits"):
            ablated_logits = ablated_logits.logits

    return kl_divergence_all_positions(orig_logits, ablated_logits)


# ---------------------------------------------------------------------------
# 3. Quantile histogram — turns per-position KL into the 50-bin profile that
#    drives the live KL-CDF widget on each weight's detail page.
# ---------------------------------------------------------------------------

def quantile_histogram(kl_flat, n_bins=50, near_zero=1e-8):
    """Group per-position KL values into log10 bins.

    Args:
        kl_flat: ``[N]`` flattened per-position KL (across prompts × positions).
        n_bins:  how many log10 bins to use.
        near_zero: positions with KL < this are dropped (numerical noise).

    Returns:
        dict with ``bin_edges_log10``, ``counts``, plus summary stats used by
        the live histogram and the "select 90% CDF" button.
    """
    kl = np.asarray(kl_flat).reshape(-1)
    nonzero = kl[kl > near_zero]
    if nonzero.size == 0:
        return {"n_bins": n_bins, "counts": [0] * n_bins, "n_total": kl.size,
                "n_near_zero": int(kl.size), "max_kl": 0.0, "mean_kl": 0.0}

    log_kl = np.log10(nonzero)
    edges = np.linspace(log_kl.min(), log_kl.max(), n_bins + 1)
    counts, _ = np.histogram(log_kl, bins=edges)

    # KL-mass per bin (used for the 90%-CDF selection on the live site)
    bin_mass = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (log_kl >= lo) & (log_kl < hi)
        bin_mass.append(float(nonzero[mask].sum()))

    return {
        "n_bins":          n_bins,
        "bin_edges_log10": edges.tolist(),
        "counts":          counts.tolist(),
        "kl_mass_per_bin": bin_mass,
        "n_total":         int(kl.size),
        "n_near_zero":     int(kl.size - nonzero.size),
        "max_kl":          float(kl.max()),
        "mean_kl":         float(kl.mean()),
    }


# ---------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------

def profile_one_weight(param_name, indices, n_prompts=1_000, seq_len=64):
    """Reproduce a single weight's detail-page card.

    Picks one weight (e.g. ``param_name="blocks.0.mlp.W_in"``,
    ``indices=(2199, 723)``), ablates it, measures per-position KL across a
    corpus batch, and returns the quantile histogram + top-affected tokens.
    """
    model, tok = load_model_and_tokeniser()
    tokens = load_prompts(tok, n=n_prompts, seq_len=seq_len)

    kl_per_pos = measure_kl(model, tokens, param_name, indices)  # [B, T]
    histogram = quantile_histogram(kl_per_pos.cpu().numpy())

    # Highest-KL position across the batch — used for "what tokens move?"
    flat_idx = int(kl_per_pos.flatten().argmax())
    b = flat_idx // tokens.shape[1]
    t = flat_idx %  tokens.shape[1]

    orig = model(tokens[b:b + 1])
    orig_logits = orig.logits if hasattr(orig, "logits") else orig
    with weight_intervention(model, param_name,
                             lambda w: ablate_weight_entry(w, indices)):
        pert = model(tokens[b:b + 1])
        pert_logits = pert.logits if hasattr(pert, "logits") else pert
    movers = top_k_affected_tokens(orig_logits, pert_logits, tok,
                                   position=t, k=10)

    return {
        "param_name":      param_name,
        "indices":         list(indices),
        "max_kl":          histogram["max_kl"],
        "kl_histogram":    histogram,
        "top_movers":      movers,
        "argmax_position": {"prompt": b, "token_pos": t},
    }


if __name__ == "__main__":
    # Example invocation — replace with your weight of interest.
    profile = profile_one_weight(
        param_name="blocks.0.mlp.W_in",
        indices=(2199, 723),
        n_prompts=1_000,
        seq_len=64,
    )
    print(profile["max_kl"], profile["top_movers"])
