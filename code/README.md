# Weight-Level Interpretability Pipeline (anonymous code excerpt)

This directory contains a small, representative slice of the analysis pipeline
that produced the visualisations on the live site
(<https://anonymous121512.github.io>). The full pipeline and case studies will
be released alongside the paper after de-anonymisation.

## Question

Are individual weight matrix entries in **weight-sparse transformers**
interpretable? When you ablate (set to zero) a single weight entry, does the
model's output distribution shift on a sparse, semantically coherent set of
tokens?

This builds on Gao et al. (2025), "Weight-sparse transformers have
interpretable circuits" ([arXiv:2511.13653](https://arxiv.org/abs/2511.13653)),
which established that components (neurons, heads, channels) of the
`csp_yolo1` model are basis-aligned. We drill down to the **individual-weight**
level.

## Pipeline (one weight)

For each nonzero weight entry `W[i, j]`:

1. **Ablate** — temporarily set the entry to zero.
2. **Compare** — run the original and ablated models on the same prompts; for
   every (prompt, token) position, compute the KL divergence between the two
   output distributions.
3. **Bin** — group positions by `log10(KL)` into ~50 quantile bins; record
   example tokens for each bin.
4. **Inspect** — for the highest-KL positions, look at *which tokens are most
   affected* (top-K probability shifts). If the affected tokens form a
   coherent semantic group (e.g. all numerals, all string delimiters), the
   weight is monosemantic.

The detail page of any weight on the live site is exactly this: KL histogram
with per-bin example prompts, plus probability-shift readouts.

## Files

| File | What it does |
|---|---|
| `patching.py` | The `weight_intervention` context manager. Clones the original weight, applies a modifier, runs your forward pass, restores the weight on exit. |
| `metrics.py` | KL divergence (per position and last-position), CE loss increase, and a `top_k_affected_tokens` helper. |
| `example.py` | End-to-end mini-demo: load a sparse weight, ablate one entry, measure KL across a batch of prompts, build the quantile histogram. |

## Quick example

```python
from patching import weight_intervention, ablate_weight_entry
from metrics import kl_divergence_all_positions
import torch

with torch.no_grad():
    orig_logits = model(tokens).logits

with weight_intervention(model, "blocks.0.mlp.W_in",
                         lambda w: ablate_weight_entry(w, (row, col))):
    with torch.no_grad():
        pert_logits = model(tokens).logits

kl = kl_divergence_all_positions(orig_logits, pert_logits)  # [B, T]
```

The `kl` tensor is the per-position effect — feed it through your favourite
quantile binner and you have everything you need to reproduce a weight's card
on the live site.

## Notes

- Designed for the `csp_yolo1` weight-sparse transformer (12 layers,
  d_model=1024, ~95% zero weights, tinypython_2k tokeniser). The mechanism
  generalises to any model whose nonzero weights are accessible by name.
- Code paths assume PyTorch + a model loaded with named parameters. No
  TransformerLens-specific hooks are required for the examples here.
- Full pipeline (corpus management, batching, neuron-level case studies,
  cross-corpus held-out generalisation) will accompany the paper.
