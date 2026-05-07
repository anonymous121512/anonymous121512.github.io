"""Metrics for evaluating weight intervention effects.

All metrics operate on model logits and compare original vs perturbed behavior.
"""
import torch
import torch.nn.functional as F


def ce_loss_on_tokens(logits, target_tokens):
    """Cross-entropy loss between logits and target tokens.

    Args:
        logits: [batch, seq_len, vocab] model output logits
        target_tokens: [batch, seq_len] ground truth token IDs (shifted by 1)

    Returns:
        [batch] per-example CE loss (averaged over positions)
    """
    B, T, V = logits.shape
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_tokens[:, 1:].contiguous()
    # Per-token loss
    loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_targets.view(-1),
        reduction="none",
    )
    loss = loss.view(B, T - 1)
    return loss  # [batch, seq_len-1]


def ce_loss_increase(original_logits, perturbed_logits, target_tokens):
    """Per-token CE loss increase from a weight perturbation.

    Args:
        original_logits: [batch, seq_len, vocab]
        perturbed_logits: [batch, seq_len, vocab]
        target_tokens: [batch, seq_len]

    Returns:
        [batch, seq_len-1] per-position CE loss increase (positive = worse)
    """
    orig_loss = ce_loss_on_tokens(original_logits, target_tokens)
    pert_loss = ce_loss_on_tokens(perturbed_logits, target_tokens)
    return pert_loss - orig_loss


def kl_divergence(original_logits, perturbed_logits, position=-1):
    """KL divergence from original to perturbed prediction distribution.

    KL(original || perturbed) — measures how much information is lost.

    Args:
        original_logits: [batch, seq_len, vocab]
        perturbed_logits: [batch, seq_len, vocab]
        position: which sequence position to evaluate (-1 = last)

    Returns:
        [batch] KL divergence at the specified position
    """
    orig_logprobs = F.log_softmax(original_logits[:, position, :], dim=-1)
    pert_logprobs = F.log_softmax(perturbed_logits[:, position, :], dim=-1)
    orig_probs = orig_logprobs.exp()
    kl = (orig_probs * (orig_logprobs - pert_logprobs)).sum(dim=-1)
    return kl  # [batch]


def kl_divergence_all_positions(original_logits, perturbed_logits):
    """KL divergence at every position.

    Returns:
        [batch, seq_len] KL divergence per position
    """
    orig_logprobs = F.log_softmax(original_logits, dim=-1)
    pert_logprobs = F.log_softmax(perturbed_logits, dim=-1)
    orig_probs = orig_logprobs.exp()
    kl = (orig_probs * (orig_logprobs - pert_logprobs)).sum(dim=-1)
    return kl  # [batch, seq_len]


def logit_diff(logits, pos_token, neg_token, position=-1):
    """Difference between logits for two specific tokens.

    Args:
        logits: [batch, seq_len, vocab]
        pos_token: int, the "positive" token ID
        neg_token: int, the "negative" token ID
        position: which sequence position

    Returns:
        [batch] logit_diff = logit(pos) - logit(neg)
    """
    return logits[:, position, pos_token] - logits[:, position, neg_token]


def top_k_affected_tokens(original_logits, perturbed_logits, tokenizer,
                          position=-1, k=10):
    """Find tokens whose predicted probability changed most.

    Returns list of (token_str, prob_change, orig_prob, pert_prob) tuples.
    """
    orig_probs = F.softmax(original_logits[:, position, :], dim=-1).mean(dim=0)
    pert_probs = F.softmax(perturbed_logits[:, position, :], dim=-1).mean(dim=0)
    diff = (pert_probs - orig_probs).abs()
    topk = diff.topk(k)
    results = []
    for val, idx in zip(topk.values, topk.indices):
        tid = idx.item()
        tok_str = tokenizer.decode([tid])
        results.append((
            tok_str,
            val.item(),
            orig_probs[tid].item(),
            pert_probs[tid].item(),
        ))
    return results
