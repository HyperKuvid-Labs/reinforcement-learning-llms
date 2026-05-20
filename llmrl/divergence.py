from __future__ import annotations

def _safe_log(x):
    import torch

    return torch.log(x.clamp_min(1e-12))


def binary_tv(mu_chosen, pi_chosen):
    import torch

    return (mu_chosen - pi_chosen).abs()


def binary_kl(mu_chosen, pi_chosen):
    import torch

    mu_other = 1.0 - mu_chosen
    pi_other = 1.0 - pi_chosen
    return mu_chosen * (_safe_log(mu_chosen) - _safe_log(pi_chosen)) + mu_other * (
        _safe_log(mu_other) - _safe_log(pi_other)
    )


def topk_tv(mu_topk, pi_topk):
    import torch

    return 0.5 * (mu_topk - pi_topk).abs().sum(dim=-1)


def topk_kl(mu_topk, pi_topk):
    return (mu_topk * (_safe_log(mu_topk) - _safe_log(pi_topk))).sum(dim=-1)


def old_reduced_distribution(topk_indices, topk_probs, sampled_indices, sampled_probs):
    import torch

    sampled_in_topk = (topk_indices == sampled_indices.unsqueeze(-1)).any(dim=-1, keepdim=True)
    sampled_extra = torch.where(sampled_in_topk, sampled_probs.new_zeros(sampled_probs.shape[0], 1), sampled_probs.unsqueeze(-1))
    other_prob = (1.0 - topk_probs.sum(dim=-1, keepdim=True) - sampled_extra).clamp_min(0.0)
    return torch.cat([topk_probs, sampled_extra, other_prob], dim=-1)


def current_reduced_distribution(topk_indices, sampled_indices, current_probs):
    import torch

    topk_gathered = current_probs.gather(1, topk_indices)
    sampled_probs = current_probs.gather(1, sampled_indices.unsqueeze(-1))
    sampled_in_topk = (topk_indices == sampled_indices.unsqueeze(-1)).any(dim=-1, keepdim=True)
    sampled_extra = torch.where(sampled_in_topk, sampled_probs.new_zeros(sampled_probs.shape), sampled_probs)
    other_prob = (1.0 - topk_gathered.sum(dim=-1, keepdim=True) - sampled_extra).clamp_min(0.0)
    return torch.cat([topk_gathered, sampled_extra, other_prob], dim=-1)


def full_distribution_tv(mu_probs, pi_probs):
    return 0.5 * (mu_probs - pi_probs).abs().sum(dim=-1)


def full_distribution_kl(mu_probs, pi_probs):
    return (mu_probs * (_safe_log(mu_probs) - _safe_log(pi_probs))).sum(dim=-1)
