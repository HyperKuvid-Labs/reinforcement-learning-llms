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


def old_reduced_distribution(topk_indices, topk_probs, sampled_indices):
    import torch

    expanded_sampled = sampled_indices.unsqueeze(-1)
    merged_indices = torch.cat([topk_indices, expanded_sampled], dim=-1)
    merged_probs = torch.cat(
        [topk_probs, topk_probs.new_zeros(topk_probs.shape[0], 1)],
        dim=-1,
    )
    rows = []
    for row_indices, row_probs, sampled in zip(merged_indices, merged_probs, sampled_indices):
        seen = {}
        for token_id, prob in zip(row_indices.tolist(), row_probs.tolist()):
            seen[token_id] = max(seen.get(token_id, 0.0), prob)
        if sampled.item() not in seen:
            seen[sampled.item()] = 0.0
        values = torch.tensor(list(seen.values()), device=topk_probs.device, dtype=topk_probs.dtype)
        other_prob = (1.0 - values.sum()).clamp_min(0.0)
        rows.append(torch.cat([values, other_prob.unsqueeze(0)], dim=0))
    max_len = max(row.shape[0] for row in rows)
    padded = []
    for row in rows:
        if row.shape[0] < max_len:
            pad = row.new_zeros(max_len - row.shape[0])
            padded.append(torch.cat([row, pad], dim=0))
        else:
            padded.append(row)
    return torch.stack(padded, dim=0)


def current_reduced_distribution(topk_indices, sampled_indices, current_probs):
    import torch

    expanded_sampled = sampled_indices.unsqueeze(-1)
    merged = torch.cat([topk_indices, expanded_sampled], dim=-1)
    rows = []
    unique_probs = []
    for row_indices, row_probs in zip(merged, current_probs):
        seen = set()
        row_unique = []
        for token_id in row_indices.tolist():
            if token_id not in seen:
                row_unique.append(token_id)
                seen.add(token_id)
        index_tensor = torch.tensor(row_unique, device=current_probs.device, dtype=torch.long)
        gathered = row_probs.gather(0, index_tensor)
        other_prob = (1.0 - gathered.sum()).clamp_min(0.0)
        row_with_other = torch.cat([gathered, other_prob.unsqueeze(0)], dim=0)
        rows.append(index_tensor)
        unique_probs.append(row_with_other)
    max_len = max(prob.shape[0] for prob in unique_probs)
    padded = []
    for row in unique_probs:
        if row.shape[0] < max_len:
            pad = row.new_zeros(max_len - row.shape[0])
            padded.append(torch.cat([row, pad], dim=0))
        else:
            padded.append(row)
    return torch.stack(padded, dim=0)


def full_distribution_tv(mu_probs, pi_probs):
    return 0.5 * (mu_probs - pi_probs).abs().sum(dim=-1)


def full_distribution_kl(mu_probs, pi_probs):
    return (mu_probs * (_safe_log(mu_probs) - _safe_log(pi_probs))).sum(dim=-1)
