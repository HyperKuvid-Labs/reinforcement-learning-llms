from __future__ import annotations

from dataclasses import dataclass

from .divergence import binary_tv, topk_tv


@dataclass(slots=True)
class LossBundle:
    loss: object
    metrics: dict[str, float]


def compute_group_advantages(rewards, mode: str):
    import torch

    rewards = rewards.float()
    mean = rewards.mean(dim=1, keepdim=True)
    if mode == "grpo":
        std = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (rewards - mean) / std
    return rewards - mean


def ppo_loss(current_logprobs, old_logprobs, advantages, clip_eps: float) -> LossBundle:
    import torch

    ratio = torch.exp(current_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    surrogate = torch.minimum(unclipped, clipped)
    clip_fraction = ((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps)).float().mean()
    loss = -surrogate.mean()
    return LossBundle(loss=loss, metrics={"train/clip_fraction": float(clip_fraction.item())})


def dppo_loss(
    current_logprobs,
    old_logprobs,
    advantages,
    current_chosen_probs,
    old_chosen_probs,
    current_topk_probs,
    old_topk_probs,
    approx: str,
    delta: float,
) -> LossBundle:
    import torch

    ratio = torch.exp(current_logprobs - old_logprobs)
    if approx == "binary":
        divergence = binary_tv(old_chosen_probs, current_chosen_probs)
        signed_shift = current_chosen_probs - old_chosen_probs
        mask = torch.ones_like(divergence)
        mask = torch.where((advantages > 0) & (signed_shift > delta), torch.zeros_like(mask), mask)
        mask = torch.where((advantages < 0) & ((-signed_shift) > delta), torch.zeros_like(mask), mask)
    else:
        divergence = topk_tv(old_topk_probs, current_topk_probs)
        mask = torch.where(divergence > delta, torch.zeros_like(divergence), torch.ones_like(divergence))

    surrogate = ratio * advantages * mask
    loss = -surrogate.mean()
    return LossBundle(
        loss=loss,
        metrics={
            "train/dppo_mask_fraction": float((1.0 - mask).mean().item()),
            f"divergence/{approx}": float(divergence.mean().item()),
        },
    )
