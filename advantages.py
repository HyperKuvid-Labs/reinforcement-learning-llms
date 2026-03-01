import torch

# from the paper: https://arxiv.org/abs/2602.02710

def compute_grpo(rewards: torch.Tensor) -> torch.Tensor:
  """
  grpo: (r - μ) / σ
  here we are taking only k, which is basically only successful trajectories
  rewards here come as: rewards = (y_samples == y_star).float()
  shape: rewards [batch_size, k] → advantages [batch_size, k]
  """

  # compute mean and std for each x
  mean_rewards = rewards.mean(dim=1, keepdim=True)
  std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8  # add small value to avoid division by zero

  adv = (rewards - mean_rewards) / std_rewards

  return adv

def compute_reinforce(rewards: torch.Tensor) -> torch.Tensor:
  """
  vanilla (group-relative) reinforce baseline: r - μ
  here we are taking only k, which is basically only successful trajectories
  rewards here come as: rewards = (y_samples == y_star).float()
  """

  # compute mean for each x
  mean_rewards = rewards.mean(dim=1, keepdim=True)

  adv = rewards - mean_rewards

  return adv

def compute_maxrl(rewards: torch.Tensor) -> torch.Tensor:
  """
  maxrl advantage: (r - μ) / μ
  we divide by μ (mean reward), not σ
  when μ is very small (many wrong answers), this gives very large weight → focuses on hard examples
  """

  mean_rewards = rewards.mean(dim=1, keepdim=True)
  mean_reward_denom = mean_rewards + 1e-8  # add small value to avoid division by zero

  adv = (rewards - mean_rewards) / mean_reward_denom

  return adv
