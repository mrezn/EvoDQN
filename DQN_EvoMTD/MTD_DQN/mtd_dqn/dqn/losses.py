from __future__ import annotations

from typing import Iterable

import torch


def huber(td_error: torch.Tensor, kappa: float) -> torch.Tensor:
    abs_error = torch.abs(td_error)
    quadratic = torch.minimum(abs_error, torch.tensor(kappa, device=td_error.device))
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + kappa * linear


def td_loss(predicted: torch.Tensor, target: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    td_error = predicted - target
    return huber(td_error, kappa).mean()


def entropy_penalty(q_values: torch.Tensor, tau: float) -> torch.Tensor:
    logits = q_values / tau
    policy = torch.softmax(logits, dim=-1)
    log_policy = torch.log(policy + 1e-8)
    entropy = -(policy * log_policy).sum(dim=-1)
    return -entropy.mean()


def weight_decay(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    params = list(parameters)
    if not params:
        return torch.tensor(0.0)
    total = torch.zeros(1, device=params[0].device)
    for param in params:
        total = total + torch.sum(param ** 2)
    return total


def combined_loss(
    predicted_q: torch.Tensor,
    target_q: torch.Tensor,
    q_values: torch.Tensor,
    tau: float,
    beta_ent: float,
    weight_decay_coeff: float,
    parameters: Iterable[torch.nn.Parameter],
    kappa: float = 1.0,
) -> torch.Tensor:
    params = list(parameters)
    td = td_loss(predicted_q, target_q, kappa=kappa)
    ent = entropy_penalty(q_values, tau) if beta_ent > 0 else torch.tensor(0.0, device=q_values.device)
    wd = weight_decay(params) if weight_decay_coeff > 0 else torch.tensor(0.0, device=q_values.device)
    return td + beta_ent * ent + weight_decay_coeff * wd
