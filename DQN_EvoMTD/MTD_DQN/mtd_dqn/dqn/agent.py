from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .. import payoffs
from ..uncertainty import Scenario
from .losses import combined_loss, entropy_penalty
from .networks import DuelingQNetwork
from .replay import ReplayBuffer, Transition


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


@dataclass
class AgentConfig:
    gamma: float
    tau_soft: float
    tau_D: float
    beta_ent: float
    huber_kappa: float
    weight_decay: float
    lr: float


class DQNAgent:
    """Implements the robust Double DQN update described in the prompt."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        config: AgentConfig,
        layer_norm: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = DuelingQNetwork(state_dim, action_dim, hidden_dims, layer_norm=layer_norm).to(self.device)
        self.target = DuelingQNetwork(state_dim, action_dim, hidden_dims, layer_norm=layer_norm).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = Adam(self.online.parameters(), lr=config.lr, weight_decay=0.0)

    def select_action(self, state: np.ndarray, mode: str = "train") -> int:
        state_tensor = _to_tensor(state, self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state_tensor).squeeze(0)
        if mode == "eval":
            return int(torch.argmax(q_values).item())
        policy = torch.softmax(q_values / self.config.tau_D, dim=-1)
        action = torch.multinomial(policy, num_samples=1)
        return int(action.item())

    @staticmethod
    def _defender_reward_from_summary(summary: dict, scenario: Scenario) -> float:
        sap_scaled = summary["sap"] * scenario.defender_scale
        return payoffs.defender_payoff(
            sap_scaled,
            summary["defender_cost"],
            summary["alpha_I"],
            summary["incentive"],
        )

    @staticmethod
    def _attacker_reward_from_summary(summary: dict, scenario: Scenario) -> float:
        sal_scaled = summary["sal"] * scenario.attacker_scale
        return payoffs.attacker_payoff(
            sal_scaled,
            summary["attacker_cost"],
            summary["guidance_penalty"],
        )

    def _compute_targets(
        self, batch: List[Transition]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float], List[float]]:
        states = torch.stack([_to_tensor(tr.state, self.device) for tr in batch])
        next_states = torch.stack([_to_tensor(tr.next_state, self.device) for tr in batch])
        actions = torch.tensor([tr.action for tr in batch], device=self.device, dtype=torch.long)
        dones = torch.tensor([float(tr.done) for tr in batch], device=self.device)

        q_values = self.online(states)
        predicted = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.online(next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target(next_states).gather(1, next_actions).squeeze(1)

        targets = []
        defender_proxies: List[float] = []
        attacker_proxies: List[float] = []

        for idx, transition in enumerate(batch):
            scenario_returns: List[float] = []
            scenario_attacker: List[float] = []
            discount_factor = (1.0 - dones[idx].item())
            for scenario in transition.scenarios:
                reward_def = self._defender_reward_from_summary(transition.summary, scenario)
                reward_att = self._attacker_reward_from_summary(transition.summary, scenario)
                bootstrap = float(next_q_target[idx].item() * discount_factor)
                scenario_returns.append(reward_def + self.config.gamma * bootstrap)
                scenario_attacker.append(reward_att)
            defender_value = float(np.min(scenario_returns))
            attacker_value = float(np.min(scenario_attacker))
            defender_proxies.append(defender_value)
            attacker_proxies.append(attacker_value)
            targets.append(defender_value)

        target_tensor = torch.tensor(targets, device=self.device, dtype=torch.float32)
        return predicted, target_tensor, q_values, defender_proxies, attacker_proxies

    def update(self, buffer: ReplayBuffer, batch_size: int) -> dict:
        batch = buffer.sample(batch_size)
        predicted, target_tensor, q_values, defender_proxies, attacker_proxies = self._compute_targets(batch)

        loss = combined_loss(
            predicted_q=predicted,
            target_q=target_tensor,
            q_values=q_values,
            tau=self.config.tau_D,
            beta_ent=self.config.beta_ent,
            weight_decay_coeff=self.config.weight_decay,
            parameters=self.online.parameters(),
            kappa=self.config.huber_kappa,
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self._soft_update()

        entropy_value = -float(entropy_penalty(q_values.detach(), self.config.tau_D).item())
        return {
            "loss": float(loss.item()),
            "entropy": entropy_value,
            "defender_proxies": defender_proxies,
            "attacker_proxies": attacker_proxies,
        }

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
                target_param.data.mul_(1.0 - self.config.tau_soft)
                target_param.data.add_(self.config.tau_soft * online_param.data)

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
