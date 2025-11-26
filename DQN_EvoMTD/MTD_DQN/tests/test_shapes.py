import numpy as np
import torch

from mtd_dqn.dqn.agent import AgentConfig, DQNAgent
from mtd_dqn.dqn.networks import DuelingQNetwork
from mtd_dqn.dqn.replay import ReplayBuffer
from mtd_dqn.uncertainty import Scenario


def test_dueling_q_output_shape() -> None:
    net = DuelingQNetwork(state_dim=12, action_dim=3, hidden_dims=[32, 32])
    batch = torch.randn(5, 12)
    out = net(batch)
    assert out.shape == (5, 3)


def test_replay_buffer_sampling() -> None:
    buffer = ReplayBuffer(capacity=4)
    scenario = Scenario(defender_scale=1.0, attacker_scale=1.0, transition=np.eye(2))
    state = np.zeros(3)
    buffer.add(state, 0, 0.0, state, False, 0, {"sap": 1.0, "sal": 1.0, "defender_cost": 0.0, "attacker_cost": 0.0, "guidance_penalty": 0.0, "alpha_I": 0.1, "incentive": 1.0}, [scenario])
    sample = buffer.sample(1)
    assert len(sample) == 1
    assert sample[0].state.shape == (3,)


def test_agent_target_shapes() -> None:
    agent = DQNAgent(
        state_dim=6,
        action_dim=2,
        hidden_dims=[16, 16],
        config=AgentConfig(gamma=0.95, tau_soft=0.01, tau_D=0.7, beta_ent=0.0, huber_kappa=1.0, weight_decay=0.0, lr=1e-3),
    )
    buffer = ReplayBuffer(capacity=10)
    scenario = Scenario(defender_scale=1.0, attacker_scale=1.0, transition=np.eye(2))
    summary = {"sap": 1.0, "sal": 1.0, "defender_cost": 0.1, "attacker_cost": 0.2, "guidance_penalty": 0.1, "alpha_I": 0.1, "incentive": 1.0}
    state = np.ones(6)
    buffer.add(state, 0, 0.5, state, False, 0, summary, [scenario])
    output = agent.update(buffer, batch_size=1)
    assert "loss" in output
