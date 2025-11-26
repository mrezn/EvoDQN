from pathlib import Path

import yaml

from mtd_dqn.env_edgecloud import EdgeCloudMTDEnv


def _load_config() -> dict:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "base.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_environment_reset_and_step() -> None:
    config = _load_config()
    env = EdgeCloudMTDEnv(config)
    state = env.reset()
    assert state.features.ndim == 1
    assert state.features.size > 0

    scenarios = env.sample_uncertainty(0, 0, count=2)
    assert len(scenarios) == 2
    next_state, attacker_reward, defender_reward, info = env.step(0, 0, scenarios[0])
    assert next_state.features.shape == state.features.shape
    assert isinstance(attacker_reward, float)
    assert isinstance(defender_reward, float)
    assert "sap" in info
