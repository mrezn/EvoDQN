from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..uncertainty import Scenario


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    attacker_action: int
    summary: Dict[str, float]
    scenarios: List[Scenario]


class ReplayBuffer:

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._storage: List[Transition] = []
        self._position = 0

    def __len__(self) -> int:  
        return len(self._storage)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        attacker_action: int,
        summary: Dict[str, float],
        scenarios: Sequence[Scenario],
    ) -> None:
        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
            attacker_action=int(attacker_action),
            summary=dict(summary),
            scenarios=list(scenarios),
        )
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size > len(self._storage):
            raise ValueError("Not enough samples in buffer to satisfy batch size.")
        idx = np.random.choice(len(self._storage), size=batch_size, replace=False)
        return [self._storage[i] for i in idx]

    def prioritised_sample(self, batch_size: int) -> List[Transition]: 
        return self.sample(batch_size)
