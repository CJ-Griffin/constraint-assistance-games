from abc import ABC

from src.formalisms.decision_process import DecisionProcess
from src.formalisms.distributions import Distribution


class MDP(DecisionProcess, ABC):
    K: int = 0
    initial_state_dist: Distribution = None

    def c(self, k: int) -> float:
        raise NotImplementedError

    def check_init_dist_is_valid(self):
        for s in self.initial_state_dist.support():
            if s not in self.S:
                raise ValueError(f"state s={s} is s.t. I(s) = "
                                 f"{self.initial_state_dist.get_probability(s)} but s is not in self.S={self.S}")

    def test_cost_for_sinks(self):
        pass

    def check_is_instantiated(self):
        if self.initial_state_dist is None:
            raise ValueError("init dist hasn't been instantiated!")
        super().check_is_instantiated()
