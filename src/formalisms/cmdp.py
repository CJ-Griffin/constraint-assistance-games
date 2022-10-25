from src.formalisms.decision_process import DecisionProcess
from src.formalisms.distributions import *


class CMDP(DecisionProcess, ABC):
    initial_state_dist: Distribution = None

    @abstractmethod
    def C(self, k: int, s, a) -> float:
        raise NotImplementedError

    def check_init_dist_is_valid(self):
        for s in self.initial_state_dist.support():
            if s not in self.S:
                raise ValueError(f"state s={s} is s.t. I(s) = "
                                 f"{self.initial_state_dist.get_probability(s)} but s is not in self.S={self.S}")

    def test_cost_for_sinks(self):
        sinks = {s for s in self.S if self.is_sink(s)}
        for s in sinks:
            for a in self.A:
                for k in range(self.K):
                    if self.C(k, s, a) != 0.0:
                        raise ValueError("Cost should be 0 at a sink")

    def check_is_instantiated(self):
        if self.initial_state_dist is None:
            raise ValueError("init dist hasn't been instantiated!")
        super().check_is_instantiated()


