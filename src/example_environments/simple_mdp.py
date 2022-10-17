from src.formalisms.distributions import Distribution, KroneckerDistribution
from src.formalisms.mdp import MDP
from src.formalisms.spaces import FiniteSpace, Space


class SimpleMDP(MDP):
    S: Space = FiniteSpace({0, 1, 2})
    A: set = {0, 1}
    gamma: float = 0.9

    initial_state_dist: Distribution = KroneckerDistribution(1)

    def T(self, s, a) -> Distribution:
        if s in {1, 2}:
            return KroneckerDistribution(s)
        else:
            return KroneckerDistribution(s + a + 1)

    def R(self, s, a) -> float:
        if s in {1, 2}:
            return 0.0
        else:
            return float(a)

    def is_sink(self, s) -> bool:
        return s in {1, 2}

    def c(self, k: int) -> float:
        raise NotImplementedError
