from src.formalisms.distributions import Distribution, KroneckerDistribution
from src.formalisms.abstract_decision_processes import MDP
from src.formalisms.primitives import IntState, IntAction, Space, FiniteSpace


class SimpleMDP(MDP):
    S: Space = FiniteSpace({IntState(0), IntState(1), IntState(2)})
    A: set = {IntAction(0), IntAction(1)}
    gamma: float = 0.9

    initial_state_dist: Distribution = KroneckerDistribution(IntState(1))

    def _inner_T(self, s: IntState, a: IntAction) -> Distribution:
        if s.n in {1, 2}:
            return KroneckerDistribution(s)
        else:
            return KroneckerDistribution(IntState(s.n + a.n + 1))

    def _inner_R(self, s: IntState, a: IntAction) -> float:
        if s.n in {1, 2}:
            return 0.0
        else:
            return float(a.n)

    def is_sink(self, s: IntState) -> bool:
        return s.n in {1, 2}
