from src.formalisms.abstract_decision_processes import CMDP
from src.formalisms.distributions import KroneckerDistribution, DiscreteDistribution
from src.formalisms.primitives import IntState, IntAction, FiniteSpace


class ACMDPNeedingStochasticity(CMDP):

    def __init__(self):
        self.S = FiniteSpace({IntState(0), IntState(1), IntState(2)})
        self.A = frozenset({IntAction(1), IntAction(2)})
        self.gamma = 1.0
        self.K = 1
        self.initial_state_dist = KroneckerDistribution(IntState(0))
        self.c_tuple = (1.0,)

    def _inner_T(self, s, a):
        if s == IntState(0):
            if a == IntAction(1):
                return KroneckerDistribution(IntState(1))
            elif a == IntAction(2):
                return KroneckerDistribution(IntState(2))
            else:
                raise ValueError
        else:
            return KroneckerDistribution(s)

    def _inner_R(self, s, a):
        s_next_dist = self.T(s, a)
        # If T is deterministic, we know the next state
        if s_next_dist.is_degenerate():
            s_next = s_next_dist.sample()
        else:
            raise ValueError

        if s == IntState(0) and s_next == IntState(1):
            return 100.0
        elif s == IntState(0) and s_next == IntState(2):
            return 0.0
        elif s in [IntState(1), IntState(2)]:
            return 0.0
        else:
            raise NotImplementedError

    def _inner_C(self, k, s, a):
        s_next_dist = self.T(s, a)
        # If T is deterministic, we know the next state
        if s_next_dist.is_degenerate():
            s_next = s_next_dist.sample()
        else:
            raise ValueError

        if s == IntState(0) and s_next == IntState(1):
            return 3.0
        elif s == IntState(0) and s_next == IntState(2):
            return 0.0
        elif s in [IntState(1), IntState(2)]:
            return 0.0
        else:
            raise NotImplementedError

    def is_sink(self, s) -> bool:
        return s == IntState(1) or s == IntState(2)


class ASecondCMDPNeedingStochasticity(CMDP):

    def __init__(self):
        self.S = FiniteSpace({IntState(0), IntState(1)})
        self.A = frozenset({IntAction(1), IntAction(2)})
        self.gamma = 1.0
        self.K = 1
        self.initial_state_dist = KroneckerDistribution(IntState(0))
        self.c_tuple = (1.0,)

    def _inner_T(self, s, a):
        if s == IntState(0):
            if a == IntAction(1):
                return DiscreteDistribution({IntState(0): 0.9, IntState(1): 0.1})
            elif a == IntAction(2):
                return KroneckerDistribution(IntState(1))
            else:
                raise ValueError(a)
        elif s == IntState(1):
            return KroneckerDistribution(s)
        else:
            raise ValueError

    def _inner_R(self, s, a):
        if s == IntState(0):
            return 1.0
        else:
            return 0.0

    def _inner_C(self, k, s, a):
        if s == IntState(0):
            return 0.3
        else:
            return 0.0

    def is_sink(self, s) -> bool:
        return s == IntState(1)
