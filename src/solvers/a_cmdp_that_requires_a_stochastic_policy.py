from src.formalisms.cmdp import CMDP
from src.formalisms.distributions import KroneckerDistribution, DiscreteDistribution
from src.formalisms.spaces import FiniteSpace


class ACMDPNeedingStochasiticity(CMDP):

    def __init__(self):
        self.S = FiniteSpace({"s0", "s1", "s2"})
        self.A = {1, 2}
        self.gamma = 1.0
        self.K = 1
        self.I = KroneckerDistribution("s0")
        self.perform_checks()

    def T(self, s, a):
        assert s == "s0"
        if a == 1:
            return KroneckerDistribution("s1")
        else:
            assert a == 2
            return KroneckerDistribution("s2")

    def R(self, s, a):
        # Only works because this is deterministic!
        s_next = self.T(s,a).sample()
        if s_next == "s1":
            return 100.0
        elif s_next == "s2":
            return 0.0
        else:
            raise ValueError

    def C(self, k, s, a):
        # Only works because this is deterministic!
        s_next = self.T(s,a).sample()
        assert k == 0
        if s_next == "s1":
            return 3.0
        elif s_next == "s2":
            return 0.0
        else:
            raise ValueError

    def c(self, k: int) -> float:
        return 1.0

    def is_sink(self, s) -> bool:
        return s == "s1" or s == "s2"


class ASecondCMDPNeedingStochasiticity(CMDP):

    def __init__(self):
        self.S = FiniteSpace({"s0", "s1"})
        self.A = {1, 2}
        self.gamma = 1.0
        self.K = 1
        self.I = KroneckerDistribution("s0")
        self.perform_checks()

    def T(self, s, a):
        assert s == "s0"
        if a == 1:
            return DiscreteDistribution({"s0": 0.9, "s1": 0.1})
        elif a == 2:
            return KroneckerDistribution("s1")
        else:
            raise ValueError(a)

    def R(self, s, a):
        # Only works because this is deterministic!
        s_next = self.T(s,a).sample()
        if s_next == "s0":
            return 3.0
        elif s_next == "s1":
            return 0.0
        else:
            raise ValueError

    def C(self, k, s, a):
        # Only works because this is deterministic!
        s_next = self.T(s,a).sample()
        assert k == 0
        if s_next == "s1":
            return 0.0
        elif s_next == "s0":
            return 0.3
        else:
            raise ValueError

    def c(self, k: int) -> float:
        return 1.0

    def is_sink(self, s) -> bool:
        return s == "s1"
