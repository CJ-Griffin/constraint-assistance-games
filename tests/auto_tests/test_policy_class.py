from unittest import TestCase

from src.formalisms.distributions import UniformDiscreteDistribution
from src.formalisms.policy import FiniteCMDPPolicy
from src.formalisms.spaces import FiniteSpace


class TestPolicyClass(TestCase):
    def setUp(self):
        self.S = FiniteSpace({1, 2, 3})
        self.A = {0, 1}
        self.uni_dist = UniformDiscreteDistribution(self.A)
        self.map = {s: self.uni_dist for s in self.S}

    def test_normal(self):
        policy = FiniteCMDPPolicy(self.S, self.A, self.map)
        dist = policy(1)
        a = dist.sample()
        if a not in self.A:
            raise ValueError
