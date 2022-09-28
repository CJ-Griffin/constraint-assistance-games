import numpy as np

from src.formalisms import CMDP
from src.formalisms.distributions import *
from abc import ABC, abstractmethod

from src.formalisms.spaces import Space


class MDP(ABC):
    S: Space = None
    A: set = None
    gamma: float = None

    I: Distribution = None

    state = None

    def perform_checks(self):
        self.check_is_instantiated()
        self.check_I_is_valid()

    @abstractmethod
    def T(self, s, a) -> Distribution | None:
        pass

    @abstractmethod
    def R(self, s, a, next_s) -> float:
        pass

    @abstractmethod
    def is_sink(self, s) -> bool:
        # this should be
        # assert s in self.S, f"s={s} is not in S={self.S}"
        raise NotImplementedError

    def check_is_instantiated(self):
        components = [
            self.S,
            self.A,
            self.I,
            self.gamma,
        ]
        if None in components:
            raise ValueError("Something hasn't been instantiated!")

    def check_I_is_valid(self):
        for s in self.I.support():
            if s not in self.S:
                raise ValueError(f"state s={s} is s.t. I(s) = "
                                 f"{self.I.get_probability(s)} but s is not in self.S={self.S}")
        return True

    def render_state_as_string(self, s) -> str:
        return str(s)


class Lagrangian_CMDP_to_MDP(MDP):
    def __init__(self, cmdp: CMDP, lagrange_multiplier: list):
        self.cmdp = cmdp

        self.S = cmdp.S
        self.A = cmdp.A
        self.gamma = cmdp.gamma
        self.I = cmdp.I

        self.perform_checks()

        self.lagrange_multiplier = lagrange_multiplier
        if isinstance(self.lagrange_multiplier, float) or isinstance(self.lagrange_multiplier, int):
            self.lagrange_multiplier = [self.lagrange_multiplier]
        if len(self.lagrange_multiplier) != cmdp.K:
            raise ValueError
        if any([x < 0 for x in self.lagrange_multiplier]):
            raise ValueError

    def R(self, s, a, s_next) -> float:
        costs = [self.cmdp.C(k, s, a, s_next) for k in range(self.cmdp.K)]
        weighted_costs = [
            self.lagrange_multiplier[k] * costs[k] for k in range(self.cmdp.K)
        ]
        reward = self.cmdp.R(s, a, s_next)
        return reward - sum(weighted_costs)

    def T(self, s, a):
        return self.cmdp.T(s, a)

    def is_sink(self, s) -> bool:
        return self.cmdp.is_sink(s)
