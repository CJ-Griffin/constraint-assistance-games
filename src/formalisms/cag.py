from src.formalisms.distributions import *
from abc import ABC, abstractmethod


class CAG(ABC):
    S: set = None
    h_A: set = None
    r_A: set = None
    Theta: set = None
    gamma: float = None
    K: int = None

    s_0 = None
    I: Distribution = None

    state = None

    def perform_checks(self):
        self.check_is_instantiated()
        self.check_s0_is_valid()
        self.check_I_is_valid()

    @abstractmethod
    def T(self, s, h_a, r_a) -> Distribution | None:
        pass

    @abstractmethod
    def R(self, s, h_a, r_a, next_s) -> float:
        pass

    @abstractmethod
    def C(self, k: int, theta, s, h_a, r_a, next_s) -> float:
        assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
        raise NotImplementedError

    @abstractmethod
    def c(self, k: int) -> float:
        # this should be
        # assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
        raise NotImplementedError

    @abstractmethod
    def is_sink(self, s) -> bool:
        # this should be
        # assert s in self.S, f"s={s} is not in S={self.S}"
        raise NotImplementedError

    def check_is_instantiated(self):
        components = [
            self.S,
            self.h_A,
            self.r_A,
            self.I,
            self.Theta,
            self.gamma,
            self.K,
            self.I
        ]
        if None in components:
            raise ValueError("Something hasn't been instantiated!")

    def check_s0_is_valid(self):
        if self.s_0 not in self.S:
            raise ValueError("s_0 not in S")

    def check_I_is_valid(self):
        supp = set(self.I.support())
        S_cross_Theta = {(s, theta) for s in self.S for theta in self.Theta}
        if not supp.issubset(S_cross_Theta):
            raise ValueError("I should only have support over S x Theta!")

    def render_state_as_string(self, s) -> str:
        return str(s)


