from abc import ABC, abstractmethod

from src.formalisms.spaces import Space
from src.formalisms.distributions import Distribution


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
    def T(self, s, a) -> Distribution:  # | None:
        pass

    @abstractmethod
    def R(self, s, a) -> float:
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
