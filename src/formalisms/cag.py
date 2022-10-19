from abc import ABC, abstractmethod
from itertools import product
from typing import Tuple

from src.formalisms.abstract_process import AbstractProcess, validate_T, validate_R
from src.formalisms.distributions import Distribution


class CAG(AbstractProcess, ABC):
    h_A: set = None
    r_A: set = None

    initial_state_theta_dist: Distribution = None

    s_0 = None
    Theta: set = None

    @property
    def A(self):
        return set(product(self.h_A, self.r_A))

    @validate_T
    def T(self, s, action_pair: Tuple[object, object]) -> Distribution:
        if not isinstance(action_pair, Tuple):
            raise TypeError
        elif len(action_pair) != 2:
            raise TypeError
        else:
            h_a, r_a = action_pair
            return self.split_T(s, h_a, r_a)

    @abstractmethod
    def split_T(self, s, h_a, r_a) -> Distribution:
        pass

    @validate_R
    def R(self, s, action_pair: Tuple[object, object]) -> float:
        if not isinstance(action_pair, Tuple):
            raise TypeError
        elif len(action_pair) != 2:
            raise TypeError
        else:
            h_a, r_a = action_pair
            return self.split_R(s, h_a, r_a)

    @abstractmethod
    def split_R(self, s, h_a, r_a) -> float:
        pass

    @abstractmethod
    def C(self, k: int, theta, s, h_a, r_a) -> float:
        raise NotImplementedError

    def check_init_dist_is_valid(self):
        supp = set(self.initial_state_theta_dist.support())
        for x in supp:
            try:
                s, theta = x
            except TypeError as e:
                raise ValueError("cag.I should only have support over {s_0} x Theta!", e)
            except Exception as e:
                raise e

            if s != self.s_0:
                raise ValueError("init dist should only be supported on a single state")
            if theta not in self.Theta:
                raise ValueError(f"theta={theta} not in cag.Theta={self.Theta}")

    def check_is_instantiated(self):
        if self.Theta is None:
            raise ValueError("Something hasn't been instantiated!")

        if self.initial_state_theta_dist is None:
            raise ValueError("init dist hasn't been instantiated!")

        super().check_is_instantiated()

    def test_cost_for_sinks(self):
        sinks = {s for s in self.S if self.is_sink(s)}
        for s in sinks:
            for h_a in self.h_A:
                for r_a in self.r_A:
                    for theta in self.Theta:
                        for k in range(self.K):
                            cost = self.C(k, theta, s, h_a, r_a)
                            if cost != 0.0:
                                raise ValueError("Cost should be 0 at a sink")
