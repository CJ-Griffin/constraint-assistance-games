from abc import ABC, abstractmethod
from itertools import product
from random import shuffle
from typing import FrozenSet, Tuple

from src.formalisms.distributions import Distribution
from src.formalisms.primitives import State, Action, ActionPair, FiniteSpace


class DecisionProcess(ABC):
    S: FiniteSpace = None
    A: FrozenSet[Action] = None
    gamma: float = None

    c_tuple: Tuple[float]

    should_debug: bool = True

    @property
    def K(self):
        return len(self.c_tuple)

    def T(self, s: State, a: Action) -> Distribution:
        if not self.should_debug:
            return self._inner_T(s, a)
        else:
            self.check_s(s)
            self.check_a(a)

            dist = self._inner_T(s, a)

            if not isinstance(dist, Distribution):
                raise TypeError
            else:
                sample_state = dist.sample()
                if self.is_sink(s):
                    if not dist.is_degenerate():
                        raise ValueError
                    elif sample_state != s:
                        raise ValueError
                else:
                    if not isinstance(sample_state, State):
                        raise TypeError
                    elif s not in self.S:
                        raise ValueError
            return dist

    @abstractmethod
    def _inner_T(self, s: State, a: Action) -> Distribution:
        pass

    def R(self, s: State, a: Action) -> float:
        if not self.should_debug:
            return self._inner_R(s, a)
        else:
            self.check_s(s)
            self.check_a(a)

            r = self._inner_R(s, a)

            if not isinstance(r, float):
                raise TypeError
            elif self.is_sink(s) and r != 0.0:
                raise ValueError
            return r

    @abstractmethod
    def _inner_R(self, s: State, a: Action) -> float:
        pass

    def c(self, k: int) -> float:
        if self.should_debug:
            if not isinstance(k, int):
                raise TypeError
            elif not (0 <= k < self.K):
                raise ValueError
        return self.c_tuple[k]

    @abstractmethod
    def is_sink(self, s: State) -> bool:
        raise NotImplementedError

    def check_s(self, s: State):
        if not isinstance(s, State):
            raise TypeError
        elif s not in self.S:
            raise ValueError

    def check_a(self, a: Action):
        if not isinstance(a, Action):
            raise TypeError
        elif a not in self.A:
            raise ValueError

    def perform_checks(self):
        for attr in self.__dict__.values():
            if isinstance(attr, DecisionProcess):
                attr.perform_checks()
        self.check_is_instantiated()
        self.check_init_dist_is_valid()
        self.check_transition_function()

    def check_c_tuple(self):
        if not isinstance(self.c_tuple, tuple):
            raise TypeError
        elif any(not isinstance(ck, float) for ck in self.c_tuple):
            raise TypeError
        elif any(not ck >= 0 for ck in self.c_tuple):
            raise ValueError

    def check_is_instantiated(self):
        components = [
            self.S,
            self.A,
            self.gamma,
            self.K,
            self.c_tuple,
        ]
        if None in components:
            raise ValueError("Something hasn't been instantiated!")

    def check_transition_function(self):
        pairs = [(s, a) for s in self.S for a in self.A]
        shuffle(pairs)
        for s, a in pairs[:min(100, len(pairs))]:
            _ = self.T(s, a)

    @abstractmethod
    def check_init_dist_is_valid(self):
        pass

    def get_size_string(self):
        if hasattr(self, "Theta"):
            x, y, z = len(self.S), len(self.A), len(self.Theta)
            size_str = f"|S|={x}, |A|={y}, |ϴ|={z}. |SxAxϴ| = {x * y * z}"
        else:
            x, y = len(self.S), len(self.A)
            size_str = f"|S|={x}, |A|={y} |SxA| = {x * y}"

        return size_str


def validate_T(process_T):
    def wrapper(process_self: DecisionProcess, s, a, *args, **kwargs):
        if len(args) > 0 or len(kwargs):
            raise TypeError("Excessive arguments to T: expected T(s,a) but got", s, a, "and then", args, kwargs)
        else:
            if not isinstance(s, State):
                raise TypeError
            elif s not in process_self.S:
                raise ValueError(f"s={s} not in S={process_self.S}")
            elif not isinstance(a, Action):
                raise TypeError
            elif a not in process_self.A:
                raise ValueError(f"a={a} not in A={process_self.A}")
            else:
                next_s_dist = process_T(process_self, s, a)

                if not isinstance(next_s_dist, Distribution):
                    raise TypeError
                else:
                    next_s = next_s_dist.sample()
                    if not isinstance(next_s, State):
                        raise TypeError
                    elif next_s not in process_self.S:
                        raise ValueError(f"s={s} not in S={process_self.S}")
                    else:
                        return next_s_dist

    return wrapper


def validate_R(process_R):
    def wrapper(process_self: DecisionProcess, s, a, *args, **kwargs):
        if len(args) > 0 or len(kwargs):
            raise TypeError("Excessive arguments to R: expected R(s,a) but got", s, a, "and then", args, kwargs)
        else:
            if not isinstance(s, State):
                raise TypeError
            elif s not in process_self.S:
                raise ValueError(f"s={s} not in S={process_self.S}")
            elif not isinstance(a, Action):
                raise TypeError
            elif a not in process_self.A:
                raise ValueError(f"a={a} not in A={process_self.A}")
            else:
                reward = process_R(process_self, s, a)

                if not isinstance(reward, float):
                    raise TypeError
                elif process_self.is_sink(s) and reward != 0.0:
                    raise ValueError("There should be no reward for acting in a sink state!")
                else:
                    return reward

    return wrapper


def validate_c(process_c):
    def wrapper(process_self: DecisionProcess, k, *args, **kwargs):
        if len(args) > 0 or len(kwargs):
            raise TypeError("Excessive arguments to R: expected R(s,a) but got", args, kwargs)
        else:
            if k not in range(process_self.K):
                raise ValueError(f"k={k} not in range(K)={list(range(process_self.K))}")
            else:
                c = process_c(process_self, k)

                if not isinstance(c, float):
                    raise TypeError
                elif c < 0:
                    raise ValueError("costs must be positive")
                else:
                    return c

    return wrapper


class CAG(DecisionProcess, ABC):
    h_A: frozenset = None
    r_A: frozenset = None

    initial_state_theta_dist: Distribution = None

    s_0 = None
    Theta: set = None

    @property
    def A(self):
        return set(
            ActionPair(ha, ra)
            for ha, ra in product(self.h_A, self.r_A)
        )

    @abstractmethod
    def _split_inner_T(self, s: State, h_a: Action, r_a: Action) -> Distribution:
        pass

    def _inner_T(self, s: State, a: ActionPair) -> Distribution:
        if isinstance(a, ActionPair):
            h_a, r_a = a
        else:
            raise TypeError
        return self._split_inner_T(s, h_a, r_a)

    @validate_R
    def _inner_R(self, s: State, action_pair: ActionPair) -> float:
        if not isinstance(action_pair, ActionPair):
            raise TypeError
        else:
            h_a = action_pair[0]
            r_a = action_pair[1]
            return self.split_R(s, h_a, r_a)

    @abstractmethod
    def split_R(self, s: State, h_a: Action, r_a: Action) -> float:
        pass

    def C(self, k: int, theta, s: State, h_a: Action, r_a: Action) -> float:
        if not self.should_debug:
            return self._inner_C(k, theta, s, h_a, r_a)
        else:
            self.check_s(s)
            self.check_a(ActionPair(h_a, r_a))

            if not isinstance(k, int):
                raise TypeError
            elif not (0 <= k < self.K):
                raise ValueError
            elif theta not in self.Theta:
                raise ValueError

            c = self._inner_C(k, theta, s, h_a, r_a)

            if not isinstance(c, float):
                raise TypeError
            elif self.is_sink(s) and c != 0.0:
                raise ValueError
            return c

    @abstractmethod
    def _inner_C(self, k: int, theta, s: State, h_a: Action, r_a: Action) -> float:
        raise NotImplementedError

    def check_init_dist_is_valid(self):
        supp = set(self.initial_state_theta_dist.support())
        for x in supp:
            if not isinstance(x, tuple):
                raise TypeError("cag.I should only have support over {s_0} x Theta!")
            else:
                s, theta = x
                self.check_s(s)
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


class CMDP(DecisionProcess, ABC):
    initial_state_dist: Distribution = None

    def C(self, k: int, s: State, a: Action) -> float:
        if not self.should_debug:
            return self._inner_C(k, s, a)
        else:
            self.check_s(s)
            self.check_a(a)

            if not isinstance(k, int):
                raise TypeError
            elif not (0 <= k < self.K):
                raise ValueError

            c = self._inner_C(k, s, a)

            if not isinstance(c, float):
                raise TypeError
            elif self.is_sink(s) and c != 0.0:
                raise ValueError
            return c

    @abstractmethod
    def _inner_C(self, k: int, s: State, a) -> float:
        raise NotImplementedError

    def check_init_dist_is_valid(self):
        for s in self.initial_state_dist.support():
            if s not in self.S:
                raise ValueError(f"state s={s} is s.t. I(s) = "
                                 f"{self.initial_state_dist.get_probability(s)} but s is not in self.S={self.S}")

    def check_is_instantiated(self):
        if self.initial_state_dist is None:
            raise ValueError("init dist hasn't been instantiated!")
        super().check_is_instantiated()


class MDP(DecisionProcess, ABC):
    initial_state_dist: Distribution = None
    c_tuple = tuple()

    def check_init_dist_is_valid(self):
        for s in self.initial_state_dist.support():
            if s not in self.S:
                raise ValueError(f"state s={s} is s.t. I(s) = "
                                 f"{self.initial_state_dist.get_probability(s)} but s is not in self.S={self.S}")

    def check_is_instantiated(self):
        if self.initial_state_dist is None:
            raise ValueError("init dist hasn't been instantiated!")
        super().check_is_instantiated()
