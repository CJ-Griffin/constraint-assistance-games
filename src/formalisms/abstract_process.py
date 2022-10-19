from abc import ABC, abstractmethod

from src.formalisms.distributions import Distribution
from src.formalisms.spaces import Space


class AbstractProcess(ABC):
    S: Space = None
    A: set = None
    gamma: float = None
    K: int = None

    @abstractmethod
    def T(self, s, a) -> Distribution:
        pass

    @abstractmethod
    def R(self, s, a) -> float:
        pass

    @abstractmethod
    def c(self, k: int) -> float:
        pass

    @abstractmethod
    def is_sink(self, s) -> bool:
        raise NotImplementedError

    def enable_debug_mode(self):
        self.perform_checks()
        self.enable_input_output_validators()
        for attr in self.__dict__.values():
            if isinstance(attr, AbstractProcess):
                attr.enable_debug_mode()

    def perform_checks(self):
        self.check_is_instantiated()
        self.check_init_dist_is_valid()
        self.check_sinks()
        self.check_transition_function()

    def check_sinks(self):
        self.test_cost_for_sinks()
        sinks = {s for s in self.S if self.is_sink(s)}
        for s in sinks:
            for a in self.A:
                dist = self.T(s, a)
                p = dist.get_probability(s)
                if p < 1.0:
                    raise ValueError("T should be self loop at a sink")
                r = self.R(s, a)
                if r != 0.0:
                    raise ValueError("Reward should be 0 at a sink")

    @abstractmethod
    def test_cost_for_sinks(self):
        pass

    def check_is_instantiated(self):
        components = [
            self.S,
            self.A,
            self.gamma,
            self.K,
        ]
        if None in components:
            raise ValueError("Something hasn't been instantiated!")

    def check_transition_function(self):
        for s in self.S:
            for a in self.A:
                next_state_dist = self.T(s, a)
                for s_next in next_state_dist.support():
                    if s_next not in self.S:
                        raise ValueError

    @abstractmethod
    def check_init_dist_is_valid(self):
        pass

    def enable_input_output_validators(self):
        pass

    # TODO try to remove this method when possible (make rendering a property of the state)
    def render_state_as_string(self, s) -> str:
        return str(s)


def validate_T(process_T):
    def wrapper(process_self: AbstractProcess, s, a, *args, **kwargs):
        if len(args) > 0 or len(kwargs):
            raise TypeError("Excessive arguments to T: expected T(s,a) but got", s, a, "and then", args, kwargs)
        else:
            if s not in process_self.S:
                raise ValueError(f"s={s} not in S={process_self.S}")
            elif a not in process_self.A:
                raise ValueError(f"a={a} not in a={process_self.A}")
            else:
                next_s_dist = process_T(process_self, s, a)

                if not isinstance(next_s_dist, Distribution):
                    raise TypeError
                else:
                    next_s = next_s_dist.sample()
                    if next_s not in process_self.S:
                        raise ValueError(f"s={s} not in S={process_self.S}")
                    else:
                        return next_s_dist

    return wrapper


def validate_R(process_R):
    def wrapper(process_self: AbstractProcess, s, a, *args, **kwargs):
        if len(args) > 0 or len(kwargs):
            raise TypeError("Excessive arguments to R: expected R(s,a) but got", s, a, "and then", args, kwargs)
        else:
            if s not in process_self.S:
                raise ValueError(f"s={s} not in S={process_self.S}")
            elif a not in process_self.A:
                raise ValueError(f"a={a} not in a={process_self.A}")
            else:
                reward = process_R(process_self, s, a)

                if not isinstance(reward, float):
                    raise TypeError
                else:
                    return reward

    return wrapper


def validate_c(process_c):
    def wrapper(process_self: AbstractProcess, k, *args, **kwargs):
        if len(args) > 0 or len(kwargs):
            raise TypeError("Excessive arguments to R: expected R(s,a) but got", s, a, "and then", args, kwargs)
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
