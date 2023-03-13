from abc import ABC, abstractmethod, ABCMeta
from typing import Callable

import numpy as np

from src.formalisms.primitives import Plan
from src.utils.renderer import render


class Distribution(ABC):
    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def get_probability(self, x):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    def is_degenerate(self):
        sup = list(self.support())
        if len(sup) == 1:
            return True
        else:
            return False

    def is_almost_degenerate(self, tolerance: float = 1e-6):
        sup = list(self.support())
        sup.sort(key=self.get_probability, reverse=True)
        if self.get_probability(sup[0]) >= 1.0 - tolerance:
            return True
        else:
            return False


class ContinuousDistribution(Distribution, metaclass=ABCMeta):
    def support(self):
        raise ValueError("Cannot ask for support of a continuous distribution")

    def get_probability(self, x):
        raise ValueError("Cannot get probability of an element of a continuous distribution")


# TODO make this more generic, s.t. the dictionary is only used in .support() and .get_probability()
#   This way we can make another implementation that uses as state -> ind map and np array
class DiscreteDistribution(Distribution):
    def __hash__(self):
        vals = frozenset([
            (x, self.get_probability(x))
            for x in self.support()
        ])
        return hash(vals)

    def __init__(self, option_prob_map):
        if type(option_prob_map) == np.ndarray:
            assert len(option_prob_map.shape) == 1
            option_prob_map = dict(zip(range(len(option_prob_map)), option_prob_map))
        elif type(option_prob_map) == dict:
            pass
        else:
            raise ValueError

        self.option_prob_map = option_prob_map
        self.check_sums_to_1_and_positive()

    def get_mode(self):
        return max(self.support(), key=self.get_probability)

    @property
    def supported_option_prob_map(self):
        return {k: p for k, p in self.option_prob_map.items() if p > 0}

    def __eq__(self, other):
        if not isinstance(other, Distribution):
            return False
        else:
            sup1 = set(self.support())
            sup2 = set(other.support())
            if sup1 != sup2:
                return False
            else:
                for x in sup1:
                    p1 = self.get_probability(x)
                    p2 = other.get_probability(x)
                    if not np.isclose(p1, p2):
                        return False
                return True

    def check_sums_to_1_and_positive(self, tolerance: float = 1e-6):
        sum_of_probs = sum(self.option_prob_map.values())
        if sum_of_probs != 1.0:
            if np.abs(sum_of_probs - 1.0) < tolerance:
                pass
            else:
                op_map = self.option_prob_map.items()
                raise ValueError("probabilities' sum is not close to 1!", self.option_prob_map, sum_of_probs)
        if any([x < 0 for x in self.option_prob_map.values()]):
            for k in self.option_prob_map.keys():
                if self.option_prob_map[k] < 0:
                    if self.option_prob_map[k] > -tolerance:
                        self.option_prob_map[k] = 0
                    else:
                        raise ValueError(k, self.option_prob_map[k])

    def support(self):
        for x, p in self.option_prob_map.items():
            if p > 0:
                yield x

    def get_nonzero_probability_list(self):
        return [self.option_prob_map[k] for k in self.option_prob_map if self.option_prob_map[k] > 0]

    def get_probability(self, option):
        return self.option_prob_map.get(option, 0.0)

    def sample(self):
        options, probs = zip(*self.option_prob_map.items())
        try:
            idx = np.random.choice(len(options), p=probs)
        except ValueError as ve:
            sum_probs = sum(probs)
            if sum_probs != 1.0:
                raise ValueError(sum_probs, ve)
            else:
                raise ve
        return options[idx]

    def __str__(self):
        if self.is_degenerate():
            x = self.sample()
            if type(x) is tuple:
                x = [str(z) for z in x]
            return f"<KDist : {str(x)}>"
        d = {str(v): self.option_prob_map[v] for v in self.support()}
        return f"<Distribution : {d}>"

    def __repr__(self) -> object:
        n = len(list(self.support()))
        if n < 10:
            if n == 1:
                sup_str = str(list(self.support())[0]) + "~ 1"
            else:
                sup_str = ", ".join([f"{x}~ {self.get_probability(x)}" for x in list(self.support())])
            return repr(f"<{type(self).__name__}= [{sup_str}]>")
        else:
            return super.__repr__(self)

    def expectation(self, f):
        pairs = [
            (self.get_probability(x), f(x))
            for x in self.support()
        ]
        return sum([x * y for (x, y) in pairs])

    def render(self):
        r_str = f"{type(self).__name__}"
        sup = list(self.support())
        if len(sup) > 1:
            for x in sup:
                xstr = render(x)
                p = self.get_probability(x)
                r_str += f"\n |  {p: 4.3f} ~> {xstr}"
        else:
            r_str += f"(1~> {render(sup[0])})"
        return r_str


"""
A special kind of distribution for more easily representing beliefs over parameters.
It is equivalent to a DiscreteDistribution, but can be updated more easily.
It is specifically used when there is a prior discrete distribution β_0 over Theta,
and subsequent distributions β_t are s.t. β_t(theta) is proportional to β_0(theta) or 0.
"""


class FiniteParameterDistribution(DiscreteDistribution):
    def __init__(self, beta_0: Distribution, subset: frozenset):
        if not isinstance(subset, frozenset):
            raise NotImplementedError
        if len(subset) == 0:
            raise ValueError

        if not subset.issubset(set(beta_0.support())):
            raise ValueError
        self.subset: frozenset = subset
        self.beta_0 = beta_0
        self.norm_const = sum([self.beta_0.get_probability(x) for x in self.subset])
        self.option_prob_map = {
            x: self.beta_0.get_probability(x) / self.norm_const
            for x in self.subset
        }
        super().__init__(self.option_prob_map)

    def get_collapsed_distribution_from_filter_func(self, filter_func: Callable[[object], bool]):
        new_subset = frozenset({
            x
            for x in self.subset
            if filter_func(x)
        })
        if len(new_subset) == 0:
            raise ValueError
        return FiniteParameterDistribution(beta_0=self.beta_0, subset=new_subset)

    def get_collapsed_distribution_from_lambda_ah(self, h_lambda: Plan, ah: object):
        new_subset = frozenset({
            theta
            for theta in self.subset
            if h_lambda(theta) == ah
        })
        if len(new_subset) == 0:
            raise ValueError
        return FiniteParameterDistribution(beta_0=self.beta_0, subset=new_subset)

    def get_collapsed_distribution_from_lambda_theta(self, h_lambda: Plan, theta: object):
        new_subset = frozenset({
            theta_other
            for theta_other in self.subset
            if h_lambda(theta_other) == h_lambda(theta)
        })
        if len(new_subset) == 0:
            raise ValueError
        return FiniteParameterDistribution(beta_0=self.beta_0, subset=new_subset)

    def render(self):
        # r_str = f"{type(self).__name__}"
        sup = list(self.support())
        r_str = ""
        for x in sup:
            xstr = render(x)
            p = self.get_probability(x)
            r_str += f"\n| β({xstr}) = {p: 4.3f}"
        return r_str


class KroneckerDistribution(DiscreteDistribution):
    def __init__(self, x):
        super().__init__({x: 1.0})
        self.x = x

    def support(self):
        yield self.x

    def get_probability(self, x):
        return 1.0 if x == self.x else 0.0

    def sample(self):
        return self.x


class UniformDiscreteDistribution(DiscreteDistribution):
    def __init__(self, options):
        p = 1.0 / len(options)
        super().__init__({option: p for option in options})


def split_initial_dist_into_s_and_beta(joint_initial_dist: Distribution) -> (object, Distribution):
    if isinstance(joint_initial_dist, FiniteParameterDistribution):
        raise NotImplementedError
    elif isinstance(joint_initial_dist, DiscreteDistribution):
        sup = joint_initial_dist.support()
        support_over_states = {
            s for (s, theta) in sup
        }
        if len(support_over_states) != 1:
            raise ValueError(f"Reduction to coordination BCMDP only supported when s_0 is deterministic:"
                             f" dist.support()={sup}")
        else:
            s = list(support_over_states)[0]

        theta_map = {
            theta: joint_initial_dist.get_probability((s, theta))
            for _, theta in joint_initial_dist.support()
        }

        b = DiscreteDistribution(theta_map)
        beta = FiniteParameterDistribution(b, frozenset(b.support()))

        return (s, beta)
    else:
        raise NotImplementedError
