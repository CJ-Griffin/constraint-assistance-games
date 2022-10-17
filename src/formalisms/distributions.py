from abc import ABC, abstractmethod
from itertools import product
from typing import Callable
import numpy as np


# TODO: currently this is only well-defined for finite distributions, make more general or change name
# Taken from assistance-games
# TODO ask Justin about the math.isclose nastiness here
class Distribution(ABC):
    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def get_probability(self, x):
        pass

    def sample(self):
        sample = np.random.random()
        total_prob = 0
        for x in self.support():
            total_prob += self.get_probability(x)
            if total_prob >= sample:
                return x

        raise ValueError("Total probability was less than 1")

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

    def __hash__(self):
        vals = frozenset([
            (x, self.get_probability(x))
            for x in self.support()
        ])
        return hash(vals)


class ContinuousDistribution(Distribution):
    def support(self):
        raise ValueError("Cannot ask for support of a continuous distribution")

    def get_probability(self, x):
        raise ValueError("Cannot get probability of an element of a continuous distribution")


class DiscreteDistribution(Distribution):
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

    def check_sums_to_1_and_positive(self, tolerance: float = 1e-6):
        sum_of_probs = sum(self.option_prob_map.values())
        if sum_of_probs != 1.0:
            if np.abs(sum_of_probs - 1.0) < tolerance:
                pass
            else:
                map = self.option_prob_map.items()
                raise ValueError("probabilities' sum is not close to 1!", self.option_prob_map, sum_of_probs)
        if any([x < 0 for x in self.option_prob_map.values()]):
            for k in self.option_prob_map.keys():
                if self.option_prob_map[k] < 0:
                    if self.option_prob_map[k] > -tolerance:
                        self.option_prob_map[k] = 0
                    else:
                        raise ValueError

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
        if len(list(self.support())) == 1:
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


"""
A special kind of distribution for more easily representing beliefs over parameters.
It is equivalent to a DiscreteDistribution, but can be updated more easily.
It is specifically used when there is a prior distribution β_0 over Theta, and subsequent distributions
β_t are s.t. β_t(theta) is proportional to β_0(theta) or 0.
"""


class FiniteParameterDistribution(DiscreteDistribution):
    def __init__(self, beta_0, subset: frozenset):
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

    def get_collapsed_distribution(self, filter_func: Callable[[object], bool]):
        new_subset = frozenset({
            x
            for x in self.subset
            if filter_func(x)
        })
        return FiniteParameterDistribution(beta_0=self.beta_0, subset=new_subset)


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


class PairOfIndependentDistributions(Distribution):

    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def support(self):
        return product(self.d1.support(), self.d2.support())

    def get_probability(self, x):
        x1, x2 = x
        return self.d1.get_probability(x1) * self.d2.get_probability(x2)


class UniformDiscreteDistribution(DiscreteDistribution):
    def __init__(self, options):
        p = 1.0 / len(options)
        super().__init__({option: p for option in options})


class UniformContinuousDistribution(ContinuousDistribution):
    def __init__(self, lows, highs):
        self.lows = lows
        self.highs = highs

    def sample(self):
        return np.random.uniform(self.lows, self.highs)


class MapDistribution(Distribution):
    def __init__(self, f, finv, base_dist):
        self.f = f
        self.finv = f
        self.base_dist = base_dist

    def support(self):
        for x in self.base_dist.support():
            yield self.f(x)

    def get_probability(self, option):
        return self.base_dist.get_probability(self.finv(option))

    def sample(self):
        return self.f(self.base_dist.sample())
