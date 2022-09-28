from abc import ABC, abstractmethod
from itertools import product

import numpy as np


# TODO: currently this is only well-defined for finite distributions, make more general or change name
# Taken from assistance-games
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
                    if self.get_probability(x) != other.get_probability(x):
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
    def __init__(self, option_prob_map, is_leniant: bool = True):
        if type(option_prob_map) == np.ndarray:
            assert len(option_prob_map.shape) == 1
            option_prob_map = dict(zip(range(len(option_prob_map)), option_prob_map))
        elif type(option_prob_map) != dict:
            raise ValueError

        self.option_prob_map = option_prob_map
        self.check_sums_to_1(is_leniant)

    def check_sums_to_1(self, tolerance: float = 0.0001):
        sum_of_probs = sum(self.option_prob_map.values())
        option_prob_map = self.option_prob_map
        if sum_of_probs != 1.0:
            if np.abs(sum_of_probs - 1.0) < tolerance:
                pass
                # # Either let it slide or try to force
                # if is_leniant:
                #     pass
                # else:
                #     sum_before = sum(option_prob_map.values())
                #     first_key = next(iter(option_prob_map.keys()))
                #     first_val_before = option_prob_map[first_key]
                #     sum_except_first = sum(option_prob_map.values()) - option_prob_map[first_key]
                #     first_val_after = 1.0 - sum_except_first
                #     option_prob_map[first_key] = first_val_after
                #     sum_after = sum(option_prob_map.values())
                #     if sum_after != 1.0:
                #         raise ValueError
                #     else:
                #         self.option_prob_map = option_prob_map
            else:
                raise ValueError

    def support(self):
        for x, p in self.option_prob_map.items():
            if p > 0:
                yield x

    def get_probability(self, option):
        return self.option_prob_map.get(option, 0.0)

    def sample(self):
        options, probs = zip(*self.option_prob_map.items())
        idx = np.random.choice(len(options), p=probs)
        return options[idx]

    def __str__(self):
        return f"<Distribution : {self.option_prob_map}>"


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
