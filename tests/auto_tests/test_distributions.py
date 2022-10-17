import unittest

from src.formalisms.distributions import DiscreteDistribution


class TestDistributions(unittest.TestCase):
    def test_equality(self):
        dist1 = DiscreteDistribution({"a": 0.2, "b": 0.3, "c": 0.5})
        dist2 = DiscreteDistribution({"a": 0.2, "b": 0.3, "c": 0.5, "d": 0.0})
        if dist1 != dist2:
            raise ValueError(f"{dist1} != {dist2}")

    def test_discrete_sums_to_over_1(self):
        # Too high
        try:
            dist3 = DiscreteDistribution({"a": 0.2, "b": 0.3, "c": 0.5, "d": 0.1})
            x = None
        except ValueError as e:
            x = e
        except Exception as e:
            raise e

        if not isinstance(x, ValueError):
            raise ValueError

    def test_discrete_sums_to_under_1(self):
        # Too low
        try:
            dist4 = DiscreteDistribution({"a": 0.1, "b": 0.3, "c": 0.5, "d": 0.0})
            x = None
        except ValueError as e:
            x = e
        except Exception as e:
            raise e

        if not isinstance(x, ValueError):
            raise ValueError
