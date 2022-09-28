import collections.abc

from src.formalisms.cpomdp import CPOMDP
from src.formalisms.cag import CAG
from src.formalisms.distributions import Distribution, KroneckerDistribution, PairOfIndependentDistributions
from itertools import product


def check_I_support_over_single_state(I: Distribution):
    sup = I.support()
    support_over_states = {
        s for (s, theta) in sup
    }
    if len(support_over_states) != 1:
        raise ValueError(f"Reduction to coordination CPOMDP only supported when s_0 is deterministic:"
                         f" I.support()={sup}")


class Plan(collections.abc.Mapping):

    def __init__(self, dict_map: dict):
        self._d = dict_map

    def __getitem__(self, k):
        return self._d[k]

    def __len__(self) -> int:
        return len(self._d)

    def __iter__(self):
        return list(self._d.items())

    def __hash__(self):
        return hash(tuple(sorted(self._d.items())))

    def __eq__(self, other):
        if isinstance(other, Plan):
            return tuple(sorted(self._d.items())) == tuple(sorted(other._d.items()))
        else:
            return False

    def __str__(self):
        # strs = [f"{k} -> {v}" for k, v in self._d.items()]
        return f"<Plan: {self._d} >"

    def __call__(self, x):
        return self[x]


class CoordinationCPOMDP(CPOMDP):

    def __init__(self, cag: CAG):
        self.cag = cag
        self.S = {
            (s, theta)
            for s in self.cag.S
            for theta in self.cag.Theta
        }

        self.Lambda: set = {
            Plan({
                theta: ordering[i]
                for i, theta in enumerate(self.cag.Theta)
            })
            for ordering in product(self.cag.h_A, repeat=len(self.cag.Theta))
        }

        self.A = {
            (conditional_action, a_r)
            for conditional_action in self.Lambda
            for a_r in self.cag.r_A
        }

        self.Omega: set = {
            (a_h, s_next)
            for a_h in self.cag.h_A
            for s_next in self.cag.S
        }

        check_I_support_over_single_state(self.cag.I)
        self.b_0: Distribution = self.cag.I

        self.gamma = self.cag.gamma
        self.K = self.cag.K

    def T(self, s_and_theta, a_pair) -> Distribution | None:
        assert s_and_theta in self.S
        if a_pair not in self.A:
            # TODO fix this
            pass
            # raise ValueError(self.A, a_pair)
        s_concrete, theta = s_and_theta
        plan, a_r = a_pair
        a_h = plan(theta)
        return PairOfIndependentDistributions(
            self.cag.T(s_concrete, a_h, a_r),
            KroneckerDistribution(theta)
        )

    def R(self, s_pair, a_pair, next_s_pair) -> float:
        s_concrete, theta = s_pair
        s_concrete_next, _ = s_pair
        h_plan, r_a = a_pair

        return self.cag.R(s_concrete, h_plan(theta), r_a, s_concrete_next)

    def O(self, a_pair, s_pair_next) -> Distribution:
        h_plan, r_a = a_pair
        concrete_state, theta = s_pair_next
        h_a = h_plan(theta)
        return KroneckerDistribution((h_a, concrete_state))

    def C(self, k: int, s_pair, a_pair, next_s_pair) -> float:
        s_concrete, theta = s_pair
        s_concrete_next, _ = s_pair
        h_plan, r_a = a_pair

        return self.cag.C(k, theta, s_concrete, h_plan(theta), r_a, s_concrete_next)

    def c(self, k: int) -> float:
        return self.cag.c(k)

    def is_sink(self, s) -> bool:
        s_concrete, theta = s
        return self.cag.is_sink(s_concrete)
