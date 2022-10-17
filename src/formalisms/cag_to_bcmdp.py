import collections.abc
from itertools import chain, combinations
from itertools import product

from src.formalisms.cag import CAG
from src.formalisms.cmdp import FiniteCMDP
from src.formalisms.distributions import Distribution, KroneckerDistribution, \
    DiscreteDistribution, FiniteParameterDistribution
from src.formalisms.spaces import FiniteSpace


# Adapted from from https://stackoverflow.com/questions/18035595/powersets-in-python-using-itertools
def powerset(s: set, min_size: int = 0) -> set:
    s = list(s)
    subsets = set(chain.from_iterable(combinations(s, r) for r in range(min_size, len(s) + 1)))
    return {frozenset(subset) for subset in subsets}


def increase_entry(d: dict, k, v: float):
    if k in d.keys():
        k[d] += v
    else:
        k[d] = v


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

        return s, beta
    else:
        raise NotImplementedError


class Plan(collections.abc.Mapping):

    def __init__(self, dict_map: dict):
        self._d = dict_map

    def __getitem__(self, k):
        return self._d[k]

    def get_keys(self):
        return self._d.keys()

    def get_values(self):
        return self._d.values()

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
        return f"<Plan: {self._d} >"

    def __call__(self, x):
        return self[x]


def get_all_plans(Theta, h_A):
    Lambda: set = {
        Plan({
            theta: ordering[i]
            for i, theta in enumerate(Theta)
        })
        for ordering in product(h_A, repeat=len(Theta))
    }
    return Lambda


class CAGtoBCMDP(FiniteCMDP):
    def __init__(self, cag: CAG, is_debug_mode: bool = False):
        self.cag = cag
        self.is_debug_mode = is_debug_mode
        # check I is only supported over a single s
        self.concrete_s_0, self.beta_0 = split_initial_dist_into_s_and_beta(self.cag.initial_state_theta_dist)
        # the initial belief state should be deterministic (Kronecker)
        self.initial_state_dist: Distribution = KroneckerDistribution((self.concrete_s_0, self.beta_0))

        self.S = FiniteSpace({
            (s, FiniteParameterDistribution(beta_0=self.beta_0,
                                            subset=sset))
            for s in self.cag.S
            for sset in powerset(self.beta_0.support(), min_size=1)
        })

        self.Lambda: set = get_all_plans(self.cag.Theta, self.cag.h_A)
        self.A = {
            (conditional_action, a_r)
            for conditional_action in self.Lambda
            for a_r in self.cag.r_A
        }

        self.gamma = self.cag.gamma
        self.K = self.cag.K

    def T(self, s_and_beta, coordinator_action) -> Distribution:
        if self.is_debug_mode:
            if s_and_beta not in self.S:
                raise ValueError
            elif coordinator_action not in self.A:
                raise ValueError
        s, beta = s_and_beta
        h_lambda, r_a = coordinator_action

        if self.cag.is_sink(s):
            return KroneckerDistribution(s_and_beta)
        else:
            next_probs = {}
            for theta in beta.support():
                h_a = h_lambda(theta)

                next_state_dist_given_h_a = self.cag.split_T(s, h_a, r_a)
                next_beta = self.get_belief_update(beta, h_lambda, h_a)
                for next_s in next_state_dist_given_h_a.support():
                    next_b = (next_s, next_beta)

                    if self.is_debug_mode and next_b not in self.S:
                        raise ValueError

                    prob_a_h_given_beta_lambda = sum([
                        beta.get_probability(theta_prime)
                        for theta_prime in beta.support()
                        if h_lambda(theta_prime) == h_a
                    ])

                    next_probs[next_b] = next_state_dist_given_h_a.get_probability(next_s) \
                                         * prob_a_h_given_beta_lambda
            dist = DiscreteDistribution(next_probs)
            return dist

    def get_belief_update(self, beta: FiniteParameterDistribution, h_lambda: Plan, h_a) -> Distribution:

        if self.is_debug_mode:
            for theta in beta.support():
                if theta not in self.cag.Theta:
                    raise ValueError
            if h_lambda not in self.Lambda:
                raise ValueError
            if h_a not in self.cag.h_A:
                raise ValueError

        def filter_funct(theta):
            return h_lambda(theta) == h_a

        if isinstance(beta, FiniteParameterDistribution):
            return beta.get_collapsed_distribution(filter_funct)
        else:
            raise TypeError

    def R(self, s_and_beta, a) -> float:
        """
        :param s_and_beta:
        :param a:
        :return: float:: RB(b, (λ, r_a)) = E_{θ ~ β} R(s, λ(θ), ar)
        """
        s, beta = s_and_beta
        h_lambda, r_a = a
        if self.cag.is_sink(s):
            return 0.0
        else:
            def get_R_given_theta(theta):
                return self.cag.split_R(s, h_lambda(theta), r_a)

            return beta.expectation(get_R_given_theta)

    def C(self, k: int, s_and_beta, a) -> float:
        s, beta = s_and_beta
        h_lambda, r_a = a
        if self.cag.is_sink(s):
            return 0.0
        else:
            def get_C_given_theta(theta):
                return self.cag.C(k, theta, s, h_lambda(theta), r_a)

            return beta.expectation(get_C_given_theta)

    def c(self, k: int) -> float:
        return self.cag.c(k)

    def is_sink(self, s_and_beta: Distribution) -> bool:
        s, _ = s_and_beta
        return self.cag.is_sink(s)
