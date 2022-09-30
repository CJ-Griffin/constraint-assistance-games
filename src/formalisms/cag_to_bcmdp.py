import collections.abc
from itertools import product

import numpy as np

from src.formalisms.spaces import FiniteSpace, AllDistributionsOverFiniteSet
from src.formalisms.cmdp import CMDP
from src.formalisms.cag import CAG
from src.formalisms.distributions import Distribution, KroneckerDistribution, PairOfIndependentDistributions, \
    DiscreteDistribution
from itertools import chain, combinations


# Adapted from from https://stackoverflow.com/questions/18035595/powersets-in-python-using-itertools
def powerset(s: set, min_size: int = 0) -> set:
    s = list(s)
    return set(chain.from_iterable(combinations(s, r) for r in range(min_size, len(s) + 1)))


def increase_entry(d: dict, k, v: float):
    if k in d.keys():
        k[d] += v
    else:
        k[d] = v


def split_b_into_s_and_beta(I: Distribution) -> (object, Distribution):
    sup = I.support()
    support_over_states = {
        s for (s, theta) in sup
    }
    if len(support_over_states) != 1:
        raise ValueError(f"Reduction to coordination CPOMDP only supported when s_0 is deterministic:"
                         f" I.support()={sup}")
    else:
        s = list(support_over_states)[0]

    theta_map = {
        theta: I.get_probability((s, theta))
        for _, theta in I.support()
    }

    beta = DiscreteDistribution(theta_map)

    return s, beta


def join_s_and_beta_into_b(s, beta: Distribution) -> Distribution:
    p_map = {
        (s, theta): beta.get_probability(theta)
        for theta in beta.support()
    }

    return DiscreteDistribution(p_map)


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


def get_all_plans(Theta, h_A):
    Lambda: set = {
        Plan({
            theta: ordering[i]
            for i, theta in enumerate(Theta)
        })
        for ordering in product(h_A, repeat=len(Theta))
    }
    return Lambda


class CAG_to_BMDP(CMDP):

    def __init__(self, cag: CAG):
        self.cag = cag

        self.S = self._get_reachable_state_space()

        self.Lambda: set = get_all_plans(self.cag.Theta, self.cag.h_A)

        self.A = {
            (conditional_action, a_r)
            for conditional_action in self.Lambda
            for a_r in self.cag.r_A
        }

        self.gamma = self.cag.gamma
        self.K = self.cag.K

        # check I is only supported over a single s
        _, _ = split_b_into_s_and_beta(self.cag.I)
        # the initial belief state should be deterministic (Kronecker)
        # P(b_0 = b) = 1.0 when b = I
        self.I: Distribution = KroneckerDistribution(self.cag.I)

    def T(self, s, a, is_debug_mode: bool = True) -> Distribution:  # | None:
        h_lambda, r_a = a
        if s not in self.S:
            raise ValueError
        if h_lambda not in self.Lambda:
            raise ValueError(h_lambda, self.Lambda)
        if r_a not in self.cag.r_A:
            raise ValueError(r_a, self.cag.r_A)
        s_t, beta_t = split_b_into_s_and_beta(s)

        next_probs = {}

        for theta in beta_t.support():
            h_a = h_lambda(theta)

            s_tp1_dist_given_h_a = self.cag.T(s_t, h_a, r_a)
            beta_tp1 = self.get_belief_update(beta_t, h_lambda, h_a)
            for s_tp1 in s_tp1_dist_given_h_a.support():
                b_tp1 = join_s_and_beta_into_b(s_tp1, beta_tp1)

                if is_debug_mode and b_tp1 not in self.S:
                    close_values = {
                        dist for dist in self.S
                        if dist.support() == b_tp1.support()
                    }
                    raise ValueError

                prob_a_h_given_beta_lambda = sum([
                    beta_t.get_probability(theta_prime)
                    for theta_prime in beta_t.support()
                    if h_lambda(theta_prime) == h_a
                ])

                next_probs[b_tp1] = s_tp1_dist_given_h_a.get_probability(s_tp1) \
                                    * prob_a_h_given_beta_lambda

        return DiscreteDistribution(next_probs)

    def get_belief_update(self, beta: Distribution, h_lambda: Plan, h_a) -> Distribution:

        for theta in beta.support():
            if theta not in self.cag.Theta:
                raise ValueError
        if h_lambda not in self.Lambda:
            raise ValueError
        if h_a not in self.cag.h_A:
            raise ValueError

        prev_support = beta.support()
        next_support = {
            theta
            for theta in prev_support
            if h_lambda(theta) == h_a
        }
        Z = sum([
            beta.get_probability(theta)
            for theta in next_support
        ])

        if Z == 0.0:
            raise ValueError

        beta_tp1_map = {
            theta: beta.get_probability(theta) / Z
            for theta in next_support
        }

        return DiscreteDistribution(beta_tp1_map)

    def R(self, s, a) -> float:
        # NOTE: this only works because this cmdp is deterministic!
        next_s = self.T(s, a).sample()

        h_lambda, r_a = a
        s_t, beta_t = split_b_into_s_and_beta(s)
        s_tp1, beta_tp1 = split_b_into_s_and_beta(next_s)

        h_a = self.get_true_h_a(beta_t, beta_tp1, h_lambda)

        r = self.cag.R(s_t, h_a, r_a)
        return r

    def C(self, k: int, s, a) -> float:
        # NOTE: this only works because this cmdp is deterministic!
        next_s = self.T(s, a).sample()
        h_lambda, r_a = a
        s_t, beta_t = split_b_into_s_and_beta(s)
        s_tp1, beta_tp1 = split_b_into_s_and_beta(next_s)

        h_a = self.get_true_h_a(beta_t, beta_tp1, h_lambda)

        expectation_term_prob_pairs = [
            (self.cag.C(k, theta, s_t, h_a, r_a),
             beta_tp1.get_probability(theta))

            for theta in beta_tp1.support()
        ]

        cost = sum([v * p for v, p in expectation_term_prob_pairs])

        return cost

    def get_true_h_a(self, beta_t, beta_tp1, h_lambda):
        candidate_h_a_s_first = list({
            h_lambda(theta)
            for theta in beta_t.support()
        })
        candidate_h_a_s = [
            h_a
            for h_a in candidate_h_a_s_first
            if beta_tp1 == self.get_belief_update(beta_t, h_lambda, h_a)
        ]
        if len(candidate_h_a_s) > 1:
            raise ValueError
        elif len(candidate_h_a_s) == 0:
            # Maybe return 0, -inf, inf, None instead?
            raise ValueError
        else:
            h_a = candidate_h_a_s[0]
        return h_a

    def c(self, k: int) -> float:
        return self.cag.c(k)

    def is_sink(self, s: Distribution) -> bool:
        s_t, beta = split_b_into_s_and_beta(s)
        return self.cag.is_sink(s_t)

    def render_state_as_string(self, s) -> str:
        s_t, beta = split_b_into_s_and_beta(s)
        s1 = str(s_t)
        s2 = str(beta)
        return s1 + "\n" + s2

    def _get_reachable_state_space(self):
        s_0, beta_0 = split_b_into_s_and_beta(self.cag.I)
        possible_betas = {
            self._get_updated_beta(beta_0, sset)
            for sset in powerset(self.cag.Theta, min_size=1)
        }

        possible_bs = {
            join_s_and_beta_into_b(s=s, beta=beta)
            for s in self.cag.S
            for beta in possible_betas
        }
        return FiniteSpace(possible_bs)

    @staticmethod
    def _get_updated_beta(beta_0: DiscreteDistribution, compatible_thetas: set):
        elems = list(compatible_thetas)
        priors = np.array([beta_0.get_probability(e) for e in compatible_thetas])
        probs = priors / priors.sum()

        return DiscreteDistribution({
            elems[i]: probs[i] for i in range(len(elems))
        })
