from dataclasses import dataclass
from typing import FrozenSet

import numpy as np

from src.formalisms.abstract_decision_processes import CAG, CMDP
from src.formalisms.distributions import Distribution
from src.formalisms.distributions import KroneckerDistribution, DiscreteDistribution
from src.formalisms.finite_processes import FiniteCMDP, FiniteCAG
from src.formalisms.policy import FiniteCAGPolicy
from src.formalisms.primitives import IntAction, State, get_all_plans, FiniteSpace


def _get_random_uniform_limited_precision(num_bits: int = 8) -> float:
    max_int = 2 ** num_bits
    x = np.random.randint(0, max_int + 1)
    x = float(x) / max_int
    return x


def _get_random_discrete_distribution_bounded_precision(
        items: set,
        num_bits: int = 8
) -> DiscreteDistribution:
    """
    Randomly generates a discrete distribution over n items.
    The distribution treats all items (a priori) indistinguisably.
    We split the probability space into 2**num_bits events, and assign each randomly to one of the n items.
    (This gives a distribtuion whose probabilities are themselves sampled as in the multinomial distribution)

    :param items: the items over which to form a distribution
    :param num_bits: the number of bits of allowed to represent each probability ()
    :return: a discrete distribution over the set of items
    """
    items = list(items)
    n = len(items)

    num_balls = 2 ** num_bits
    ball_samples = np.random.randint(0, n, size=num_balls)
    bit_array = np.zeros((n, num_balls))
    bit_array[ball_samples, range(len(ball_samples))] = 1.0

    num_balls_per_item = bit_array.sum(axis=1)
    assert num_balls_per_item.shape == (n,)

    probs = num_balls_per_item / num_balls
    assert probs.sum() == 1.0  # Not just "is close" but "is actually"

    return DiscreteDistribution({
        items[i]: probs[i] for i in range(n)
    })


@dataclass(frozen=True, eq=True)
class NumlineState(State):
    x: int
    t: int

    def render(self):
        return f"NLS(x={self.x}, y={self.t})"


@dataclass(frozen=True, eq=True, order=True)
class Parameterisation:
    theta: int

    def __str__(self):
        return f"parameterisation: theta=({self.theta})"


class RandomisedCAG(FiniteCAG):
    def __init__(self,
                 max_x: int = 3,
                 max_steps: int = 4,
                 num_h_a: int = 2,
                 num_r_a: int = 3,
                 size_Theta: int = 3,
                 gamma: float = 0.9,
                 K: int = 3,
                 seed: int = None
                 ):

        if seed is not None:
            np.random.seed(seed)

        self.max_x = max_x
        self.max_steps = max_steps
        self.num_h_a = num_h_a
        self.num_r_a = num_r_a
        self.size_Theta = size_Theta
        self.c_tuple = tuple([1.0 * max_steps] * K)
        self.gamma = gamma

        self.S = FiniteSpace({
            (NumlineState(x, t))
            for x in range(1, max_x + 1)
            for t in range(0, max_steps + 1)
        })

        self.h_A = frozenset({
            (IntAction(i))
            for i in range(1, num_h_a + 1)
        })

        self.r_A = frozenset({
            (IntAction(i))
            for i in range(1, num_r_a + 1)
        })

        self.Theta: set = {
            Parameterisation(i)
            for i in range(1, size_Theta + 1)
        }

        self.s_0 = NumlineState(1, 0)

        self.initial_state_theta_dist: Distribution = _get_random_discrete_distribution_bounded_precision({
            (self.s_0, theta) for theta in self.Theta
        })

        self.transition_dist_map = {
            (st, h_a, r_a): self.get_next_state_dist(st, h_a, r_a)
            for st in self.S
            for h_a in self.h_A
            for r_a in self.r_A
        }

        self.reward_map = {
            (st, h_a, r_a,): (0.25 * _get_random_uniform_limited_precision()) if st.t < self.max_steps else 0.0
            for st in self.S
            for h_a in self.h_A
            for r_a in self.r_A
        }

        probability_any_cost = 0.8

        def get_random_cost(s: NumlineState):
            if s.t == self.max_steps:
                return 0.0
            elif np.random.binomial(1, probability_any_cost) == 0.0:
                return 0.0
            else:
                return _get_random_uniform_limited_precision()

        self.cost_map = {
            (k, theta, st, h_a, r_a): get_random_cost(st)
            for k in range(self.K)
            for theta in self.Theta
            for st in self.S
            for h_a in self.h_A
            for r_a in self.r_A
        }

        self.check_is_instantiated()

    def _split_inner_T(self, s, h_a, r_a) -> Distribution:  # | None:
        return self.transition_dist_map[(s, h_a, r_a)]

    def _inner_split_R(self, s, h_a, r_a) -> float:
        return self.reward_map[(s, h_a, r_a)]

    def _inner_C(self, k: int, theta, s, h_a, r_a) -> float:
        return self.cost_map[(k, theta, s, h_a, r_a)]

    def is_sink(self, s: NumlineState) -> bool:
        return s.t == self.max_steps

    def get_next_state_dist(self, st: NumlineState, h_a: IntAction, r_a: IntAction):
        if st.t == self.max_steps:
            return KroneckerDistribution(st)
        else:
            tp1 = st.t + 1
            next_states = {
                NumlineState(i, tp1) for i in range(1, self.max_x + 1)
            }
            return _get_random_discrete_distribution_bounded_precision(next_states)


class RandomisedCMDP(FiniteCMDP):
    def __init__(self,
                 max_x: int = 3,
                 max_steps: int = 5,
                 num_a: int = 3,
                 K: int = 4,
                 gamma: float = 0.9,
                 seed: int = None
                 ):
        if seed is not None:
            np.random.seed(seed)
        self.max_x = max_x
        self.max_steps = max_steps
        self.num_a = num_a

        self.gamma = gamma
        self.c_tuple = tuple([1.0] * K)

        self.S = FiniteSpace({
            (NumlineState(x, t))
            for x in range(1, max_x + 1)
            for t in range(0, max_steps + 1)
        })

        self.A: FrozenSet = frozenset({
            (IntAction(i))
            for i in range(1, num_a + 1)
        })

        self.s_0 = NumlineState(1, 0)

        self.initial_state_dist: Distribution = KroneckerDistribution(self.s_0)

        self.transition_dist_map = {
            (st, a): self.get_next_state_dist(st, a)
            for st in self.S
            for a in self.A
        }

        self.reward_map = {
            (st, a): np.random.uniform(0.0, 0.3) if st.t < self.max_steps else 0.0
            for st in self.S
            for a in self.A
        }

        self.fixed_cost = 2.0 / max_steps

        self.cost_map = dict()
        for st in self.S:
            if st.t == self.max_steps:
                for k in range(self.K):
                    for a in self.A:
                        self.cost_map[(k, st, a)] = 0.0
            else:
                for k in range(self.K):
                    no_cost_actions = np.random.choice(list(self.A), size=num_a // 2, replace=False)
                    for a in self.A:
                        if a in no_cost_actions:
                            self.cost_map[(k, st, a)] = 0.0
                        else:
                            self.cost_map[(k, st, a)] = self.fixed_cost
        self.check_is_instantiated()

    def _inner_T(self, s, a) -> Distribution:  # | None:
        return self.transition_dist_map[(s, a)]

    def _inner_R(self, s, a) -> float:
        return self.reward_map[(s, a)]

    def _inner_C(self, k: int, s, a) -> float:
        return self.cost_map[(k, s, a)]

    def is_sink(self, s: NumlineState) -> bool:
        return s.t == self.max_steps

    def get_next_state_dist(self, st: NumlineState, a: IntAction):
        if st.t == self.max_steps:
            return KroneckerDistribution(st)
        else:
            tp1 = st.t + 1
            next_states = {
                NumlineState(i, tp1) for i in range(1, self.max_x + 1)
            }
            return _get_random_discrete_distribution_bounded_precision(next_states)


# We're going to assume that the start state is predetermined.
# This maps CAGs more directly onto CPOMDPs

def sample_from_set(a: set):
    l = list(a)
    return l[np.random.randint(len(l))]


class RandJointPolicy(FiniteCAGPolicy):
    def __init__(self, cag: CAG):
        raise NotImplementedError("This needs to be updated to the new kind of policy")
        self.cag = cag
        self.r_A = cag.r_A
        self.Lambda = get_all_plans(cag.Theta, cag.h_A)
        empty_history = tuple()
        self.history_map = {empty_history: self._generate_random_action_pair()}

    def get_action_pair(self, history: tuple):
        self.verify_history(history)
        if history not in self.history_map:
            self.history_map[history] = self._generate_random_action_pair()
        return self.history_map[history]

    def verify_history(self, history: tuple):
        for triplet in history:
            self.verify_triplet(triplet)

    def verify_triplet(self, triplet):
        if len(triplet) != 3:
            raise ValueError
        elif not isinstance(triplet[0], IntAction):
            raise ValueError
        elif not isinstance(triplet[1], IntAction):
            raise ValueError
        elif not isinstance(triplet[2], NumlineState):
            raise ValueError
        else:
            return

    def _generate_random_action_pair(self):
        r_a = sample_from_set(self.r_A)
        h_lambda = sample_from_set(self.Lambda)
        return (h_lambda, r_a)
