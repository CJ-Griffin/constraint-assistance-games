from dataclasses import dataclass

import numpy as np

from src.formalisms.distributions import Distribution
from src.formalisms.cag import CAG
from src.formalisms.cag_to_bcmdp import get_all_plans
from src.formalisms.cmdp import CMDP
from src.formalisms.distributions import KroneckerDistribution, DiscreteDistribution
from src.formalisms.spaces import FiniteSpace


def _get_random_discrete_distribution(items: set) -> DiscreteDistribution:
    items = list(items)
    n = len(items)

    # Here are 3 ways to create vals that sum to 1
    # It's not clear which to use as each has difficulty summing to 1

    def get_vals_from_dirichlet():
        return np.random.dirichlet(np.ones(n), size=1).flatten()

    def get_vals_and_round(num_digits=5):
        vals = np.random.rand(n)
        vals /= vals.sum()
        vals = [round(v, num_digits) for v in vals]
        vals[-1] = 1.0 - sum(vals[0:-1])
        return vals

    def get_vals32_and_convert_to_63():
        rng = np.random.default_rng()
        vals = rng.random(size=n, dtype=np.float32)
        vals = vals.astype(np.float64)
        vals /= vals.sum()
        return vals

    g_vals = get_vals_and_round()

    return DiscreteDistribution({
        items[i]: g_vals[i] for i in range(n)
    })


@dataclass(frozen=True, eq=True)
class NumlineState:
    x: int
    t: int

    def __str__(self):
        return f"state: x=({self.x}) t=({self.t})"


@dataclass(frozen=True, eq=True)
class Action:
    a: int

    def __str__(self):
        return f"action: a=({self.a})"


@dataclass(frozen=True, eq=True, order=True)
class Parameterisation:
    theta: int

    def __str__(self):
        return f"parameterisation: theta=({self.theta})"


class RandomisedCAG(CAG):
    def __init__(self,
                 max_x: int = 5,
                 max_steps: int = 10,
                 num_h_a: int = 2,
                 num_r_a: int = 4,
                 size_Theta: int = 4,
                 gamma: float = 0.9,
                 K: int = 2
                 ):
        self.max_x = max_x
        self.max_steps = max_steps
        self.num_h_a = num_h_a
        self.num_r_a = num_r_a
        self.size_Theta = size_Theta

        self.gamma = gamma

        self.S = FiniteSpace({
            (NumlineState(x, t))
            for x in range(1, max_x + 1)
            for t in range(0, max_steps + 1)
        })

        self.h_A = {
            (Action(i))
            for i in range(1, num_h_a + 1)
        }
        self.r_A: set = {
            Action(i)
            for i in range(1, num_r_a + 1)
        }
        self.Theta: set = {
            Parameterisation(i)
            for i in range(1, size_Theta + 1)
        }

        self.K: int = K

        self.s_0 = NumlineState(1, 0)

        self.initial_state_theta_dist: Distribution = _get_random_discrete_distribution({
            (self.s_0, theta) for theta in self.Theta
        })

        self.transition_dist_map = {
            (st, h_a, r_a): self.get_next_state_dist(st, h_a, r_a)
            for st in self.S
            for h_a in self.h_A
            for r_a in self.r_A
        }

        self.reward_map = {
            (st, h_a, r_a,): np.random.uniform(0.0, 0.3)
            for st in self.S
            for h_a in self.h_A
            for r_a in self.r_A
        }

        self.cost_map = {
            (k, theta, st, h_a, r_a): np.random.uniform(0.0, 0.3)
            for k in range(self.K)
            for theta in self.Theta
            for st in self.S
            for h_a in self.h_A
            for r_a in self.r_A
        }

        self.check_is_instantiated()

    def split_T(self, s, h_a, r_a) -> Distribution:  # | None:
        return self.transition_dist_map[(s, h_a, r_a)]

    def split_R(self, s, h_a, r_a) -> float:
        return self.reward_map[(s, h_a, r_a)]

    def C(self, k: int, theta, s, h_a, r_a) -> float:
        return self.cost_map[(k, theta, s, h_a, r_a)]

    def c(self, k: int) -> float:
        # TODO consider making variable
        return 1.0

    def is_sink(self, s: NumlineState) -> bool:
        return s.t == self.max_steps

    def get_next_state_dist(self, st: NumlineState, h_a: Action, r_a: Action):
        if st.t == 10:
            return None
        else:
            tp1 = st.t + 1
            next_states = {
                NumlineState(i, tp1) for i in range(1, self.max_x + 1)
            }
            return _get_random_discrete_distribution(next_states)


class RandomisedCMDP(CMDP):
    def __init__(self,
                 max_x: int = 5,
                 max_steps: int = 10,
                 num_a: int = 2,
                 K: int = 2,
                 gamma: float = 0.9
                 ):
        self.max_x = max_x
        self.max_steps = max_steps
        self.num_a = num_a

        self.gamma = gamma
        self.K = K

        self.S = FiniteSpace({
            (NumlineState(x, t))
            for x in range(1, max_x + 1)
            for t in range(0, max_steps + 1)
        })

        self.A: set = {
            (Action(i))
            for i in range(1, num_a + 1)
        }

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

        self.fixed_cost = 1.0

        self.cost_map = dict()
        for st in self.S:
            for k in range(self.K):
                no_cost_actions = np.random.choice(list(self.A), size=2, replace=False)
                for a in self.A:
                    if a in no_cost_actions:
                        self.cost_map[(k, st, a)] = 0.0
                    else:
                        self.cost_map[(k, st, a)] = self.fixed_cost

        self.check_is_instantiated()

    def T(self, s, a) -> Distribution:  # | None:
        return self.transition_dist_map[(s, a)]

    def R(self, s, a) -> float:
        return self.reward_map[(s, a)]

    def C(self, k: int, s, a) -> float:
        return self.cost_map[(k, s, a)]

    def c(self, k: int) -> float:
        # TODO consider making variable
        return 1.0

    def is_sink(self, s: NumlineState) -> bool:
        return s.t == self.max_steps

    def get_next_state_dist(self, st: NumlineState, a: Action):
        if st.t == self.max_steps:
            return KroneckerDistribution(st)
        else:
            tp1 = st.t + 1
            next_states = {
                NumlineState(i, tp1) for i in range(1, self.max_x + 1)
            }
            return _get_random_discrete_distribution(next_states)


# We're going to assume that the start state is predetermined.
# This maps CAGs more directly onto CPOMDPs

def sample_from_set(a: set):
    l = list(a)
    return l[np.random.randint(len(l))]


class RandJointPolicy:
    def __init__(self, cag: CAG):
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
        elif not isinstance(triplet[0], Action):
            raise ValueError
        elif not isinstance(triplet[1], Action):
            raise ValueError
        elif not isinstance(triplet[2], NumlineState):
            raise ValueError
        else:
            return

    def _generate_random_action_pair(self):
        r_a = sample_from_set(self.r_A)
        h_lambda = sample_from_set(self.Lambda)
        return (h_lambda, r_a)
