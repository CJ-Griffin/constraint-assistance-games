from functools import lru_cache
from itertools import chain, combinations
from typing import Callable

import numpy as np
from tqdm import tqdm

from src.formalisms.cag import CAG, FiniteCAG
from src.formalisms.decision_process import validate_T, validate_R
from src.formalisms.distributions import Distribution, KroneckerDistribution, \
    DiscreteDistribution, FiniteParameterDistribution
from src.formalisms.finite_cmdp import FiniteCMDP
from src.formalisms.plans import Plan, get_all_plans
from src.formalisms.primitive_utils import split_initial_dist_into_s_and_beta
from src.formalisms.spaces import FiniteSpace
from src.utils import time_function


# Adapted from from https://stackoverflow.com/questions/18035595/powersets-in-python-using-itertools
def powerset(s: set, min_size: int = 0) -> set:
    s = list(s)
    subsets = set(chain.from_iterable(combinations(s, r) for r in range(min_size, len(s) + 1)))
    return {frozenset(subset) for subset in subsets}


class CAGtoBCMDP(FiniteCMDP):
    def __init__(self, cag: CAG, is_debug_mode: bool = False, should_print_size: bool = True):
        self.cag = cag
        self.is_debug_mode = is_debug_mode
        # check I is only supported over a single s
        self.concrete_s_0, self.beta_0 = split_initial_dist_into_s_and_beta(self.cag.initial_state_theta_dist)
        # the initial belief state should be deterministic (Kronecker)
        self.initial_state_dist: Distribution = KroneckerDistribution((self.concrete_s_0, self.beta_0))

        self.Beta = FiniteSpace({
            FiniteParameterDistribution(beta_0=self.beta_0,
                                        subset=sset)
            for sset in powerset(self.beta_0.support(), min_size=1)
        })

        self.S = FiniteSpace({
            (s, beta)
            for s in self.cag.S
            for beta in self.Beta
        })

        self.Lambda: set = get_all_plans(self.cag.Theta, self.cag.h_A)
        self.A = {
            (conditional_action, a_r)
            for conditional_action in self.Lambda
            for a_r in self.cag.r_A
        }

        self.gamma = self.cag.gamma
        self.K = self.cag.K

        if should_print_size:
            print("-" * 20)
            print(f"|B| = 2^|Θ| = 2^{len(self.cag.Theta)}")
            print(f"|S_CMDP| = |B|*|S_CAG|={len(self.S)}")
            print(f"|A_CMDP| = (|A_h|^|Theta|) * |A_r| = "
                  f"({len(self.cag.h_A)}^{len(self.cag.Theta)}) * {len(self.cag.r_A)}"
                  f" = {len(self.A)}")
            print("-" * 20)

    @validate_T
    def T(self, s_and_beta, coordinator_action) -> Distribution:
        """
        When β′ = β[ah*, λ] for some ah*
        - TB ((s′, β′) | (s, β), (λ, ar)) = T (s′ | s, ah, ar ) · P[ah | β, λ]
        - where P[ah | β, λ] = ∑_{θ∈Θ} 1[λ(θ) = ah] · β(θ)
        :param s_and_beta:
        :param coordinator_action:
        :return: T((s, β), (λ, ar)) a Distribution over the set {(s′, β′) | s' ∈ cag.S, β′ ∈ B}
        """
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

    @validate_R
    def R(self, s_and_beta, a) -> float:
        """
        :param s_and_beta:
        :param a:
        :return: float:: RB(b, (λ, r_a)) = E_{θ ~ β} R(s, λ(θ), ar)
        """
        s, beta = s_and_beta
        h_lambda, r_a = a

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


class CheckTheta(Callable):
    def __init__(self, h_lambda, h_a):
        self.h_lambda = h_lambda
        self.h_a = h_a

    def __call__(self, theta):
        return self.h_lambda(theta) == self.h_a


class MatrixCAGtoBCMDP(CAGtoBCMDP):
    cag: FiniteCAG

    def __init__(self, cag: FiniteCAG, is_debug_mode: bool = False, should_print_size: bool = True):
        cag.generate_matrices()
        super().__init__(cag, is_debug_mode, should_print_size)

    @lru_cache(maxsize=None)
    def get_beta_vec(self, beta: FiniteParameterDistribution) -> np.array:
        return np.array([beta.get_probability(theta) for theta in self.cag.theta_list])

    @lru_cache(maxsize=None)
    def matrixify_plan(self, h_lambda: Plan) -> np.array:
        iter_of_h_a_indeces = [
            self.cag.human_action_to_ind_map[h_lambda(theta)]
            for theta in self.cag.theta_list
        ]
        iter_of_theta_indeces = range(len(self.cag.Theta))

        bool_array = np.zeros((len(self.cag.Theta), len(self.cag.h_A)), dtype=bool)
        bool_array[iter_of_theta_indeces, iter_of_h_a_indeces] = True

        return np.array(
            [
                [
                    h_lambda(theta) == h_a
                    for theta in self.cag.theta_list
                ]
                for h_a in self.cag.human_action_list
            ],
            dtype=bool
        )

    @lru_cache(maxsize=None)
    def _get_beta_update(self, beta, h_lambda, h_a):
        return beta.get_collapsed_distribution(CheckTheta(h_lambda, h_a))

    @lru_cache(maxsize=None)
    def get_x_given_beta_lambda(self, beta, h_lambda):
        """
        Get x given λ, β
        x[a_h] = P[a_h | β, λ]
        :param beta: FiniteParameterDistribution
        :param h_lambda: Plan
        :return: np.array of size (|Ah|,)
        """
        matrix_lambda = self.matrixify_plan(h_lambda)
        x = matrix_lambda @ self.get_beta_vec(beta)
        return x

    def get_vector_s_next(self, s_and_beta, coordinator_action) -> np.ndarray:
        """
        When β′ = β[ah*, λ] for some ah*
        - TB ((s′, β′) | (s, β), (λ, ar)) = T (s′ | s, ah, ar ) · P[ah | β, λ]
        - where P[ah | β, λ] = ∑_{θ∈Θ} 1[λ(θ) = ah] · β(θ)
        :param s_and_beta:
        :param coordinator_action:
        :return: A vector P_{b,c_a} ∈ ℝ^{|B|}
        this represents the distribution T((s, β), (λ, ar)) over the set B
        where B = {(s′, β′) | s' ∈ cag.S, β′ ∈ param belief space}
        """
        if self.is_sink(s_and_beta):
            m = np.zeros(len(self.state_list))
            m[self.state_to_ind_map[s_and_beta]] = 1.0
            return m
        else:
            h_lambda, r_a = coordinator_action
            s, beta = s_and_beta
            """
            Get x given λ, β
            x[a_h] = P[a_h | β, λ]
            """
            x = self.get_x_given_beta_lambda(beta, h_lambda)

            """
            Get Y given b=(s, β) and a_coord = (λ, ar)
            """
            s_ind = self.cag.state_to_ind_map[s]
            r_a_ind = self.cag.robot_action_to_ind_map[r_a]
            # Y[s′, β′, a_h] = P[moving to b | a_h, s, a_r]
            # Y[s′, β′, a_h] = T'( s' | s, a_h, a_r ) * 1[ β′ = β[λ. a_h] ]

            Y = np.zeros(shape=(len(self.cag.state_list), len(self.beta_list), len(self.cag.human_action_list)))

            for h_a_ind in range(len(x)):
                if x[h_a_ind] > 0:
                    h_a = self.cag.human_action_list[h_a_ind]
                    beta_next = self._get_beta_update(beta, h_lambda, h_a)
                    beta_next_ind = self.beta_to_ind_map[beta_next]
                    h_a_ind = self.cag.human_action_to_ind_map[h_a]
                    Y[:, beta_next_ind, h_a_ind] = self.cag.transition_matrix_s_ha_ra_sn[s_ind, h_a_ind, r_a_ind, :]

            matrix_dist_over_s_next = Y @ x

            # M[s', β′] = P[s', β′ | (s, β), (λ, ar) ]
            M = matrix_dist_over_s_next

            # m[(s', β′)] = M[s', β′]
            m = M.flatten()

            is_debug = False
            if is_debug:
                self._check_m_against_M(M, m)

            return m

    def _check_m_against_M(self, M, m):
        for i in range(10):
            s_ind = np.random.randint(0, len(self.cag.state_list))
            beta_ind = np.random.randint(0, len(self.beta_list))
            s = self.cag.state_list[s_ind]
            beta = self.beta_list[beta_ind]
            b = (s, beta)
            b_ind = self.state_to_ind_map[b]
            val1 = M[s_ind, beta_ind]
            val2 = m[b_ind]
            if val1 != val2:
                raise ValueError

    def _initialise_orders(self):
        self.beta_list = list(self.Beta)
        self.beta_to_ind_map = {
            self.beta_list[i]: i
            for i in range(len(self.beta_list))
        }

        self.state_list = [(s_concrete, beta) for s_concrete in self.cag.S for beta in self.beta_list]
        self.state_to_ind_map = {
            self.state_list[i]: i for i in range(len(self.state_list))
        }

        self.action_list = list(self.A)
        self.action_to_ind_map = {
            self.action_list[i]: i for i in range(len(self.action_list))
        }

    @time_function
    def _initialise_matrices_new(self):
        self._initialise_orders()

        self.reward_matrix = np.zeros((self.n_states, self.n_actions))
        self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.cost_matrix = np.zeros((self.K, self.n_states, self.n_actions))
        self.start_state_matrix = np.zeros(self.n_states)

        for state in self.state_list:
            for action in self.action_list:
                vec = self.get_vector_s_next(s_and_beta=state, coordinator_action=action)
                self.transition_matrix[self.state_to_ind_map[state], self.action_to_ind_map[action], :] = vec

    @time_function
    def _initialise_matrices_old(self):
        self._initialise_orders()

        self.reward_matrix = np.zeros((self.n_states, self.n_actions))
        self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.cost_matrix = np.zeros((self.K, self.n_states, self.n_actions))
        self.start_state_matrix = np.zeros(self.n_states)

        sm = self.state_to_ind_map
        am = self.action_to_ind_map
        for s in tqdm(self.S, desc="creating CMDP matrices statewise"):
            self.start_state_matrix[sm[s]] = self.initial_state_dist.get_probability(s)
            for a in self.A:
                self.reward_matrix[sm[s], am[a]] = self.R(s, a)
                dist = self.T(s, a)
                s_ind = sm[s]
                a_ind = am[a]
                for sp in dist.support():
                    sp_ind = sm[sp]
                    self.transition_matrix[s_ind, a_ind, sp_ind] = dist.get_probability(sp)

                for k in range(self.K):
                    self.cost_matrix[k, sm[s], am[a]] = self.C(k, s, a)

    def initialise_matrices(self):
        if self.transition_matrix is not None:
            return None
        else:
            self._initialise_matrices_old()
