from dataclasses import dataclass
from functools import lru_cache
from itertools import chain, combinations

import numpy as np
from tqdm import tqdm

from src.formalisms.distributions import Distribution, KroneckerDistribution, \
    DiscreteDistribution, FiniteParameterDistribution, split_initial_dist_into_s_and_beta
from src.formalisms.finite_processes import FiniteCMDP, FiniteCAG
from src.formalisms.primitives import State, ActionPair, Plan, get_all_plans, FiniteSpace


# Adapted from from https://stackoverflow.com/questions/18035595/powersets-in-python-using-itertools
def powerset(s: set, min_size: int = 0) -> set:
    s = list(s)
    subsets = set(chain.from_iterable(combinations(s, r) for r in range(min_size, len(s) + 1)))
    return {frozenset(subset) for subset in subsets}


@dataclass(frozen=True, eq=True)
class BeliefState(State):
    s: State
    beta: FiniteParameterDistribution

    def render(self):
        return f"b=(s,β) where s=...\n{self.s.render()} \n and β=...{self.beta.render()}"

    def __repr__(self):
        return f"<b=(s={repr(self.s)}, β={repr(self.beta)}?"


class CAGtoBCMDP(FiniteCMDP):
    def __init__(self, cag: FiniteCAG, is_debug_mode: bool = True, should_print_size: bool = False):
        self.cag = cag
        self.is_debug_mode = is_debug_mode
        # check I is only supported over a single s
        self.concrete_s_0, self.beta_0 = split_initial_dist_into_s_and_beta(self.cag.initial_state_theta_dist)
        # the initial belief state should be deterministic (Kronecker)
        self.initial_state_dist: Distribution = KroneckerDistribution(BeliefState(self.concrete_s_0, self.beta_0))

        self.Beta = FiniteSpace({
            FiniteParameterDistribution(beta_0=self.beta_0,
                                        subset=sset)
            for sset in powerset(self.beta_0.support(), min_size=1)
        })

        self.S = FiniteSpace({
            BeliefState(s, beta)
            for s in self.cag.S
            for beta in self.Beta
        })

        self.Lambda: set = get_all_plans(self.cag.Theta, self.cag.h_A)
        self.A = frozenset({
            ActionPair(conditional_action, a_r)
            for conditional_action in self.Lambda
            for a_r in self.cag.r_A
        })

        self.gamma = self.cag.gamma
        self.c_tuple = self.cag.c_tuple

        if should_print_size:
            print("-" * 20)
            print(f"|B| = 2^|Θ| = 2^{len(self.cag.Theta)}")
            print(f"|S_CMDP| = |B|*|S_CAG|={len(self.S)}")
            print(f"|A_CMDP| = (|A_h|^|Theta|) * |A_r| = "
                  f"({len(self.cag.h_A)}^{len(self.cag.Theta)}) * {len(self.cag.r_A)}"
                  f" = {len(self.A)}")
            print("-" * 20)

    # @validate_T
    def _inner_T(self, s_and_beta: BeliefState, coordinator_action) -> Distribution:
        """
        When β′ = β[ah*, λ] for some ah*
        - TB ((s′, β′) | (s, β), (λ, ar)) = T (s′ | s, ah, ar ) · P[ah | β, λ]
        - where P[ah | β, λ] = ∑_{θ∈Θ} 1[λ(θ) = ah] · β(θ)
        :param s_and_beta:
        :param coordinator_action:
        :return: T((s, β), (λ, ar)) a Distribution over the set {(s′, β′) | s' ∈ cag.S, β′ ∈ B}
        """
        s, beta = s_and_beta.s, s_and_beta.beta
        h_lambda, r_a = coordinator_action

        if self.cag.is_sink(s):
            return KroneckerDistribution(s_and_beta)
        else:
            next_probs = {}
            for theta in beta.support():
                h_a = h_lambda(theta)

                next_state_dist_given_h_a = self.cag._split_inner_T(s, h_a, r_a)
                next_beta = self.get_belief_update(beta, h_lambda, h_a)
                for next_s in next_state_dist_given_h_a.support():
                    next_b = BeliefState(next_s, next_beta)

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

    def get_belief_update(self, beta: FiniteParameterDistribution, h_lambda: Plan, h_a) -> FiniteParameterDistribution:

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
            return beta.get_collapsed_distribution_from_filter_func(filter_funct)
        else:
            raise TypeError

    # @validate_R
    def _inner_R(self, s_and_beta: BeliefState, a) -> float:
        """
        :param s_and_beta:
        :param a:
        :return: float:: RB(b, (λ, r_a)) = E_{θ ~ β} R(s, λ(θ), ar)
        """
        s, beta = s_and_beta.s, s_and_beta.beta
        h_lambda, r_a = a

        def get_R_given_theta(theta):
            return self.cag._inner_split_R(s, h_lambda(theta), r_a)

        return beta.expectation(get_R_given_theta)

    def _inner_C(self, k: int, s_and_beta: BeliefState, a) -> float:
        s, beta = s_and_beta.s, s_and_beta.beta
        h_lambda, r_a = a
        if self.cag.is_sink(s):
            return 0.0
        else:
            def get_C_given_theta(theta):
                return self.cag.C(k, theta, s, h_lambda(theta), r_a)

            return beta.expectation(get_C_given_theta)

    def is_sink(self, s_and_beta: BeliefState) -> bool:
        s, _ = s_and_beta.s, s_and_beta.beta
        return self.cag.is_sink(s)

    def _initialise_orders(self):
        self.cag.initialise_object_to_ind_maps()
        self.beta_list = list(self.Beta)
        self.beta_to_ind_map = {
            self.beta_list[i]: i
            for i in range(len(self.beta_list))
        }

        self.state_list = tuple(BeliefState(s_concrete, beta) for s_concrete in self.cag.S for beta in self.beta_list)
        self.state_to_ind_map = {
            self.state_list[i]: i for i in range(len(self.state_list))
        }

        self.lambda_list = tuple(self.Lambda)
        self.lambda_to_ind_map = {
            self.lambda_list[i]: i for i in range(len(self.lambda_list))
        }

        self.action_list = tuple(
            ActionPair(h_lambda, r_a)
            for h_lambda in self.lambda_list
            for r_a in self.cag.robot_action_list
        )
        self.action_to_ind_map = {
            self.action_list[i]: i for i in range(len(self.action_list))
        }


class MatrixCAGtoBCMDP(CAGtoBCMDP):
    cag: FiniteCAG

    def __init__(self, cag: FiniteCAG, is_debug_mode: bool = True, should_print_size: bool = False,
                 should_tqdm: bool = True):
        self._should_tqdm = should_tqdm
        cag.generate_matrices(should_tqdm=should_tqdm)
        super().__init__(cag, is_debug_mode, should_print_size)

    @lru_cache(maxsize=None)
    def get_beta_vec(self, beta: FiniteParameterDistribution) -> np.array:
        return np.array([beta.get_probability(theta) for theta in self.cag.theta_list])

    @lru_cache(maxsize=None)
    def vectorise_plan(self, h_lambda: Plan) -> np.array:
        return np.array([
            self.cag.human_action_to_ind_map[h_lambda(theta)]
            for theta in self.cag.theta_list
        ])

    # There's no real point in caching, since each is called once
    # @lru_cache(maxsize=None)
    def get_beta_next_ind(self, lambda_ind: int, beta_ind: int, theta_ind: int):
        h_lambda = self.lambda_list[lambda_ind]
        beta = self.beta_list[beta_ind]
        theta = self.cag.theta_list[int(theta_ind)]
        beta_next = beta.get_collapsed_distribution_from_lambda_theta(h_lambda, theta)
        return self.beta_to_ind_map[beta_next]

    def get_beta_next_ind_by_action(self, lambda_ind: int, beta_ind: int, ah_ind: int):
        h_lambda = self.lambda_list[lambda_ind]
        beta = self.beta_list[beta_ind]
        ah = self.cag.human_action_list[int(ah_ind)]
        beta_next = beta.get_collapsed_distribution_from_lambda_ah(h_lambda, ah)
        return self.beta_to_ind_map[beta_next]

    # @time_function
    def initialise_matrices(self, should_tqdm: bool = False):
        self._initialise_orders()

        should_tqdm = should_tqdm or self._should_tqdm

        self.start_state_vector = np.zeros(self.n_states)
        for s_and_beta in self.initial_state_dist.support():
            prob = self.initial_state_dist.get_probability(s_and_beta)
            self.start_state_vector[self.state_to_ind_map[s_and_beta]] = prob

        self.transition_matrix = np.zeros((len(self.state_list), len(self.action_list), len(self.state_list)))
        T_s_beta_lambda_ar_sp_betap = self.transition_matrix.reshape((
            len(self.cag.state_list),
            len(self.beta_list),
            len(self.lambda_list),
            len(self.cag.robot_action_list),
            len(self.cag.state_list),
            len(self.beta_list),
        ))

        self.reward_matrix = np.zeros((self.n_states, self.n_actions))
        R_s_beta_lambda_ar = self.reward_matrix.reshape((
            len(self.cag.state_list),
            len(self.beta_list),
            len(self.lambda_list),
            len(self.cag.robot_action_list)
        ))

        self.cost_matrix = np.zeros((self.K, self.n_states, self.n_actions))
        C_k_s_beta_lambda_ar = self.cost_matrix.reshape((
            self.K,
            len(self.cag.state_list),
            len(self.beta_list),
            len(self.lambda_list),
            len(self.cag.robot_action_list)
        ))

        # Define reshaped tensors that are easier to work with
        T_s_beta_ar_sp_betap_lambda = np.moveaxis(T_s_beta_lambda_ar_sp_betap, 2, -1)
        T_s_ar_sp_beta_betap_lambda = np.moveaxis(T_s_beta_ar_sp_betap_lambda, 1, -3)
        R_s_ar_beta_lambda = np.moveaxis(R_s_beta_lambda_ar, 3, 1)
        C_k_s_ar_beta_lambda = np.moveaxis(C_k_s_beta_lambda_ar, 4, 2)

        # Similarly, reshape matrices from the CAG to be easier to work with
        T_s_ar_sp_ah = np.moveaxis(self.cag.transition_matrix_s_ha_ra_sn, 1, -1)
        R_s_ar_ha = np.moveaxis(self.cag.reward_matrix_s_ha_ra, 1, -1)
        C_k_s_ah_ar_theta = np.moveaxis(self.cag.cost_matrix_k_theta_s_ha_ra, 1, -1)
        C_k_s_ar_theta_ah = np.moveaxis(C_k_s_ah_ar_theta, 2, -1)

        # beta_mat ∈ R^{|Θ| x |Β|}
        # beta_mat[θ_ind, β_ind] = β(θ)
        beta_mat = np.stack([self.get_beta_vec(beta) for beta in self.beta_list], axis=1)
        theta_inds = list(range(len(self.cag.theta_list)))

        iterator = iter(self.lambda_list) if not should_tqdm else tqdm(self.lambda_list,
                                                                       desc="constructing MatrixCAGtoBCMDP λ-wise")

        # TODO if things are too slow, look into multiprocessing
        for h_lambda in iterator:
            # vec_lambda[theta_ind] = ha_ind where λ(θ) = ha
            vec_lambda = self.vectorise_plan(h_lambda)
            lambda_ind = self.lambda_to_ind_map[h_lambda]

            # TODO if things are too slow, look into refactoring out these two while loops
            for beta in self.beta_list:
                beta_ind = self.beta_to_ind_map[beta]
                beta_vec = beta_mat[:, beta_ind]

                poss_theta_inds = beta_vec.nonzero()[0]

                for poss_theta_ind in poss_theta_inds:
                    poss_h_a_ind = vec_lambda[poss_theta_ind]
                    poss_betap_ind = self.get_beta_next_ind_by_action(lambda_ind, beta_ind, poss_h_a_ind)

                    T_s_ar_sp_given_theta = T_s_ar_sp_ah[:, :, :, poss_h_a_ind]
                    prob_theta_given_beta = float(beta_vec[poss_theta_ind])
                    T_s_ar_sp = T_s_ar_sp_given_theta * prob_theta_given_beta

                    T_s_ar_sp_beta_betap_lambda[:, :, :, beta_ind, poss_betap_ind, lambda_ind] += T_s_ar_sp

            # hint: R_s_ar_theta[s, ar, θ] = cag.R(s, λ(θ), ar)
            R_s_ar_theta = R_s_ar_ha[:, :, vec_lambda]
            R_s_ar_beta = R_s_ar_theta @ beta_mat
            R_s_ar_beta_lambda[:, :, :, lambda_ind] = R_s_ar_beta

            # hint: C_k_s_ar_theta[s, ar, θ] = cag.C(k, θ, s, ar, λ(θ))
            C_k_s_ar_theta = C_k_s_ar_theta_ah[:, :, :, theta_inds, vec_lambda]
            C_k_s_ar_beta = C_k_s_ar_theta @ beta_mat
            C_k_s_ar_beta_lambda[:, :, :, :, lambda_ind] = C_k_s_ar_beta

        # Stop the agent from getting any additional information about β once it reaches the sink
        sinks = [b for b in self.state_list if self.is_sink(b)]
        sink_inds = [self.state_to_ind_map[b] for b in sinks]
        self.transition_matrix[sink_inds, :, :] = 0.0
        self.transition_matrix[sink_inds, :, sink_inds] = 1.0
