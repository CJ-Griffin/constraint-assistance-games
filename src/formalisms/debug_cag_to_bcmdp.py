from functools import lru_cache

import numpy as np
from tqdm import tqdm

from src.formalisms.cag_to_bcmdp import MatrixCAGtoBCMDP
from src.formalisms.plans import Plan
from src.utils import time_function


class DebuggingMatrixCAGtoBCMDP(MatrixCAGtoBCMDP):

    @lru_cache(maxsize=None)
    def matrixify_plan(self, h_lambda: Plan) -> np.array:
        """
        Mλ [ah, θ] = 1 iff λ(θ) = ah else 0
        :param h_lambda:
        :return:
        """
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
                    beta_next = beta.get_collapsed_distribution_from_lambda_ah(h_lambda, h_a)
                    beta_next_ind = self.beta_to_ind_map[beta_next]
                    h_a_ind = self.cag.human_action_to_ind_map[h_a]
                    Y[:, beta_next_ind, h_a_ind] = self.cag.transition_matrix_s_ha_ra_sn[s_ind, h_a_ind, r_a_ind, :]

            matrix_dist_over_s_next = Y @ x

            # M[s', β′] = P[s', β′ | (s, β), (λ, ar) ] Σ_{h_a}
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

        self.lambda_list = list(self.Lambda)
        self.lambda_to_ind_map = {
            self.lambda_list[i]: i for i in range(len(self.lambda_list))
        }

        self.action_list = [(h_lambda, r_a) for h_lambda in self.lambda_list for r_a in self.cag.robot_action_list]
        self.action_to_ind_map = {
            self.action_list[i]: i for i in range(len(self.action_list))
        }

    # There's no real point in caching, since each is called once
    # @lru_cache(maxsize=None)
    def get_beta_next_ind_by_action(self, lambda_ind: int, beta_ind: int, ah_ind: int):
        h_lambda = self.lambda_list[lambda_ind]
        beta = self.beta_list[beta_ind]
        ah = self.cag.human_action_list[int(ah_ind)]
        beta_next = beta.get_collapsed_distribution_from_lambda_ah(h_lambda, ah)
        return self.beta_to_ind_map[beta_next]

    @time_function
    def verify_transition_matrix_against_old_method(self):
        old_transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s_and_beta in self.state_list:
            for coord_action in self.action_list:
                vec = self.get_vector_s_next(s_and_beta=s_and_beta, coordinator_action=coord_action)
                old_transition_matrix[self.state_to_ind_map[s_and_beta], self.action_to_ind_map[coord_action], :] = vec
        m1 = self.transition_matrix
        m2 = old_transition_matrix
        locs = np.argwhere(m1 != m2)
        for i in range(locs.shape[0]):
            triplet = locs[i]
            v1 = m1[tuple(triplet)]
            v2 = m2[tuple(triplet)]
            v3a = self.state_list[triplet[0]][0]
            v3b = self.state_list[triplet[0]][1]
            v4a = self.action_list[triplet[1]][0]
            v4b = self.action_list[triplet[1]][1]
            v5a = self.state_list[triplet[2]][0]
            v5b = self.state_list[triplet[2]][1]
            raise ValueError

    @time_function
    def _initialise_matrices_old(self):
        self._initialise_orders()

        self.reward_matrix = np.zeros((self.n_states, self.n_actions))
        self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.cost_matrix = np.zeros((self.K, self.n_states, self.n_actions))
        self.start_state_matrix = np.zeros(self.n_states)

        sm = self.state_to_ind_map
        am = self.action_to_ind_map
        for s in tqdm(self.S, desc="creating OLD MatrixCAGtoBCMDP matrices statewise"):
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
