from abc import ABC

import numpy as np
from tqdm import tqdm

from src.formalisms.abstract_decision_processes import CAG, CMDP
from src.formalisms.distributions import Distribution, split_initial_dist_into_s_and_beta
from src.formalisms.primitives import State, Action, ActionPair, FiniteSpace
from src.utils import time_function


class FiniteCMDP(CMDP, ABC):
    S: FiniteSpace = None

    transition_matrix: np.array = None
    reward_matrix: np.array = None
    cost_matrix: np.array = None
    start_state_matrix: np.array = None
    state_to_ind_map: dict = None
    action_to_ind_map: dict = None

    state_list: tuple = None
    action_list: tuple = None

    @property
    def n_states(self):
        return len(self.S)

    @property
    def n_actions(self):
        return len(self.A)

    @property
    def transition_probabilities(self) -> np.array:
        if self.transition_matrix is None:
            self.initialise_matrices()
        return self.transition_matrix

    @property
    def rewards(self) -> np.array:
        if self.reward_matrix is None:
            self.initialise_matrices()
        return self.reward_matrix

    @property
    def costs(self) -> np.array:
        if self.cost_matrix is None:
            self.initialise_matrices()
        return self.cost_matrix

    @property
    def start_state_probabilities(self) -> np.array:
        if self.start_state_matrix is None:
            self.initialise_matrices()
        return self.start_state_matrix

    def initialise_matrices(self):
        if self.transition_matrix is not None:
            return None
        else:
            self._initialise_orders()

            self.reward_matrix = np.zeros((self.n_states, self.n_actions))
            self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
            self.cost_matrix = np.zeros((self.K, self.n_states, self.n_actions))
            self.start_state_matrix = np.zeros(self.n_states)

            sm = self.state_to_ind_map
            am = self.action_to_ind_map
            for s in tqdm(self.S, desc="creating FiniteCMDP matrices statewise"):
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

    def _initialise_orders(self):
        self.state_list = tuple(self.S)
        self.state_to_ind_map = {
            self.state_list[i]: i for i in range(len(self.state_list))
        }
        self.action_list = tuple(self.A)
        self.action_to_ind_map = {
            self.action_list[i]: i for i in range(len(self.action_list))
        }

    @time_function
    def check_matrices(self):
        assert self.n_states is not None
        assert self.n_actions is not None

        assert self.S is not None
        assert self.A is not None
        assert self.rewards is not None
        assert self.transition_probabilities is not None
        assert self.start_state_probabilities is not None

        assert self.rewards.shape == (self.n_states, self.n_actions)
        assert self.transition_probabilities.shape == (self.n_states, self.n_actions, self.n_states)
        assert self.cost_matrix.shape == (self.K, self.n_states, self.n_actions)
        assert self.start_state_probabilities.shape == (self.n_states,)

        assert self.is_stochastic_on_nth_dim(self.transition_probabilities, 2)
        assert self.is_stochastic_on_nth_dim(self.start_state_probabilities, 0)
        self.perform_checks()
        self.stoch_check_if_matrices_match()

    @staticmethod
    def is_stochastic_on_nth_dim(arr: np.ndarray, n: int):
        collapsed = arr.sum(axis=n)
        bools = collapsed == 1.0
        return bools.all()

    def stoch_check_if_matrices_match(self, num_checks=100):
        num_checks = min([self.n_actions, self.n_states, num_checks])
        sm = self.state_to_ind_map
        am = self.action_to_ind_map

        s_ind_list = np.random.choice(self.n_states, size=num_checks, replace=False)
        a_ind_list = np.random.choice(self.n_actions, size=num_checks, replace=False)
        for i in range(num_checks):
            s = self.state_list[s_ind_list[i]]
            a = self.action_list[a_ind_list[i]]
            s_next_dist = self.T(s, a)
            s_ind = sm[s]
            a_ind = am[a]

            reward_from_R = self.R(s, a)
            reward_from_mat = self.reward_matrix[s_ind, a_ind]
            if reward_from_R != reward_from_mat:
                raise ValueError

            for k in range(self.K):
                cost_matrix = self.cost_matrix[k, s_ind, a_ind]
                cost_from_C = self.C(k, s, a)

                if cost_matrix != cost_from_C:
                    raise ValueError

            for s_next in self.state_list:
                prob_T = s_next_dist.get_probability(s_next)
                sn_ind = sm[s_next]
                prob_matrix = self.transition_matrix[s_ind, a_ind, sn_ind]
                if prob_T != prob_matrix:
                    raise ValueError


class FiniteCAG(CAG, ABC):
    are_maps_initialised: bool = False
    are_matrices_initialised: bool = False

    state_list: tuple = None
    human_action_list: list = None
    robot_action_list: list = None
    theta_list: list

    state_to_ind_map: dict = None
    human_action_to_ind_map: dict = None
    robot_action_to_ind_map: dict = None
    theta_to_ind_map: dict = None

    reward_matrix_s_ha_ra: np.array = None
    transition_matrix_s_ha_ra_sn: np.array = None
    cost_matrix_k_theta_s_ha_ra: np.array = None
    initial_beta_matrix: np.array = None

    beta_0: Distribution = None

    def T(self, s: State, a: Action) -> Distribution:
        dist = super().T(s, a)
        if self.are_matrices_initialised and self.should_debug:
            self.cross_reference_transition(a, dist, s)
        return dist

    def cross_reference_transition(self, a, dist, s):
        h_a, r_a = a
        s_ind = self.state_to_ind_map[s]
        h_a_ind = self.human_action_to_ind_map[h_a]
        r_a_ind = self.robot_action_to_ind_map[r_a]
        sample = dist.sample()
        s_next_ind = self.state_to_ind_map[sample]
        prob = dist.get_probability(sample)
        if prob != self.transition_matrix_s_ha_ra_sn[s_ind, h_a_ind, r_a_ind, s_next_ind]:
            raise ValueError

    def initialise_object_to_ind_maps(self):
        if not self.are_maps_initialised:
            self.state_list = tuple(self.S)
            self.state_to_ind_map = {
                self.state_list[i]: i for i in range(len(self.state_list))
            }

            self.human_action_list = list(self.h_A)
            self.human_action_to_ind_map = {
                self.human_action_list[i]: i for i in range(len(self.human_action_list))
            }

            self.robot_action_list = list(self.r_A)
            self.robot_action_to_ind_map = {
                self.robot_action_list[i]: i for i in range(len(self.robot_action_list))
            }

            self.theta_list = list(self.Theta)
            self.theta_to_ind_map = {
                self.theta_list[i]: i for i in range(len(self.theta_list))
            }
            try:
                self.s_0, self.beta_0 = split_initial_dist_into_s_and_beta(self.initial_state_theta_dist)
            except ValueError as e:
                raise NotImplementedError("FiniteCAG only supported when s_0 is deterministic", e,
                                          self.initial_state_theta_dist)
            self.are_maps_initialised = True

    def generate_matrices(self):
        self.initialise_object_to_ind_maps()

        if not self.are_matrices_initialised:
            sm = self.state_to_ind_map
            ham = self.human_action_to_ind_map
            ram = self.robot_action_to_ind_map
            tm = self.theta_to_ind_map

            n_S = len(self.S)
            n_h_A = len(self.h_A)
            n_r_A = len(self.r_A)
            n_Theta = len(self.Theta)

            self.reward_matrix_s_ha_ra = np.zeros((n_S, n_h_A, n_r_A))
            self.transition_matrix_s_ha_ra_sn = np.zeros((n_S, n_h_A, n_r_A, n_S))
            self.cost_matrix_k_theta_s_ha_ra = np.zeros((self.K, n_Theta, n_S, n_h_A, n_r_A))
            self.initial_beta_matrix = np.zeros(n_Theta)

            for theta in self.Theta:
                self.initial_beta_matrix[tm[theta]] = self.beta_0.get_probability(theta)

            for s in tqdm(self.S, desc="creating FiniteCAG matrices statewise"):
                for h_a in self.h_A:
                    for r_a in self.r_A:
                        self.reward_matrix_s_ha_ra[sm[s], ham[h_a], ram[r_a]] = self.split_R(s, h_a, r_a)
                        dist = self._split_inner_T(s, h_a, r_a)

                        for sp in dist.support():
                            s_ind = sm[s]
                            h_a_ind = ham[h_a]
                            r_a_ind = ram[r_a]
                            sp_ind = sm[sp]
                            self.transition_matrix_s_ha_ra_sn[s_ind, h_a_ind, r_a_ind, sp_ind] = dist.get_probability(
                                sp)

                        for k in range(self.K):
                            for theta in self.Theta:
                                cost = self.C(k, theta, s, h_a, r_a)
                                self.cost_matrix_k_theta_s_ha_ra[k, tm[theta], sm[s], ham[h_a], ram[r_a]] = cost
        self.are_maps_initialised = True

    def check_matrices(self):
        assert self.state_list is not None
        assert self.human_action_list is not None
        assert self.robot_action_list is not None

        assert self.reward_matrix_s_ha_ra is not None
        assert self.transition_matrix_s_ha_ra_sn is not None
        assert self.cost_matrix_k_theta_s_ha_ra is not None
        assert self.initial_beta_matrix is not None

        n_states = len(self.state_list)
        n_h_actions = len(self.human_action_list)
        n_r_actions = len(self.robot_action_list)
        n_theta = len(self.theta_list)

        assert self.transition_matrix_s_ha_ra_sn.shape == (n_states, n_h_actions, n_r_actions, n_states)
        assert self.reward_matrix_s_ha_ra.shape == (n_states, n_h_actions, n_r_actions)
        assert self.cost_matrix_k_theta_s_ha_ra.shape == (self.K, n_theta, n_states, n_h_actions, n_r_actions)
        assert self.initial_beta_matrix.shape == (n_theta,)

        assert self.is_stochastic_on_nth_dim(self.transition_matrix_s_ha_ra_sn, 3)
        assert self.is_stochastic_on_nth_dim(self.initial_beta_matrix, 0)
        self.perform_checks()
        self.stoch_check_if_matrices_match()

    @staticmethod
    def is_stochastic_on_nth_dim(arr: np.ndarray, n: int):
        collapsed = arr.sum(axis=n)
        bools = collapsed == 1.0
        return bools.all()

    def stoch_check_if_matrices_match(self, num_checks=100):
        n_states = len(self.state_list)
        n_h_actions = len(self.human_action_list)
        n_r_actions = len(self.robot_action_list)
        n_theta = len(self.theta_list)

        num_checks = min([n_h_actions, n_r_actions, n_states, num_checks])
        sm = self.state_to_ind_map
        ham = self.human_action_to_ind_map
        ram = self.robot_action_to_ind_map
        tm = self.theta_to_ind_map

        s_ind_list = np.random.choice(n_states, size=num_checks, replace=False)
        h_a_ind_list = np.random.choice(n_h_actions, size=num_checks, replace=False)
        r_a_ind_list = np.random.choice(n_r_actions, size=num_checks, replace=False)
        theta_ind_list = np.random.choice(n_theta, size=num_checks, replace=True)
        for i in range(num_checks):
            s = self.state_list[s_ind_list[i]]
            h_a = self.human_action_list[h_a_ind_list[i]]
            r_a = self.robot_action_list[r_a_ind_list[i]]
            theta = self.theta_list[theta_ind_list[i]]
            s_next_dist = self.T(s, ActionPair(h_a, r_a))
            s_ind = sm[s]
            h_a_ind = ham[h_a]
            r_a_ind = ram[r_a]
            theta_ind = tm[theta]

            reward_from_R = self.R(s, ActionPair(h_a, r_a))
            reward_from_mat = self.reward_matrix_s_ha_ra[s_ind, h_a_ind, r_a_ind]
            if reward_from_R != reward_from_mat:
                raise ValueError

            for k in range(self.K):
                cost_matrix = self.cost_matrix_k_theta_s_ha_ra[k, theta_ind, s_ind, h_a_ind, r_a_ind]
                cost_from_C = self.C(k, theta, s, h_a, r_a)

                if cost_matrix != cost_from_C:
                    raise ValueError

            for s_next in self.state_list:
                prob_T = s_next_dist.get_probability(s_next)
                sn_ind = sm[s_next]
                prob_matrix = self.transition_matrix_s_ha_ra_sn[s_ind, h_a_ind, r_a_ind, sn_ind]
                if prob_T != prob_matrix:
                    raise ValueError
