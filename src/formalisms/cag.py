from abc import ABC, abstractmethod
from itertools import product
from typing import Tuple

import numpy as np
from tqdm import tqdm

from src.formalisms.decision_process import DecisionProcess, validate_T, validate_R
from src.formalisms.distributions import Distribution
from src.formalisms.primitive_utils import split_initial_dist_into_s_and_beta


class CAG(DecisionProcess, ABC):
    h_A: set = None
    r_A: set = None

    initial_state_theta_dist: Distribution = None

    s_0 = None
    Theta: set = None

    @property
    def A(self):
        return set(product(self.h_A, self.r_A))

    @validate_T
    def T(self, s, action_pair: Tuple[object, object]) -> Distribution:
        if not isinstance(action_pair, Tuple):
            raise TypeError
        elif len(action_pair) != 2:
            raise TypeError
        else:
            h_a, r_a = action_pair
            return self.split_T(s, h_a, r_a)

    @abstractmethod
    def split_T(self, s, h_a, r_a) -> Distribution:
        pass

    @validate_R
    def R(self, s, action_pair: Tuple[object, object]) -> float:
        if not isinstance(action_pair, Tuple):
            raise TypeError
        elif len(action_pair) != 2:
            raise TypeError
        else:
            h_a, r_a = action_pair
            return self.split_R(s, h_a, r_a)

    @abstractmethod
    def split_R(self, s, h_a, r_a) -> float:
        pass

    @abstractmethod
    def C(self, k: int, theta, s, h_a, r_a) -> float:
        raise NotImplementedError

    def check_init_dist_is_valid(self):
        supp = set(self.initial_state_theta_dist.support())
        for x in supp:
            try:
                s, theta = x
            except TypeError as e:
                raise ValueError("cag.I should only have support over {s_0} x Theta!", e)
            except Exception as e:
                raise e

            if s != self.s_0:
                raise ValueError("init dist should only be supported on a single state")
            if theta not in self.Theta:
                raise ValueError(f"theta={theta} not in cag.Theta={self.Theta}")

    def check_is_instantiated(self):
        if self.Theta is None:
            raise ValueError("Something hasn't been instantiated!")

        if self.initial_state_theta_dist is None:
            raise ValueError("init dist hasn't been instantiated!")

        super().check_is_instantiated()

    def test_cost_for_sinks(self):
        sinks = {s for s in self.S if self.is_sink(s)}
        for s in sinks:
            for h_a in self.h_A:
                for r_a in self.r_A:
                    for theta in self.Theta:
                        for k in range(self.K):
                            cost = self.C(k, theta, s, h_a, r_a)
                            if cost != 0.0:
                                raise ValueError("Cost should be 0 at a sink")


class FiniteCAG(CAG, ABC):
    are_maps_initialised: bool = False
    are_matrices_initialised: bool = False

    state_list: list = None
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

    def initialise_object_to_ind_maps(self):
        if not self.are_maps_initialised:
            self.state_list = list(self.S)
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

            for s in tqdm(self.S, desc="creating CMDP matrices statewise"):
                for h_a in self.h_A:
                    for r_a in self.r_A:
                        self.reward_matrix_s_ha_ra[sm[s], ham[h_a], ram[r_a]] = self.split_R(s, h_a, r_a)
                        dist = self.split_T(s, h_a, r_a)

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
            s_next_dist = self.T(s, (h_a, r_a))
            s_ind = sm[s]
            h_a_ind = ham[h_a]
            r_a_ind = ram[r_a]
            theta_ind = tm[theta]

            reward_from_R = self.R(s, (h_a, r_a))
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
