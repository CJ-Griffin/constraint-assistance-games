import copy
import random
from typing import Set, List

import numpy as np
from frozendict import frozendict

from src.formalisms.distributions import KroneckerDistribution
from src.formalisms.finite_processes import FiniteCMDP
from src.formalisms.primitives import State, Action
from src.formalisms.policy import FinitePolicyForFixedCMDP
from src.solution_methods.linear_programming.cplex_dual_cmdp_solver import solve_CMDP_for_policy
from src.concrete_decision_processes.maze_cmdp import RoseMazeCMDP


def generate_any_reduction_policy_from_occupancy_measures(
        oms: np.ndarray,
        cmdp: FiniteCMDP,
        should_choose_most_probable_actions: bool = True) -> FinitePolicyForFixedCMDP:
    nonzero_om_mask = ~np.isclose(oms, 0.0)

    policy_matrix = np.zeros((cmdp.n_states, cmdp.n_actions))
    for s in range(cmdp.n_states):
        num_actions = nonzero_om_mask[s].sum()
        if num_actions == 1:
            policy_matrix[s, nonzero_om_mask[s]] = 1.0
        elif num_actions > 1:
            row = oms[s, :].flatten()
            action_ind = np.argmax(row)
            policy_matrix[s, action_ind] = 1.0
        else:
            assert num_actions == 0
            policy_matrix[s, -1] = 1.0

    return FinitePolicyForFixedCMDP.fromPolicyMatrix(cmdp=cmdp, policy_matrix=policy_matrix)


def generate_any_reduction_policy(
        sigma: FinitePolicyForFixedCMDP,
        cmdp: FiniteCMDP,
        should_choose_most_probable_actions: bool = True):
    phi_1_policy_matrix = sigma.policy_matrix.copy()
    is_stochastic_state_mask = ((phi_1_policy_matrix > 0).sum(axis=1) != 1)
    states_inds_to_be_split = np.where(is_stochastic_state_mask)[0]

    for s_ind in states_inds_to_be_split:
        action_probs = phi_1_policy_matrix[s_ind, :]
        if should_choose_most_probable_actions:
            action_ind = np.argmax(action_probs)
        else:
            action_ind = np.where(action_probs > 0.0)[0][0]
        new_action_probs = np.zeros(cmdp.n_actions)
        new_action_probs[action_ind] = 1.0
        phi_1_policy_matrix[s_ind, :] = new_action_probs

    # Check row_stochastic
    assert np.allclose(phi_1_policy_matrix.sum(axis=1), 1.0), (phi_1_policy_matrix, phi_1_policy_matrix.sum(axis=1))
    # Check deterministic
    assert ((phi_1_policy_matrix > 0).sum(axis=1) == 1).all()
    return FinitePolicyForFixedCMDP.fromPolicyMatrix(cmdp=cmdp, policy_matrix=phi_1_policy_matrix)


def A(policy: FinitePolicyForFixedCMDP, s: State) -> Set[Action]:
    return set(a for a in policy.A if policy(s).get_probability(a) > 0.0)


def get_set_of_impossible_stochastic_states(cmdp, A_star, q):
    """
    This returns the set $U$ from Feinberg and Rothblum Algorithm 1
    # TODO make sure we have citations in this repo
    :param cmdp:
    :param A_star:
    :param q:
    :return:
    """
    return {
        s
        for s in cmdp.S
        if len(A_star[s]) > 1 and q[s] == 0
    }


def get_set_of_possible_stochastic_states(cmdp, A_star, q):
    """
    This returns the set $V$ from Feinberg and Rothblum Algorithm 1
    :param cmdp:
    :param A_star:
    :param q:
    :return:
    """
    return {
        s
        for s in cmdp.S
        if len(A_star[s]) > 1 and q[s] > 0
    }


def get_updated_deterministic_policy(
        policy: FinitePolicyForFixedCMDP,
        s: State,
        a: Action
) -> FinitePolicyForFixedCMDP:
    s_ind = policy.cmdp.state_to_ind_map[s]
    a_ind = policy.cmdp.action_to_ind_map[a]
    modified_policy_matrix = policy.policy_matrix.copy()
    modified_policy_matrix[s_ind, :] = 0.0
    modified_policy_matrix[s_ind, a_ind] = 1.0
    return FinitePolicyForFixedCMDP.fromPolicyMatrix(policy.cmdp, modified_policy_matrix)


def split_policy(
        sigma: FinitePolicyForFixedCMDP,
        cmdp: FiniteCMDP,
        should_assume_deterministic_on_impossilbe_states: bool = True
) -> (List[FinitePolicyForFixedCMDP], List[float]):
    phis, alphas = split_policy_simply(sigma, cmdp, should_assume_deterministic_on_impossilbe_states)
    return phis, alphas


def get_m_random_number_from_policy(sigma: FinitePolicyForFixedCMDP) -> int:
    is_nonzero_mask = ~np.isclose(sigma.policy_matrix, 0.0)
    num_supported_state_action_pairs = int(is_nonzero_mask.sum())
    num_states = sigma.policy_matrix.shape[0]
    return num_supported_state_action_pairs - num_states


def split_policy_simply(
        sigma: FinitePolicyForFixedCMDP,
        cmdp: FiniteCMDP,
        should_assume_deterministic_on_impossilbe_states: bool = True
) -> (List[FinitePolicyForFixedCMDP], List[float]):
    m_sigma = get_m_random_number_from_policy(sigma)

    if m_sigma == 0:
        assert sigma.get_is_policy_deterministic()
        return [sigma], [1.0]

    assert not sigma.get_is_policy_deterministic()

    phi = generate_any_reduction_policy(sigma, cmdp)
    assert phi.get_is_policy_deterministic()

    # Based on equation 37 from Feinberg
    potential_alphas = []
    for s in [s for s in cmdp.state_list if sigma.get_state_occupancy_measure(s) > 0]:
        numerator = sigma.get_occupancy_measure(s, phi.get_deterministic_action(s))
        denomentator = phi.get_state_occupancy_measure(s)
        # If denomenator is 0, this gives infty
        potential_alpha = numerator / denomentator
        assert potential_alpha >= 0
        potential_alphas.append(potential_alpha)

    alpha = min(potential_alphas)
    assert 0 < alpha < 1

    beta = 1 - alpha

    pi_occupancy_measure_matrix = (sigma.occupancy_measure_matrix - (alpha * phi.occupancy_measure_matrix)) / beta
    pi = FinitePolicyForFixedCMDP.fromOccupancyMeasureMatrix(cmdp, pi_occupancy_measure_matrix)

    assert np.allclose(
        sigma.occupancy_measure_matrix,
        (alpha * phi.occupancy_measure_matrix) + (beta * pi.occupancy_measure_matrix)
    )
    m_pi = get_m_random_number_from_policy(pi)

    # Since the m_number is greater than 0, so long as the number keeps decreasing, we can keep recursing
    assert m_pi < m_sigma

    pi_split_policies, pi_split_weightings = split_policy_simply(pi, cmdp)

    alphas = [alpha] + [beta * weight for weight in pi_split_weightings]
    phis = [phi] + pi_split_policies

    assert np.allclose(
        sigma.occupancy_measure_matrix,
        sum([alpha * pol.occupancy_measure_matrix for pol, alpha in zip(phis, alphas)])
    )

    return phis, alphas


def get_support_size(sigma: FinitePolicyForFixedCMDP):
    nonzero_probability_mask = ~np.isclose(sigma.policy_matrix, 0.0)
    return int(nonzero_probability_mask.sum())


# def split_policy_cg_2(
#         sigma: FinitePolicyForFixedCMDP,
#         cmdp: FiniteCMDP,
#         should_assume_deterministic_on_impossilbe_states: bool = True
# ) -> (List[FinitePolicyForFixedCMDP], List[float]):
#     m = get_support_size(sigma) - cmdp.n_states
#
#     phis: List[FinitePolicyForFixedCMDP] = [None] * (m + 2)
#     alphas: List[float] = [None] * (m + 2)
#
#     j = 0
#
#     M_j = sigma.occupancy_measure_matrix
#
#     # Loop invariant 1:
#     #   Mj = σ.occupancy_measure_matrix - ∑ αj * ϕj.occupancy_measure_matrix
#
#     while not np.allclose(M_j, 0):
#         # M_j = σ.occupancy_measure_matrix - ∑ αj * ϕj.occupancy_measure_matrix
#         assert np.allclose(
#             M_j,
#             sigma.occupancy_measure_matrix - sum([alphas[i] * phis[i].occupancy_measure_matrix for i in range(j)])
#         )
#
#         phis[j] = generate_any_reduction_policy_from_occupancy_measures((1 - sum(alphas[:j])) * M_j, cmdp)
#         alphas[j] = min([
#             M_j[s_ind, phis[j].get_deterministic_action_index(s)]
#             / phis[j].get_state_occupancy_measure(s)
#
#             for s_ind, s in enumerate(cmdp.state_list)
#             if M_j[s_ind, :].sum() > 0
#         ])
#
#         j += 1
#
#         M_j = sigma.occupancy_measure_matrix - sum([alphas[i] * phis[i].occupancy_measure_matrix for i in range(j)])
#
#     return phis[:j], alphas[:j]


def split_policy_feinberg(
        sigma: FinitePolicyForFixedCMDP,
        cmdp: FiniteCMDP,
        should_assume_deterministic_on_impossilbe_states: bool = True
) -> (List[FinitePolicyForFixedCMDP], List[float]):
    if should_assume_deterministic_on_impossilbe_states:
        assert sigma.is_deterministic_on_impossible_states()

    phi_1 = generate_any_reduction_policy(sigma, cmdp)

    q = {
        cmdp.state_list[s_ind]: sigma.state_occupancy_measure_vector[s_ind]
        for s_ind in range(cmdp.n_states)
    }

    A_star = {
        s: A(sigma, s)
        for s in cmdp.S
    }

    Q = {
        (s, a): sigma.occupancy_measure_matrix[cmdp.state_to_ind_map[s], cmdp.action_to_ind_map[a]]
        for s in get_set_of_possible_stochastic_states(cmdp, A_star, q)
        for a in A_star[s]
    }

    phis = [phi_1]
    alphas = []

    if len(get_set_of_impossible_stochastic_states(cmdp, A_star, q)) != 0:
        raise NotImplementedError("The implementation is in split_on_impossible_states, but is untested, "
                                  "since we default to originally creating "
                                  "policiies that are deterministic on impossilbe states")

    # Then address the othe states
    V_values = []

    while len(get_set_of_possible_stochastic_states(cmdp, A_star, q)) > 0:
        V_values.append(get_set_of_possible_stochastic_states(cmdp, A_star, q))
        if len(V_values) > 1:
            assert len(V_values[-2]) > len(V_values[-1]), "don't loop forever!"

        candidate_ajs = {
            s: Q[(s, phis[-1].get_deterministic_action(s))] / phis[-1].get_state_occupancy_measure(s)
            for s in list(get_set_of_possible_stochastic_states(cmdp, A_star, q))
        }
        alpha = min(candidate_ajs.values())
        alphas.append(alpha)

        # Make a list of all the states overwhich there is still difference
        G = [
            s
            for s in candidate_ajs.keys()
            if candidate_ajs[s] == alpha
        ]

        # We create a sequence of ϕ's, each differing from the last on a single state, such that the final
        #   ϕ disagrees with the original on everything in G
        for s in G:
            prev_phi = phis[-1]
            prev_phis_action_on_s = {prev_phi.get_deterministic_action(s)}
            action_choice = random.sample(A(sigma, s) - prev_phis_action_on_s, 1)[0]
            phis.append(get_updated_deterministic_policy(prev_phi, s, action_choice))

        # All but the final phi in this sequence are given weighting
        for i in range(len(G) - 1):
            alphas.append(0)

        # phi <- phis[j + k - 1] == phis[-1]
        for s in G:
            A_star[s].remove(phis[-1].get_deterministic_action(s))

        for x in get_set_of_possible_stochastic_states(cmdp, A_star, q):
            Q_x_phi_x = phis[-1].get_occupancy_measure(x, phis[-1].get_deterministic_action(x))
            Q[(x, phis[-1].get_deterministic_action(x))] -= alpha * Q_x_phi_x

        # TODO Write what I think π would be, and see if it corresponds to Q
        # It seems that pi and Q are equivalent for a lot of the pairs, but not all
        # assert len(phis) == len(alphas) + 1
        # beta = 1 - sum(alphas)
        # weighted_omms = [alphas[j] * phis[j].occupancy_measure_matrix for j in range(len(phis[:-1]))]
        # pi_occ_matrix = (sigma.occupancy_measure_matrix - sum(weighted_omms)) / beta
        # pi = FinitePolicyForFixedCMDP.fromOccupancyMeasureMatrix(cmdp, pi_occ_matrix)
        #
        # for s, a in Q.keys():
        #     if s in get_set_of_possible_stochastic_states(cmdp, A_star, q):
        #         s_ind, a_ind = cmdp.state_to_ind_map[s], cmdp.action_to_ind_map[a]
        #         print(Q[(s, a)], pi.policy_matrix[s_ind, a_ind] * beta, "--")

    alphas.append(1 - sum(alphas))
    assert len(alphas) == len(phis)
    return phis, alphas


def split_on_impossible_states(A_star, alphas, cmdp, j, phi, phis, q):
    while len(get_set_of_impossible_stochastic_states(cmdp, A_star, q)) > 0:
        z = random.sample(get_set_of_impossible_stochastic_states(cmdp, A_star, q), 1)[0]

        phi_j_of_z = phis[j](z).sample()
        potential_new_actions = A_star[z] - {phi_j_of_z}
        a = random.sample(potential_new_actions, 1)[0]
        alphas[j] = 0
        phis[j + 1] = FinitePolicyForFixedCMDP.fromPolicyDict(cmdp, {
            s: phis[j](s) if s != z else KroneckerDistribution(a)
            for s in cmdp.S
        })
        A_star[z] = A_star[z] - {phi_j_of_z}
        phi = get_updated_deterministic_policy(phi, z, a)
        j = j + 1
    return j, phi
