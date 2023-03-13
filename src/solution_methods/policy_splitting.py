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


def generate_phi_1(sigma: FinitePolicyForFixedCMDP, cmdp: FiniteCMDP):
    phi_1_policy_matrix = sigma.policy_matrix.copy()
    is_stochastic_state_mask = ((phi_1_policy_matrix > 0).sum(axis=1) != 1)
    states_inds_to_be_split = np.where(is_stochastic_state_mask)[0]

    for s_ind in states_inds_to_be_split:
        action_probs = phi_1_policy_matrix[s_ind, :]
        first_action_ind = np.where(action_probs > 0.0)[0][0]
        new_action_probs = np.zeros(cmdp.n_actions)
        new_action_probs[first_action_ind] = 1.0
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
    phis_2, alphas_2 = split_policy_2(sigma, cmdp, should_assume_deterministic_on_impossilbe_states)
    phis_1, alphas_1 = split_policy_1(sigma, cmdp, should_assume_deterministic_on_impossilbe_states)
    for phi_a, phi_b in zip(phis_1, phis_2):
        assert np.allclose(phi_a.policy_matrix, phi_b.policy_matrix)
        assert np.allclose(phi_a.occupancy_measure_matrix, phi_b.occupancy_measure_matrix)
    assert alphas_1 == alphas_2
    return phis_2, alphas_2


def split_policy_1(
        sigma: FinitePolicyForFixedCMDP,
        cmdp: FiniteCMDP,
        should_assume_deterministic_on_impossilbe_states: bool = True
) -> (List[FinitePolicyForFixedCMDP], List[float]):
    if should_assume_deterministic_on_impossilbe_states:
        assert sigma.is_deterministic_on_impossible_states()

    phi_1 = generate_phi_1(sigma, cmdp)

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


def split_policy_2(
        sigma: FinitePolicyForFixedCMDP,
        cmdp: FiniteCMDP,
        should_assume_deterministic_on_impossilbe_states: bool = True
) -> (List[FinitePolicyForFixedCMDP], List[float]):
    if should_assume_deterministic_on_impossilbe_states:
        assert sigma.is_deterministic_on_impossible_states()

    phi_1 = generate_phi_1(sigma, cmdp)

    q = frozendict({
        cmdp.state_list[s_ind]: sigma.state_occupancy_measure_vector[s_ind]
        for s_ind in range(cmdp.n_states)
    })

    A_star = {
        s: A(sigma, s)
        for s in cmdp.S
    }

    phi = copy.deepcopy(phi_1)

    j = 1

    Q = {
        (s, a): sigma.occupancy_measure_matrix[cmdp.state_to_ind_map[s], cmdp.action_to_ind_map[a]]
        for s in get_set_of_possible_stochastic_states(cmdp, A_star, q)
        for a in A_star[s]
    }

    # First, address states with no probability of occuring
    alphas: List[float] = [None] * 10
    phis: List[FinitePolicyForFixedCMDP] = [None] * 10

    phis[1] = phi_1
    if len(get_set_of_impossible_stochastic_states(cmdp, A_star, q)) != 0:
        raise NotImplementedError("The implementation is in split_on_impossible_states, bu is untested, "
                                  "since we default to originally creating "
                                  "policiies that are deterministic on impossilbe states")
        j, phi = split_on_impossible_states(A_star, alphas, cmdp, j, phi, phis, q)

    # Then address the othe states
    while len(get_set_of_possible_stochastic_states(cmdp, A_star, q)) > 0:
        candidate_ajs = {
            s: Q[(s, phi.get_deterministic_action(s))] / phi.get_state_occupancy_measure(s)
            for s in list(get_set_of_possible_stochastic_states(cmdp, A_star, q))
        }

        alphas[j] = min(candidate_ajs.values())

        G = [
            s
            for s in candidate_ajs.keys()
            if candidate_ajs[s] == alphas[j]

        ]
        k = len(G)

        for i in range(1, k + 1):
            prev_phi = phis[j + i - 1]
            s_i = G[i - 1]
            action_choice = random.sample(A(sigma, s_i) - {prev_phi.get_deterministic_action(s_i)}, 1)[0]
            next_phi = get_updated_deterministic_policy(prev_phi, s_i, action_choice)
            phis[j + i] = next_phi

        for i in range(1, k - 1 + 1):
            alphas[j + i] = 0

        # Sets phi = phi[j+k] and updates A_star
        for s in G:
            A_star[s].remove(phi.get_deterministic_action(s))
            phi = get_updated_deterministic_policy(phi, s, phis[j + k].get_deterministic_action(s))
        assert np.allclose(phi.policy_matrix, phis[j + k].policy_matrix)

        j = j + k
        for x in get_set_of_possible_stochastic_states(cmdp, A_star, q):
            q_x_phi_x = phi.get_occupancy_measure(x, phi.get_deterministic_action(x))
            relevant_alpha = alphas[j - k]
            assert relevant_alpha is not None
            Q[(x, phi.get_deterministic_action(x))] -= relevant_alpha * q_x_phi_x

    m = j - 1
    alphas[m + 1] = 1 - sum(alphas[1:m + 1])

    return phis[1:m + 2], alphas[1:m + 2]


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
