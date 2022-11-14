import time

import cplex
import numpy as np
from tqdm import tqdm

from src.formalisms.distributions import DiscreteDistribution, split_initial_dist_into_s_and_beta
from src.formalisms.finite_processes import FiniteCMDP, FiniteCAG
from src.formalisms.policy import FiniteCMDPPolicy, CAGPolicyFromCMDPPolicy, FiniteCAGPolicy
from src.reductions.cag_to_bcmdp import MatrixCAGtoBCMDP
from src.utils.utils import open_log_debug, time_function


def __set_variables(c, cmdp):
    # NOTE: a.flatten()[i*m + j] = a[i,j] where a \in R(m,n) / a.shape() == (m,n)
    # AND" rewards[s,a] = rewards.flatten()[s*N + a]
    sa = [
        (s, a) for s in range(cmdp.n_states) for a in range(cmdp.n_actions)
    ]
    names = [
        f"y({s}, {a})" for (s, a) in sa
    ]
    c.variables.add(types=[c.variables.type.continuous] * (cmdp.n_states * cmdp.n_actions),
                    names=names)


def __set_objective(c, memory_mdp):
    flat = memory_mdp.reward_matrix.flatten()
    c.objective.set_linear([(i, flat[i])
                            for i in range(memory_mdp.n_states * memory_mdp.n_actions)])
    c.objective.set_sense(c.objective.sense.maximize)


def __set_transition_constraints(c, cmdp, should_tqdm: bool, should_debug=False):
    # each constraint will use all (s, a) occupancy measures as possible predecessor states
    variables = range(cmdp.n_states * cmdp.n_actions)

    if should_debug:
        # lin_expr, rhs, names = get_coefficients_slow(cmdp, variables)
        lin_expr2, rhs2, names2 = get_coefficients_fast(cmdp, variables, should_tqdm=should_tqdm)
        lin_expr3, rhs3, names3 = get_coefficients_faster(cmdp, variables, should_tqdm=should_tqdm)
        # assert lin_expr == lin_expr2
        # assert rhs == rhs2
        # assert names == names2
        for i in range(len(lin_expr2)):
            v1 = list(lin_expr3[i][1])
            v2 = lin_expr2[i][1]

            assert v1 == v2
        assert rhs3 == rhs2
        assert names3 == names2
        lin_expr, rhs, names = lin_expr3, rhs3, names3
    else:
        lin_expr, rhs, names = get_coefficients_faster(cmdp, variables, should_tqdm=should_tqdm)

    # add all flow constraints to CPLEX at once, "E" for equality constraints
    batch_size = 20
    inds = list(range(0, len(lin_expr), batch_size))
    if inds[-1] != len(lin_expr):
        inds.append(len(lin_expr))

    batch_iter = range(len(inds) - 1)
    if should_tqdm:
        batch_iter = tqdm(batch_iter, desc="Batches of constraints added")
    for i in batch_iter:
        j = inds[i]
        k = inds[i + 1]
        shorter_lin_expr = lin_expr[j: k]
        c.linear_constraints.add(lin_expr=shorter_lin_expr,
                                 rhs=rhs[j:k],
                                 senses=["E"] * len(shorter_lin_expr),
                                 names=names[j:k])


def __set_non_negative_constraints(c, cmdp):
    variables = range(cmdp.n_states * cmdp.n_actions)
    # non-negative constraints
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[i], val=[1]) for i in variables],
                             rhs=[0] * len(variables), senses=["G"] * len(variables),
                             names=[f"0<={c.variables.get_names(i)}" for i in variables])


# @time_function
def get_coefficients_slow(cmdp, variables, should_tqdm: bool = False):
    lin_expr = []
    rhs = []
    names = []

    # one constraint for each state
    iterator = range(cmdp.n_states) if not should_tqdm else tqdm(range(cmdp.n_states),
                                                                 desc="constructing statewise LP constraints")
    for next_s_ind in iterator:

        """
        the constraint for state s' takes the form:
        Σ_{a'} μ(s',a') =  d(s') + Σ_{s, a} (γ* T(s, a, s') * μ(s,a))
        
        rearranging we get:
        Σ_{a} (1) μ(s',a')
            + Σ_{s, a} [ (- γ * T(s, a, s')) * μ(s,a) ] 
            = d(s')
        where the RHS (d(s')) is a constant.
        
        therefore the coefficient for μ(s,a) is:
        1[s == s'] - γ * T(s, a, s')
        
        """
        coefficients_slow = []
        # each constraint refers to all (s, a) variables as possible predecessors
        # each coefficient depends on whether the preceding state is the current state or not
        for s_ind in range(cmdp.n_states):
            for a_ind in range(cmdp.n_actions):
                indicator = 1 if s_ind == next_s_ind else 0
                trans_coeff = - cmdp.gamma * cmdp.transition_probabilities[s_ind, a_ind, next_s_ind]
                coefficient = indicator + trans_coeff
                coefficients_slow.append(coefficient)

        # append linear constraint
        lin_expr.append([variables, coefficients_slow])

        # rhs is the start state probability
        rhs.append(float(cmdp.start_state_probabilities[next_s_ind]))
        names.append(f"dynamics constraint s={next_s_ind}")
    return lin_expr, rhs, names


# @time_function
def get_coefficients_fast(cmdp: FiniteCMDP, variables, should_tqdm: bool = False):
    lin_expr = []
    rhs = []
    names = []

    # one constraint for each state
    """
    the constraint for state s' takes the form:
    Σ_{a'} μ(s',a') =  d(s') + Σ_{s, a} (γ* T(s, a, s') * μ(s,a))

    rearranging we get:
    Σ_{a} (1) μ(s',a')
        + Σ_{s, a} [ (- γ * T(s, a, s')) * μ(s,a) ]
        = d(s')
    where the RHS (d(s')) is a constant.

    therefore the coefficient for μ(s,a) is:
    1[s == s'] - γ * T(s, a, s')

    """
    trans_coeff_array = - cmdp.gamma * cmdp.transition_matrix[:, :, :]
    next_s_inds = range(cmdp.n_states)
    trans_coeff_array[next_s_inds, :, next_s_inds] += 1.0

    iterator = range(cmdp.n_states) if not should_tqdm else tqdm(range(cmdp.n_states),
                                                                 desc="fast constructing statewise LP constraints")
    for next_s_ind in iterator:  # append linear constraint
        lin_expr.append([variables, list(trans_coeff_array[:, :, next_s_ind].flatten())])

        # rhs is the start state probability
        rhs.append(float(cmdp.start_state_probabilities[next_s_ind]))
        names.append(f"dynamics constraint s={next_s_ind}")

    return lin_expr, rhs, names


# @time_function
def get_coefficients_faster(cmdp: FiniteCMDP, variables, should_tqdm: bool = False):
    lin_expr = []
    rhs = []
    names = []

    # one constraint for each state
    """
    the constraint for state s' takes the form:
    Σ_{a'} μ(s',a') =  d(s') + Σ_{s, a} (γ* T(s, a, s') * μ(s,a))

    rearranging we get:
    Σ_{a} (1) μ(s',a')
        + Σ_{s, a} [ (- γ * T(s, a, s')) * μ(s,a) ]
        = d(s')
    where the RHS (d(s')) is a constant.

    therefore the coefficient for μ(s,a) is:
    1[s == s'] - γ * T(s, a, s')

    """
    trans_coeff_array = - cmdp.gamma * cmdp.transition_matrix[:, :, :]
    next_s_inds = range(cmdp.n_states)
    trans_coeff_array[next_s_inds, :, next_s_inds] += 1.0

    old_shape = trans_coeff_array.shape
    new_shape = (old_shape[0] * old_shape[1], old_shape[2])

    trans_coeff_2darray = trans_coeff_array.reshape(new_shape)

    iterator = range(cmdp.n_states) if not should_tqdm else tqdm(range(cmdp.n_states),
                                                                 desc="fastest constructing statewise LP constraints")
    for next_s_ind in iterator:  # append linear constraint
        lin_expr.append([variables, trans_coeff_2darray[:, next_s_ind]])

        # rhs is the start state probability
        rhs.append(float(cmdp.start_state_probabilities[next_s_ind]))
        names.append(f"dynamics constraint s={next_s_ind}")

    return lin_expr, rhs, names


def __set_kth_cost_constraint(c: cplex.Cplex, cmdp: FiniteCMDP, k: int):
    # iterate over states, then actions
    # flow constraints
    rhs = [cmdp.c(k)]

    # each constraint will use all (s, a) occupancy measures as possible predecessor states
    variables = range(cmdp.n_states * cmdp.n_actions)
    coefficients = [
        cmdp.costs[k, i, j]
        for i in range(cmdp.n_states)
        for j in range(cmdp.n_actions)
    ]
    lin_expr = [[variables, coefficients]]

    c.linear_constraints.add(lin_expr=lin_expr, rhs=rhs, senses=["L"], names=[f"C_{k}"])


def __get_deterministic_int_policy_dict(occupancy_measures, cmdp) -> dict:
    occupancy_measures = np.array(occupancy_measures).reshape((cmdp.n_states, cmdp.n_actions))
    policy = np.argmax(occupancy_measures, axis=1)
    return {s_int: policy[s_int] for s_int in cmdp.n_states}


def __get_stochastic_int_policy_dict(occupancy_measures, cmdp) -> dict:
    occupancy_measures = np.array(occupancy_measures).reshape((cmdp.n_states, cmdp.n_actions))
    states_occupancy_measures = occupancy_measures.sum(axis=1)
    normalised_occupancy_measures = occupancy_measures / states_occupancy_measures[:, np.newaxis]
    policy_map = {
        s: None if np.isnan(normalised_occupancy_measures[s]).any() else DiscreteDistribution(
            {
                a: normalised_occupancy_measures[s, a]
                for a in range(cmdp.n_actions)
            }
        )
        for s in range(cmdp.n_states)
    }
    return policy_map


def __get_program(cmdp: FiniteCMDP,
                  should_tqdm: bool,
                  optimality_tolerance: float = 1e-9,
                  ):
    cmdp = cmdp
    cmdp.initialise_matrices()
    cmdp.check_matrices()

    c = cplex.Cplex()
    c.parameters.simplex.tolerances.optimality.set(optimality_tolerance)

    __set_variables(c, cmdp)
    __set_objective(c, cmdp)
    __set_transition_constraints(c, cmdp, should_tqdm=should_tqdm)
    __set_non_negative_constraints(c, cmdp)
    for k in range(cmdp.K):
        __set_kth_cost_constraint(c, cmdp, k)

    return c, cmdp


def solve_CAG(cag: FiniteCAG,
              should_force_deterministic_cmdp_solution: bool = True,
              should_tqdm: bool = False) -> (FiniteCAGPolicy, dict):
    if not isinstance(cag, FiniteCAG):
        raise NotImplementedError("solver only works on FiniteCMDPs, try converting")
    bcmdp = MatrixCAGtoBCMDP(cag, should_tqdm=should_tqdm)
    bcmdp_pol, details = solve_CMDP(bcmdp, should_force_deterministic=should_force_deterministic_cmdp_solution,
                                    should_tqdm=should_tqdm)
    _, beta_0 = split_initial_dist_into_s_and_beta(cag.initial_state_theta_dist)
    cag_pol = CAGPolicyFromCMDPPolicy(bcmdp_pol, beta_0)
    return cag_pol, details


@time_function
def _solve_in_cplex(c):
    c.solve()


# @time_function
def solve_CMDP(cmdp: FiniteCMDP, should_force_deterministic: bool = False, should_tqdm: bool = False) \
        -> (FiniteCMDPPolicy, dict):
    if not isinstance(cmdp, FiniteCMDP):
        raise NotImplementedError("solver only works on FiniteCMDPs, try converting")

    c, cmdp = __get_program(cmdp, should_tqdm=should_tqdm)

    time_string = time.strftime("%Y_%m_%d__%H:%M:%S")
    fn = 'dual_mdp_result_' + time_string + '.log'
    with open_log_debug(fn, 'a+') as results_file:

        c.set_results_stream(results_file)

        print(f"Entering cplex: view {fn} for info")
        _solve_in_cplex(c)
        print("Exiting cplex")

        basis = c.solution.basis

        constraint_names = [f"C_{k}" for k in range(cmdp.K)]
        objective_value = c.solution.get_objective_value()
        constraint_values = {
            name: f"{c.solution.get_activity_levels(name)} {c.linear_constraints.get_senses(name)} {c.linear_constraints.get_rhs(name)}"
            for name in constraint_names
        }
        occupancy_measures = c.solution.get_values()
        if should_force_deterministic:
            int_policy_dict = __get_deterministic_int_policy_dict(occupancy_measures, cmdp)
        else:
            int_policy_dict = __get_stochastic_int_policy_dict(occupancy_measures, cmdp)
    c.solution.write('logs/dual_mdp_solution_' + time_string + '.mst')

    policy_object = get_polict_object_from_int_policy(cmdp, int_policy_dict)

    state_occ_arr = np.array(occupancy_measures).reshape((cmdp.n_states, cmdp.n_actions)).sum(axis=1)
    solution_details = {
        'objective_value': objective_value,
        'occupancy_measures': {(cmdp.state_list[i // cmdp.n_actions],
                                cmdp.action_list[i % cmdp.n_actions]): occupancy_measure
                               for i, occupancy_measure in enumerate(occupancy_measures)},
        "state_occupancy_measures": {
            cmdp.state_list[i]: state_occ_arr[i]
            for i in range(cmdp.n_states)
        },
        "constraint_values": constraint_values
    }

    return policy_object, solution_details


def get_polict_object_from_int_policy(cmdp: FiniteCMDP, int_policy_dict: dict) -> FiniteCMDPPolicy:
    policy_dict = {}
    for state in range(cmdp.n_states):
        if int_policy_dict[state] is None:
            policy_dict[cmdp.state_list[state]] = None
        else:
            policy_dict[cmdp.state_list[state]] = DiscreteDistribution({
                cmdp.action_list[action]: int_policy_dict[state].get_probability(action)
                for action in range(cmdp.n_actions)
            })
    object_pol = FiniteCMDPPolicy(S=cmdp.S, A=cmdp.A, state_to_dist_map=policy_dict)
    return object_pol
