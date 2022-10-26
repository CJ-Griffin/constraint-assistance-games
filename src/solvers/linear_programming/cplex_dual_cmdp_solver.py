import time

import cplex
import numpy as np
from tqdm import tqdm

from src.formalisms.finite_cmdp import FiniteCMDP
from src.formalisms.distributions import DiscreteDistribution
from src.formalisms.policy import FiniteCMDPPolicy
from src.utils import open_debug


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
    c.objective.set_linear([(i, memory_mdp.rewards.flatten()[i])
                            for i in range(memory_mdp.n_states * memory_mdp.n_actions)])
    c.objective.set_sense(c.objective.sense.maximize)


def __set_transition_constraints(c, cmdp):
    # iterate over states, then actions
    # flow constraints
    lin_expr = []
    rhs = []
    names = []

    # each constraint will use all (s, a) occupancy measures as possible predecessor states
    variables = range(cmdp.n_states * cmdp.n_actions)

    # one constraint for each state
    for k in tqdm(range(cmdp.n_states), desc="constructing statewise LP constraints"):
        coefficients = []
        # each constraint refers to all (s, a) variables as possible predecessors
        # each coefficient depends on whether the preceding state is the current state or not
        for i in range(cmdp.n_states):
            for j in range(cmdp.n_actions):
                # if previous state is the same as the current state
                if k == i:
                    coefficient = 1 - cmdp.gamma * cmdp.transition_probabilities[i, j, k]
                else:
                    coefficient = - cmdp.gamma * cmdp.transition_probabilities[i, j, k]
                coefficients.append(coefficient)

        # append linear constraint
        lin_expr.append([variables, coefficients])

        # rhs is the start state probability
        rhs.append(float(cmdp.start_state_probabilities[k]))
        names.append(f"dynamics constraint s={k}")

    # add all flow constraints to CPLEX at once, "E" for equality constraints
    c.linear_constraints.add(lin_expr=lin_expr, rhs=rhs, senses=["E"] * len(rhs), names=names)

    # non-negative constraints
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[i], val=[1]) for i in variables],
                             rhs=[0] * len(variables), senses=["G"] * len(variables),
                             names=[f"0<={c.variables.get_names(i)}" for i in variables])


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
                  optimality_tolerance: float = 1e-9):
    cmdp = cmdp
    cmdp.initialise_matrices()
    cmdp.check_matrices()

    c = cplex.Cplex()
    c.parameters.simplex.tolerances.optimality.set(optimality_tolerance)

    __set_variables(c, cmdp)
    __set_objective(c, cmdp)
    __set_transition_constraints(c, cmdp)
    for k in range(cmdp.K):
        __set_kth_cost_constraint(c, cmdp, k)

    return c, cmdp


def solve(cmdp: FiniteCMDP, should_force_deterministic: bool = False) -> (FiniteCMDPPolicy, dict):
    if not isinstance(cmdp, FiniteCMDP):
        raise NotImplementedError("solver only works on FiniteCMDPs, try converting")

    c, cmdp = __get_program(cmdp)

    time_string = time.strftime("%Y_%m_%d__%H:%M:%S")

    with open_debug('logs/dual_mdp_result_' + time_string + '.log', 'a+') as results_file:

        c.set_results_stream(results_file)

        c.solve()

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
