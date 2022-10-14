import os

import cplex
import numpy as np
import time

# from src.solvers.linear_programming.memory_mdp import MatrixCMDP
from src.formalisms.cmdp import FiniteCMDP
from tqdm import tqdm

from src.formalisms.distributions import DiscreteDistribution

TIMING_ENABLED = True


def __set_variables(c, memory_mdp):
    # NOTE: a.flatten()[i*m + j] = a[i,j] where a \in R(m,n) / a.shape() == (m,n)
    # AND" rewards[s,a] = rewards.flatten()[s*N + a]
    sa = [
        (s, a) for s in range(memory_mdp.n_states) for a in range(memory_mdp.n_actions)
    ]
    names = [
        f"y({s}, {a})" for (s, a) in sa
    ]
    c.variables.add(types=[c.variables.type.continuous] * (memory_mdp.n_states * memory_mdp.n_actions),
                    names=names)


def __set_objective(c, memory_mdp):
    c.objective.set_linear([(i, memory_mdp.rewards.flatten()[i])
                            for i in range(memory_mdp.n_states * memory_mdp.n_actions)])
    c.objective.set_sense(c.objective.sense.maximize)


# @timeit
def __set_transition_constraints(c, memory_mdp, gamma):
    # iterate over states, then actions
    # flow constraints
    lin_expr = []
    rhs = []
    names = []

    # each constraint will use all (s, a) occupancy measures as possible predecessor states
    variables = range(memory_mdp.n_states * memory_mdp.n_actions)

    # one constraint for each state
    for k in tqdm(range(memory_mdp.n_states)):
        coefficients = []
        # each constraint refers to all (s, a) variables as possible predecessors
        # each coefficient depends on whether the preceding state is the current state or not
        for i in range(memory_mdp.n_states):
            for j in range(memory_mdp.n_actions):
                # if previous state is the same as the current state
                if k == i:
                    coefficient = 1 - gamma * memory_mdp.transition_probabilities[i, j, k]
                else:
                    coefficient = - gamma * memory_mdp.transition_probabilities[i, j, k]
                coefficients.append(coefficient)

        # append linear constraint
        lin_expr.append([variables, coefficients])

        # rhs is the start state probability
        rhs.append(float(memory_mdp.start_state_probabilities[k]))
        names.append(f"dynamics constraint s={k}")

    # add all flow constraints to CPLEX at once, "E" for equality constraints
    c.linear_constraints.add(lin_expr=lin_expr, rhs=rhs, senses=["E"] * len(rhs), names=names)

    # non-negative constraints
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[i], val=[1]) for i in variables],
                             rhs=[0] * len(variables), senses=["G"] * len(variables),
                             names=[f"0<={c.variables.get_names(i)}" for i in variables])


def __set_kth_cost_constraint(c: cplex.Cplex, memory_mdp: FiniteCMDP, gamma, k: int):
    # iterate over states, then actions
    # flow constraints
    rhs = [memory_mdp.c(k)]

    # each constraint will use all (s, a) occupancy measures as possible predecessor states
    variables = range(memory_mdp.n_states * memory_mdp.n_actions)
    coefficients = [
        memory_mdp.costs[k, i, j]
        for i in range(memory_mdp.n_states)
        for j in range(memory_mdp.n_actions)
    ]
    lin_expr = [[variables, coefficients]]

    # # one constraint for each state
    # for k in range(memory_mdp.n_states):
    #     coefficients = []
    #     # each constraint refers to all (s, a) variables as possible predecessors
    #     # each coefficient depends on whether the preceding state is the current state or not
    #     for i in range(memory_mdp.n_states):
    #         for j in range(memory_mdp.n_actions):
    #             # if previous state is the same as the current state
    #             if k == i:
    #                 coefficient = 1 - gamma * memory_mdp.transition_probabilities[i, j, k]
    #             else:
    #                 coefficient = - gamma * memory_mdp.transition_probabilities[i, j, k]
    #     coefficients.append(coefficient)
    #
    #     # append linear constraint
    #     lin_expr.append([variables, coefficients])
    #
    #     # rhs is the start state probability
    #     rhs.append(float(memory_mdp.start_state_probabilities[k]))

    c.linear_constraints.add(lin_expr=lin_expr, rhs=rhs, senses=["L"], names=[f"C_{k}"])


# @timeit
def __get_deterministic_policy(occupancy_measures, memory_mdp, gamma) -> list:
    # return list of best action for each state
    occupancy_measures = np.array(occupancy_measures).reshape((memory_mdp.n_states, memory_mdp.n_actions))
    policy = np.argmax(occupancy_measures, axis=1)
    return list(policy)


def __get_stochastic_policy(occupancy_measures, memory_mdp) -> dict:
    occupancy_measures = np.array(occupancy_measures).reshape((memory_mdp.n_states, memory_mdp.n_actions))
    states_occupancy_measures = occupancy_measures.sum(axis=1)
    normalised_occupancy_measures = occupancy_measures / states_occupancy_measures[:, np.newaxis]
    policy = {
        s: None if np.isnan(normalised_occupancy_measures[s]).any() else DiscreteDistribution(
            {
                a: normalised_occupancy_measures[s, a]
                for a in range(memory_mdp.n_actions)
            }
        )
        for s in range(memory_mdp.n_states)
    }
    return policy


def __get_program(mdp: FiniteCMDP, gamma, parallelize=False, transformer=None):
    if TIMING_ENABLED:
        ts = time.time()

    # memory_mdp = MatrixCMDP(mdp, parallelize=parallelize)
    memory_mdp = mdp
    memory_mdp.validate()

    if TIMING_ENABLED:
        te = time.time()
        print('%r %2.2f sec' % ('set up and validate memory mdp', te - ts))

    c = cplex.Cplex()

    __set_variables(c, memory_mdp)
    __set_objective(c, memory_mdp)
    __set_transition_constraints(c, memory_mdp, gamma)
    for k in range(memory_mdp.K):
        __set_kth_cost_constraint(c, memory_mdp, gamma, k)

    if TIMING_ENABLED:
        ts = time.time()

    if transformer:
        transformer.transform(c, memory_mdp)

    if TIMING_ENABLED:
        te = time.time()
        print('%r %2.2f sec' % ('add ethical constraints', te - ts))

    return c, memory_mdp


def solve(mdp: FiniteCMDP, gamma: "float", parallelize: "bool" = False, transformer=None):
    c, memory_mdp = __get_program(mdp, gamma, parallelize=parallelize, transformer=transformer)

    print("===== Program Details =============================================")
    print("{} variables".format(c.variables.get_num()))
    print("{} sense".format(c.objective.sense[c.objective.get_sense()]))
    print("{} linear coefficients".format(len(c.objective.get_linear())))
    print("{} linear constraints".format(c.linear_constraints.get_num()))

    time_string = time.strftime("%Y_%m_%d__%H:%M:%S")
    with open('logs/dual_mdp_result_' + time_string + '.log', 'a+') as results_file:
        # log_file = open('src/logs/dual_mdp_log_' + time_string + '.log', 'w')

        c.set_results_stream(results_file)
        # c.set_log_stream(log_file)

        if TIMING_ENABLED:
            ts = time.time()

        print("===== CPLEX Details ===============================================")
        c.solve()
        print("===================================================================")

        basis = c.solution.basis

        if TIMING_ENABLED:
            te = time.time()
            print('%r %2.2f sec' % ('solve LP', te - ts))

        constraint_names = [f"C_{k}" for k in range(mdp.K)]
        objective_value = c.solution.get_objective_value()
        constraint_values = {
            name: f"{c.solution.get_activity_levels(name)} {c.linear_constraints.get_senses(name)} {c.linear_constraints.get_rhs(name)}"
            for name in constraint_names
        }
        occupancy_measures = c.solution.get_values()
        # policy = __get_deterministic_policy(occupancy_measures, memory_mdp, gamma)
        policy = __get_stochastic_policy(occupancy_measures, memory_mdp)
    # results_file.close()
    c.solution.write('logs/dual_mdp_solution_' + time_string + '.mst')
    # log_file.close()

    noninteger_policy = {}
    for state in range(memory_mdp.n_states):
        if policy[state] is None:
            noninteger_policy[memory_mdp.states[state]] = None
        else:
            noninteger_policy[memory_mdp.states[state]] = DiscreteDistribution({
                memory_mdp.actions[action]: policy[state].get_probability(action)
                for action in range(memory_mdp.n_actions)
            })
    state_occ_arr = np.array(occupancy_measures).reshape((memory_mdp.n_states, memory_mdp.n_actions)).sum(axis=1)
    return {
        'objective_value': objective_value,
        'occupancy_measures': {(memory_mdp.states[i // memory_mdp.n_actions],
                                memory_mdp.actions[i % memory_mdp.n_actions]): occupancy_measure
                               for i, occupancy_measure in enumerate(occupancy_measures)},
        "state_occupancy_measures": {
            memory_mdp.states[i]: state_occ_arr[i]
            for i in range(memory_mdp.n_states)
        },
        'policy': noninteger_policy,
        "constraint_values": constraint_values
    }
