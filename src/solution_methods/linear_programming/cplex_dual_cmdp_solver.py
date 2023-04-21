import time

import cplex
import numpy as np
from tqdm import tqdm

from src.formalisms.finite_processes import FiniteCMDP
from src.formalisms.policy import FinitePolicyForFixedCMDP
from src.reductions.cag_to_bcmdp import CAGtoBCMDP
from src.solution_methods.linear_programming.cplex_transition_constraints import __set_transition_constraints, \
    __set_transition_constraints_batchwise
from src.utils.utils import open_log_debug


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
    objective_list = [
        (i, flat[i])
        for i in range(memory_mdp.n_states * memory_mdp.n_actions)
    ]
    c.objective.set_linear(objective_list)
    c.objective.set_sense(c.objective.sense.maximize)


def __set_non_negative_constraints(c, cmdp):
    variables = range(cmdp.n_states * cmdp.n_actions)
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


def __get_program(cmdp: FiniteCMDP,
                  should_tqdm: bool,
                  optimality_tolerance: float = 1e-9,
                  ):
    cmdp = cmdp
    cmdp.initialise_matrices()
    cmdp.check_matrices()

    c = cplex.Cplex()
    c.parameters.simplex.tolerances.optimality.set(optimality_tolerance)
    print(f"Creating LP for CMDP with |S|={cmdp.n_states} and |A|={cmdp.n_actions}")
    if isinstance(cmdp, CAGtoBCMDP):
        print(f"(Original CAG had |S|={len(cmdp.cag.state_list)}, |Ah|={len(cmdp.cag.human_action_list)},"
              f" |Ar|={len(cmdp.cag.robot_action_list)}, |Θ|={len(cmdp.cag.theta_list)})")
    __set_variables(c, cmdp)
    __set_objective(c, cmdp)
    __set_transition_constraints_batchwise(c, cmdp, should_tqdm=should_tqdm)
    __set_non_negative_constraints(c, cmdp)

    for k in (tqdm(range(cmdp.K), "") if should_tqdm else range(cmdp.K)):
        __set_kth_cost_constraint(c, cmdp, k)
    print("LP Created")
    return c, cmdp


def solve_CMDP_for_occupancy_measures(cmdp, should_tqdm: bool = False):
    c, cmdp = __get_program(cmdp, should_tqdm=should_tqdm)
    time_string = time.strftime("%Y%m%d_%H%M%S")
    fn = 'dual_mdp_result_' + time_string + '.log'
    with open_log_debug(fn, 'a+') as results_file:
        c.set_results_stream(results_file)

        print(f"Entering cplex: view {fn} for info")

        print("solving!")

        c.solve()
        print("Exiting cplex")

        constraint_names = [f"C_{k}" for k in range(cmdp.K)]
        objective_value = c.solution.get_objective_value()
        constraint_values = {
            name: f"{c.solution.get_activity_levels(name)} {c.linear_constraints.get_senses(name)} {c.linear_constraints.get_rhs(name)}"
            for name in constraint_names
        }
        occupancy_measures = c.solution.get_values()
        variable_names = c.variables.get_names()

        fn_mst = 'dual_mdp_solution_' + time_string + '.mst'
        with open_log_debug(fn_mst, 'a+') as sol_file:
            directory = sol_file.name
            c.solution.write(directory)

    return constraint_values, objective_value, occupancy_measures, variable_names


# @time_function
def solve_CMDP_for_policy(
        cmdp: FiniteCMDP,
        should_tqdm: bool = True,
        should_round_small_values: bool = True,
) -> (FinitePolicyForFixedCMDP, dict):
    if not isinstance(cmdp, FiniteCMDP):
        raise NotImplementedError("solver only works on FiniteCMDPs, try converting")

    constraint_vals, objective_value, occupancy_measures, variable_names = solve_CMDP_for_occupancy_measures(
        cmdp,
        should_tqdm
    )

    occupancy_measure_matrix = np.array(occupancy_measures).reshape((cmdp.n_states, cmdp.n_actions))
    if should_round_small_values:
        near_zero_mask = np.isclose(occupancy_measure_matrix, 0.0)
        occupancy_measure_matrix[near_zero_mask] = 0.0

    policy_object = FinitePolicyForFixedCMDP.fromOccupancyMeasureMatrix(cmdp, occupancy_measure_matrix)

    solution_details = {
        'variable_names': variable_names,
        'occupancy_measure_matrix': policy_object.occupancy_measure_matrix,
        'objective_value': objective_value,
        "constraint_vals": constraint_vals
    }

    return policy_object, solution_details
