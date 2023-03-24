from ctypes import Union

from src.formalisms.distributions import split_initial_dist_into_s_and_beta
from src.formalisms.finite_processes import FiniteCMDP, FiniteCAG
from src.formalisms.policy import CMDPPolicy, CMDPPolicyMixture, FiniteCMDPPolicy, FiniteCAGPolicy, \
    CAGPolicyFromCMDPPolicy, FiniteCAGPolicyMixture, CAGPolicyMixtureFromCMDPPolicyMixture
from src.reductions.cag_to_bcmdp import MatrixCAGtoBCMDP
from src.solution_methods.lagrangian_cmdp_solver import get_value_function_using_naive_lagrangian_cmdp_solver
from src.solution_methods.linear_programming.cplex_dual_cmdp_solver import solve_CMDP_for_policy
from src.solution_methods.policy_splitting import split_cmdp_policy


def get_policy_solution_to_FiniteCMDP(
        cmdp: FiniteCMDP,
        method_name: str = "linear_programming"
) -> FiniteCMDPPolicy:
    """
    Returns some optimal policy to a given CMDP,
    :param method_name:
    :param cmdp:
    :return: π
    """
    if method_name == "linear_programming":
        pol, _ = solve_CMDP_for_policy(cmdp)
        return pol
    elif method_name == "lagrangian":
        raise NotImplementedError
        # return get_value_function_using_naive_lagrangian_cmdp_solver(cmdp)
    else:
        raise NotImplementedError(f"No solution method for CMDPs found for {method_name}")


def get_mixed_policy_solution_to_FiniteCMDP(
        cmdp: FiniteCMDP,
        cmdp_solution_method_name: str = "linear_programming"
) -> CMDPPolicyMixture:
    """
    Returns a policy mixture solution to a given CMDP: i.e. a distribution over deterministic policies.
    Sampling from this distribution at the start of an episode gives optimal expected return without
        exceeding the constraints in expectation
    :param cmdp_solution_method_name:
    :param cmdp:
    :return: π
    """
    stochastic_policy = get_policy_solution_to_FiniteCMDP(cmdp, cmdp_solution_method_name)
    phis, alphas = split_cmdp_policy(stochastic_policy, cmdp)
    return CMDPPolicyMixture({phis[i]: alphas[i] for i in range(len(phis))})


def get_mixed_solution_to_FiniteCAG(cag: FiniteCAG) -> FiniteCAGPolicyMixture:
    if not isinstance(cag, FiniteCAG):
        raise NotImplementedError("solver only works on FiniteCAGs, try converting")
    _, beta_0 = split_initial_dist_into_s_and_beta(cag.initial_state_theta_dist)
    bcmdp = MatrixCAGtoBCMDP(cag)
    mixed_cmdp_policy = get_mixed_policy_solution_to_FiniteCMDP(bcmdp)
    mixed_cag_policy = CAGPolicyMixtureFromCMDPPolicyMixture(mixed_cmdp_policy, beta_0)
    return mixed_cag_policy
