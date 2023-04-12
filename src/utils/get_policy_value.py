from typing import Tuple

from src.formalisms.abstract_decision_processes import CMDP
from src.formalisms.finite_processes import FiniteCMDP
from src.formalisms.policy import CMDPPolicy, FinitePolicyForFixedCMDP


def get_cmdp_policy_value_and_costs(cmdp: CMDP, policy: CMDPPolicy) -> (float, Tuple[float]):
    if isinstance(policy, FinitePolicyForFixedCMDP) and cmdp == policy.cmdp:
        assert isinstance(cmdp, FiniteCMDP)
        value = (policy.occupancy_measure_matrix * cmdp.reward_matrix).sum()
        costs = tuple(
            (policy.occupancy_measure_matrix * cmdp.cost_matrix[k]).sum()
            for k in range(cmdp.K)
        )
        return value, costs
    else:
        raise NotImplementedError("This case hasn't been defined yet.")
