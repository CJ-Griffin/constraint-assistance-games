from src.formalisms.abstract_decision_processes import CMDP, MDP


class LagrangianCMDPtoMDP(MDP):

    def __init__(self, cmdp: CMDP, lagrange_multiplier: list):
        self.cmdp = cmdp

        self.S = cmdp.S
        self.A = cmdp.A
        self.gamma = cmdp.gamma
        self.initial_state_dist = cmdp.initial_state_dist

        self.lagrange_multiplier = lagrange_multiplier

        if isinstance(self.lagrange_multiplier, float) or isinstance(self.lagrange_multiplier, int):
            self.lagrange_multiplier = [self.lagrange_multiplier]

        if len(self.lagrange_multiplier) != cmdp.K:
            raise ValueError

        if any([x < 0 for x in self.lagrange_multiplier]):
            raise ValueError

    def _inner_R(self, s, a) -> float:
        costs = [self.cmdp.C(k, s, a) for k in range(self.cmdp.K)]
        weighted_costs = [
            self.lagrange_multiplier[k] * costs[k] for k in range(self.cmdp.K)
        ]
        reward = self.cmdp.R(s, a)
        return reward - sum(weighted_costs)

    def _inner_T(self, s, a):
        return self.cmdp.T(s, a)

    def is_sink(self, s) -> bool:
        return self.cmdp.is_sink(s)
