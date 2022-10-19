from src.formalisms.appr_grid_cag import ApprenticeshipStaticGridCAG
from src.formalisms.distributions import UniformDiscreteDistribution

"""
=====HUMAN================ROBOT=====
| strt | slow |      | strt | slow |
| (0,0)| (1,0)|      | (0,0)| (1,0)|
--------------- 
| rose | goal |      | rose | goal |
| (0,1)| (1,1)|      | (0,1)| (1,1)|
====================================

gets R=-0.5 for going through the slow state
gets C=3.0 for going through the roses
"""


class SimplestCAG(ApprenticeshipStaticGridCAG):
    K = 1
    Theta = {"imprm", "prm"}

    def __init__(self, budget: float = 0.0):
        super().__init__(
            h_height=2,
            h_width=2,
            h_start=(0, 0),
            h_sinks={(1, 1)},
            r_height=2,
            r_width=2,
            r_start=(0, 0),
            r_sinks={(1, 1)},
            goal_reward=1,
            gamma=0.9,
            dud_action_penalty=-0.2
        )
        # TODO - this is hacky - plz remove
        self.h_A = {
            (0, 1),  # Down
            (1, 0),  # Right
        }
        self.initial_state_theta_dist = UniformDiscreteDistribution({(self.s_0, theta) for theta in self.Theta})
        self.r_A = self.h_A.copy()
        self.check_is_instantiated()
        self.budget = budget

    def C(self, k: int, theta, s, h_a, r_a) -> float:
        if theta == "prm":
            return 0.0
        else:
            if s.whose_turn == "h":
                if s.h_xy == (0, 0) and h_a == (0, 1):
                    return 3.0
                else:
                    return 0.0
            elif s.whose_turn == "r":
                if s.r_xy == (0, 0) and r_a == (0, 1):
                    return 3.0
                else:
                    return 0.0
            else:
                raise ValueError

    def c(self, k: int) -> float:
        assert k == 0
        return self.budget

    def split_R(self, s, h_a, r_a) -> float:
        r_base = super().split_R(s, h_a, r_a)
        if s.whose_turn == "h":
            if s.h_xy == (0, 0) and h_a == (1, 0):
                r_penalty = - 0.5
            else:
                r_penalty = 0.0
        elif s.whose_turn == "r":
            if s.r_xy == (0, 0) and r_a == (1, 0):
                r_penalty = - 0.5
            else:
                r_penalty = 0.0
        else:
            raise ValueError
        return r_base + r_penalty
