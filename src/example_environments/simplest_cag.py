from src.formalisms.cag import CAG
from src.formalisms.distributions import Distribution, KroneckerDistribution, UniformDiscreteDistribution
from src.formalisms.appr_grid_cag import ApprenticeshipStaticGridCAG

"""
The human and robot's state spaces each look like this:
         _->- (1) ->-_
(0) -->-{              }-->--(3)
         -->- (2) ->--

On (1) there is a fragile rose that it is either permissible or impermissible to damage
Walking through (2) takes slightly longer and therefore incurs a time cost of -0.5
There is a reward of 1.0 for reaching (3)
"""


# TODO, make this a simple grid

class SimplestCAG(CAG):
    S: set = {
        (0, 0, "h"),
        (1, 0, "h"),
        (2, 0, "h"),
        (3, 0, "r"),
        (3, 1, "r"),
        (3, 2, "r"),
        (3, 3, "r")
    }
    h_A: set = {0, 1}
    r_A: set = {0, 1}
    Theta: set = {"perm", "imperm"}
    gamma: float = 1.0
    K: int = 1

    s_0 = None
    I: Distribution = UniformDiscreteDistribution({
        ((0, 0, "h"), "perm"),
        ((0, 0, "h"), "imperm"),
    })

    def T(self, s, h_a, r_a) -> Distribution:
        h_s, r_s, whose_turn = s
        if whose_turn == "h":
            if h_s == 0:
                if h_a == 0:
                    return KroneckerDistribution((1, 0, "h"))
                elif h_a == 1:
                    return KroneckerDistribution((2, 0, "h"))
                else:
                    raise ValueError
            elif h_s in [1, 2]:
                return KroneckerDistribution((3, 0, "r"))
            else:
                raise ValueError
        elif whose_turn == "r":
            if r_s == 0:
                if r_a == 0:
                    return KroneckerDistribution((3, 1, "r"))
                elif r_a == 1:
                    return KroneckerDistribution((3, 2, "r"))
                else:
                    raise ValueError
            elif r_s in [1, 2]:
                return KroneckerDistribution((3, 3, "r"))
            elif r_s == 3:
                return KroneckerDistribution((3, 3, "r"))
            else:
                raise ValueError
        else:
            raise ValueError

    def R(self, s, h_a, r_a) -> float:
        h_s, r_s, whose_turn = s
        n_h_s, n_r_s, _ = self.T(s, h_a, r_a).sample()
        if whose_turn == "h":
            if n_h_s == 2:
                return -0.5
            elif n_h_s == 3:
                return 1.0
            else:
                return 0.0
        elif whose_turn == "r":
            if n_r_s == 2:
                return -0.5
            elif n_r_s == 3:
                return 1.0
            else:
                return 0.0
        else:
            raise ValueError

    def C(self, k: int, theta, s, h_a, r_a) -> float:
        assert k == 0
        if theta == "perm":
            return 0.0
        else:
            h_s, r_s, whose_turn = s
            if whose_turn == "h" and h_s == 0 and h_a == 0:
                return 3.0
            elif whose_turn == "r" and r_s == 0 and r_a == 0:
                return 3.0
            else:
                return 0.0

    def c(self, k: int) -> float:
        assert k == 0
        return 0

    def is_sink(self, s) -> bool:
        h_s, r_s, whose_turn = s
        return r_s == 3


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


class SimplestCAG2(ApprenticeshipStaticGridCAG):
    K = 1
    Theta = {"imprm", "prm"}

    def C(self, k: int, theta, s, h_a, r_a) -> float:
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
        return 0.0

    def R(self, s, h_a, r_a) -> float:
        r_base = super().R(s, h_a, r_a)
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
        print(s, h_a, r_a, r_base + r_penalty)
        return r_base + r_penalty

    def __init__(self):
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
            gamma=0.99,
            dud_action_penalty=-0.2
        )
        # TODO - this is hacky - plz remove
        self.h_A = {
            (0, 1),  # Down
            (1, 0),  # Right
        }
        self.I = UniformDiscreteDistribution({(self.s_0, theta) for theta in self.Theta})
        self.r_A = self.h_A.copy()
        self.check_is_instantiated()
