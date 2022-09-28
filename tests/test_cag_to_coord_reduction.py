import random
from copy import copy, deepcopy

from src.appr_grid_cag import ASGState
from src.env_wrapper import EnvCPOMDP
from src.formalisms import CoordinationCPOMDP
from src.formalisms.cag_to_cpomdp import Plan
from src.particulars.rose_garden_cag import RoseGarden


def test_policy(obs: tuple[ASGState, str]):
    if obs is None:
        return (Plan({"prm": (0, -1), "imprm": (1, 0)}), (0,1))
    else:
        h_a, s_concrete = obs
        if s_concrete.whose_turn == "h":
            plan_dict = ({
                "prm": (0, -1),
            })
            if s_concrete.h_xy == (0, 2):
                plan_dict["imprm"] = (1, 0)
            elif s_concrete.h_xy in [(1, 2), (1, 1)]:
                plan_dict["imprm"] = (0, -1)
            elif s_concrete.h_xy == (1, 0):
                plan_dict["imprm"] = (-1, 0)
            else:
                plan_dict["imprm"] = (0, 1)

            a_r = (0, 1)
            return Plan(plan_dict), a_r
        else:
            h_a = Plan({"prm": (1, 0), "imprm": (1, 0)})
            if s_concrete.r_xy == (0, 1):
                r_a = (0, -1)
            elif s_concrete.r_xy in [(0, 0), (1, 0)]:
                r_a = (1, 0)
            elif s_concrete.r_xy == (2, 0):
                r_a = (0, 1)

            else:
                r_a = (0, 1)
            return (h_a, r_a)


if __name__ == "__main__":
    cag = RoseGarden()
    coord = CoordinationCPOMDP(deepcopy(cag))
    b0 = coord.b_0
    assert cag.I == b0
    s_0 = coord.b_0.sample()
    a_0 = random.choice(list(coord.A))
    s1_dist = coord.T(s_0, a_0)

    done = False
    env = EnvCPOMDP(coord)
    obs = env.reset()
    env.render()
    while not done:
        obs, r, done, inf = env.step(test_policy(obs))
        env.render()
    pass



    # b1_dist = coord.T(b0, 1)
    # b1 = b1_dist.sample()
    # reward = coord.R(b0, 1, b1)
    pass
    # bcmdp.T()

    # g1 = RoseMazeCMDP()
    # env = EnvCMDP(g1)

    # control_scheme = {
    #     "w": 3,
    #     "a": 2,
    #     "s": 1,
    #     "d": 0,
    #     "e": 4
    # }
    #
    # def get_action_pair():
    #     x = control_scheme[input()]
    #     return x
    #
    # done = False
    #
    # env.reset()
    # env.render()
    # while not done:
    #     obs, r, done, inf = env.step(get_action_pair())
    #     env.render()
    #     print(obs)
    # pass
