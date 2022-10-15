import random
import unittest
from copy import deepcopy, copy
from typing import Tuple

from archive.CAGs import Distribution
from src.example_environments.simplest_cag import SimplestCAG2
from src.formalisms.appr_grid_cag import ASGState
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.example_environments.rose_garden_cag import RoseGarden
from src.example_environments.maze_cpomdp import RoseMazeCPOMDP
from src.env_wrapper import EnvCAG, EnvCPOMDP, EnvCMDP
from src.formalisms.cag_to_bcmdp import Plan, CAG_to_BMDP
from src.formalisms.cag_to_cpomdp import CoordinationCPOMDP


def coord_rose_garden_test_policy(obs: Tuple[Tuple, ASGState]):
    if obs is None:
        return (Plan({"prm": (0, -1), "imprm": (1, 0)}), (0, 1))
    else:
        h_a = obs[0]
        s_concrete: ASGState = obs[1]
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


def bcmdp_rose_garden_test_policy(bstate: Tuple[ASGState, Distribution]):
    s_concrete, beta = bstate
    sup = list(beta.support())
    if len(sup) == 1:
        theta = sup[0]
    else:
        sup = sorted(sup)
        theta = sup[0]
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


class TestEnvWrappers(unittest.TestCase):
    def test_cag_wrapper(self):
        g1 = SimplestCAG2()
        env = EnvCAG(g1)
        control_scheme = {
            "8": (0, -1),
            "5": (0, 0),
            "2": (0, 1),
            "4": (-1, 0),
            "6": (1, 0),
            "w": (0, -1),
            "q": (0, 0),
            "s": (0, 1),
            "a": (-1, 0),
            "d": (1, 0)
        }
        moves = list(control_scheme.values())
        done = False

        def get_action_pair(is_human=False):
            if is_human:
                x = control_scheme[input()]
            else:
                x = random.choice(moves)
            return x, x

        env.reset()
        env.render()
        while not done:
            state, r, done, inf = env.step(get_action_pair())
            env.render()
        pass

    def test_cag_wrapper_on_rose_garden(self):
        g1 = RoseGarden()
        env = EnvCAG(g1)
        control_scheme = {
            "8": (0, -1),
            "5": (0, 0),
            "2": (0, 1),
            "4": (-1, 0),
            "6": (1, 0),
            "w": (0, -1),
            "q": (0, 0),
            "s": (0, 1),
            "a": (-1, 0),
            "d": (1, 0)
        }
        moves = list(control_scheme.values())
        done = False

        def get_action_pair(is_human=True):
            if is_human:
                x = control_scheme[input()]
            else:
                x = random.choice(moves)
            return x, x

        env.reset()
        env.render()
        while not done:
            state, r, done, inf = env.step(get_action_pair())
            env.render()
        pass

    def test_cpomdp_wrapper(self):
        g1 = RoseMazeCPOMDP()
        env = EnvCPOMDP(g1)

        control_scheme = {
            "w": 3,
            "a": 2,
            "s": 1,
            "d": 0,
            "e": 4
        }
        moves = list(control_scheme.values())

        def get_cpomdp_action(is_human=False):
            if is_human:
                x = control_scheme[input()]
            else:
                x = random.choice(moves)
            return x

        done = False

        env.reset()
        # env.render()
        while not done:
            obs, r, done, inf = env.step(get_cpomdp_action())
            env.render()
        pass

    def test_cmdp_wrapper(self):
        g1 = RoseMazeCMDP()
        x = g1.transition_probabilities
        g1.validate()
        env = EnvCMDP(g1)

        control_scheme = {
            "w": 3,
            "a": 2,
            "s": 1,
            "d": 0,
            # "e": 4
        }
        moves = list(control_scheme.values())

        def get_cmdp_action(is_human=False):
            if is_human:
                x = control_scheme[input()]
            else:
                x = random.choice(moves)
            return x

        done = False

        env.reset()
        env.render()
        while not done:
            a = get_cmdp_action()
            obs, r, done, inf = env.step(a)
            env.render()
            print(obs)
        pass

    def test_cpomdp_wrapper_with_coord_cpomdp(self):
        cag = RoseGarden()
        coord = CoordinationCPOMDP(deepcopy(cag))
        b0 = coord.b_0
        # assert cag.I == b0
        s_0 = coord.b_0.sample()
        a_0 = random.choice(list(coord.A))
        s1_dist = coord.T(s_0, a_0)

        done = False
        env = EnvCPOMDP(coord)
        obs = env.reset()
        env.render()
        while not done:
            obs, r, done, inf = env.step(coord_rose_garden_test_policy(obs))
            env.render()
        pass

    def test_rose_garden_bcmdp_cag_wrapper(self):
        cag = RoseGarden()
        bcmdp = CAG_to_BMDP(copy(cag))
        sup = list(bcmdp.I.support())
        if len(sup) != 1:
            raise ValueError
        # else:
        #     b0 = sup[0]
        #     if b0 != cag.I:
        #         raise ValueError
        done = False
        with EnvCMDP(bcmdp) as env:
            b_t = env.reset()
            env.render()
            while not done:
                b_t, r, done, inf = env.step(bcmdp_rose_garden_test_policy(b_t))
                env.render()

    def test_simplest2_reduction_to_cmdp_with_envwrapper(self):
        control_scheme = {
            "s": (0, 1),
            "d": (1, 0)
        }
        moves = list(control_scheme.values())

        def get_cmdp_action(is_human: bool = False):
            if is_human:
                x1 = control_scheme[input("human imprm")]
                x2 = control_scheme[input("human prm")]
                x3 = control_scheme[input("robot")]
            else:
                x1 = random.choice(moves)
                x2 = random.choice(moves)
                x3 = random.choice(moves)
            return Plan({"imprm": x1, "prm": x2}), x3

        cag = SimplestCAG2()
        bcmdp = CAG_to_BMDP(copy(cag))
        sup = list(bcmdp.I.support())
        if len(sup) != 1:
            raise ValueError

        done = False
        with EnvCMDP(bcmdp) as env:
            b_t = env.reset()
            env.render()
            while not done:
                b_t, r, done, inf = env.step(get_cmdp_action())
                env.render()
