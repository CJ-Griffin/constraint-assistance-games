import random
import unittest
from copy import deepcopy, copy
from typing import Tuple

import numpy as np

from src.example_environments.simple_mdp import SimpleMDP
from src.example_environments.simplest_cag import SimplestCAG
from src.formalisms.appr_grid_cag import ASGState
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.example_environments.rose_garden_cag import RoseGarden
# from src.example_environments.maze_cpomdp import RoseMazeCPOMDP
from src.env_wrapper import EnvWrapper
# from src.env_wrapper import EnvCPOMDP
from src.formalisms.cag_to_bcmdp import Plan, CAGtoBCMDP
# from src.formalisms.cag_to_cpomdp import CoordinationCPOMDP
from src.formalisms.distributions import Distribution


def coord_rose_garden_test_policy(obs: Tuple[Tuple, ASGState]):
    if obs is None:
        return Plan({"prm": (0, -1), "imprm": (1, 0)}), (0, 1)
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
            return h_a, r_a


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
            "prm": (0, 1),
        })
        if s_concrete.h_xy == (0, 0):
            plan_dict["imprm"] = (1, 0)
        elif s_concrete.h_xy in [(1, 0), (1, 1), (0, 1)]:
            plan_dict["imprm"] = (0, 1)
        elif s_concrete.h_xy == (1, 2):
            plan_dict["imprm"] = (-1, 0)
        else:
            raise NotImplementedError

        a_r = (0, 0)
        return Plan(plan_dict), a_r
    else:
        h_a = Plan({"prm": (0, 0), "imprm": (0, 0)})
        if s_concrete.r_xy == (0, 0):
            r_a = (0, 1)
        elif s_concrete.r_xy in [(0, 1), (1, 1)]:
            r_a = (1, 0)
        elif s_concrete.r_xy == (2, 1):
            r_a = (0, -1)
        else:
            raise NotImplementedError
        return h_a, r_a


class TestEnvWrappers(unittest.TestCase):
    def test_cag_wrapper(self):
        g1 = SimplestCAG()
        env = EnvWrapper(g1)
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
        env = EnvWrapper(g1)
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

    def test_cmdp_wrapper(self):
        g1 = RoseMazeCMDP()
        x = g1.transition_probabilities
        g1.validate()
        env = EnvWrapper(g1)

        control_scheme = {
            "w": 3,
            "a": 2,
            "s": 1,
            "d": 0,
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

    def test_rose_garden_bcmdp_cag_wrapper(self):
        cag = RoseGarden()
        bcmdp = CAGtoBCMDP(copy(cag))
        sup = list(bcmdp.initial_state_dist.support())
        if len(sup) != 1:
            raise ValueError
        done = False
        with EnvWrapper(bcmdp) as env:
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

        cag = SimplestCAG()
        bcmdp = CAGtoBCMDP(copy(cag))
        sup = list(bcmdp.initial_state_dist.support())
        if len(sup) != 1:
            raise ValueError

        done = False
        with EnvWrapper(bcmdp) as env:
            b_t = env.reset()
            env.render()
            while not done:
                b_t, r, done, inf = env.step(get_cmdp_action())
                env.render()

    def test_simple_mdp(self):
        mdp = SimpleMDP()
        list_of_actions = list(mdp.A)

        def get_mdp_action(is_human: bool = False):
            if is_human:
                raise NotImplementedError
            else:
                return np.random.choice(list_of_actions)

        done = False
        with EnvWrapper(mdp) as env:
            s_t = env.reset()
            env.render()
            while not done:
                s_t, r, done, inf = env.step(get_mdp_action())
                env.render()
