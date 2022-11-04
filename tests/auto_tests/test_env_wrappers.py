import random
import unittest
from copy import copy
from typing import Tuple

import numpy as np

from src.appr_grid_cag import ASGState, A_NORTH, A_SOUTH, A_EAST, A_WEST, A_NOOP
from src.concrete_processes.maze_cmdp import RoseMazeCMDP
from src.concrete_processes.simple_mdp import SimpleMDP
from src.concrete_processes.simplest_cag import SimplestCAG
from src.env_wrapper import EnvWrapper
from src.formalisms.primitives import ActionPair, IntAction, Plan
from src.reductions.cag_to_bcmdp import CAGtoBCMDP, BeliefState


def coord_rose_garden_test_policy(obs: Tuple[Tuple, ASGState]):
    if obs is None:
        return Plan({"prm": A_NORTH, "imprm": A_EAST}), A_SOUTH
    else:
        h_a = obs[0]
        s_concrete: ASGState = obs[1]
        if s_concrete.whose_turn == "h":
            plan_dict = ({
                "prm": A_NORTH,
            })
            if s_concrete.h_xy == (0, 2):
                plan_dict["imprm"] = A_EAST
            elif s_concrete.h_xy in [(1, 2), (1, 1)]:
                plan_dict["imprm"] = A_NORTH
            elif s_concrete.h_xy == A_EAST:
                plan_dict["imprm"] = A_WEST
            else:
                plan_dict["imprm"] = A_SOUTH

            a_r = A_SOUTH
            return ActionPair(Plan(plan_dict), a_r)
        else:
            h_a = Plan({"prm": A_EAST, "imprm": A_EAST})
            if s_concrete.r_xy == (0, 1):
                r_a = A_NORTH
            elif s_concrete.r_xy in [(0, 0), (1, 0)]:
                r_a = A_EAST
            elif s_concrete.r_xy == (2, 0):
                r_a = A_SOUTH

            else:
                r_a = A_SOUTH
            return ActionPair(h_a, r_a)


def bcmdp_rose_garden_test_policy(bstate: BeliefState):
    s_concrete, beta = bstate.s, bstate.beta
    sup = list(beta.support())
    if not isinstance(s_concrete, ASGState):
        raise ValueError
    if len(sup) == 1:
        theta = sup[0]
    else:
        sup = sorted(sup)
        theta = sup[0]
    if s_concrete.whose_turn == "h":
        plan_dict = ({
            "prm": A_SOUTH,
        })
        if s_concrete.h_xy == (0, 0):
            plan_dict["imprm"] = A_EAST
        elif s_concrete.h_xy in [(1, 0), (1, 1), (0, 1)]:
            plan_dict["imprm"] = A_SOUTH
        elif s_concrete.h_xy == (1, 2):
            plan_dict["imprm"] = A_WEST
        else:
            raise NotImplementedError

        a_r = A_NOOP
        return ActionPair(Plan(plan_dict), a_r)
    else:
        h_a = Plan({"prm": A_NOOP, "imprm": A_NOOP})
        if s_concrete.r_xy == (0, 0):
            r_a = A_SOUTH
        elif s_concrete.r_xy in [(0, 1), (1, 1)]:
            r_a = A_EAST
        elif s_concrete.r_xy == (2, 1):
            r_a = A_NORTH
        else:
            raise NotImplementedError
        return ActionPair(h_a, r_a)


class TestEnvWrappers(unittest.TestCase):
    def test_cag_wrapper(self):
        g1 = SimplestCAG()
        env = EnvWrapper(g1)
        control_scheme = {
            "w": A_NORTH,
            "q": A_NOOP,
            "s": A_EAST,
            "a": A_WEST,
            "d": A_SOUTH
        }
        moves = list(control_scheme.values())
        done = False

        def get_action_pair(is_human=False):
            if is_human:
                x = control_scheme[input()]
            else:
                x = random.choice(moves)
            return ActionPair(x, x)

        env.reset()
        env.render()
        while not done:
            state, r, done, inf = env.step(get_action_pair())
            env.render()
        pass

    def test_cmdp_wrapper(self):
        g1 = RoseMazeCMDP()
        x = g1.transition_probabilities
        g1.check_matrices()
        env = EnvWrapper(g1)

        control_scheme = {
            "w": IntAction(3),
            "a": IntAction(2),
            "s": IntAction(1),
            "d": IntAction(0),
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

    def test_simplest2_reduction_to_cmdp_with_envwrapper(self):
        control_scheme = {
            "s": A_SOUTH,
            "d": A_EAST
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
            return ActionPair(Plan({"imprm": x1, "prm": x2}), x3)

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
            s = env.reset()
            env.render()
            while not done:
                s, r, done, inf = env.step(get_mdp_action())
                env.render()
