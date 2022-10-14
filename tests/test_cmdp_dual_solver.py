import unittest
from abc import ABC, abstractmethod

from unittest import TestCase

from src.env_wrapper import EnvCMDP, EnvCAG
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.example_environments.simplest_cag import SimplestCAG, SimplestCAG2
from src.formalisms.cag_to_bcmdp import CAG_to_BMDP
from src.formalisms.cmdp import FiniteCMDP
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve
from src.example_environments.rose_garden_cag import RoseGarden
import pprint

GRID_WORLD_WIDTH = 5
GRID_WORLD_HEIGHT = 5
WALL_PROBABILITY = 0.2


class TestCMDPSolver(TestCase):
    cmdp: FiniteCMDP = None
    pp = pprint.PrettyPrinter(indent=4)

    def setUp(self):
        self.cmdp = RoseMazeCMDP()
        self.cmdp.validate()

    def test_solve(self):
        solution = solve(self.cmdp, 0.99)
        soms = solution["state_occupancy_measures"]
        pol = solution["policy"]
        reached_states = [s for s in soms.keys() if soms[s] > 0]
        reached_states.sort(key=str)

        for state in reached_states:
            print(state)
            print(soms[state])
            print(pol[state])
            print()

        print(f"Value = {solution['objective_value']}")
        c_val_dict = solution["constraint_values"]
        for constr_name in c_val_dict:
            print(f"{constr_name} = {c_val_dict[constr_name]}")

        for s_0 in self.cmdp.I.support():
            done = False
            env = EnvCMDP(self.cmdp)
            obs = env.reset(s_0)
            env.render()
            while not done:
                a = pol[obs].sample()
                obs, r, done, inf = env.step(a)
                env.render()
                # print(obs)
            pass


class TestDualSolveRoseGarden(TestCMDPSolver):
    def setUp(self):
        self.cag = RoseGarden()
        self.cmdp = CAG_to_BMDP(self.cag)
        self.cmdp.validate()


class TestDualSolveSimpleCAG(TestCMDPSolver):
    def setUp(self):
        self.cag = SimplestCAG()
        self.cmdp = CAG_to_BMDP(self.cag)
        self.cmdp.validate()


class TestDualSolveSimpleCAG2(TestCMDPSolver):
    def setUp(self):
        self.cag = SimplestCAG2()
        self.cmdp = CAG_to_BMDP(self.cag)
        self.cmdp.validate()
