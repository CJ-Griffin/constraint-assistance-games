import unittest

from unittest import TestCase

from src.env_wrapper import EnvCMDP, EnvCAG
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.example_environments.simplest_cag import SimplestCAG
from src.formalisms.cag_to_bcmdp import CAG_to_BMDP
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve
from src.example_environments.rose_garden_cag import RoseGarden
import pprint

GRID_WORLD_WIDTH = 5
GRID_WORLD_HEIGHT = 5
WALL_PROBABILITY = 0.2


class TestCMDPSolver(TestCase):
    def setUp(self):
        self.pp = pprint.PrettyPrinter(indent=4)
        self.cmdp = RoseMazeCMDP()

    def test_solve(self):
        solution = solve(self.cmdp, 0.99)
        self.pp.pprint(solution)


class TestCMDPSolverOnCAGReduction(TestCase):
    def setUp(self):
        self.pp = pprint.PrettyPrinter(indent=4)
        self.cag = RoseGarden()
        pickle_path = "cag_to_bmdp.pick"
        self.cmdp = CAG_to_BMDP(self.cag)
        self.cmdp.validate()

    def test_solve(self):
        solution = solve(self.cmdp, 0.99)
        self.pp.pprint(solution)


def get_rollout(cmdp, policy):
    init = cmdp.I
    hist = {}
    hist["s_0"] = cmdp.I


class TestCMDPSolverOnCAGReduction2(TestCase):
    def setUp(self):
        self.pp = pprint.PrettyPrinter(indent=4)
        self.cag = SimplestCAG()
        pickle_path = "cag_to_bmdp.pick"
        self.cmdp = CAG_to_BMDP(self.cag)
        self.cmdp.validate()

    def test_solve(self):
        solution = solve(self.cmdp, 0.99)
        soms = solution["state_occupancy_measures"]
        pol = solution["policy"]
        reached_states = [s for s in soms.keys() if soms[s] > 0]
        reached_states.sort(key=str)

        for state in reached_states:
            x = pol[state]
            print(state)
            print(soms[state])
            print(pol[state])
            print()

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


