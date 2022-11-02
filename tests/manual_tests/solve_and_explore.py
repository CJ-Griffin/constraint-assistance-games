import pprint
from abc import abstractmethod, ABC

from src.concrete_processes.maze_cmdp import RoseMazeCMDP
from src.concrete_processes.rose_garden_cag import RoseGarden
from src.concrete_processes.simplest_cag import SimplestCAG
from src.reductions.cag_to_bcmdp import CAGtoBCMDP
from src.formalisms.finite_processes import FiniteCMDP
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve
from src.policy_analysis import explore_CMDP_solution_with_trajectories


class TestCMDPSolver(ABC):
    cmdp: FiniteCMDP = None
    pp = pprint.PrettyPrinter(indent=4)

    @abstractmethod
    def setUp(self):
        pass

    def test_solve(self):
        policy, solution_details = solve(self.cmdp)
        self.explore_solution(policy, solution_details)

    def explore_solution(self, policy, solution_details):
        explore_CMDP_solution_with_trajectories(policy, self.cmdp, solution_details)


class SolveRoseMazeCMDP(TestCMDPSolver):
    def setUp(self):
        self.cmdp = RoseMazeCMDP()
        self.cmdp.check_matrices()


class SolveRoseGarden(TestCMDPSolver):
    def setUp(self):
        cag = RoseGarden()
        self.cmdp = CAGtoBCMDP(cag)
        self.cmdp.check_matrices()


class SolveSimpleCAG(TestCMDPSolver):
    def setUp(self):
        cag = SimplestCAG()
        self.cmdp = CAGtoBCMDP(cag)
        self.cmdp.check_matrices()
