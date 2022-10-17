from abc import abstractmethod, ABC

from src.env_wrapper import EnvCMDP
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.example_environments.simplest_cag import SimplestCAG
from src.formalisms.cag_to_bcmdp import CAGtoBCMDP
from src.formalisms.cmdp import FiniteCMDP
from src.get_traj_dist import get_traj_dist
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve
from src.example_environments.rose_garden_cag import RoseGarden
from src.renderer import render
import pprint

GRID_WORLD_WIDTH = 5
GRID_WORLD_HEIGHT = 5
WALL_PROBABILITY = 0.2


class TestCMDPSolver(ABC):
    cmdp: FiniteCMDP = None
    pp = pprint.PrettyPrinter(indent=4)

    @abstractmethod
    def setUp(self):
        pass

    def test_solve(self):
        solution = solve(self.cmdp)
        self.explore_solution(solution)

    def explore_solution(self, solution):
        self.explore_solution_with_trajectories(solution)

    def explore_solution_with_trajectories(self, solution, tol_min_prob: float = 1e-6):
        traj_dist = get_traj_dist(
            cmdp=self.cmdp,
            pol=solution["policy"]
        )

        fetch_prob = (lambda tr: traj_dist.get_probability(tr))
        filter_func = (lambda tr: traj_dist.get_probability(tr) > tol_min_prob)
        filtered_trajs = filter(filter_func, traj_dist.support())
        sorted_trajs = sorted(filtered_trajs, key=fetch_prob, reverse=True)
        for traj in sorted_trajs:
            print(render(traj))
            print(f"Prob = {traj_dist.get_probability(traj)}")
            print()

    def explore_solution_old(self, solution):
        soms = solution["state_occupancy_measures"]
        pol = solution["policy"]
        reached_states = [s for s in soms.keys() if soms[s] > 0]
        reached_states.sort(key=(lambda x: str(x[1]) + str(x[0])))

        print("=" * 100)

        for state in reached_states:
            print()
            print("STATE:", render(state))
            print("STATE OCC. MEASURE:", render(soms[state]))
            print("POLICY:", render(pol[state]))
            print()

        print("=" * 100)

        print(f"Value = {solution['objective_value']}")
        c_val_dict = solution["constraint_values"]
        for constr_name in c_val_dict:
            print(f"{constr_name} => {c_val_dict[constr_name]}")

        print("=" * 100)

        for s_0 in self.cmdp.initial_state_dist.support():
            done = False
            env = EnvCMDP(self.cmdp)
            obs = env.reset()
            env.render()
            while not done:
                a = pol[obs].sample()
                obs, r, done, inf = env.step(a)
                env.render()
                # print(obs)
            pass


class SolveRoseMazeCMDP(TestCMDPSolver):
    def setUp(self):
        self.cmdp = RoseMazeCMDP()
        self.cmdp.validate()


class SolveRoseGarden(TestCMDPSolver):
    def setUp(self):
        cag = RoseGarden()
        self.cmdp = CAGtoBCMDP(cag)
        self.cmdp.validate()


class SolveSimpleCAG(TestCMDPSolver):
    def setUp(self):
        cag = SimplestCAG()
        self.cmdp = CAGtoBCMDP(cag)
        self.cmdp.validate()
