from abc import abstractmethod

import numpy as np

from src.formalisms.distributions import KroneckerDistribution, DiscreteDistribution
from src.formalisms.ecas_cags import EthicalContext
from src.formalisms.finite_processes import FiniteCAG
from src.formalisms.primitives import FiniteSpace
from src.abstract_gridworlds.grid_world_primitives import *
from src.utils.renderer import render


class StaticGridWorldCAG(FiniteCAG):
    Theta: set = None
    initial_theta_dist: DiscreteDistribution = None
    c_tuple: Tuple[float]

    CONTROL_SCHEME = control_scheme = {
        "w": A_NORTH,
        "q": A_NOOP,
        "d": A_EAST,
        "a": A_WEST,
        "s": A_SOUTH
    }

    def __init__(self, grid: np.ndarray,
                 gamma: float = 0.90,
                 dud_action_penalty: float = -10.0,
                 goal_reward: float = 1.0
                 ):
        assert dud_action_penalty <= 0
        self.dud_action_penalty = dud_action_penalty

        self.gamma = gamma
        self.goal_reward = goal_reward

        self.start_grid = grid.copy()

        poss_human_start_locs = np.argwhere(self.start_grid == "h")
        assert len(poss_human_start_locs) == 1
        self.human_start_loc = (poss_human_start_locs[0][1], poss_human_start_locs[0][0])

        poss_robot_start_locs = np.argwhere(self.start_grid == "r")
        assert len(poss_robot_start_locs) == 1
        self.robot_start_loc = (poss_robot_start_locs[0][1], poss_robot_start_locs[0][0])

        self.background_grid = self.start_grid.copy()
        self.background_grid[self.human_start_loc[1], self.human_start_loc[0]] = " "
        self.background_grid[self.robot_start_loc[1], self.robot_start_loc[0]] = " "

        self.tuple_grid = tuple(tuple(row) for row in self.background_grid)
        self.grid_coords = [
            (x, y)
            for x in range(self.start_grid.shape[1])
            for y in range(self.start_grid.shape[0]) if self.background_grid[y, x] != "#"
        ]

        self.goal_coords = [(x, y) for (x, y) in self.grid_coords if self.start_grid[y, x] == "*"]

        set_of_states = {
            self._create_state(h_xy, r_xy, h_can_act, r_can_act)
            for h_xy in self.grid_coords
            for r_xy in self.grid_coords if r_xy != h_xy
            for h_can_act in [True, False]
            for r_can_act in [True, False]
        }

        self.h_A = DIR_ACTIONS
        self.r_A = DIR_ACTIONS

        self.s_0 = self._create_state(self.human_start_loc, self.robot_start_loc, *self._get_start_who_can_act())
        self.S: FiniteSpace = FiniteSpace(set_of_states)
        reachable_state_set, reachable_state_tree = self.get_reachable_state_set_and_tree(self.s_0)
        # print_tree(reachable_state_tree)
        self.S: FiniteSpace = FiniteSpace(reachable_state_set)
        sinks = [s for s in self.S if self.is_sink(s)]
        if len(sinks) == 0:
            if (self.background_grid == "*").sum() < 2:
                raise ValueError("There's only one sink, but two agents!")
            else:
                raise ValueError("No sink state is reachable")

        super().__init__()

        if self.initial_theta_dist is not None:
            self._infer_state_theta_dist()

    def set_theta_dist(self, dist: DiscreteDistribution):
        self.initial_theta_dist = dist
        self._infer_state_theta_dist()

    def _infer_state_theta_dist(self):
        self.initial_state_theta_dist = DiscreteDistribution({
            (self.s_0, theta): self.initial_theta_dist.get_probability(theta)
            for theta in self.initial_theta_dist.support()
        })

    def _create_state(self, *args, **kwargs) -> StaticGridState:
        return StaticGridState(*args, _background_grid_tuple=self.tuple_grid, **kwargs)

    def _split_inner_T(self, s: StaticGridState, h_a: GridAction, r_a: GridAction) -> DiscreteDistribution:
        if self.is_sink(s):
            return KroneckerDistribution(s)
        else:
            next_h_s = self._get_next_indiv_s(h_a, s, who="h")
            next_r_s = self._get_next_indiv_s(r_a, s, who="r")
            if next_r_s == next_h_s and next_r_s:
                if next_r_s == s.r_xy:
                    next_h_s = s.h_xy
                else:
                    next_r_s = s.r_xy
            next_who_can_act = self._get_next_who_can_act(s, next_h_s, next_r_s, h_a, r_a)
            next_state = self._create_state(next_h_s, next_r_s, *next_who_can_act)
            return KroneckerDistribution(next_state)

    def _get_next_indiv_s(self, indiv_a: GridAction, s: StaticGridState, who: str):
        assert who in ["h", "r"]
        curr_pos = s.h_xy if who == "h" else s.r_xy
        can_they_act = s.h_can_act if who == "h" else s.r_can_act
        if can_they_act:
            a_vec = indiv_a.vector()
            poss_dest = (curr_pos[0] + a_vec[0], curr_pos[1] + a_vec[1])
            if poss_dest in self.grid_coords:
                actual_dest = poss_dest
            else:
                actual_dest = curr_pos
        else:
            actual_dest = curr_pos
        return actual_dest

    def is_sink(self, s: StaticGridState) -> bool:
        return not s.h_can_act and not s.r_can_act

    def _inner_split_R(self, s: StaticGridState, h_a: GridAction, r_a: GridAction) -> float:
        if self.is_sink(s):
            return 0.0
        else:
            next_dist = self.split_T(s, h_a, r_a)
            if next_dist.is_degenerate():
                next_s = next_dist.sample()
            else:
                raise ValueError

            h_dud_action_penalty = self.dud_action_penalty if h_a != A_NOOP and next_s.h_xy == s.h_xy else 0.0
            r_dud_action_penalty = self.dud_action_penalty if r_a != A_NOOP and next_s.r_xy == s.r_xy else 0.0

            if s.h_can_act and next_s.h_xy in self.goal_coords:
                human_reaches_goal_reward = self.goal_reward
            else:
                human_reaches_goal_reward = 0.0

            if s.r_can_act and next_s.r_xy in self.goal_coords:
                robot_reaches_goal_reward = self.goal_reward
            else:
                robot_reaches_goal_reward = 0.0

            return h_dud_action_penalty + r_dud_action_penalty + human_reaches_goal_reward + robot_reaches_goal_reward

    @abstractmethod
    def _get_start_who_can_act(self) -> Tuple[bool, bool]:
        pass

    @abstractmethod
    def _get_next_who_can_act(self, s: StaticGridState, next_h_s: Tuple[int, int], next_r_s: Tuple[int, int],
                              h_a: GridAction, r_a: GridAction) -> Tuple[bool, bool]:
        pass

    @abstractmethod
    def _inner_C(self, k: int, theta, s: StaticGridState, h_a: GridAction, r_a: GridAction) -> float:
        pass

    def render(self):
        st = ""
        st += render(self.s_0)
        st += "\n"
        st += "Ah = Ar = {↑, ↓, ←, →, _} \n"
        theta_list = list(self.Theta)
        if isinstance(theta_list[0], EthicalContext):
            st += "E"
        else:
            st += "Θ"
        st += " = {" + ",".join(render(theta) for theta in theta_list) + "}"
        st += "\n"


class CoordinationStaticGridCAG(StaticGridWorldCAG):

    @abstractmethod
    def _inner_C(self, k: int, theta, s: StaticGridState, h_a: GridAction, r_a: GridAction) -> float:
        pass

    def _get_start_who_can_act(self) -> Tuple[bool, bool]:
        return True, True

    def _get_next_who_can_act(self,
                              s: StaticGridState,
                              next_h_s: Tuple[int, int],
                              next_r_s: Tuple[int, int],
                              h_a: GridAction,
                              r_a: GridAction) -> Tuple[bool, bool]:
        return (next_h_s not in self.goal_coords), (next_r_s not in self.goal_coords)


class ApprenticeshipStaticGridCAG(StaticGridWorldCAG):

    @abstractmethod
    def _inner_C(self, k: int, theta, s: StaticGridState, h_a: GridAction, r_a: GridAction) -> float:
        pass

    def _get_start_who_can_act(self) -> Tuple[bool, bool]:
        return True, False

    def _get_next_who_can_act(self, s: StaticGridState, next_h_s: Tuple[int, int], next_r_s: Tuple[int, int],
                              h_a: GridAction, r_a: GridAction) -> Tuple[bool, bool]:
        if s.h_can_act:
            if next_h_s in self.goal_coords:
                return False, True
            else:
                return True, False
        else:
            assert s.r_can_act
            if next_r_s in self.goal_coords:
                return False, False
            else:
                return False, True
