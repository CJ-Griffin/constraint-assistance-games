from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Set

import numpy as np

from src.formalisms.finite_processes import FiniteCAG
from src.formalisms.distributions import Distribution, KroneckerDistribution
from src.formalisms.primitives import State, Action, FiniteSpace


@dataclass(frozen=True, eq=True)
class GridAction(Action):
    name: str

    _CHAR_DICT = {
        "north": "↑",
        "south": "↑",
        "east": "↑",
        "west": "↑",
        "noop": "_",
        "interact": "☚"
    }

    _VECTOR_DICT = {
        "north": (0, -1),
        "south": (0, 1),
        "east": (1, 0),
        "west": (-1, 0),
        "noop": (0, 0),
        "interact": ValueError
    }

    def __post_init__(self):
        if self.name not in ["north", "south", "east", "west", "noop", "interact"]:
            raise ValueError

    def render(self):
        return self._CHAR_DICT[self.name]

    def vector(self):
        return self._VECTOR_DICT[self.name]

    def __repr__(self):
        return f"<{self._CHAR_DICT[self.name]}>"

    def __getitem__(self, item):
        return self.vector()[item]


A_NORTH = GridAction("north")
A_SOUTH = GridAction("south")
A_EAST = GridAction("east")
A_WEST = GridAction("west")
A_NOOP = GridAction("noop")
A_INTERACT = GridAction("interact")

DIR_ACTIONS = frozenset({A_NORTH, A_SOUTH, A_EAST, A_WEST, A_NOOP})


@dataclass(frozen=True, eq=True)
class ASGState(State):
    h_xy: Tuple[int, int]
    r_xy: Tuple[int, int]
    whose_turn: str

    _grid: np.ndarray

    def __str__(self, short=False):
        if short:
            return f"<ASGState: h={self.h_xy},  r=({self.r_xy}), t={self.whose_turn}>"
        else:
            return f"<ASGState: h={self.h_xy},  r=({self.r_xy}), t={self.whose_turn}>"

    def render(self):
        return str(self)

    def __repr__(self):
        return repr(str(self))


class ApprenticeshipStaticGridCAG(FiniteCAG, ABC):
    """
    Leaves properties Theta, K and I undefined.
    Also leaves methods C and c undefined.
    """

    def __init__(self,
                 h_height: int,
                 h_width: int,
                 h_start: (int, int),
                 h_sinks: Set[Tuple[int, int]],
                 r_height: int,
                 r_width: int,
                 r_start: (int, int),
                 r_sinks: Set[Tuple[int, int]],
                 goal_reward: float,
                 gamma: float,
                 dud_action_penalty: float = 0.0,
                 grid: np.ndarray = None
                 ):

        assert dud_action_penalty <= 0
        self.dud_action_penalty = dud_action_penalty

        self.h_height = h_height
        self.h_width = h_width
        self.h_start = h_start
        self.h_sinks = h_sinks

        self.r_height = r_height
        self.r_width = r_width
        self.r_start = r_start
        self.r_sinks = r_sinks

        self.goal_reward = goal_reward
        self.gamma = gamma

        self.h_S = [(x, y) for x in range(h_width) for y in range(h_height)]

        self.r_S = [(x, y) for x in range(r_width) for y in range(r_height)]

        self.grid = grid

        states_where_human_is_next = {
            ASGState(h_s, self.r_start, "h", _grid=self.grid)
            for h_s in self.h_S if h_s not in self.h_sinks
        }
        states_where_robot_is_next = {
            ASGState(h_s, r_s, "r", _grid=self.grid)
            for h_s in self.h_sinks
            for r_s in self.r_S
        }

        set_of_states = states_where_human_is_next | states_where_robot_is_next

        self.S: FiniteSpace = FiniteSpace(set_of_states)

        self.h_A = DIR_ACTIONS

        self.r_A = DIR_ACTIONS

        self.s_0: ASGState = ASGState(self.h_start, self.r_start, "h", _grid=self.grid)
        super().__init__()

    def _split_inner_T(self, s: ASGState, h_a, r_a) -> Distribution:  # | None:
        if h_a not in self.h_A:
            raise ValueError
        if r_a not in self.r_A:
            raise ValueError

        if not isinstance(s, ASGState):
            raise ValueError
        elif s not in self.S:
            raise ValueError
        elif self.is_sink(s):
            return KroneckerDistribution(s)
        else:
            h_s = s.h_xy
            r_s = s.r_xy
            whose_turn = s.whose_turn

            next_state = None

            if whose_turn == "h":

                poss_dest = (h_s[0] + h_a[0], h_s[1] + h_a[1])
                if poss_dest in self.h_S:
                    next_h_s = poss_dest
                else:
                    next_h_s = h_s

                if next_h_s in self.h_sinks:
                    next_state = (ASGState(next_h_s, r_s, "r", _grid=self.grid))
                else:
                    next_state = (ASGState(next_h_s, r_s, "h", _grid=self.grid))

            elif whose_turn == "r":
                poss_dest = (r_s[0] + r_a[0], r_s[1] + r_a[1])
                if poss_dest in self.r_S:
                    next_r_s = poss_dest
                else:
                    next_r_s = r_s

                next_state = (ASGState(h_s, next_r_s, "r", _grid=self.grid))

            else:
                raise ValueError(f'{whose_turn} should be either "h" or "s"')

            if next_state not in self.S:
                raise ValueError
            return KroneckerDistribution(next_state)

    def split_R(self, s: ASGState, h_a, r_a) -> float:
        if not isinstance(s, ASGState):
            raise ValueError
        elif self.is_sink(s):
            return 0.0
        else:
            next_dist = self._split_inner_T(s, h_a, r_a)
            if next_dist.is_degenerate():
                next_s = next_dist.sample()
            else:
                raise ValueError

            if s.whose_turn == "h" and next_s.h_xy in self.h_sinks:
                reached_goal_reward = 1.0
            elif s.whose_turn == "r" and next_s.r_xy in self.r_sinks:
                reached_goal_reward = 1.0
            else:
                reached_goal_reward = 0.0

            if s == next_s:
                dud_penalty = self.dud_action_penalty
            else:
                dud_penalty = 0.0

            if s.whose_turn == "r" and h_a != (0, 0):
                not_humans_turn_penalty = self.dud_action_penalty
            else:
                not_humans_turn_penalty = 0.0

            if s.whose_turn == "h" and r_a != (0, 0):
                not_robots_turn_penalty = self.dud_action_penalty
            else:
                not_robots_turn_penalty = 0.0

            return reached_goal_reward + dud_penalty + not_humans_turn_penalty + not_robots_turn_penalty

    def is_sink(self, s: ASGState):
        assert s in self.S, f"s={s} is not in S={self.S}"
        return s.r_xy in self.r_sinks


class MirrorApprentishipCAG(ApprenticeshipStaticGridCAG, ABC):
    def __init__(self, height: int, width: int, start: (int, int), sinks: Set[Tuple[int, int]], goal_reward: float,
                 gamma: float):
        super().__init__(height, width, start, sinks, height, width, start, sinks, goal_reward, gamma)
