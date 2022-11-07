# from abc import ABC
# from dataclasses import dataclass
# from typing import Tuple, Set
#
# import numpy as np
# from tabulate import TableFormat, Line, DataRow, tabulate
#
# from src.formalisms.distributions import Distribution, KroneckerDistribution
# from src.formalisms.finite_processes import FiniteCAG
# from src.formalisms.primitives import State, Action, FiniteSpace
# from src.grid_world_primitives import A_NOOP, DIR_ACTIONS
# from src.utils import colors
#
# """
# ╭─┬─┬─┬─╮
# ├ 0 R * ┤
# ├   R   ┤
# ╰─┴─┴─┴─╯
# ╭─┬─╮
# │.│.│
# ├─┼─┤
# │.│.│
# ╰─┴─╯
# """
#
#
# @dataclass(frozen=True)
# class ASGState(State):
#     h_xy: Tuple[int, int]
#     r_xy: Tuple[int, int]
#     whose_turn: str
#
#     _human_background_grid_tuple: Tuple[Tuple[str, ...], ...]
#     _robot_background_grid_tuple: Tuple[Tuple[str, ...], ...]
#
#     _INNER_GRID_FORMAT = TableFormat(
#         lineabove=Line("╭─", "┬", "─", "─╮"),
#         datarow=DataRow("├ ", " ", " ┤"),
#         linebelow=Line("╰─", "┴", "─", "─╯"),
#         linebelowheader=None,
#         linebetweenrows=None,
#         headerrow=None,
#         padding=0,  # Changed to 0 from 1
#         with_header_hide=["lineabove"],
#     )
#
#     def __str__(self, short=False):
#         if short:
#             return f"<ASGState: h={self.h_xy},  r=({self.r_xy}), t={self.whose_turn}>"
#         else:
#             return f"<ASGState: h={self.h_xy},  r=({self.r_xy}), t={self.whose_turn}>"
#
#     def __eq__(self, other):
#         if isinstance(other, ASGState):
#             return self.__dict__ == other.__dict__
#         else:
#             return False
#
#     def render(self):
#         if self._human_background_grid_tuple is not None:
#             hgrid = np.array(self._human_background_grid_tuple)
#             rgrid = np.array(self._robot_background_grid_tuple)
#             hgrid[self.h_xy[1], self.h_xy[0]] = "h"
#             rgrid[self.r_xy[1], self.r_xy[0]] = "r"
#             h_st = self._inner_array2d_to_str(hgrid)
#             r_st = self._inner_array2d_to_str(rgrid)
#             colourless_st = self._outer_array2d_to_str([[h_st, r_st]])
#             colourfull_st = self._get_colourful_unicode_str(colourless_st)
#             return colourfull_st
#         else:
#             return str(self)
#
#     def __repr__(self):
#         return repr(str(self))
#
#     @staticmethod
#     def _inner_array2d_to_str(array2d) -> str:
#         if (array2d == ";").any():
#             raise ValueError("this is reserved for preserving spaces")
#         array2d[array2d == " "] = ";"
#         str_arr = tabulate(array2d, tablefmt=ASGState._INNER_GRID_FORMAT)
#         return str_arr.replace(";", " ")
#
#     @staticmethod
#     def _outer_array2d_to_str(array2d) -> str:
#         return tabulate(array2d, tablefmt="plain")
#
#     @staticmethod
#     def _get_colourful_unicode_str(st: str):
#         return "".join(ASGState._char_to_colorful_unicode(list(st)))
#
#     @staticmethod
#     @np.vectorize
#     def _char_to_colorful_unicode(c: str):
#         if c == "R":
#             return colors.red("⌘")
#         elif c == "D":
#             return colors.yellow("⌘")
#         elif c == "L":
#             return colors.pink("⌘")
#         elif c == "0":
#             return " "
#         elif c == "*":
#             return colors.yellow("*")
#         # elif c == "h":
#         #     return "≗"
#         # elif c == "r":
#         #     return colors.light_cyan("r")
#         else:
#             return c

#
# class ApprenticeshipStaticGridCAG(FiniteCAG, ABC):
#     """
#     Leaves properties Theta, K and I undefined.
#     Also leaves methods C and c undefined.
#     """
#
#     def __init__(self,
#                  h_height: int,
#                  h_width: int,
#                  h_start: (int, int),
#                  h_sinks: Set[Tuple[int, int]],
#                  r_height: int,
#                  r_width: int,
#                  r_start: (int, int),
#                  r_sinks: Set[Tuple[int, int]],
#                  goal_reward: float,
#                  gamma: float,
#                  dud_action_penalty: float = -10.0,
#                  human_bg_grid: np.ndarray = None,
#                  robot_bg_grid: np.ndarray = None
#                  ):
#
#         assert dud_action_penalty <= 0
#         self.dud_action_penalty = dud_action_penalty
#
#         self.h_height = h_height
#         self.h_width = h_width
#         self.h_start = h_start
#         self.h_sinks = h_sinks
#
#         self.r_height = r_height
#         self.r_width = r_width
#         self.r_start = r_start
#         self.r_sinks = r_sinks
#
#         self.goal_reward = goal_reward
#         self.gamma = gamma
#
#         self.h_S = [(x, y) for x in range(h_width) for y in range(h_height)]
#
#         self.r_S = [(x, y) for x in range(r_width) for y in range(r_height)]
#
#         self.human_bg_grid_tuple = None if human_bg_grid is None else tuple(
#             tuple(row)
#             for row in human_bg_grid
#         )
#
#         self.robot_bg_grid_tuple = None if human_bg_grid is None else tuple(
#             tuple(row)
#             for row in robot_bg_grid
#         )
#
#         class CustASGState(ASGState):
#             def __init__(inner_self, *args, **kwargs):
#                 ASGState.__init__(inner_self, *args,
#                                   _human_background_grid_tuple=self.human_bg_grid_tuple,
#                                   _robot_background_grid_tuple=self.robot_bg_grid_tuple,
#                                   **kwargs)
#
#         self.ASGState = CustASGState
#
#         states_where_human_is_next = {
#             self.ASGState(h_s, self.r_start, "h")
#             for h_s in self.h_S if h_s not in self.h_sinks
#         }
#         states_where_robot_is_next = {
#             self.ASGState(h_s, r_s, "r")
#             for h_s in self.h_sinks
#             for r_s in self.r_S
#         }
#
#         set_of_states = states_where_human_is_next | states_where_robot_is_next
#
#         self.S: FiniteSpace = FiniteSpace(set_of_states)
#
#         self.h_A = DIR_ACTIONS
#
#         self.r_A = DIR_ACTIONS
#
#         self.s_0: ASGState = self.ASGState(self.h_start, self.r_start, "h")
#         super().__init__()
#
#     def _split_inner_T(self, s: ASGState, h_a: Action, r_a: Action) -> Distribution:  # | None:
#         if self.is_sink(s):
#             return KroneckerDistribution(s)
#         else:
#             h_s = s.h_xy
#             r_s = s.r_xy
#             whose_turn = s.whose_turn
#
#             next_state = None
#
#             if whose_turn == "h":
#
#                 poss_dest = (h_s[0] + h_a[0], h_s[1] + h_a[1])
#                 if poss_dest in self.h_S:
#                     next_h_s = poss_dest
#                 else:
#                     next_h_s = h_s
#
#                 if next_h_s in self.h_sinks:
#                     next_state = (self.ASGState(next_h_s, r_s, "r"))
#                 else:
#                     next_state = (self.ASGState(next_h_s, r_s, "h"))
#
#             elif whose_turn == "r":
#                 poss_dest = (r_s[0] + r_a[0], r_s[1] + r_a[1])
#                 if poss_dest in self.r_S:
#                     next_r_s = poss_dest
#                 else:
#                     next_r_s = r_s
#
#                 next_state = (self.ASGState(h_s, next_r_s, "r"))
#
#             else:
#                 raise ValueError(f'{whose_turn} should be either "h" or "s"')
#
#             if next_state not in self.S:
#                 raise ValueError
#             return KroneckerDistribution(next_state)
#
#     def _inner_split_R(self, s: ASGState, h_a, r_a) -> float:
#         if not isinstance(s, ASGState):
#             raise ValueError
#         elif self.is_sink(s):
#             return 0.0
#         else:
#             next_dist = self._split_inner_T(s, h_a, r_a)
#             if next_dist.is_degenerate():
#                 next_s = next_dist.sample()
#             else:
#                 raise ValueError
#
#             if s.whose_turn == "h" and next_s.h_xy in self.h_sinks:
#                 reached_goal_reward = 1.0
#             elif s.whose_turn == "r" and next_s.r_xy in self.r_sinks:
#                 reached_goal_reward = 1.0
#             else:
#                 reached_goal_reward = 0.0
#
#             if s == next_s:
#                 dud_penalty = self.dud_action_penalty
#             else:
#                 dud_penalty = 0.0
#
#             if s.whose_turn == "r" and h_a != A_NOOP:
#                 not_humans_turn_penalty = self.dud_action_penalty
#             else:
#                 not_humans_turn_penalty = 0.0
#
#             if s.whose_turn == "h" and r_a != A_NOOP:
#                 not_robots_turn_penalty = self.dud_action_penalty
#             else:
#                 not_robots_turn_penalty = 0.0
#
#             return reached_goal_reward + dud_penalty + not_humans_turn_penalty + not_robots_turn_penalty
#
#     def is_sink(self, s: ASGState):
#         assert s in self.S, f"s={s} is not in S={self.S}"
#         return s.r_xy in self.r_sinks
#
#
# class MirrorApprentishipCAG(ApprenticeshipStaticGridCAG, ABC):
#     def __init__(self, height: int, width: int, start: (int, int), sinks: Set[Tuple[int, int]], goal_reward: float,
#                  gamma: float, grid_array: np.ndarray = None):
#         super().__init__(height, width, start, sinks, height, width, start, sinks, goal_reward, gamma,
#                          human_bg_grid=grid_array, robot_bg_grid=grid_array)
