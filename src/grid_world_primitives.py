from dataclasses import dataclass
from typing import Tuple, Union, List

from src.formalisms.primitives import Action, State


@dataclass(frozen=True, eq=True)
class GridAction(Action):
    name: str

    _CHAR_DICT = {
        "north": "↑",
        "south": "↓",
        "east": "→",
        "west": "←",
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


@dataclass(frozen=True)
class StaticGridState(State):
    h_xy: Tuple[int, int]
    r_xy: Tuple[int, int]
    h_can_act: bool
    r_can_act: bool

    _background_grid_tuple: Tuple[Tuple[str, ...], ...]

    from tabulate import TableFormat, Line, DataRow
    _TABULATE_FORMAT = TableFormat(
        lineabove=Line("╭─", "┬", "─", "─╮"),
        datarow=DataRow("├ ", " ", " ┤"),
        linebelow=Line("╰─", "┴", "─", "─╯"),
        linebelowheader=None,
        linebetweenrows=None,
        headerrow=None,
        padding=0,  # Changed to 0 from 1
        with_header_hide=["lineabove"],
    )

    def __post_init__(self):
        width = len(self._background_grid_tuple[0])
        assert all(len(row) == width for row in self._background_grid_tuple)
        assert 0 <= self.h_xy[0] < width
        assert 0 <= self.r_xy[0] < width

        height = len(self._background_grid_tuple)
        assert 0 <= self.h_xy[1] < height
        assert 0 <= self.r_xy[1] < height

        assert self.h_xy != self.r_xy

    def __str__(self, short=False):
        who_str = '' + ('h' if self.h_can_act else '') + ('r' if self.r_can_act else '')
        return f"<s:h=({self.h_xy[0]},{self.h_xy[1]}),r=({self.r_xy[0]},{self.r_xy[1]}),t={who_str}>"

    def __repr__(self):
        who_str = '' + ('h' if self.h_can_act else '') + ('r' if self.r_can_act else '')
        st = f"<{self.__class__.__name__}:"
        st += f"h=({self.h_xy[0]},{self.h_xy[1]}),r=({self.r_xy[0]},{self.r_xy[1]}),"
        st += f"t={who_str},"
        st += f"bgg=np.ndarray"
        st += f">"
        return st

    def __eq__(self, other):
        if isinstance(other, StaticGridState):
            return self.__dict__ == other.__dict__
        else:
            return False

    def render(self, rend_who_str: bool = False) -> str:
        import numpy as np
        grid = np.array(self._background_grid_tuple)
        grid[self.h_xy[1], self.h_xy[0]] = "h"
        grid[self.r_xy[1], self.r_xy[0]] = "r"
        grid_st = self._array2d_to_str(grid)
        colourfull_st = self._get_colourful_unicode_str(grid_st)
        if rend_who_str:
            colourfull_st += "\nwho=" + ('h' if self.h_can_act else '') + ('r' if self.r_can_act else '')
        return colourfull_st

    def get_human_cell(self):
        return self._background_grid_tuple[self.h_xy[1]][self.h_xy[0]]

    def get_robot_cell(self):
        return self._background_grid_tuple[self.r_xy[1]][self.r_xy[0]]

    @staticmethod
    def _array2d_to_str(array2d) -> str:
        if (array2d == ";").any():
            raise ValueError("this is reserved for preserving spaces")
        array2d[array2d == " "] = ";"
        from tabulate import tabulate
        str_arr = tabulate(array2d, tablefmt=StaticGridState._TABULATE_FORMAT)
        return str_arr.replace(";", " ")

    def _get_colourful_unicode_str(self, st: str):
        return "".join(self._char_to_colorful_unicode(list(st)))

    def _char_to_colorful_unicode(self, c: Union[str, List[str]]):
        if isinstance(c, list):
            return [self._char_to_colorful_unicode(x) for x in c]
        elif isinstance(c, str) and len(c) == 1:
            from src.utils import colors
            if c == "R":
                return colors.term.red("⌘")
            elif c == "D":
                return colors.term.yellow("⌘")
            elif c == "L":
                return colors.term.pink("⌘")
            elif c == "0":
                return " "
            elif c == "*":
                return colors.term.yellow("*")
            elif c == "h" and not self.h_can_act:
                return colors.term.grey("h")
            elif c == "r" and not self.r_can_act:
                return colors.term.grey("r")
            else:
                return c
        else:
            raise TypeError
