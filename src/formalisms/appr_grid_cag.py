from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Set
from src.formalisms.cag import CAG
from src.formalisms.distributions import Distribution, KroneckerDistribution


@dataclass(frozen=True, eq=True)
class ASGState:
    h_xy: Tuple[int, int]
    r_xy: Tuple[int, int]
    whose_turn: str

    def __str__(self):
        return f"state: h_xy=({self.h_xy}) r_xy=({self.r_xy}) turn={self.whose_turn}"


class ApprenticeshipStaticGridCAG(CAG, ABC):
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
                 dud_action_penalty: float = 0.0
                 ):
        super().__init__()

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

        self.S = set([
            ASGState(h_s, r_s, whose_turn)
            for h_s in self.h_S
            for r_s in self.r_S
            for whose_turn in ["h", "r"]
        ])

        self.h_A = {
            (0, -1),  # UP
            (0, 1),  # DOWN
            # (0, 0),  # NoMove
            (-1, 0),  # Left
            (1, 0),  # Right
        }

        self.r_A = self.h_A.copy()

        self.s_0: ASGState = ASGState(self.h_start, self.r_start, "h")

    def T(self, s: ASGState, h_a, r_a) -> Distribution: # | None:
        if not isinstance(s, ASGState):
            raise ValueError
        h_s = s.h_xy
        r_s = s.r_xy
        whose_turn = s.whose_turn

        if whose_turn == "h":

            poss_dest = (h_s[0] + h_a[0], h_s[1] + h_a[1])
            if poss_dest in self.h_S:
                next_h_s = poss_dest
            else:
                next_h_s = h_s

            if next_h_s in self.h_sinks:
                return KroneckerDistribution(ASGState(next_h_s, r_s, "r"))
            else:
                return KroneckerDistribution(ASGState(next_h_s, r_s, "h"))

        elif whose_turn == "r":
            poss_dest = (r_s[0] + r_a[0], r_s[1] + r_a[1])
            if poss_dest in self.r_S:
                next_r_s = poss_dest
            else:
                next_r_s = r_s

            return KroneckerDistribution(ASGState(h_s, next_r_s, "r"))

        else:
            raise ValueError(f'{whose_turn} should be either "h" or "s"')

    def R(self, s: ASGState, h_a, r_a) -> float:
        # NOTE: this only works because this CAG is deterministic!
        next_s = self.T(s, h_a, r_a).sample()
        if next_s is None:
            return self.goal_reward
        else:
            if not isinstance(s, ASGState):
                raise ValueError
            if s == next_s:
                dud_penalty = self.dud_action_penalty
            else:
                dud_penalty = 0.0
            whose_turn = s.whose_turn
            next_whose_turn = next_s.whose_turn
            # If the humans turn is ending and the robots is beginning, the human reached the goal
            if whose_turn != next_whose_turn:
                return self.goal_reward + dud_penalty
            else:
                return 0.0 + dud_penalty

    def is_sink(self, s):
        assert s in self.S, f"s={s} is not in S={self.S}"
        return s.r_xy in self.r_sinks
