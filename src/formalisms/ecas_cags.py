from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, Hashable, Set

from src.formalisms.cag import CAG
from src.formalisms.decision_process import validate_c


class EthicalContext(Hashable, ABC):
    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class EthicalComplianceAG(CAG, ABC):
    Theta: Set[EthicalContext]

    @abstractmethod
    def C(self, k: int, theta: EthicalContext, s, h_a, r_a) -> float:
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class DCTEthicalContext:
    forbidden_states: FrozenSet


class DivineCommandTheoryCAG(EthicalComplianceAG, ABC):

    def C(self, k: int, theta: DCTEthicalContext, s, h_a, r_a) -> float:
        if theta not in self.Theta:
            raise ValueError
        elif k != 0:
            raise ValueError
        else:
            next_state_dist = self.T(s, (h_a, r_a))
            forbidden = {f for f in next_state_dist.support() if f in theta.forbidden_states}
            return sum(next_state_dist.get_probability(f) for f in forbidden)

    @validate_c
    def c(self, k: int) -> float:
        return 0.0

    def __init__(self, poss_Fs: FrozenSet[FrozenSet]):
        self.Theta = set({DCTEthicalContext(F) for F in poss_Fs})
        for poss_F in poss_Fs:
            for f in poss_F:
                if f not in self.S:
                    raise ValueError

                # class PrimaFacieDuties(CAG, ABC):
#     def C(self, k: int, theta: FrozenSet, s, h_a, r_a) -> float:
#         if theta not in self.Theta:
#             raise ValueError
#         elif k != 0:
#             raise ValueError
#         else:
#             next_state_dist = self.T(s, (h_a, r_a))
#             forbidden = {f for f in next_state_dist.support() if f in theta}
#             return sum(next_state_dist.get_probability(f) for f in forbidden)
#
#     @abstractmethod
#     def c(self, k: int) -> float:
#         return 0.0
#
#     def __init__(self, poss_Fs: FrozenSet[FrozenSet]):
#         self.Theta = set(poss_Fs)
#         for poss_F in poss_Fs:
#             for f in poss_F:
#                 if f not in self.S:
#                     raise ValueError
