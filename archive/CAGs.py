from abc import ABC, abstractmethod, abstractproperty
from typing import TypeVar, Generic, Set

StateType = TypeVar('StateType')
HumanActionType = TypeVar('HumanActionType')
RobotActionType = TypeVar('RobotActionType')
ThetaType = TypeVar('ThetaType')

SampleType = TypeVar('SampleType')


class Distribution(ABC, Generic[SampleType]):

    @abstractmethod
    def sample(self) -> SampleType:
        raise NotImplementedError

    @abstractmethod
    def support(self) -> Set[SampleType]:
        raise NotImplementedError

    @abstractmethod
    def get_probability(self, x: SampleType) -> float:
        raise NotImplementedError


# Assume it is finite, fully-observable (and deterministic?)
class AssistanceGame(ABC, Generic[StateType, HumanActionType, RobotActionType, ThetaType]):

    @property
    @abstractmethod
    def state(self) -> StateType:
        raise NotImplementedError

    @property
    @abstractmethod
    def state_space(self) -> Set[StateType]:
        raise NotImplementedError

    @property
    @abstractmethod
    def human_action_space(self) -> Set[HumanActionType]:
        raise NotImplementedError

    @property
    @abstractmethod
    def robot_action_space(self) -> Set[RobotActionType]:
        raise NotImplementedError

    @abstractmethod
    def step(self,
             a_h: HumanActionType,
             a_r: RobotActionType) -> (StateType, float, bool, dict):
        raise NotImplementedError

    @property
    @abstractmethod
    def gamma(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> StateType:
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError


# class NumberLine(AssistanceGame[int, int, int, bool]):
#     state_space = set(range(-100, 101))
#     human_action_space = {-1, 0, 1}
#     robot_action_space = {-1, 0, 1}
#     gamma: float = None
#     state: StateType = None
#
#     def __init__(self,
#                  gamma: float = 0.9):
#         self.gamma = gamma
#
#     def step(self, a_h: HumanActionType, a_r: RobotActionType) \
#             -> (int, float, bool, dict):
#         s = self.state
#         sp = s + a_h + a_r
#         reward = 1 if (sp == 10) else 0
#         done = bool(reward)
#         info = {}
#         self.state = spz
#         `
#         return sp, reward, done, info
#
#     def reset(self) -> bool:
#         self.state = 0
#         return self.state
#
#     def render(self):
#         pass


if __name__ == "__main__":
    nl = NumberLine()
