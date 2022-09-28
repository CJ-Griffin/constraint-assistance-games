from abc import ABC, abstractmethod, abstractproperty
from typing import TypeVar, Generic

StateType = TypeVar('StateType')
HumanActionType = TypeVar('HumanActionType')
RobotActionType = TypeVar('RobotActionType')
ThetaType = TypeVar('ThetaType')


class TransitionFunction(ABC, Generic[StateType, HumanActionType, RobotActionType]):

    @abstractmethod
    def get_prob(self, s: StateType, a_h: HumanActionType, a_r: RobotActionType, sp: StateType) -> float:
        pass

    @abstractmethod
    def sample_successor(self, s: StateType, a_h: HumanActionType, a_r: RobotActionType) -> StateType:
        pass


class RewardFunction(ABC, Generic[StateType, HumanActionType, RobotActionType]):

    @abstractmethod
    def get_reward(self, s: StateType, a_h: HumanActionType, a_r: RobotActionType, sp: StateType) -> float:
        pass


class CostFunction(ABC, Generic[StateType, HumanActionType, RobotActionType, ThetaType]):
    @abstractmethod
    def get_cost(self, theta: ThetaType, s: StateType, a_h: HumanActionType, a_r: RobotActionType,
                 sp: StateType) -> float:
        pass


class ConstrainedAssistanceGame(ABC, Generic[StateType, HumanActionType, RobotActionType, ThetaType]):

    @property
    @abstractmethod
    def transition_function(self) -> TransitionFunction:
        pass

    @property
    @abstractmethod
    def reward_function(self) -> RewardFunction:
        pass

    @property
    @abstractmethod
    def cost_function(self) -> CostFunction:
        pass

    @abstractmethod
    def reset_env(self) -> StateType:
        pass

    @abstractmethod
    def step(self, a_h: HumanActionType, a_r: RobotActionType) -> StateType:
        pass
