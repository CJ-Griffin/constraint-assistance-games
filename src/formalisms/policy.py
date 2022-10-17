from abc import abstractmethod

from src.formalisms.distributions import Distribution
from src.formalisms.spaces import Space, FiniteSpace


class MemorylessPolicy:
    def __init__(self, S: Space, A: set):
        self.S: Space = S
        self.A: set = A

    @abstractmethod
    def _get_distribution(self, s) -> Distribution:
        pass

    def __call__(self, s) -> Distribution:
        if s not in self.S:
            raise ValueError
        else:
            return self._get_distribution(s)


class FiniteStateSpaceMemorylessPolicy(MemorylessPolicy):

    def __init__(self, S: Space, A: set, state_to_dist_map: dict, should_validate=True):
        if not isinstance(S, FiniteSpace):
            raise ValueError
        super().__init__(S, A)
        self._state_to_dist_map = state_to_dist_map

    def _get_distribution(self, s) -> Distribution:
        return self._state_to_dist_map[s]

    def validate(self):
        for s in self.S:
            if s not in self._state_to_dist_map:
                raise ValueError
            else:
                for a in self._state_to_dist_map[s].support():
                    if a not in self.A:
                        raise ValueError


class FiniteCAGPolicy:
    def __init__(self, S: Space, A: set):
        self.S: Space = S
        self.A: set = A
        raise NotImplementedError

    # @abstractmethod
    # def _get_distribution(self, s) -> Distribution:
    #     pass
    #
    # def __call__(self, ) -> Distribution:
    #     if s not in self.S:
    #         raise ValueError
    #     else:
    #         return self._get_distribution(s)
