import collections.abc
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from itertools import product
from typing import Iterable, Sized

from src.utils.renderer import render


@dataclass(frozen=True, eq=True)
class State(ABC):
    """
    A State object should be immutable, and implement __hash__, __eq__ and __str__
    """

    # @abstractmethod
    def __str__(self, short=False) -> str:
        rend_str = f"{self.__class__.__name__}("
        for field in fields(self):
            rend_str += f"{field.name}:{getattr(self, field.name)}, "
        rend_str += ")"
        return rend_str

    @abstractmethod
    def render(self) -> str:
        pass

    # @abstractmethod
    # def __repr__(self):
    #     pass

    # @abstractmethod
    # def __hash__(self):
    #     pass
    #
    # @abstractmethod
    # def __eq__(self, other):
    #     pass


@dataclass(frozen=True, eq=True)
class Action(ABC):
    """
    An Action object should be immutable, and implement __hash__, __eq__ and __str__
    """

    # @abstractmethod
    # def __str__(self, short=False) -> str:
    #     rend_str = f"{self.__class__.__name__}("
    #     for field in fields(self):
    #         rend_str += f"{field.name}:{getattr(self, field.name)}, "
    #     rend_str += ")"
    #     return rend_str

    @abstractmethod
    def render(self):
        pass


@dataclass(frozen=True, eq=True)
class ActionPair(Action):
    a0: Action
    a1: Action

    def render(self):
        return f"ä=({self.a0.render()}, {self.a1.render()})"

    #
    # def __repr__(self):
    #     return f"<a0={repr(self.a0)} ,a1={repr(self.a1)}>"

    def __getitem__(self, item):
        if item == 0:
            return self.a0
        elif item == 1:
            return self.a1
        else:
            raise IndexError

    def __len__(self):
        return 2

    def __iter__(self):
        return iter([self.a0, self.a1])


@dataclass(frozen=True, eq=True)
class IntState(State, ABC):
    n: int

    def render(self) -> str:
        return f"(s={self.n}"


@dataclass(frozen=True, eq=True)
class IntAction(Action):
    n: int

    def render(self):
        return f"(a={self.n})"

    def __repr__(self):
        return f"<IntAction({self.n})>"


@dataclass(frozen=True)
class Plan(collections.abc.Mapping, Action):
    _d: dict

    def __getitem__(self, k):
        return self._d[k]

    def get_keys(self):
        return self._d.keys()

    def get_values(self):
        return self._d.values()

    def __len__(self) -> int:
        return len(self._d)

    def __iter__(self):
        return list(self._d.items())

    def __hash__(self):
        items = self._d.items()
        hashes = [hash(item) for item in items]
        return hash(tuple(sorted(hashes)))

    def __eq__(self, other):
        if isinstance(other, Plan):
            if set(self._d.keys()) != set(other._d.keys()):
                return False
            else:
                return all(
                    self._d[key] == other._d[key]
                    for key in self._d.keys()
                )
        else:
            return False

    def __str__(self):
        return f"<Plan: {self._d} >"

    def __call__(self, x):
        return self[x]

    def render(self):
        # string = f"<_Plan"
        string = ""
        for key in self.get_keys():
            string += f"\n| λ({render(key)}) = {render(self(key))}"
        string += "\n"
        return string


def get_all_plans(Theta, h_A):
    Lambda: set = {
        Plan({
            theta: ordering[i]
            for i, theta in enumerate(Theta)
        })
        for ordering in product(h_A, repeat=len(Theta))
    }
    return Lambda


class Space(ABC, Iterable):

    @abstractmethod
    def __contains__(self, y):  # real signature unknown; restored from __doc__
        """ x.__contains__(y) <==> y in x. """
        pass


class FiniteSpace(Space, Iterable, Sized):
    is_finite = True

    def __init__(self, set_of_elements):
        super().__init__()
        self._set = set_of_elements
        self._len = len(set_of_elements)

    def __contains__(self, item):
        return item in self._set

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self._set)


class CountableSpace(Space, Iterable):

    @abstractmethod
    def __contains__(self, y):  # real signature unknown; restored from __doc__
        """ x.__contains__(y) <==> y in x. """
        pass

    @abstractmethod
    def __iter__(self):
        pass
