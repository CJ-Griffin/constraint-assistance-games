import collections.abc
from itertools import product


class Plan(collections.abc.Mapping):

    def __init__(self, dict_map: dict):
        self._d = dict_map

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


def get_all_plans(Theta, h_A):
    Lambda: set = {
        Plan({
            theta: ordering[i]
            for i, theta in enumerate(Theta)
        })
        for ordering in product(h_A, repeat=len(Theta))
    }
    return Lambda
