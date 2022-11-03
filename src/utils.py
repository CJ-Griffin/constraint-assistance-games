import functools
import time
from typing import TextIO

import numpy as np
from colorama import Fore, Style


# Adapted from https://towardsdatascience.com/a-simple-way-to-time-code-in-python-a9a175eb0172
def time_function(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time_taken = end_time - start_time
        print(f"{func.__name__}(args={args}, kwargs={kwargs})")
        print(f" \\__ took time t={total_time_taken}")
        return result

    return inner


def open_debug(file_name: str, *args, **kwargs) -> TextIO:
    try:
        file = open(file_name, *args, **kwargs)
    except FileNotFoundError as fnfe:
        import os
        cwd = os.getcwd()
        ls = os.listdir(cwd)
        raise fnfe
    return file


def raise_exception_at_difference_in_arrays(m1: np.ndarray, m2: np.ndarray):
    if not m1.shape == m2.shape:
        s1 = m1.shape
        s2 = m2.shape
        raise Exception
    if not (m1 == m2).all():
        import numpy as np
        locs = np.argwhere(m1 != m2)
        for i in range(locs.shape[0]):
            triplet = locs[i]
            v1 = m1[tuple(triplet)]
            v2 = m2[tuple(triplet)]
            raise ValueError


class colors:
    @staticmethod
    def green(s: str) -> str:
        return Fore.GREEN + s + Style.RESET_ALL

    @staticmethod
    def red(s: str) -> str:
        return Fore.RED + s + Style.RESET_ALL

    @staticmethod
    def blue(s: str) -> str:
        return Fore.BLUE + s + Style.RESET_ALL

    @staticmethod
    def pink(s: str) -> str:
        return Fore.MAGENTA + s + Style.RESET_ALL

    @staticmethod
    def yellow(s: str) -> str:
        return Fore.YELLOW + s + Style.RESET_ALL

    @staticmethod
    def light_cyan(s: str) -> str:
        return Fore.LIGHTCYAN_EX + s + Style.RESET_ALL
