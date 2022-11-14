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


def open_log_debug(file_name: str, *args, **kwargs) -> TextIO:
    proj_root_path = get_root_path()
    import os
    if os.sep in file_name:
        raise ValueError("This looks like a path, not a file name")
    path = os.path.join(proj_root_path, "logs", f"{file_name}")
    return open_debug(path, *args, **kwargs)


def get_root_path():
    import os
    cwd = os.getcwd()
    split_path = os.path.normpath(cwd).split(os.path.sep)
    cag_ind = split_path.index("cags")
    proj_root_path = os.path.join(os.sep, *split_path[:cag_ind + 1])
    if not os.path.exists(proj_root_path):
        raise ValueError
    return proj_root_path


def get_path_relative_to_root(p: str):
    import os
    root_path = get_root_path()
    path = os.path.join(root_path, p)
    return path


def open_debug(path_name: str, *args, **kwargs) -> TextIO:
    try:
        file = open(path_name, *args, **kwargs)
    except FileNotFoundError as fnfe:
        import os
        cwd = os.getcwd()
        path = os.path.normpath(path_name)
        split_path = path.split(os.sep)
        for i in range(len(split_path)):
            subpath = os.path.join(*split_path[:i + 1])
            if not os.path.exists(subpath):
                raise ValueError
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
    class term:
        dct = {
            "green": Fore.GREEN,
            "red": Fore.RED,
            "blue": Fore.BLUE,
            "purple": Fore.MAGENTA,
            "yellow": Fore.YELLOW,
            "black": Fore.BLACK,
            "grey": Fore.LIGHTBLACK_EX,
        }

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
        def purple(s: str) -> str:
            return Fore.MAGENTA + s + Style.RESET_ALL

        @staticmethod
        def yellow(s: str) -> str:
            return Fore.YELLOW + s + Style.RESET_ALL

        @staticmethod
        def black(s: str) -> str:
            return Fore.BLACK + s + Style.RESET_ALL

        @staticmethod
        def grey(s: str) -> str:
            return Fore.LIGHTBLACK_EX + s + Style.RESET_ALL

    class html:
        background_hex = "#2E3440"
        black_hex = "#3B4252"
        blue_hex = "#81A1C1"
        cyan_hex = "#88C0D0"
        foreground_hex = "#D8DEE9"
        green_hex = "#A3BE8C"
        brightBlack_hex = "#8490a7"
        brightBlue_hex = "#81A1C1"
        brightCyan_hex = "#8FBCBB"
        brightGreen_hex = "#A3BE8C"
        brightPurple_hex = "#B48EAD"
        brightWhite_hex = "#ECEFF4"
        brightYellow_hex = "#EBCB8B"
        purple_hex = "#B48EAD"
        red_hex = "#BF616A"
        white_hex = "#E5E9F0"
        yellow_hex = "#EBCB8B"

        color_dct = {
            "green": green_hex,
            "red": red_hex,
            "blue": blue_hex,
            "purple": purple_hex,
            "yellow": yellow_hex,
            "black": black_hex,
            "grey": brightBlack_hex
        }

        @staticmethod
        def red(s: str) -> str:
            return f'<span style="color:{colors.html.red_hex};">{s}</span>'

        @staticmethod
        def green(s: str) -> str:
            return f'<span style="color:{colors.html.green_hex};">{s}</span>'

        @staticmethod
        def blue(s: str) -> str:
            return f'<span style="color:{colors.html.blue_hex};">{s}</span>'

        @staticmethod
        def purple(s: str) -> str:
            return f'<span style="color:{colors.html.purple_hex};">{s}</span>'

        @staticmethod
        def yellow(s: str) -> str:
            return f'<span style="color:{colors.html.yellow_hex};">{s}</span>'

        @staticmethod
        def black(s: str) -> str:
            return f'<span style="color:{colors.html.black_hex};">{s}</span>'

        @staticmethod
        def grey(s: str) -> str:
            return f'<span style="color:{colors.html.brightBlack_hex};">{s}</span>'

        @staticmethod
        def get_start_str(col_name: str) -> str:
            return f'<span style="color:{colors.html.color_dct[col_name]};">'

        @staticmethod
        def get_end_str() -> str:
            return '</span>'

    @staticmethod
    def term_to_html(s: str) -> str:
        for col in colors.term.dct.keys():
            s = s.replace(colors.term.dct[col], colors.html.get_start_str(col))
        s = s.replace(Style.RESET_ALL, colors.html.get_end_str())
        return s


def write_to_html(st, path):
    st = colors.term_to_html(st)
    start = f"""
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<!-- This file was created with the aha Ansi HTML Adapter. https://github.com/theZiz/aha -->
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="application/xml+xhtml; charset=UTF-8"/>
<title>stdin</title>
</head>
<body style="background-color:{colors.html.background_hex}; font-family: monospace; color: {colors.html.white_hex};">
<pre>"""
    end = """
</pre>
</body>
</html>
"""
    with open_debug(path, "a+") as file:
        file.write(start + st + end)
        print(f"Written to {path}")
