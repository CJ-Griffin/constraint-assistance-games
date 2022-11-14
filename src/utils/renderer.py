_PRIMITIVES = (int, str, bool, float)
_BASIC_COMPOSITES = (list, tuple, set, frozenset)


def render(x: object) -> str:
    if hasattr(x, "render"):
        return x.render()
    else:
        if isinstance(x, _PRIMITIVES):
            return str(x)
        elif isinstance(x, _BASIC_COMPOSITES):
            return _render_basic_composite(x)
        else:
            raise NotImplementedError(x)


def _add_indents(s: str):
    return s.replace("\n", "\n    ")


def _render_tuple(t: tuple):
    return "(" + ", ".join([render(y) for y in t]) + ")"


def _render_list(xs: list):
    return "[" + ", ".join([render(y) for y in xs]) + "]"


def _render_set(t: set):
    return "{" + ", ".join([render(y) for y in t]) + "}"


def _render_basic_composite(xs):
    if isinstance(xs, tuple):
        return _render_tuple(xs)
    elif isinstance(xs, list):
        return _render_list(xs)
    elif isinstance(xs, set):
        return _render_set(xs)
    else:
        raise NotImplementedError


def construct_children(top_node, d):
    from pptree import Node
    for key in d.keys():
        child = Node(str(key), top_node)
        construct_children(child, d[key])


def print_tree(d: dict):
    from pptree import Node, print_tree
    assert len(d) == 1
    root_state = next(iter(d.keys()))
    root_node = Node(str(root_state))
    construct_children(root_node, d[root_state])
    # import sys
    # with open('trees.txt', 'w') as sys.stdout:
    #     print()
    #     print(datetime.datetime.now())
    print_tree(root_node)
