from src.particulars.rose_garden_cag import RoseGarden
from src.env_wrapper import EnvCAG

if __name__ == "__main__":
    g1 = RoseGarden()
    env = EnvCAG(g1)
    control_scheme = {
        "8": (0, -1),
        "5": (0, 0),
        "2": (0, 1),
        "4": (-1, 0),
        "6": (1, 0),
        "w": (0, -1),
        "q": (0, 0),
        "s": (0, 1),
        "a": (-1, 0),
        "d": (1, 0)
    }
    done = False


    def get_action_pair():
        x = control_scheme[input()]
        return x, x

    env.reset()
    env.render()
    while not done:
        state, r, done, inf = env.step(get_action_pair())
        env.render()
    pass
