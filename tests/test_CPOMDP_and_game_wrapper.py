from src.particulars.maze_cpomdp import RoseMazeCPOMDP
from src.env_wrapper import EnvCAG, EnvCPOMDP

if __name__ == "__main__":
    g1 = RoseMazeCPOMDP()
    env = EnvCPOMDP(g1)

    control_scheme = {
        "w": 3,
        "a": 2,
        "s": 1,
        "d": 0,
        "e": 4
    }


    def get_action_pair():
        x = control_scheme[input()]
        return x

    done = False

    env.reset()
    # env.render()
    while not done:
        obs, r, done, inf = env.step(get_action_pair())
        # env.render()
        print(obs)
    pass
