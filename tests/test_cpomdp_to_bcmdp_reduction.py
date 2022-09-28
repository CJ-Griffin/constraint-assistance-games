from copy import copy
from src.formalisms import BCMDP
from src.particulars.maze_cpomdp import RoseMazeCPOMDP

if __name__ == "__main__":
    cpomdp = RoseMazeCPOMDP()
    bcmdp = BCMDP(copy(cpomdp))
    b0 = bcmdp.I.sample()
    assert cpomdp.b_0 == b0
    b1_dist = bcmdp.T(b0, 1)
    b1 = b1_dist.sample()
    reward = bcmdp.R(b0, 1, b1)
    pass
    # bcmdp.T()


    # g1 = RoseMazeCMDP()
    # env = EnvCMDP(g1)

    # control_scheme = {
    #     "w": 3,
    #     "a": 2,
    #     "s": 1,
    #     "d": 0,
    #     "e": 4
    # }
    #
    # def get_action_pair():
    #     x = control_scheme[input()]
    #     return x
    #
    # done = False
    #
    # env.reset()
    # env.render()
    # while not done:
    #     obs, r, done, inf = env.step(get_action_pair())
    #     env.render()
    #     print(obs)
    # pass
