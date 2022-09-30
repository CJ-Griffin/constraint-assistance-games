# from abc import ABC, abstractmethod
#
# import cplex
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor
#
# from src.formalisms import CMDP
#
# NUM_PROCESSES = 72
#
#
# def task(mdp, states, memory_mdp):
#     transition_probabilities = {}
#     for state in states:
#         for action in range(memory_mdp.n_actions):
#             for successor_state in range(memory_mdp.n_states):
#                 if state not in transition_probabilities:
#                     transition_probabilities[state] = np.zeros(shape=(memory_mdp.n_actions, memory_mdp.n_states))
#                     transition_probabilities[state][action, successor_state] = mdp.transition_function(
#                         memory_mdp.states[state], memory_mdp.actions[action], memory_mdp.states[successor_state])
#     return transition_probabilities
#
#
# def get_partitions(n_states, n_processes):
#     state_space_partitions = []
#     min_size = n_states // n_processes
#     remainder = n_states % n_processes
#     current_index = 0
#
#     for i in range(remainder):
#         end_index = current_index + min_size + 1
#         state_space_partitions.append(list(range(current_index, end_index)))
#         current_index = end_index
#
#     for i in range(n_processes - remainder):
#         end_index = current_index + min_size
#         state_space_partitions.append(list(range(current_index, end_index)))
#         current_index = end_index
#
#     return state_space_partitions
#
#
# class CMDPwithMatrixSupport(ABC, CMDP):
#     @property
#     @abstractmethod
#     def transition_matrix(self) -> np.array:
#         """
#         T[i, j, k] = T(stp1 = state k | state i and action j)
#         """
#         pass
#
#     @property
#     @abstractmethod
#     def reward_matrix(self) -> np.array:
#         """
#         R(i,j) = R(state i, action j)
#         """
#         pass
#
#     @property
#     @abstractmethod
#     def constraint_matrix(self) -> np.array:
#         """
#         C(k, i, j) = C_k (state i, action j)
#         """
#         pass
#
#     def validate(self):
#         with self.constraint_matrix.shape() as cshape:
#             assert len(cshape) == 3
#             assert cshape[0] == self.K
#             assert cshape[1] == len(self.S)
#             assert cshape[2] == len(self.A)
#         with self.reward_matrix.shape() as rshape:
#             assert len(rshape) == 2
#             assert rshape[0] == len(self.S)
#             assert rshape[1] == len(self.A)
#         with self.transition_matrix.shape() as tshape:
#             assert len(rshape) == 3
#             assert tshape[0] == len(self.S)
#             assert tshape[1] == len(self.A)
#             assert tshape[2] == len(self.S)
#
#         total_prob_from_s_a = self.transition_matrix.sum(axis=2)
#         pass
#
#
# class MatrixCMDP(CMDPwithMatrixSupport):
#     def __init__(self, cmdp: CMDP, parallelize=False):
#         list_of_abstr_states = list(cmdp.S)
#         self.state_to_ind = {
#             i: list_of_abstr_states[i] for i in range(len(list_of_abstr_states))
#         }
#         self.states = list(range(len(list_of_abstr_states)))
#
#         list_of_abstr_actions = list(cmdp.A)
#         self.action_to_ind = {
#             i: list_of_abstr_actions[i] for i in range(len(list_of_abstr_actions))
#         }
#         self.actions = list(range(len(list_of_abstr_actions)))
#
#         self.n_states = len(self.states)
#         self.n_actions = len(self.actions)
#
#         self.rewards = np.zeros(shape=(self.n_states, self.n_actions))
#         for state in range(self.n_states):
#             for action in range(self.n_actions):
#                 self.rewards[state, action] = cmdp.R(self.states[state], self.actions[action])
#
#         self.transition_probabilities = np.zeros(shape=(self.n_states, self.n_actions, self.n_states))
#
#         if parallelize:
#             with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as pool:
#                 partition_futures = []
#                 state_space_partitions = get_partitions(self.n_states, NUM_PROCESSES)
#
#                 for state_space in state_space_partitions:
#                     partition_future = pool.submit(task, cmdp, state_space, self)
#                     partition_futures.append(partition_future)
#
#                 for partition_future in partition_futures:
#                     result = partition_future.result()
#                     for state in result:
#                         self.transition_probabilities[state] = result[state]
#         else:
#             for state in range(self.n_states):
#                 for action in range(self.n_actions):
#                     for successor_state in range(self.n_states):
#                         dist = cmdp.T(self.states[state], self.actions[action])
#                         prob = dist.get_probability(self.states[successor_state])
#                         self.transition_probabilities[state, action, successor_state] =
#
#         self.start_state_probabilities = np.zeros(self.n_states)
#         for state in range(self.n_states):
#             self.start_state_probabilities[state] = cmdp.start_state_function(self.states[state])
#
#     def validate(self):
#         assert self.n_states is not None
#         assert self.n_actions is not None
#
#         assert self.states is not None
#         assert self.actions is not None
#         assert self.rewards is not None
#         assert self.transition_probabilities is not None
#         assert self.start_state_probabilities is not None
#
#         assert self.rewards.shape == (self.n_states, self.n_actions)
#         assert self.transition_probabilities.shape == (self.n_states, self.n_actions, self.n_states)
#         assert self.start_state_probabilities.shape == (self.n_states,)
