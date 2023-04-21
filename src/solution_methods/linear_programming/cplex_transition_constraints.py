import cplex
import numpy as np
from tqdm import tqdm


def __set_transition_constraints(c, cmdp, should_tqdm: bool):
    """
    the constraint for state s' takes the form:
    Σ_{a'} μ(s',a') =  d(s') + Σ_{s, a} (γ* T(s, a, s') * μ(s,a))

    rearranging we get:
    Σ_{a} (1) μ(s',a')
        + Σ_{s, a} [ (- γ * T(s, a, s')) * μ(s,a) ]
        = d(s')
    where the RHS (d(s')) is a constant.

    therefore the coefficient for μ(s,a) is:
    1[s == s'] - γ * T(s, a, s')

    """

    # each constraint will use all (s, a) occupancy measures as possible predecessor states
    variables = range(cmdp.n_states * cmdp.n_actions)

    trans_coeff_array = - cmdp.gamma * cmdp.transition_matrix[:, :, :]
    next_s_inds = range(cmdp.n_states)
    trans_coeff_array[next_s_inds, :, next_s_inds] += 1.0

    old_shape = trans_coeff_array.shape
    new_shape = (old_shape[0] * old_shape[1], old_shape[2])

    trans_coeff_2darray = trans_coeff_array.view(new_shape)

    lower_bound_num_bytes = (trans_coeff_2darray != 0).sum() * 8
    lower_bound_num_megabytes = lower_bound_num_bytes / (1024 ** 2)
    print(f"A lower bound for the number of megabytes needed to store the program is {lower_bound_num_megabytes} MB")

    iterator = range(cmdp.n_states) if not should_tqdm \
        else tqdm(range(cmdp.n_states), desc=" | Generating and adding transition constraints statewise")

    for next_s_ind in iterator:
        row = trans_coeff_2darray[:, next_s_ind]
        non_zero_indeces = np.nonzero(row)[0]
        non_zero_vals = row[non_zero_indeces]
        sparse_pair = cplex.SparsePair(non_zero_indeces.tolist(), non_zero_vals.tolist())
        c.linear_constraints.add(
            lin_expr=[sparse_pair],
            names=[f"dynamics constraint s={next_s_ind}"],
            senses=["E"],
            rhs=[float(cmdp.start_state_probabilities[next_s_ind])]
        )


def __set_transition_constraints_1(c, cmdp, should_tqdm: bool):
    iterator = range(cmdp.n_states) if not should_tqdm \
        else tqdm(range(cmdp.n_states), desc=" | Generating and adding transition constraints statewise")

    for next_s_ind in iterator:
        t_coeffs = - cmdp.gamma * cmdp.transition_matrix[:, :, next_s_ind]
        t_coeffs[next_s_ind, :] += 1
        s_inds, a_inds = np.nonzero(t_coeffs)
        sa_inds = [int(s_ind * cmdp.n_actions + a_ind) for s_ind, a_ind in zip(s_inds, a_inds)]
        non_zero_vals2 = t_coeffs[s_inds, a_inds]

        sparse_pair = cplex.SparsePair(sa_inds, non_zero_vals2.tolist())
        c.linear_constraints.add(
            lin_expr=[sparse_pair],
            names=[f"dynamics constraint s={next_s_ind}"],
            senses=["E"],
            rhs=[float(cmdp.start_state_probabilities[next_s_ind])]
        )


def __set_transition_constraints_batchwise(c, cmdp, should_tqdm: bool, batch_size=16):
    """
    A fundtion for adding transition constraints, that calculates coefficients in batches and uses SparsePair
    :param c:
    :param cmdp:
    :param should_tqdm:
    :param batch_size:
    :return:
    """

    num_batches = (cmdp.n_states + batch_size - 1) // batch_size
    indices = [[i + j for j in range(batch_size) if i + j < cmdp.n_states]
               for i in range(0, cmdp.n_states, batch_size)]

    iterator = indices if not should_tqdm \
        else tqdm(indices, desc=" | Generating and adding transition constraints batchwise")

    for next_s_batch_indeces in iterator:
        t_coeffs_all = - cmdp.gamma * cmdp.transition_matrix[:, :, next_s_batch_indeces]
        t_coeffs_all[next_s_batch_indeces, :, range(len(next_s_batch_indeces))] += 1
        for i, next_s_ind in enumerate(next_s_batch_indeces):
            t_coeffs = t_coeffs_all[:, :, i]
            s_inds, a_inds = np.nonzero(t_coeffs)
            sa_inds = [int(s_ind * cmdp.n_actions + a_ind) for s_ind, a_ind in zip(s_inds, a_inds)]
            non_zero_vals2 = t_coeffs[s_inds, a_inds]

            sparse_pair = cplex.SparsePair(sa_inds, non_zero_vals2.tolist())
            c.linear_constraints.add(
                lin_expr=[sparse_pair],
                names=[f"dynamics constraint s={next_s_ind}"],
                senses=["E"],
                rhs=[float(cmdp.start_state_probabilities[next_s_ind])]
            )
