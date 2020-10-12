import numpy as np

def graph_to_precision_matrix(adj_matrix,
                              pos_lims=(2, 3),
                              neg_lims=(-2, -3),
                              target_condition=100,
                              eps_bin=1e-2,
                              num_binary_search=100):

    """Find a precision matrix corresponding to a binary graph.

    Arguments
    --------
    adj_matrix (np.array): binary adjacency matrix
    pos_lims (tuple): positive limits to sample partial correlations from
    neg_lims (tuple): negative limits to sample partial correlations from
    target_condition (int): target condition number for added diagonal weight
    eps_bin (float): convergence condition for condition number search
    num_binary_search (int): max number of search iterations

    Returns
    -------
    theta (np.array): precision matrix (inverse covariance matrix) corresponding
                      to the input graph
    """

    assert len(pos_lims) == 2 and len(neg_lims) == 2, (
            'sampling limits should have length 2')

    n = adj_matrix.shape[1]
    pos_lims = sorted(pos_lims)
    neg_lims = sorted(neg_lims)

    # add diagonal to adjacency matrix
    theta = adj_matrix + np.eye(n)
    # get degree of each node
    degree_vec = np.sum(adj_matrix, axis=0)

    # get locations of (binary) edges in graph
    utri = np.triu(theta, k=1)
    nzind = (utri != 0)

    # replace binary edges with positive or negative edge weights at random,
    # sampled from the ranges given
    nnz = np.count_nonzero(nzind)
    rands = np.zeros(shape=(nnz,))
    boolind = np.random.choice([True, False], size=nnz)
    rands[boolind] = np.random.uniform(low=pos_lims[0], high=pos_lims[1],
                                       size=(np.count_nonzero(boolind),))
    rands[~boolind] = np.random.uniform(low=neg_lims[0], high=neg_lims[1],
                                        size=(np.count_nonzero(~boolind),))
    utri[nzind] = rands

    triu_ix = np.triu_indices(theta.shape[0], 1)
    tril_ix = np.tril_indices(theta.shape[0], -1)
    theta[triu_ix] = utri[triu_ix]
    theta[tril_ix] = utri[triu_ix]

    # make sure smallest eigenvalue of matrix isn't 0 (or close to 0)
    # since theta is symmetric, eigenvalues determine condition number
    evals = np.linalg.eig(theta)[0]
    min_eig, max_eig = evals.min(), evals.max()
    if min_eig < 1e-2:
        theta = theta + (abs(min_eig) * np.eye(n))

    # now find the smallest constant to add to the diagonal to satisfy
    # the desired condition number
    diag_constant = _bin_search_condition(theta, target_condition,
                                          num_binary_search, eps_bin)
    theta = theta + (diag_constant * np.eye(n))

    return theta

def _bin_search_condition(theta, target_cond, num_binary_search, eps_bin):
    """Perform a binary search to find the smallest diagonal weight that
    will bring the condition number of theta under target_cond.
    """

    n = theta.shape[0]
    curr_cond = np.linalg.cond(theta)

    if curr_cond < target_cond:
        curr_lb = -np.diag(theta).max()
        step_size = curr_lb + np.finfo(type(theta)).eps
        while curr_cond < target_cond:
            curr_cond = np.linalg.cond(theta + (step_size * np.eye(n)))
            step_size = step_size / 2
        curr_ub = step_size

    else:
        curr_lb = 0
        step_size = 0.1
        while curr_cond > target_cond:
            curr_cond = np.linalg.cond(theta + (step_size * np.eye(n)))
            step_size = step_size * 2
        curr_ub = step_size

    # find the smallest diagonal constant that satisfies the target
    # condition number (subject to the provided number of search
    # iterations)
    for i in range(1, num_binary_search+1):
        diag_const = (curr_ub + curr_lb) / 2
        curr_cond = np.linalg.cond(theta + (diag_const * np.eye(n)))
        if curr_cond < target_cond:
            curr_ub = diag_const
        else:
            curr_lb = diag_const
        if abs(curr_cond - target_cond) < eps_bin:
            break

    return diag_const


if __name__ == '__main__':
    adj_matrix = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])
    theta = graph_to_precision_matrix(adj_matrix)
    print(theta)
    print(np.linalg.inv(theta))
    print(np.linalg.cond(theta))
