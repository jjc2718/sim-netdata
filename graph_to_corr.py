import numpy as np

def graph_to_pcorr(adj_matrix,
                   pos_lims=None,
                   neg_lims=(-1, 0)):
    """Convert a binary graph to a (partial) correlation matrix.

    This is done by sampling from neg_lims (or pos_lims if provided), and
    adding ones to the diagonal.

    Arguments
    --------
    adj_matrix (np.array): binary adjacency matrix
    pos_lims (tuple): positive limits to sample elements of precision matrix
                      from, if None sample only from neg_lims
    neg_lims (tuple): negative limits to sample elements of precision matrix
                      from. Note that negative precision matrix elements
                      correspond to positive correlations.

    Returns
    -------
    theta (np.array): matrix of correlations/partial correlations corresponding
                      to the input graph
    """
    if pos_lims is None:
        assert len(neg_lims) == 2, (
                'sampling limits should have length 2')
        neg_lims = sorted(neg_lims)
    elif neg_lims is None:
        assert len(pos_lims) == 2, (
                'sampling limits should have length 2')
        pos_lims = sorted(pos_lims)
    else:
        assert len(pos_lims) == 2 and len(neg_lims) == 2, (
                'sampling limits should have length 2')
        pos_lims = sorted(pos_lims)
        neg_lims = sorted(neg_lims)

    n = adj_matrix.shape[0]

    # get degree of each node
    degree_vec = np.sum(adj_matrix, axis=0)

    # get locations of (binary) edges in graph
    utri = np.triu(adj_matrix, k=1).astype(float)
    nzind = (utri != 0)

    # replace binary edges with positive or negative edge weights at random,
    # sampled from the ranges given
    nnz = np.count_nonzero(nzind)
    if pos_lims is None:
        rands = np.random.uniform(low=neg_lims[0], high=neg_lims[1],
                                  size=(nnz,))
    elif neg_lims is None:
        rands = np.random.uniform(low=pos_lims[0], high=pos_lims[1],
                                  size=(nnz,))
    else:
        rands = np.zeros(shape=(nnz,))
        boolind = np.random.choice([True, False], size=nnz)
        rands[boolind] = np.random.uniform(low=pos_lims[0], high=pos_lims[1],
                                           size=(np.count_nonzero(boolind),))
        rands[~boolind] = np.random.uniform(low=neg_lims[0], high=neg_lims[1],
                                            size=(np.count_nonzero(~boolind),))
    utri[nzind] = rands

    # add 1 to diagonal and copy upper triangle to lower triangle
    theta = np.eye(n) + _copy_triu_to_tril(utri)

    return theta


def pcorr_to_prec(theta,
                  target_condition=50,
                  eps_bin=1e-2,
                  num_binary_search=100):
    """Convert a matrix of correlations to a valid precision matrix.

    TODO: explain condition number minimization

    Arguments
    --------
    theta (np.array): square matrix containing partial correlations between
                      elements
    target_condition (int): target condition number for added diagonal weight
    eps_bin (float): convergence condition for condition number search
    num_binary_search (int): max number of search iterations

    Returns
    -------
    theta (np.array): precision matrix (inverse covariance matrix) corresponding
                      to the input
    """
    # make sure smallest eigenvalue of matrix isn't 0 (or close to 0)
    # since theta is symmetric, eigenvalues determine condition number
    n = theta.shape[0]
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


def prec_to_cov(theta, method='inv'):
    """Convert a precision matrix to a covariance matrix.

    Arguments
    ---------
    theta (np.array): precision matrix
    method (str): method to use to invert (or pseudoinvert) theta. One of
                  'inv', 'pinv', 'chol'.

    Returns
    -------
    covariance matrix constructed by inverting (or pseudoinverting, etc)
    the input matrix
    """
    if method == 'inv':
        # just take the inverse, assumes theta is PSD
        return np.linalg.inv(theta)
    elif method == 'pinv':
        # take pseudoinverse, doesn't require theta to be PSD
        return np.linalg.pinv(theta)
    elif method == 'chol':
        # invert theta via Cholesky decomposition/backward substitution
        # assumes theta is symmetric and positive definite
        from scipy.linalg import cho_factor, cho_solve
        I_p = np.eye(theta.shape[0])
        return cho_solve(cho_factor(theta, lower=True), I_p)


def _bin_search_condition(theta, target_cond, num_binary_search, eps_bin):
    """Perform a binary search to find the smallest diagonal weight that
    will make the condition number of theta less than target_cond.
    """
    n = theta.shape[0]
    curr_cond = np.linalg.cond(theta)

    if curr_cond < target_cond:
        curr_lb = -np.diag(theta).max()
        step_size = curr_lb + np.finfo(theta.dtype).eps
        while curr_cond < target_cond:
            last_cond = curr_cond
            curr_cond = np.linalg.cond(theta + (step_size * np.eye(n)))
            step_size = step_size / 2
            if last_cond > curr_cond:
                # if repeated updates are decreasing the condition number,
                # we can stop modifying the diagonal (since the original
                # matrix has a low condition number already)
                curr_cond = last_cond
                break
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


def _copy_triu_to_tril(arr):
    """Copy upper triangle of array to lower triangle, leaving diagonal.

    Note this overwrites the lower triangle of the existing array.
    """
    return arr + arr.T - np.diag(np.diag(arr))

