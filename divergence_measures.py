"""
Created on Mon Jul 20 14:32:59 2026

@author: martin
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

# ============================================================
# Admissible penalty functions
# ============================================================

def tanh_penalty(x, a=1.0, b=1.0):
    """
    f(x) = b*tanh(a*x/2).

    For a,b >= 0, this function is nonnegative, increasing,
    concave and subadditive. The condition a*b <= 2 guarantees
    f(x) <= x.
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("a and b must be finite.")

    if a < 0 or b < 0 or a * b > 2:
        raise ValueError(
            "Require a >= 0, b >= 0, and a*b <= 2."
        )

    x = np.asarray(x, dtype=float)
    return b * np.tanh(0.5 * a * x)

def arctan_penalty(x, a=1.0, b=1.0):
    """ 
    f(x)=b*arctan(a*x)
    
    The condition a*b<=1 guarantees it satisfies the requied conditions. 
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("a and b must be finite.")
    if a < 0 or b < 0 or a * b > 1:
        raise ValueError(
            "Require a >= 0, b >= 0, and a*b <= 1."
        )
    return b * np.arctan(a * x)

def linear_penalty(x, slope=0.5):
    """f(x) = slope*x, with slope in [0,1]."""
    if not np.isfinite(slope) or not 0 <= slope <= 1:
        raise ValueError("slope must lie in [0, 1].")
    return slope * x
    
############################

def _compute_cost_matrix(dgm1, dgm2, func = arctan_penalty):
    """
    Construct the augmented cost matrix for the divergence from dgm1 to dgm2.
    
    Parameters
    ----------
    dgm1 : numpy.ndarray of shape (n, 2)
        Source persistence diagram. Each row contains birth and death
        coordinates (birth, death), with birth <= death.
    dgm2 : numpy.ndarray of shape (m, 2)
        Target persistence diagram, using the same coordinate convention.
    func : callable, optional
        Penalty function applied to the distances of unmatched points of
        dgm2 to the diagonal. Must accept a NumPy array and return an array
        of the same shape or a broadcastable scalar.
        Defaults to arctan_penalty.
    
    Returns
    -------
    c_mat : numpy.ndarray of shape (n + m, n + m)
        Augmented assignment cost matrix. Matching a point of dgm1 to a
        point of dgm2 costs their Chebyshev distance. An unmatched point
        of dgm1 costs half its persistence; an unmatched point of dgm2
        costs func applied to half its persistence. Matches between
        diagonal copies cost zero. Forbidden assignments have cost np.inf.

    """
    dist_11=cdist(dgm1, dgm2, metric='chebyshev')
    dist_12 = np.full((len(dgm1), len(dgm1)), np.inf, dtype=float,)
    np.fill_diagonal(dist_12,0.5 * (dgm1[:, 1] - dgm1[:, 0]),)
    dist_21 = np.full((len(dgm2), len(dgm2)), np.inf, dtype=float,)
    np.fill_diagonal(dist_21, func(0.5 * (dgm2[:, 1] - dgm2[:, 0])),)
    dist_22=np.zeros((len(dgm2), len(dgm1)))
    c_mat=np.concatenate((np.concatenate((dist_11, dist_12), axis=1),np.concatenate((dist_21, dist_22), axis=1)), axis=0)
    return c_mat

def _has_perfect_matching(C, t):
    """
    Check whether a square cost matrix admits a perfect matching
    using only assignments with cost at most t.

    Parameters
    ----------
    C : numpy.ndarray of shape (n, n)
        Assignment cost matrix. Entries equal to np.inf represent
        forbidden assignments.
    t : float
        Finite cost threshold.

    Returns
    -------
    bool
        True if every row and every column can be matched exactly
        once using assignments whose costs do not exceed t.
    """
    # An edge exists precisely when its cost is within the threshold.
    graph = csr_matrix(C <= t)

    # match[j] is the row matched to column j, or -1 if unmatched.
    match = maximum_bipartite_matching(graph, perm_type="row")

    # Since C is square, matching all columns also matches all rows.
    return bool(np.all(match != -1))

def _bottleneck_assignment_cost(C):
    """
    Compute the minimum bottleneck cost of a perfect assignment.
    
    The bottleneck cost of an assignment is its largest selected cost.
    This function finds the smallest threshold for which a perfect
    matching exists using only assignments at or below that threshold.
    
    Parameters
    ----------
    C : array_like of shape (n, n)
        Nonempty square cost matrix, where C[i, j] is the cost of
        matching row i to column j. Entries may be finite or np.inf,
        with np.inf representing forbidden assignments.
        The matrix must not contain NaN values.
    
    Returns
    -------
    float
        Minimum possible largest cost among assignments that match
        every row and every column exactly once. Returns np.inf
        if no perfect assignment with finite cost exists.
    
    Raises
    ------
    ValueError
        If C is not a two-dimensional square matrix.
    
    Notes
    -----
    The function performs a binary search over the distinct entries
    of C. At each threshold, feasibility is checked using maximum
    bipartite matching.
    
    The current implementation does not support an empty (0, 0)
       matrix.
    """
    C = np.asarray(C, dtype=float)
    n, m = C.shape
    if n != m:
        raise ValueError("C must be square for perfect matching.")

    # Candidate thresholds
    vals = np.unique(C)
    lo, hi = 0, len(vals) - 1
    best = vals[-1]

    while lo <= hi:
        mid = (lo + hi) // 2
        t = vals[mid]
        if _has_perfect_matching(C, t):
            best = t
            hi = mid - 1
        else:
            lo = mid + 1
    return float(best)

def _p_norm_assignment_cost(C, p):
    """
    Compute the minimum p-norm cost of a perfect assignment.

    Minimize the sum of the p-th powers of the selected costs over
    all assignments that match every row and every column exactly
    once, then take the p-th root.

    Parameters
    ----------
    C : array_like of shape (n, n)
        Square matrix of nonnegative assignment costs. Entry C[i, j]
        is the cost of matching row i to column j. Entries equal to
        np.inf represent forbidden assignments. Must not contain NaN.
    p : float or int
        Finite exponent satisfying p >= 1.

    Returns
    -------
    float
        Minimum p-norm assignment cost. Returns 0.0 for an empty
        matrix of shape (0, 0).

    Raises
    ------
    ValueError
        If the powered cost matrix contains invalid numeric entries
        or no perfect assignment with finite cost exists.
    """
    C = np.asarray(C, dtype=float)
    
    finite_edges = np.isfinite(C)

    # If every entry is finite, a finite-cost perfect assignment exists.
    if not np.all(finite_edges):
        graph = csr_matrix(finite_edges)
        match = maximum_bipartite_matching(graph, perm_type="row")
    # if there is no matching with finite cost, the assignment cost is infinite
        if np.any(match == -1):
            return np.inf

    # Minimizing the p-norm is equivalent to minimizing this sum.
    C_p = np.power(C, p)

    # Find a minimum-cost assignment using the powered costs.
    row_ind, col_ind = linear_sum_assignment(C_p)

    # Sum the selected powered costs and take the p-th root.
    sum_of_powers = C_p[row_ind, col_ind].sum()
    p_norm_cost = np.power(sum_of_powers, 1.0 / p)

    return float(p_norm_cost)

def bottleneck_divergence(dgm1, dgm2, func = arctan_penalty):
    """
    Compute the directed bottleneck divergence from dgm1 to dgm2.
    
    Parameters
    ----------
    dgm1 : numpy.ndarray of shape (n, 2)
        Source persistence diagram. Each row contains finite births and 
        finite or infinite deaths, provided the penalty evaluates correctly at 
        infinity, with birth <= death.
    dgm2 : numpy.ndarray of shape (m, 2)
        Target persistence diagram, using the same coordinate convention.
    func : callable, optional
        Penalty function applied to the distances of unmatched points
        of dgm2 to the diagonal. Must accept a NumPy array and return
        an array of the same shape or a broadcastable scalar.
        For the intended divergence, the function must be nonnegative,
        subadditive, and satisfy func(x) <= x for x >= 0.
        These properties are not checked here.
        Defaults to arctan_penalty.
    
    Returns
    -------
    divergence : float
        Minimum possible largest cost among all partial matchings
        between the diagrams. Matched pairs cost their Chebyshev
        distance. Unmatched source points cost half their persistence;
        unmatched target points cost func applied to half their
        persistence.
    """
    c_mat=_compute_cost_matrix(dgm1, dgm2, func)
    divergence = _bottleneck_assignment_cost(c_mat)
    return divergence


def wasserstein_divergence(dgm1, dgm2, func, p=1):
    """
    Compute the directed p-Wasserstein divergence from dgm1 to dgm2.

    Parameters
    ----------
    dgm1 : numpy.ndarray of shape (n, 2)
        Source persistence diagram. Each row contains finite birth
        and death coordinates (birth, death), with birth <= death.
    dgm2 : numpy.ndarray of shape (m, 2)
        Target persistence diagram, using the same coordinate convention.
    func : callable
        Penalty function applied to the distances of unmatched points
        of dgm2 to the diagonal. Must accept a NumPy array and return
        an array of the same shape or a broadcastable scalar.
        For the intended divergence, it must be nonnegative,
        subadditive, and satisfy func(x) <= x for x >= 0.
    p : float or int, optional
        Finite exponent satisfying p >= 1. Defaults to 1.

    Returns
    -------
    divergence : float
        Minimum p-norm of the costs over all partial matchings.
        Matched pairs cost their Chebyshev distance. Unmatched source
        points cost half their persistence; unmatched target points
        cost func applied to half their persistence.

    Raises
    ------
    ValueError
        If the powered assignment matrix contains invalid numeric
        entries or no perfect assignment with finite cost exists.
    """
    if p == np.inf:
        return bottleneck_divergence(dgm1, dgm2, func=func)

    if not np.isfinite(p) or p < 1:
        raise ValueError(
            "p must be at least 1; use np.inf for bottleneck."
        )
    c_mat = _compute_cost_matrix(dgm1, dgm2, func)
    divergence = _p_norm_assignment_cost(c_mat, p)
    return divergence
