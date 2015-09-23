from collections import defaultdict
import cython
import numpy as np
cimport numpy as np
from libc.math cimport log, exp, abs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.embedsignature(True)
cpdef max_sum(np.ndarray[np.double_t, ndim=2] S):
    """
    Max-sum algorithm (longest path) dynamic program on segmentation path graph 
    with score matrix ``S``.

    Input
    -----
    S : 2d-array
        segment score matrix (negative of energy matrix)

    Returns
    -------
    opt[i] : 1d-array
        optimal score of path from 0..i
    pred[i] : 1d-array
        first predecessor node on optimal path from 0..i

    """
    cdef int N = len(S) - 1
    cdef np.ndarray[np.double_t, ndim=1] opt = np.zeros(N+1, dtype=float)
    cdef np.ndarray[np.int_t, ndim=1] pred = np.zeros(N+1, dtype=int)

    cdef int i, k
    cdef double s
    opt[0] = 0.0
    for i in range(1, N+1):
        opt[i] = -np.inf
        pred[i] = i-1
        for k in range(0, i):
            s = opt[k] + S[k, i]
            if s > opt[i]:
                opt[i] = s
                pred[i] = k

    return opt, pred



# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.embedsignature(True)
# cpdef __max_sum(np.ndarray[np.double_t, ndim=2] S, np.ndarray[np.bool_t, ndim=1] edgemask):
#     """
#     Perform max-sum algorithm (longest path) dynamic program on segmentation 
#     path graph with score matrix S.

#     Input:
#         S  -  score matrix (symmetric 2D numpy array)

#     Returns:
#         opt[i]  - optimal score of path from 0..i
#         pred[i] - first predecessor node on optimal path from 0..i

#     """
#     cdef int N = len(S) - 1
#     cdef np.ndarray[np.double_t, ndim=1] opt = np.zeros(N+1, dtype=float)
#     cdef np.ndarray[np.int_t, ndim=1] pred = np.zeros(N+1, dtype=int)

#     cdef int i, k
#     cdef double s
#     opt[0] = 0.0
#     for i in range(1, N+1):
#         if edgemask[i]:
#             opt[i] = -np.inf
#             pred[i] = i-1
#             for k in range(0, i):
#                 if edgemask[k]:
#                     s = opt[k] + S[k, i]
#                     if s > opt[i]:
#                         opt[i] = s
#                         pred[i] = k
#         else:
#             opt[i] = np.nan
#             pred[i] = -1
#     return opt, pred


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.embedsignature(True)
cpdef max_sum__gapped(np.ndarray[np.double_t, ndim=2] S, 
                      double go, double ge, double gc):
    """
    Perform gapped max-sum algorithm (longest path) dynamic program on 
    segmentation path graph with score matrix ``S``.

    Input
    -----
    S : 2d-array
        score matrix (symmetric 2D numpy array)
    go : float
        gap opening score
    ge : float
        gap extension score
    gc : float
        gap closure score

    Returns
    -------
    opt : n x 2 array
        Optimal score of path from 0..i ending in domain (opt[i,0]) or
        gap (opt[i,1]) boundary.
    pred[i,:] : n x 2 array
        First predecessor node on optimal path from 0..i ending in domain
        (pred[i,0]) or gap (pred[i,1]) boundary.

    """
    cdef int N = len(S) - 1
    cdef np.ndarray[np.double_t, ndim=2] opt  = np.zeros((N+1,2), dtype=float)
    cdef np.ndarray[np.double_t, ndim=2] pred = np.zeros((N+1,2), dtype=float)
    
    cdef int i, k
    cdef double s
    opt[0, 0] = 0.0
    opt[0, 1] = go
    for i in range(1, N+1):
        # consider segment boundary predecessors
        #  - end of another segment
        #  - end of a gap
        for k in range(0, i):
            s = opt[k, 0] + S[k, i]
            if s > opt[i, 0]:
                opt[i, 0] = s
                pred[i, 0] = k
        s = opt[i-1, 0] + gc
        if s > opt[i, 0]:
            opt[i, 0] = s
            pred[i, 0] = i-1

        # consider gap boundary predecessors
        #  - gap opening
        #  - gap extension
        s = max(opt[i-1, 0] + go, opt[i-1, 1] + ge)
        opt[i, 1] = s
        pred[i, 1] = i-1
        
    return opt, pred


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.embedsignature(True)
cpdef get_path(
        np.ndarray[np.double_t, ndim=1] opt,
        np.ndarray[np.int_t, ndim=1] pred):
    """
    Backtrack over predecessor nodes to get the optimal path from max-sum.

    Input
    -----
    opt : 1d-array
        optimal score from max-sum
    pred : 1d-array
        predecessor list from max-sum

    Returns
    -------
    path: 1d-array
        optimal path of nodes from 0..N

    """
    cdef int N = len(opt) - 1
    cdef np.ndarray[np.int_t, ndim=1] path = np.zeros(N+1, dtype=int)
    cdef int i, j 
    j = 0
    i = path[j] = N
    j += 1
    while i > 0:
        i = path[j] = pred[i]
        j += 1

    return path[:j][::-1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.embedsignature(True)
cpdef get_path__gapped(
        np.ndarray[np.double_t, ndim=2] opt, 
        np.ndarray[np.int_t, ndim=2] pred):
    """
    Backtrack over predecessor nodes to get the optimal path from gapped 
    max-sum.

    Input
    -----
    opt : 1d-array
        optimal score from gapped max-sum
    pred : 1d-array
        predecessor list from gapped max-sum

    Returns
    -------
    path: 1d-array
        optimal path of nodes from 0..N

    """
    cdef int N = opt.shape[0] - 1
    cdef np.ndarray[np.int_t, ndim=1] path = np.zeros(N+1, dtype=int)
    cdef int i, j 
    j = 0
    
    if opt[-1, 0] > opt[-1, 1]:
        path[j] = N
    else:
        path[j] = -N
    i = N
    j += 1
    
    while i > 0:
        if opt[i, 0] > opt[i, 1]:
            path[j] = pred[i, 0]
            i = pred[i, 0]
        else:
            path[j] = -pred[i, 1]
            i = pred[i, 1]
        j += 1

    return path[:j][::-1]


@cython.embedsignature(True)
def optimal_segmentation(np.ndarray[np.double_t, ndim=2] S):
    """
    Find the optimal path on the segmentation path graph with score matrix
    ``S``.

    Input
    -----
    S : 2d-array
        segment score matrix (negative of energy matrix)

    Returns
    -------
    path : 1d-array
        optimal path of nodes from 0..N
    opt : 1d-array
        optimal score of path from 0..i

    """
    opt, optk = max_sum(S)
    path = get_path(opt, optk)
    return path, opt


# @cython.embedsignature(True)
# def consensus_segments(list segments, weights):
#     """
#     Returns consensus list of nonoverlapping segments.
#     Segments are 2-tuples given as half-open intervals [a,b).

#     """
#     occ = defaultdict(int)
#     for d, w in zip(segments, weights):
#         occ[d] += w

#     cdef int i, j, s_choose, s_ignore

#     # map each domain to its closest non-overlapping predecessor
#     cdef int M = len(segments)
#     cdef np.ndarray[np.int_t, ndim=1] prev = np.zeros(M, dtype=int)
#     for i in range(M-1, -1, -1):
#         d = segments[i]
#         j = i - 1
#         while j > -1:
#             if segments[j][1] <= d[0]: 
#                 prev[i] = j
#                 break
#             j -= 1

#     # weighted interval scheduling dynamic program
#     cdef np.ndarray[np.int_t, ndim=1] score = np.zeros(M, dtype=int)
#     for i in range(1, M):
#         d = segments[i]
#         s_choose = score[prev[i]] + occ[d]
#         s_ignore = score[i-1]
#         score[i] = max(s_choose, s_ignore)

#     cdef list consensus = []
#     j = M - 1
#     while j > 0:
#         if score[j] != score[j-1]:
#             consensus.append(segments[j])
#             j = prev[j]
#         else:
#             j -= 1

#     return consensus[::-1]


###
# Assuming start = 0, end = N-1 (index over borders 0..N):
# logZf[t] = ...log(
#   sum_{i=0}^{t-1}{exp(logZf[i] - beta*E[i,t])} 
# )
#
# logZf[0] = 0
# logZf[1] = log( exp(logZf[0]-beta*E[0,1]) ) = -beta*E[0,1]
# logZf[2] = log( exp(logZf[0]-beta*E[0,2]) + exp(logZf[1]-beta*E[1,2]) )
# ...
# logZf[N-1] = log( exp(logZf[0]-beta*E[0,N-1]) + exp(logZf[1]-beta*E[1,N-1]) + ... + exp(logZf[N-2]-beta*E[N-2,N-1]) )
# logZf[N]   = log( exp(logZf[0]-beta*E[0,N]) + exp(logZf[1]-beta*E[1,N]) + ... + exp(logZf[N-1]-beta*E[N-1,N]) )
###
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef np.ndarray[np.double_t, ndim=1] log_forward(
        np.ndarray[np.double_t, ndim=2] Eseg, 
        double beta,
        int start, 
        int end,
        int maxsize=-1):
    """
    Log forward subpartition functions.

    Input
    -----
    Eseg : 2d-array 
        segment energy matrix
    beta : float
        inverse temperature
    start, end: int
        first and last bin egdes to consider
    maxsize: int (experimental)
        maximum domain size to allow

    Returns
    -------
    Lf : 1d-array
        Log of sum of statistical weights of segmentation ensembles starting at
        ``start``.

    """
    cdef int N = len(Eseg) - 1 # number of nodes
    if start < 0 or end > N:
        raise IndexError("start or end out of range")
    if maxsize == -1:
        maxsize = N
    cdef int n = end - start
    cdef np.ndarray[np.double_t, ndim=1] Lfwd = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(n+1, dtype=float) 
    cdef int t, k
    cdef double a_max

    Lfwd[0] = 0.0
    for t in range(1, n+1):
        a_max = 0.0
        for k in range(max(t-maxsize, 0), t):
            a[k] = Lfwd[k] - beta*Eseg[start+k, start+t]
            if a[k] > a_max:
                a_max = a[k]

        Lfwd[t] = a_max + log(np.exp(a[:t] - a_max).sum())
    
    return Lfwd


###
# Assuming start = 0, end = N-1:
# logZb[N-t] = ...log(
#   sum_{i=0}^{t-1}{exp(logWb[N-i] - beta*E[N-t, N-i])} 
# )
#
# logZb[N]   = 0
# logZb[N-1] = log( exp(logZb[N]-beta*E[N-1,N]) )
# logZb[N-2] = log( exp(logZb[N]-beta*E[N-2,N]) + exp(logZb[N-1]-beta*E[N-2,N-1]) )
# logZb[N-3] = log( exp(logZb[N]-beta*E[N-3,N]) + exp(logZb[N-1]-beta*E[N-3,N-1]) + exp(logZb[N-2]-beta*E[N-3,N-2]))
# ...
# logZb[0] = log( exp(logZb[N]-beta*E[0,N]) + exp(logZb[N-1]-beta*E[0,N-1]) ... + exp(logZb[1]-beta*E[0,1]) )
###
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef np.ndarray[np.double_t, ndim=1] log_backward(
        np.ndarray[np.double_t, ndim=2] Eseg, 
        double beta, 
        int start, 
        int end,
        int maxsize=-1):
    """
    Log backward subpartition functions.

    Input
    -----
    Eseg : 2d-array 
        segment energy matrix
    beta : float
        inverse temperature
    start, end: int
        first and last bin egdes to consider
    maxsize: int (experimental)
        maximum domain size to allow

    Returns
    -------
    Lb : 1d-array
        Log of sum of statistical weights of segmentation ensembles ending at
        ``end``.

    """
    cdef int N = len(Eseg) - 1 # n nodes
    if start < 0 or end > N:
        raise IndexError("start or end out of range")
    if maxsize == -1:
        maxsize = N
    cdef int n = end - start
    cdef np.ndarray[np.double_t, ndim=1] Lbwd = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(n+1, dtype=float)
    cdef int k, t
    cdef double a_max
    
    Lbwd[n] = 0.0
    for t in range(1, n+1):
        a_max = 0.0
        for k in range(max(t-maxsize, 0), t):
            a[k] = Lbwd[n-k] - beta*Eseg[end-t, end-k]
            if a[k] > a_max:
                a_max = a[k]

        Lbwd[n-t] = a_max + log(np.exp(a[:t] - a_max).sum())

    return Lbwd


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def log_forward__gapped(
        np.ndarray[np.double_t, ndim=2] Eseg, 
        double beta, double go, double ge, double gc, int maxsize=-1):
    cdef int N = len(Eseg) - 1 # number of nodes
    if maxsize == -1:
        maxsize = N

    cdef np.ndarray[np.double_t, ndim=2] Lfwd = np.zeros((N+1, 2), dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(N+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] b = np.zeros(2, dtype=float)

    cdef int t, k
    cdef double a_max, b_max, c
    Lfwd[0, 0] = 0.0
    Lfwd[0, 1] = beta*go
    for t in range(1, N+1):
        a_max = 0.0
        # segment to segment
        for k in range(max(t - maxsize, 0), t):
            a[k] = Lfwd[k, 0] - beta*Eseg[k, t]
            if np.abs(a[k]) > a_max:
                a_max = a[k]
        # gap close
        c = Lfwd[t-1, 1] + beta*gc
        if c > a_max:
            a_max = c
        # gap open, gap extend
        b[0] = Lfwd[t-1, 0] + beta*go
        b[1] = Lfwd[t-1, 1] + beta*ge
        b_max = b.max()

        Lfwd[t, 0] = a_max + log(np.exp(a[:t] - a_max).sum() + np.exp(c - a_max))
        Lfwd[t, 1] = b_max + log(np.exp(b - b_max).sum())
                    
    return Lfwd


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def log_backward__gapped(
        np.ndarray[np.double_t, ndim=2] Eseg, 
        double beta, double go, double ge, double gc, int maxsize=-1):
    cdef int N = len(Eseg) - 1
    if maxsize == -1:
        maxsize = N

    cdef np.ndarray[np.double_t, ndim=2] Lbwd = np.zeros((N+1, 2), dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(N+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] b = np.zeros(2, dtype=float)
    
    cdef int t, k
    cdef double a_max, b_max, c
    Lbwd[N, 0] = 0.0
    Lbwd[N, 1] = 0.0
    for t in range(1, N+1):
        a_max = 0.0
        # segment to segment
        for k in range(max(t - maxsize, 0), t):
            a[k] = Lbwd[N-k, 0] - beta*Eseg[N-t, N-k]
            if np.abs(a[k]) > a_max:
                a_max = a[k]
        # gap open
        c = Lbwd[N-(t-1), 1] + beta*go
        if c > a_max:
            a_max = c
        # gap close, gap extend
        b[0] = Lbwd[N-(t-1), 0] + beta*gc
        b[1] = Lbwd[N-(t-1), 1] + beta*ge
        b_max = b.max()

        Lbwd[N-t, 0] = a_max + log(np.exp(a[:t] - a_max).sum() + np.exp(c - a_max))
        Lbwd[N-t, 1] = b_max + log(np.exp(b - b_max).sum())

    return Lbwd


###
# The following is equivalent to doing forward on each row, but does it one
# column at a time
### 
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef np.ndarray[np.double_t, ndim=2] log_zmatrix(
    np.ndarray[np.double_t, ndim=2] Eseg, 
    double beta):
    """
    Log subsystem partition function matrix.
    
    Input
    -----
    Eseg : 2d-array
        segment energy matrix
    beta : float
        inverse temperature

    Returns
    -------
    Lz : 2d-array
        log(Z) where Z[i,j] is the sum of statistical weights corresponding to 
        the set of subsegmentations between bin edges ``i`` and ``j``.

    """
    cdef int N = len(Eseg) - 1 # n nodes
    cdef np.ndarray[np.double_t, ndim=2] Lz = np.zeros((N+1,N+1), dtype=float)
    cdef np.ndarray[np.double_t, ndim=2] a = np.zeros((N+1,N+1), dtype=float)
    cdef int i, j, k
    cdef double a_max

    Lz[0, 0] = 0.0
    for j in range(1, N+1):
        for i in range(0, j): #range(j-1,-1,-1):
            for k in range(i, j):
                a[i, k] = Lz[i, k] - beta*Eseg[k, j]
            a_max = a[i, i:j].max()
            Lz[i, j] = Lz[j, i] = a_max + log( np.exp(a[i, i:j] - a_max).sum() )

    return Lz


@cython.embedsignature(True)
def log_boundary_marginal(
        np.ndarray[np.double_t, ndim=2] Eseg, 
        double beta):
    """
    Log of marginal domain boundary statistical weight sums.

    Input
    -----
    Eseg : 2d-array
        segment energy matrix
    beta : float
        inverse temperature

    Returns
    -------
    Lz[i] : 1d-array
        Log of sum of statistical weights of all segmentations having ``i`` as
        a domain boundary.

    """
    cdef int N = len(Eseg) - 1
    cdef np.ndarray[np.double_t, ndim=1] Lf = log_forward(Eseg, beta, 0, N)
    cdef np.ndarray[np.double_t, ndim=1] Lb = log_backward(Eseg, beta, 0, N)
    return Lf + Lb


@cython.embedsignature(True)
def log_segment_marginal(
        np.ndarray[np.double_t, ndim=2] Eseg,
        double beta):
    """
    Log of marginal domain statistical weight sums.

    Input
    -----
    Eseg : 2d-array
        segment energy matrix
    beta : float
        inverse temperature

    Returns
    -------
    Ls[a,b] : 2d-array 
        Log of sum of statistical weights of all segmentations containing the
        domain [a,b).

    Notes
    -----
    The main diagonal is filled with the boundary marginals.
    Interpretation: trivial domains [a,a) are identical to single boundary 
    occurrences.

    """
    cdef int N = len(Eseg) - 1
    cdef np.ndarray[np.double_t, ndim=1] Lf = log_forward(Eseg, beta, 0, N)
    cdef np.ndarray[np.double_t, ndim=1] Lb = log_backward(Eseg, beta, 0, N)
    return log_segment_marginal__from_forward_backward(Eseg, beta, Lf, Lb)


@cython.embedsignature(True)
def log_segment_marginal__from_forward_backward(
        np.ndarray[np.double_t, ndim=2] Eseg, 
        double beta,
        np.ndarray[np.double_t, ndim=1] Lf,
        np.ndarray[np.double_t, ndim=1] Lb):
    """
    Input
    -----
    Eseg : 2d-array
        segment energy matrix
    beta : float
        inverse temperature
    Lf: 1d-array
        log forward statistical weights
    Lb: 1d-array
        log backward statistical weights

    Returns
    -------
    Ls[a,b] : 2d-array 
        Log of sum of statistical weights of all segmentations containing the
        domain [a,b).

    Notes
    -----
    The main diagonal is filled with the boundary marginals.
    Interpretation: trivial domains [a,a) are identical to single boundary 
    occurrences.

    """
    cdef int N = len(Eseg) - 1
    cdef np.ndarray[np.double_t, ndim=2] Ls = np.zeros((N+1, N+1))
    cdef int i, j
    for i in range(N+1):
        for j in range(i, N+1):
            Ls[i,j] = Ls[j,i] = Lf[i] - beta*Eseg[i, j] + Lb[j]
    return Ls


@cython.embedsignature(True)
def log_boundary_cooccur_marginal(
        np.ndarray[np.double_t, ndim=2] Eseg, 
        double beta):
    """
    Log of statistical weight sums for pairs of bin edges simultaneously
    occurring as domain boundaries.

    Input
    -----
    Eseg : 2d-array
        segment energy matrix
    beta : float
        inverse temperature

    Returns
    -------
    Lbb[i,j] : 2d-array
        Log of sum of statistical weights of all segmentations in which both
        ``i`` and ``j`` occur as domain boundaries.

    Notes
    -----
    The main diagonal is filled with the boundary marginals.
    Interpretation: Every boundary always co-occurs with itself.

    """
    cdef np.ndarray[np.double_t, ndim=2] Lz = log_zmatrix(Eseg, beta)
    return log_boundary_cooccur_marginal__from_zmatrix(Lz)


@cython.embedsignature(True)
cpdef log_boundary_cooccur_marginal__from_zmatrix(
        np.ndarray[np.double_t, ndim=2] Lz):
    """
    Input
    -----
    Lz: 2d-array
        Log of the subsystem statistical weight matrix.

    Returns
    -------
    Lbb[i,j] : 2d-array
        Log of sum of statistical weights of all segmentations in which both
        ``i`` and ``j`` occur as domain boundaries.

    Notes
    -----
    The main diagonal is filled with the boundary marginals.
    Interpretation: Any boundary always co-occurs with itself.

    """
    cdef int N = len(Lz) - 1  
    cdef np.ndarray[np.double_t, ndim=2] Lbb = np.zeros((N+1,N+1))
    cdef int i, j
    for i in range(N+1):
        for j in range(i, N+1):
            Lbb[i,j] = Lbb[j,i] = Lz[0, i] + Lz[i, j] + Lz[j, N]
    return Lbb


@cython.embedsignature(True)
def log_segment_cooccur_marginal(
        np.ndarray[np.double_t, ndim=2] Eseg, 
        double beta):
    """
    Log of statistical weight sums for pairs of bins co-occurring within the
    same domain.

    Input
    -----
    Eseg : 2d-array
        segment energy matrix
    beta : float
        inverse temperature

    Returns
    -------
    Lbb[p,q] : 2d-array
        Log of sum of statistical weights of all segmentations in which ``p`` 
        and ``q`` occur within the same domain.

    Notes
    -----
    The main diagonal is filled with the partition function Z. 
    Interpretation: Each bin is always in the same domain as itself.

    """
    cdef np.ndarray[np.double_t, ndim=2] Ls = log_segment_marginal(Eseg, beta)
    return log_segment_cooccur_marginal__from_segment_marginal(Ls)

    # cdef double Ls_max = Ls.max()

    # cdef np.ndarray[np.double_t, ndim=2] Zseg = np.exp(Ls - Ls_max)
    # cdef double Z = Zseg[0, 0]
    
    # cdef np.ndarray[np.double_t, ndim=2] Zcos = np.zeros((N,N), dtype=float)
    # cdef double s
    # cdef int i, j
    # # j: N-1
    # Zcos[0, N-1] = Zcos[N-1, 0] = Zseg[0, N]
    # for i in range(1, N-1):
    #     Zcos[i, N-1] = Zcos[N-1, i] = Zcos[i-1, N-1] + Zseg[i, N]
    # Zcos[N-1, N-1] = Z
    # # j: N-2 to 1
    # for j in range(N-2, 0, -1):
    #     s = Zseg[0, j+1]
    #     Zcos[0, j] = Zcos[j, 0] = Zcos[0, j+1] + s
    #     for i in range(1, j):
    #         s = s + Zseg[i, j+1]
    #         Zcos[i, j] = Zcos[j, i] = Zcos[i, j+1] + s
    #     Zcos[j, j] = Z
    # # j: 0
    # Zcos[0, 0] = Z
    
    # return Ls_max + np.log(Zcos)


@cython.embedsignature(True)
cpdef log_segment_cooccur_marginal__from_segment_marginal(
        np.ndarray[np.double_t, ndim=2] Ls):
    """
    Input
    -----
    Ls: 2d-array
        log of the domain marginal matrix

    Returns
    -------
    Lbb[p,q] : 2d-array
        Sum of statistical weights of all segmentations in which ``p`` and
        ``q`` occur within the same domain.

    Notes
    -----
    The main diagonal is filled with the partition function Z. 
    Interpretation: Each bin is always in the same domain as itself.

    """
    cdef int N = len(Ls) - 1
    cdef double Ls_max = Ls.max()

    cdef np.ndarray[np.double_t, ndim=2] Zs = np.exp(Ls - Ls_max)
    cdef double Z = Zs[0, 0]
    
    cdef np.ndarray[np.double_t, ndim=2] Zss = np.zeros((N, N), dtype=float)
    cdef double s
    cdef int i, j
    # j: N-1
    Zss[0, N-1] = Zss[N-1, 0] = Zs[0, N]
    for i in range(1, N-1):
        Zss[i, N-1] = Zss[N-1, i] = Zss[i-1, N-1] + Zs[i, N]
    Zss[N-1, N-1] = Z
    # j: N-2 to 1
    for j in range(N-2, 0, -1):
        s = Zs[0, j+1]
        Zss[0, j] = Zss[j, 0] = Zss[0, j+1] + s
        for i in range(1, j):
            s = s + Zs[i, j+1]
            Zss[i, j] = Zss[j, i] = Zss[i, j+1] + s
        Zss[j, j] = Z
    # j: 0
    Zss[0, 0] = Z
    
    return Ls_max + np.log(Zss)


#@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.embedsignature(True)
cpdef log_project_insul(np.ndarray[np.double_t, ndim=1] pi):
    """
    Project a 1-D insulation profile ``pi`` into a heatmap ``P`` according to 
    the multiplicative model: P[i,j] = prod_{i<=k<-j}(1 - pi[k]).

    Input
    -----
    pi : array
        Insulation score for each bin or bin edge, between 0 and 1.

    Returns
    -------
    L = log(P)

    """
    # n bin edges or bins
    cdef n = len(pi) 
    cdef np.ndarray[np.double_t, ndim=2] L = np.zeros((n, n), dtype=float)
    cdef int i, diag

    lpi = np.log(pi)

    # base case: 0th diag
    # XXX - leave out main diag for consistency with other matrices?
    for i in range(0, n):
        L[i, i] = lpi[i]

    # base case: 1st diag
    for i in range(0, n-1):
        L[i, i+1] = L[i+1, i] = lpi[i] + lpi[i+1]

    for diag in range(2, n):
        for i in range(0, n-diag):
            L[i, i+diag] \
                = L[i+diag, i] \
                = L[i, i+diag-1] + L[i+1, i+diag] - L[i+1, i+diag-1]
    return L
