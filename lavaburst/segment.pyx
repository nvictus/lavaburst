import cython
import numpy as np
cimport numpy as np
from libc.math cimport log, exp


# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# cpdef sums_by_segment(np.ndarray[np.double_t, ndim=2] A):
#     """
#     S = sums_by_segment(A)
#     Sum the edge weights in every segment of the graph represented by A.

#     Input:
#         A : N x N (double)
#         A symmetric square matrix. A[i,j] is the weight of the edge connecting
#         nodes i and j. If there are nonzero self-weights in the graph, the 
#         values on the diagonal A[i,i] are assumed to be twice the weight.

#     Output:
#         S : N+1 x N+1 (double)
#         A symmetric square matrix of summed weights of all contiguous segments 
#         of nodes. S[a,b] is the sum of weights between all pairs of nodes in the 
#         half-open range [a..b-1] (excluding b).

#     """
#     cdef int N = len(A)
#     cdef np.ndarray[np.double_t, ndim=2] S = np.zeros((N+1,N+1), dtype=float)

#     cdef int i
#     for i in range(N):
#         S[i,i+1] = S[i+1,i] = A[i,i]/2.0

#     cdef int start, end
#     cdef np.ndarray[np.double_t, ndim=1] cumsum
#     for end in range(1, N):
#         cumsum = np.zeros(end+1, dtype=float)
#         cumsum[end] = A[end,end]

#         for start in range(end-1, -1, -1):
#             cumsum[start] = cumsum[start+1] + A[start,end]
#             S[start,end+1] = S[end+1,start] = S[start, end] + cumsum[start]

#     return S


# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# cpdef normalized_sums_by_segment(np.ndarray[np.double_t, ndim=2] A):
#     """
#     S = normalized_sums_by_segment(A)
#     Aggregate the edge weights of every segment of the graph represented by A 
#     and normalize by the total weight of the graph. If there are nonzero 
#     self-weights in the graph, the values on the diagonal A[i,i] are assumed to 
#     be twice the edge weight.

#     Input:
#         A : N x N (double)

#     Output:
#         S : N+1 x N+1 (double)

#     """
#     cdef np.ndarray[np.double_t, ndim=1] deg = A.sum(axis=0)
#     cdef double m = deg.sum()/2.0
#     cdef np.ndarray[np.double_t, ndim=2] S = sums_by_segment(A)
#     S /= m
#     return S


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef max_sum(np.ndarray[np.double_t, ndim=2] score):
    cdef int N = len(score) - 1
    cdef np.ndarray[np.double_t, ndim=1] opt = np.zeros(N+1, dtype=float)
    cdef np.ndarray[np.int_t, ndim=1] optk = np.zeros(N+1, dtype=int)

    cdef int i, k
    cdef double s
    opt[0] = 0.0
    for i in range(1, N+1):
        opt[i] = -np.inf
        optk[i] = i-1
        for k in range(0, i):
            s = opt[k] + score[k, i]
            if s > opt[i]:
                opt[i] = s
                optk[i] = k

    return opt, optk


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef get_starts(np.ndarray[np.double_t, ndim=1] opt, np.ndarray[np.int_t, ndim=1] optk):
    cdef int N = len(opt) - 1
    cdef np.ndarray[np.int_t, ndim=1] starts = np.zeros(N, dtype=int)
    cdef int j = 0
    i = N
    while i > 0:
        i = starts[j] = optk[i]
        j += 1

    return starts[:j][::-1]


def optimal_segmentation(np.ndarray[np.double_t, ndim=2] score):
    opt, optk = max_sum(score)
    starts = get_starts(opt, optk)
    return starts, optk


def consensus_segmentation(list domains, occ):
    # Returns consensus list of domains
    # domains are 2-tuples given as half-open intervals [a,b)
    cdef int i, j, s_choose, s_ignore
    cdef tuple d

    # map each domain to its closest non-overlapping predecessor
    cdef int M = len(domains)
    cdef np.ndarray[np.int_t, ndim=1] prev = np.zeros(M, dtype=int)
    for i in range(M-1, -1, -1):
        d = domains[i]
        j = i - 1
        while j > -1:
            if domains[j][1] <= d[0]: 
                prev[i] = j
                break
            j -= 1

    # weighted interval scheduling dynamic program
    cdef np.ndarray[np.int_t, ndim=1] score = np.zeros(M, dtype=int)
    for i in range(1, M):
        d = domains[i]
        s_choose = score[prev[i]] + occ[d]
        s_ignore = score[i-1]
        score[i] = max(s_choose, s_ignore)

    cdef list consensus = []
    j = M - 1
    while j > 0:
        if score[j] != score[j-1]:
            consensus.append(domains[j])
            j = prev[j]
        else:
            j -= 1

    return consensus[::-1]


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
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.double_t, ndim=1] log_forward(
        np.ndarray[np.double_t, ndim=2] Ecomm, 
        double beta, 
        int start, 
        int end):
    cdef int N = len(Ecomm) - 1 # number of nodes
    if start < 0 or end > N:
        raise IndexError("start or end out of range")
    cdef int n = end - start
    cdef np.ndarray[np.double_t, ndim=1] Lfwd = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(n+1, dtype=float) 
    cdef int t, k
    cdef double a_max

    Lfwd[0] = 0.0
    for t in range(1, n+1):
        a_max = 0.0
        for k in range(0, t):
            a[k] = Lfwd[k] - beta*Ecomm[start+k, start+t]
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
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.double_t, ndim=1] log_backward(
        np.ndarray[np.double_t, ndim=2] Ecomm, 
        double beta, 
        int start, 
        int end):
    cdef int N = len(Ecomm) - 1 # n nodes
    if start < 0 or end > N:
        raise IndexError("start or end out of range")
    cdef int n = end - start
    cdef np.ndarray[np.double_t, ndim=1] Lbwd = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(n+1, dtype=float)
    cdef int k, t
    cdef double a_max
    
    Lbwd[n] = 0.0
    for t in range(1, n+1):
        a_max = 0.0
        for k in range(t):
            a[k] = Lbwd[n-k] - beta*Ecomm[end-t, end-k]
            if a[k] > a_max:
                a_max = a[k]

        Lbwd[n-t] = a_max + log(np.exp(a[:t] - a_max).sum())

    return Lbwd

###
# The following is equivalent to doing forward on each row, but does it one
# column at a time
### 
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.double_t, ndim=2] log_zmatrix(
    np.ndarray[np.double_t, ndim=2] Ecomm, 
    double beta):
    """
    Return the subsystem Boltzmann weight matrix.
    
    Input:
        Eseg:  segment energy matrix
        beta:  inverse temperature

    Returns:
        ln(W), where
        W[a,b] = sum of Boltzmann weights corresponding to subsystem [a,b], 
                 where a and b+1 are static boundaries
    
    """
    cdef int N = len(Ecomm) - 1 # n nodes
    cdef np.ndarray[np.double_t, ndim=2] Lz = np.zeros((N+1,N+1), dtype=float)
    cdef np.ndarray[np.double_t, ndim=2] a = np.zeros((N+1,N+1), dtype=float)
    cdef int i, j, k
    cdef double a_max

    Lz[0, 0] = 0.0
    for j in range(1, N+1):
        for i in range(0, j): #range(j-1,-1,-1):
            for k in range(i, j):
                a[i, k] = Lz[i, k] - beta*Ecomm[k, j]
            a_max = a[i, i:j].max()
            Lz[i, j] = Lz[j, i] = a_max + log( np.exp(a[i, i:j] - a_max).sum() )

    return Lz


def log_boundary_marginal(np.ndarray[np.double_t, ndim=2] Ecomm, double beta, int start, int end):
    """
    For each node in subsystem [start, end], return the sum of Boltzmann weights 
    of segmentations having a boundary at that node, conditional on there being
    boundaries at start (and implicitly at end+1).
    
    Input:
        Eseg:  segment energy matrix
        beta:  inverse temperature
        start: index of first boundary node
        end:   index of final non-boundary node

    Returns:
        ln(W), where
        W[i] = marginal weight of i being a boundary in [start, end]
        Z = W[0] = W[end+1] is the partition function for subsystem [start, end].
    
    """
    cdef np.ndarray[np.double_t, ndim=1] Lf = log_forward(Ecomm, beta, start, end)
    cdef np.ndarray[np.double_t, ndim=1] Lb = log_backward(Ecomm, beta, start, end)
    return Lf + Lb


def log_segment_marginal(np.ndarray[np.double_t, ndim=2] Ecomm, double beta):
    """
    Returns the segment Boltzmann weight matrix.

    Input:
        Eseg:  segment energy matrix
        beta:  inverse temperature

    Returns:
        ln(W), where
        W[p,q] = sum of Boltmann weights corresponding to all segmentations
                 containing the segment [p, q-1]: p and q as boundaries and
                 no boundaries in between.
        NOTE: Here, indices correspond to segments via slice representation [p,q)

    """
    # NOTE: the diagonal contains the boundary marginal 
    # Interpretation: trivial segments [i,i) are single boundary occurrences.
    cdef int N = len(Ecomm) - 1 # n nodes
    cdef np.ndarray[np.double_t, ndim=1] Lf = log_forward(Ecomm, beta, 0, N)
    cdef np.ndarray[np.double_t, ndim=1] Lb = log_backward(Ecomm, beta, 0, N)
    
    cdef np.ndarray[np.double_t, ndim=2] Lms = np.zeros((N+1, N+1))
    cdef int i, j
    for i in range(N+1):
        for j in range(i, N+1):
            Lms[i,j] = Lms[j,i] = Lf[i] - beta*Ecomm[i, j] + Lb[j]
    
    return Lms


def log_boundary_cooccur_marginal(np.ndarray[np.double_t, ndim=2] Ecomm, double beta):
    # NOTE: the diagonal contains the boundary marginal (assuming Lz[i,i]==0)
    # Interpretation:  Boundary occurrences co-occur with themselves.
    cdef int N = len(Ecomm) - 1 #nodes
    cdef np.ndarray[np.double_t, ndim=2] Lz = log_zmatrix(Ecomm, beta)
     
    cdef np.ndarray[np.double_t, ndim=2] Lmm = np.zeros((N+1,N+1))
    cdef int i, j
    for i in range(N+1):
        for j in range(i, N+1):
            Lmm[i,j] = Lmm[j,i] = Lz[0, i] + Lz[i, j] + Lz[j, N]
    
    return Lmm


def log_segment_cooccur_marginal(np.ndarray[np.double_t, ndim=2] Ecomm, double beta):
    # NOTE: the diagonal contains the partition function Z.
    # Interpretation: Each node is always in the same segment as itself.
    cdef int N = len(Ecomm) - 1
    cdef np.ndarray[np.double_t, ndim=2] Ls = log_segment_marginal(Ecomm, beta)
    cdef double Ls_max = Ls.max()

    cdef np.ndarray[np.double_t, ndim=2] Zseg = np.exp(Ls - Ls_max)
    cdef double Z = Zseg[0, 0]
    
    cdef np.ndarray[np.double_t, ndim=2] Zcos = np.zeros((N,N), dtype=float)
    cdef double s
    cdef int i, j
    # j: N-1
    Zcos[0, N-1] = Zcos[N-1, 0] = Zseg[0, N]
    for i in range(1, N-1):
        Zcos[i, N-1] = Zcos[N-1, i] = Zcos[i-1, N-1] + Zseg[i, N]
    Zcos[N-1, N-1] = Z
    # j: N-2 to 1
    for j in range(N-2, 0, -1):
        s = Zseg[0, j+1]
        Zcos[0, j] = Zcos[j, 0] = Zcos[0, j+1] + s
        for i in range(1, j):
            s = s + Zseg[i, j+1]
            Zcos[i, j] = Zcos[j, i] = Zcos[i, j+1] + s
        Zcos[j, j] = Z
    # j: 0
    Zcos[0, 0] = Z
    
    return Ls_max + np.log(Zcos)


