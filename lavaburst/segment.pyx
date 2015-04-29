from collections import defaultdict
import cython
import numpy as np
cimport numpy as np
from libc.math cimport log, exp, abs


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef max_sum(np.ndarray[np.double_t, ndim=2] score):
    """
    opt, optk = max_sum(S)

    Perform max-sum algorithm (longest path) dynamic program on segmentation 
    path graph with score matrix S.

    Returns:
        opt[i]  - optimal score of path from 0..i
        optk[i] - first predecessor node on optimal path from 0..i

    """
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
cpdef get_path(np.ndarray[np.double_t, ndim=1] opt, np.ndarray[np.int_t, ndim=1] optk):
    """
    path = get_path(opt, optk)

    Backtrack over predecessor nodes to get the optimal path from max-sum.

    Returns:
        path (array) - optimal path of nodes from 0..N

    """
    cdef int N = len(opt) - 1
    cdef np.ndarray[np.int_t, ndim=1] path = np.zeros(N, dtype=int)
    cdef int j = 0
    i = path[j] = N
    j += 1
    while i > 0:
        i = path[j] = optk[i]
        j += 1

    return path[:j][::-1]


@cython.embedsignature(True)
def optimal_segmentation(np.ndarray[np.double_t, ndim=2] score):
    """
    Perform max-sum algorithm (longest path) dynamic program on segmentation 
    path graph with score matrix S.

    Returns:
        path (array) - optimal path of nodes from 0..N
        opt  (array) - optimal score of subproblems

    """
    opt, optk = max_sum(score)
    path = get_path(opt, optk)
    return path, opt


@cython.embedsignature(True)
def consensus_segments(list segments, weights):
    """
    Returns consensus list of nonoverlapping segments.
    Segments are 2-tuples given as half-open intervals [a,b).

    """
    occ = defaultdict(int)
    for d, w in zip(segments, weights):
        occ[d] += w

    cdef int i, j, s_choose, s_ignore

    # map each domain to its closest non-overlapping predecessor
    cdef int M = len(segments)
    cdef np.ndarray[np.int_t, ndim=1] prev = np.zeros(M, dtype=int)
    for i in range(M-1, -1, -1):
        d = segments[i]
        j = i - 1
        while j > -1:
            if segments[j][1] <= d[0]: 
                prev[i] = j
                break
            j -= 1

    # weighted interval scheduling dynamic program
    cdef np.ndarray[np.int_t, ndim=1] score = np.zeros(M, dtype=int)
    for i in range(1, M):
        d = segments[i]
        s_choose = score[prev[i]] + occ[d]
        s_ignore = score[i-1]
        score[i] = max(s_choose, s_ignore)

    cdef list consensus = []
    j = M - 1
    while j > 0:
        if score[j] != score[j-1]:
            consensus.append(segments[j])
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
@cython.embedsignature(True)
cpdef np.ndarray[np.double_t, ndim=1] log_forward(
        np.ndarray[np.double_t, ndim=2] Eseg, 
        double beta,
        int start, 
        int end,
        int maxsize=-1):
    """
    Lf = log_forward(Eseg, beta, start, end)

    Forward algorithm.

    Input:
        Eseg - segment energy matrix
        beta - inverse temperature
        start, end

    Returns:
        Lf (array) - forward statistical weights, length (end - start)

    """
    cdef int N = len(Eseg) - 1 # number of nodes
    if start < 0 or end > N:
        raise IndexError("start or end out of range")
    if maxsize == -1:
        maxsize = N
    cdef int n = end - start
    cdef np.ndarray[np.double_t, ndim=1] Lfwd = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(n+1, dtype=float) 
    cdef int t, k, ms
    cdef double a_max, a_min, c

    Lfwd[0] = 0.0
    for t in range(1, n+1):
        a_max = 0.0
        a_min = 0.0
        ms = max(t-maxsize, 0)
        for k in range(ms, t):
            a[k] = Lfwd[k] - beta*Eseg[start+k, start+t]
            if a[k] > a_max:
                a_max = a[k]
        #     elif a[k] < a_min:
        #         a_min = a[k]

        c = a_max #if a_max > abs(a_min) else a_min
        Lfwd[t] = c + log(np.exp(a[:t] - c).sum())
    
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
@cython.embedsignature(True)
cpdef np.ndarray[np.double_t, ndim=1] log_backward(
        np.ndarray[np.double_t, ndim=2] Eseg, 
        double beta, 
        int start, 
        int end,
        int maxsize=-1):
    """
    Lb = log_backward(Eseg, beta, start, end)

    Backward algorithm.

    Input:
        Eseg - segment energy matrix
        beta - inverse temperature
        start, end

    Returns:
        Lb (array) - forward statistical weights, length (end - start)

    """
    cdef int N = len(Eseg) - 1 # n nodes
    if start < 0 or end > N:
        raise IndexError("start or end out of range")
    if maxsize == -1:
        maxsize = N
    cdef int n = end - start
    cdef np.ndarray[np.double_t, ndim=1] Lbwd = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(n+1, dtype=float)
    cdef int k, t, ms
    cdef double a_max, a_min, c
    
    Lbwd[n] = 0.0
    for t in range(1, n+1):
        a_max = 0.0
        a_min = 0.0
        ms = max(t-maxsize, 0)
        for k in range(ms, t):
            a[k] = Lbwd[n-k] - beta*Eseg[end-t, end-k]
            if a[k] > a_max:
                a_max = a[k]
            # elif a[k] < a_min:
            #     a_min = a[k]

        c = a_max #if a_max > abs(a_min) else a_min
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
@cython.embedsignature(True)
cpdef np.ndarray[np.double_t, ndim=2] log_zmatrix(
    np.ndarray[np.double_t, ndim=2] Eseg, 
    double beta):
    """
    Lz = log_zmatrix(Eseg, beta)

    Compute the subsystem statistical weight matrix.
    
    Input:
        Eseg:  segment energy matrix
        beta:  inverse temperature

    Returns:
        ln(Z), where
        Z[a,b] = sum of statistical weights corresponding to subsystem [a,b)

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
def log_boundary_marginal(np.ndarray[np.double_t, ndim=2] Eseg, double beta, int start, int end):
    """
    Lb = log_boundary_marginal(Eseg, beta, start, end)

    Input:
        Eseg - segment energy matrix
        beta - inverse temperature
        start, end

    Returns:
        Lz[i] = sum of statistical weights of all segmentations having i as a
                segment boundary.

    """
    cdef np.ndarray[np.double_t, ndim=1] Lf = log_forward(Eseg, beta, start, end)
    cdef np.ndarray[np.double_t, ndim=1] Lb = log_backward(Eseg, beta, start, end)
    return Lf + Lb


@cython.embedsignature(True)
def log_segment_marginal(np.ndarray[np.double_t, ndim=2] Eseg, double beta):
    """
    Ls = log_segment_marginal(Eseg, beta)

    Input:
        Eseg - segment energy matrix
        beta - inverse temperature

    Returns:
        Ls[a,b] = sum of statistical weights of all segmentations containing the
                  segment [a,b).

    Returns the segment Boltzmann weight matrix.

    """
    # NOTE: the diagonal contains the boundary marginal 
    # Interpretation: trivial segments [i,i) are single boundary occurrences.
    cdef int N = len(Eseg) - 1 # n nodes
    cdef np.ndarray[np.double_t, ndim=1] Lf = log_forward(Eseg, beta, 0, N)
    cdef np.ndarray[np.double_t, ndim=1] Lb = log_backward(Eseg, beta, 0, N)
    
    cdef np.ndarray[np.double_t, ndim=2] Lms = np.zeros((N+1, N+1))
    cdef int i, j
    for i in range(N+1):
        for j in range(i, N+1):
            Lms[i,j] = Lms[j,i] = Lf[i] - beta*Eseg[i, j] + Lb[j]
    
    return Lms


@cython.embedsignature(True)
def log_boundary_cooccur_marginal(np.ndarray[np.double_t, ndim=2] Eseg, double beta):
    """
    Lbb = log_boundary_cooccur_marginal(Eseg, beta)

    Input:
        Eseg - segment energy matrix
        beta - inverse temperature

    Returns:
        Lbb[i,j] = sum of statistical weights of all segmentations in which
                   both i and j occur as segment boundaries

    """
    # NOTE: the diagonal contains the boundary marginal (assuming Lz[i,i]==0)
    # Interpretation:  Boundary occurrences co-occur with themselves.
    cdef int N = len(Eseg) - 1 #nodes
    cdef np.ndarray[np.double_t, ndim=2] Lz = log_zmatrix(Eseg, beta)
     
    cdef np.ndarray[np.double_t, ndim=2] Lmm = np.zeros((N+1,N+1))
    cdef int i, j
    for i in range(N+1):
        for j in range(i, N+1):
            Lmm[i,j] = Lmm[j,i] = Lz[0, i] + Lz[i, j] + Lz[j, N]
    
    return Lmm


@cython.embedsignature(True)
def log_segment_cooccur_marginal(np.ndarray[np.double_t, ndim=2] Eseg, double beta):
    """
    Lss = log_segment_cooccur_marginal(Eseg, beta)

    Input:
        Eseg - segment energy matrix
        beta - inverse temperature

    Returns:
        Lbb[p,q] = sum of statistical weights of all segmentations in which
                   p and q occur within the same segment

    """
    # NOTE: the diagonal contains the partition function Z.
    # Interpretation: Each node is always in the same segment as itself.
    cdef int N = len(Eseg) - 1
    cdef np.ndarray[np.double_t, ndim=2] Ls = log_segment_marginal(Eseg, beta)
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


