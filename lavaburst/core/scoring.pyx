#!python
#cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np
from scipy.linalg import toeplitz


#@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef armatus(np.ndarray[np.double_t, ndim=2] Sseg, double gamma):
    """
    Input
    -----
    Sseg : 2d-array
        a segment-aggregated matrix
    gamma: float
        scaling factor

    Returns
    -------
    Sa : 2d-array
        S[i,j] is the rescaled score for segment [i,j) 
    Mu: 2d-array 
        Mu[i, j] is the average rescaled score for a segment of length j-i.

    """
    cdef int N = Sseg.shape[0] - 1
    cdef np.ndarray[np.double_t, ndim=2] denom = toeplitz(
        np.r_[1, np.arange(1, N+1)]**gamma)
    cdef np.ndarray[np.double_t, ndim=2] Sa = Sseg / denom

    cdef np.ndarray[np.double_t, ndim=1] mu = np.zeros(N+1, dtype=float)
    cdef np.ndarray[np.int_t, ndim=1] count = np.zeros(N+1, dtype=int)
    cdef int start, end, size
    for end in range(1, N+1):
        for start in range(end-1, -1, -1):
            size = end - start
            mu[size] = (count[size]*mu[size] + Sa[start, end]) / (count[size] + 1)
            count[size] += 1

    cdef np.ndarray[np.double_t, ndim=2] Mu = np.zeros((N+1, N+1), dtype=float)
    cdef int d, i
    for d in range(N+1):
        for i in range(0, N-d):
            Mu[i, d+i] = Mu[d+i, i] = mu[d]

    return Sa,  Mu


#@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.double_t, ndim=2] arrowhead_up(
        np.ndarray[np.double_t, ndim=2] A):
    """
    Upstream Arrowhead matrix.

    ``A[i, i+d]`` gives the relative affinity that the **upstream** member 
    ``i`` has for the downstream locus ``i+d`` versus its equidistant upstream
    locus ``i-d``.

    """
    cdef int N = A.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] H = np.zeros((N,N), dtype=float)
    cdef int i, d, lo, hi
    cdef double up, dn, denom
    for i in range(N):
        for d in range(0, N-i):
            lo, hi = max(0, i-d), i+d
            if i-d > -1:
                up = A[i, lo]
            else:  
                # if no upstream locus, sample a random contact from the 
                # diagonal
                r = np.random.randint(d, N)
                up = A[r, r-d]
            dn = A[i, hi]
            denom = up + dn
            H[i, hi] = H[hi, i] = (dn - up)/denom if denom != 0 else 0
    return H


#@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.double_t, ndim=2] arrowhead_dn(
        np.ndarray[np.double_t, ndim=2] A):
    """
    Downstream Arrowhead matrix.

    ``A[i, i-d]`` gives the relative affinity that the **downstream** member
    ``i`` has for its upstream locus ``i-d`` versus its equidistant 
    downstream locus ``i+d``.

    """
    cdef np.ndarray[np.double_t, ndim=2] H = arrowhead_up(A[::-1, ::-1])
    return H[::-1, ::-1]


#@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef aggregate_by_segment(
        np.ndarray[np.double_t, ndim=2] A, 
        int offset=1,
        int normalized=0):
    """
    Aggregate the edge weights in every "segment" cluster of the graph
    represented by A.

    Input
    -----
    A : 2d-array
        Symmetric square matrix. A[i,j] is the weight of the edge connecting
        nodes i and j. If there are nonzero values on the main diagonal, those
        values are assumed be twice the self-weights.

    offset : int, optional
        Extends or reduces the shape of the resulting matrix. Used to switch
        between closed [a,b] and half open interval [a,b+1) logic. Default 
        offset is 1.

    normalized: int, optional
        Normalize by the total weight of the graph. Default is 0 (false).

    Returns
    -------
    S : 2d-array (N + offset by N + offset)
        A symmetric square matrix of summed weights of all contiguous segments 
        of nodes. 

    If offset = 1, S[a,b] is the sum of weights between all pairs of nodes in 
    the half-open range [a..b-1] (excluding b).

    """
    cdef int N = len(A)
    cdef np.ndarray[np.double_t, ndim=2] S = np.zeros(
        (N+offset,N+offset), dtype=float)

    cdef int i
    for i in range(N):
        S[i,i+offset] = S[i+offset,i] = A[i,i]/2.0

    cdef int start, end
    cdef np.ndarray[np.double_t, ndim=1] cumsum
    for end in range(1, N):
        cumsum = np.zeros(end+1, dtype=float)
        cumsum[end] = A[end,end]

        for start in range(end-1, -1, -1):
            cumsum[start] = cumsum[start+offset] + A[start,end]
            S[start,end+offset] = \
                S[end+offset,start] = S[start, end] + cumsum[start]

    cdef np.ndarray[np.double_t, ndim=1] deg
    if normalized:
        deg = A.sum(axis=0)
        S /= (deg.sum()/2.0)

    return S



#@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.double_t, ndim=2] aggregate_arrowhead_up(
        np.ndarray[np.double_t, ndim=2] H, 
        int offset=1):
    cdef int N = H.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] U = np.zeros((N+offset,N+offset), dtype=float)
    cdef int i, j, k, flag
    cdef double c
    
    U[0, offset] = U[offset, 0] = 0.25 * H[0, 0]

    for j in range(1, N):
        U[j, j + offset] = U[j + offset, j] = 0.25 * H[j, j]
        c = 0.75 * H[j-1, j]
        flag = 1
        k = 1
        for i in range(j-1, 0, -1):
            U[i, j + offset] = U[j + offset, i] = U[i, j-1 + offset] + c
            if flag:
                c = c - 0.5*H[j-k, j]
                flag = 0
            else:
                c = c - 0.25*H[j-k, j] - 0.25*H[j-k-1, j]
                k += 1
                flag = 1
            c = c + H[i-1, j]
        U[0, j + offset] = U[j + offset, 0] = U[0, j-1 + offset] + c
    
    return U

#@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.double_t, ndim=2] aggregate_arrowhead_dn(
        np.ndarray[np.double_t, ndim=2] H, 
        int offset=1):
    cdef np.ndarray[np.double_t, ndim=2] D = aggregate_arrowhead_up(H[::-1, ::-1])
    return D[::-1, ::-1]
