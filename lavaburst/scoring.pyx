import cython
import numpy as np
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef sums_by_segment(np.ndarray[np.double_t, ndim=2] A):
    """
    S = sums_by_segment(A)
    Sum the edge weights in every segment of the graph represented by A.

    Input:
        A : N x N (double)
        A symmetric square matrix. A[i,j] is the weight of the edge connecting
        nodes i and j. If there are nonzero self-weights in the graph, the 
        values on the diagonal A[i,i] are assumed to be twice the weight.

    Output:
        S : N+1 x N+1 (double)
        A symmetric square matrix of summed weights of all contiguous segments 
        of nodes. S[a,b] is the sum of weights between all pairs of nodes in the 
        half-open range [a..b-1] (excluding b).

    """
    cdef int N = len(A)
    cdef np.ndarray[np.double_t, ndim=2] S = np.zeros((N+1,N+1), dtype=float)

    cdef int i
    for i in range(N):
        S[i,i+1] = S[i+1,i] = A[i,i]/2.0

    cdef int start, end
    cdef np.ndarray[np.double_t, ndim=1] cumsum
    for end in range(1, N):
        cumsum = np.zeros(end+1, dtype=float)
        cumsum[end] = A[end,end]

        for start in range(end-1, -1, -1):
            cumsum[start] = cumsum[start+1] + A[start,end]
            S[start,end+1] = S[end+1,start] = S[start, end] + cumsum[start]

    return S


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef normalized_sums_by_segment(np.ndarray[np.double_t, ndim=2] A):
    """
    S = normalized_sums_by_segment(A)
    Aggregate the edge weights of every segment of the graph represented by A 
    and normalize by the total weight of the graph. If there are nonzero 
    self-weights in the graph, the values on the diagonal A[i,i] are assumed to 
    be twice the edge weight.

    Input:
        A : N x N (double)

    Output:
        S : N+1 x N+1 (double)

    """
    cdef np.ndarray[np.double_t, ndim=1] deg = A.sum(axis=0)
    cdef double m = deg.sum()/2.0
    cdef np.ndarray[np.double_t, ndim=2] S = sums_by_segment(A)
    S /= m
    return S


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef armatus_score(np.ndarray[np.double_t, ndim=2] Sseg, double gamma):
    """
    Rescale the segment sums.
    For each segment, subtract the mean rescaled sum for segments of the same size.

    """
    N = len(Sseg) - 1
    S = np.zeros((N+1, N+1), dtype=float)
    mu = np.zeros(N+1, dtype=float)
    count = np.zeros(N+1, dtype=int)

    for end in range(1, N+1):
        for start in range(end-1, -1, -1):
            size = end - start
            S[start, end] = Sseg[start, end] / size**gamma
            mu[size] = (count[size]*mu[size] + S[start, end])/(count[size] + 1)
            count[size] += 1

    Mu = np.zeros((N+1, N+1), dtype=float)
    for d in range(N+1):
        for k in range(d, N-d):
            Mu[d, k] = mu[d]
    return S - Mu


# def arrowhead_l(A):
#     N = len(A)
#     R = np.zeros((N,N))
#     for i in range(N):
#         for d in range(0, N-i):
#             denom = (A[i,i-d] + A[i, i+d])
#             R[i,i+d] = R[i+d,i] = (A[i,i-d] - A[i,i+d])/denom if denom != 0 else 0
#     return R


# def arrowhead_r(A):
#     N = len(A)
#     R = np.zeros((N,N))
#     for i in range(N):
#         for d in range(0, N-i):
#             denom = (A[i,i-d] + A[i, i+d])
#             R[i,i+d] = R[i+d,i] = (A[i,i+d] - A[i,i-d])/denom if denom != 0 else 0
#     return R


# def arrowhead_corner_score(ArrowheadL, ArrowheadR):
#     N = len(A)
#     aL = arrowhead_l(A)
#     aR = arrowhead_r(np.flipud(np.fliplr(A)))
#     U = np.zeros((N+1,N+1))
#     for i in xrange(0, N+1):
#         U[i,0] = 0.0
#         for j in xrange(i+1, N+1):
#             k = (i+j)//2
#             U[i,j] = U[j,i] = U[i,j-1] + aL[i:k+1, j-1].sum()
#     D = toeplitz(np.r_[1,np.arange(1,N+1)])
#     self.U = U/D
#     L = np.zeros((N+1,N+1))
#     for i in xrange(0, N+1):
#         L[i,0] = 0.0
#         for j in xrange(i+1, N+1):
#             k = (i+j)//2
#             L[i,j] = L[j,i] = L[i,j-1] + aR[i:k+1, j-1].sum()
#     L = np.flipud(np.fliplr(L))
#     self.L = L/D
#     self.Eseg = -(self.L - self.U)






