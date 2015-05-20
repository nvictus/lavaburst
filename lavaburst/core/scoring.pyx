import cython
import numpy as np
cimport numpy as np

from scipy.linalg import toeplitz


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.embedsignature(True)
cpdef sums_by_segment(
        np.ndarray[np.double_t, ndim=2] A, 
        int offset=1,
        int normalized=0):
    """
    S = sums_by_segment(A)
    Sum the edge weights in every segment of the graph represented by A.

    Input:
        A : N x N (double)
        Symmetric square matrix. A[i,j] is the weight of the edge connecting
        nodes i and j. If there are nonzero self-weights in the graph, the 
        values on the diagonal A[i,i] are assumed to be twice the weight.

        offset : int
        Extends or reduces the shape of the resulting matrix. Used to switch
        between closed [a,b] and half open interval [a,b+1) logic.

    Output:
        S : N+1 x N+1 (double)
        A symmetric square matrix of summed weights of all contiguous segments 
        of nodes. If offset = 1, S[a,b] is the sum of weights between all pairs 
        of nodes in the half-open range [a..b-1] (excluding b).

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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.embedsignature(True)
cpdef normalized_sums_by_segment(
        np.ndarray[np.double_t, ndim=2] A,
        int offset=1):
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
    cdef np.ndarray[np.double_t, ndim=2] S = sums_by_segment(A, offset)
    S /= m
    return S


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef outer_accumulate(np.ndarray[np.double_t, ndim=1] x):
    
    # N bins, n bin edges
    cdef int N = len(x) - 1
    cdef int n = N + 1 
    cdef np.ndarray[np.double_t, ndim=2] L = np.zeros((n, n), dtype=float)
    cdef int i, diag

    # base case: 0th diag
    for i in range(0, n):
        L[i, i] = x[i]

    # base case: 1st diag
    for i in range(0, n-1):
        L[i, i+1] = L[i+1, i] = x[i] + x[i+1]

    for diag in range(2, n):
        for i in range(0, n-diag):
            L[i, i+diag] \
                = L[i+diag, i] \
                = L[i, i+diag-1] + L[i+1, i+diag] - L[i+1, i+diag-1]

    return L


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.embedsignature(True)
cpdef armatus_score(np.ndarray[np.double_t, ndim=2] Sseg, double gamma):
    """
    For each segment:
        Rescale its score.
        Subtract the mean rescaled score for segments of the same size.

    """
    cdef int N = len(Sseg) - 1
    cdef np.ndarray[np.double_t, ndim=2] S = np.zeros((N+1, N+1), dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] mu = np.zeros(N+1, dtype=float)
    cdef np.ndarray[np.int_t, ndim=1] count = np.zeros(N+1, dtype=int)
    
    cdef int start, end, size
    for end in range(1, N+1):
        for start in range(end-1, -1, -1):
            size = end - start
            S[start, end] = S[end, start] = Sseg[start, end] / size**gamma
            mu[size] = (count[size]*mu[size] + S[start, end]) / (count[size] + 1)
            count[size] += 1

    cdef np.ndarray[np.double_t, ndim=2] Mu = np.zeros((N+1, N+1), dtype=float)
    cdef int diag, i
    for diag in range(N+1):
        for i in range(0, N-diag):
            Mu[i, diag+i] = Mu[diag+i, i] = mu[diag]
    return S - Mu


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.double_t, ndim=2] arrowhead(np.ndarray[np.double_t, ndim=2] A):
    cdef int N = len(A)
    cdef np.ndarray[np.double_t, ndim=2] H = np.zeros((N,N), dtype=float)
    cdef int i, d
    cdef double denom
    for i in range(N):
        for d in range(0, N-i):
            denom = (A[i,i-d] + A[i, i+d])
            H[i,i+d] = H[i+d,i] = (A[i,i-d] - A[i,i+d])/denom if denom != 0 else 0
    return H


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.double_t, ndim=2] arrowhead_r(np.ndarray[np.double_t, ndim=2] A):
    cdef int N = len(A)
    cdef np.ndarray[np.double_t, ndim=2] H = np.zeros((N,N), dtype=float)
    cdef int i, d
    cdef double denom
    for i in range(N):
        for d in range(0, N-i):
            denom = (A[i,i-d] + A[i, i+d])
            H[i,i+d] = H[i+d,i] = (A[i,i+d] - A[i,i-d])/denom if denom != 0 else 0
    return H


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef accumulate_arrowhead(
    np.ndarray[np.double_t, ndim=2] H,
    int offset=1):
    
    cdef int N = len(H)
    cdef np.ndarray[np.double_t, ndim=2] U = np.zeros(
        (N+offset,N+offset), dtype=float)
    
    cdef int i, j, k, flag
    cdef double c
    
    U[0, offset] = U[offset, 0] = 0.25 * H[0,0]

    for j in range(1, N):
        U[j, j+offset] = U[j+offset, j] = 0.25 * H[j,j]
        c = 0.75 * H[j-1,j]
        flag = 1
        k = 1
        for i in range(j-1, 0, -1):
            U[i, j+offset] = U[j+offset, i] = U[i, j-1 + offset] + c
            if flag:
                c = c - 0.5*H[j-k, j]
                flag = 0
            else:
                c = c - 0.25*H[j-k,j] - 0.25*H[j-k-1,j]
                k += 1
                flag = 1
            c = c + H[i-1, j]
        U[0, j+offset] = U[j+offset, 0] = U[0, j-1 + offset] + c
    
    return U


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.embedsignature(True)
def corner_score(np.ndarray[np.double_t, ndim=2] A):
    """
    Corner score.

    """
    cdef np.ndarray[np.double_t, ndim=2] H1 = arrowhead(A)
    cdef np.ndarray[np.double_t, ndim=2] H2 = arrowhead_r(A[::-1,::-1])
    cdef np.ndarray[np.double_t, ndim=2] U = accumulate_arrowhead(H1)
    cdef np.ndarray[np.double_t, ndim=2] L = accumulate_arrowhead(H2)[::-1,::-1]
    return L - U


def logbins(a, b, pace, N_in=0):
    "create log-spaced bins"
    from math import log
    a = int(a)
    b = int(b) 
    beg = log(a)
    end = log(b - 1)
    pace = log(pace)
    N = int((end - beg) / pace)     

    if N_in != 0: N = N_in
    if N_in > (b - a):
        raise ValueError("Cannot create more bins than elements")
    else:
        N = (b - a)

    pace = (end - beg) / N
    mas = np.arange(beg, end + 0.000000001, pace)    
    ret = np.exp(mas)
    surpass = np.arange(a,a+N)
    replace = surpass > ret[:N]-1
    ret[replace] = surpass  
    ret = np.array(ret, dtype = np.int)
    ret[-1] = b 
    return list(ret)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def diag_mean(A, mask):
    """
    Calculates averages of a contact map as a function of separation
    distance, over regions where mask==1.

    """
    _data = np.array(A, dtype=float, order="C")
    _datamask = np.array(mask == 1, dtype=int, order="C")
    cdef np.ndarray[np.double_t, ndim=2] data = _data 
    cdef np.ndarray[np.double_t, ndim=2] datamask = _datamask 
    cdef N = data.shape[0]

    _bins = logbins(1, N, 1.05)
    _bins = [(0, 1)] + [(_bins[i], _bins[i+1]) for i in xrange(len(_bins)-1)]
    _bins = np.array(_bins, dtype=int, order = "C")
    cdef np.ndarray[np.int64_t, ndim = 2] bins = _bins

    cdef int M = len(bins)
    cdef np.ndarray[np.double_t, ndim=1] avg = np.zeros(N, dtype=float)
    cdef int i, j, start, end, count, offset
    cdef double ss, meanss  
    for i in range(M):
        start, end = bins[i, 0], bins[i, 1]
        ss = 0.0
        count = 0
        for offset in range(start, end):
            for j in range(0, N-offset):
                if datamask[offset+j, j] == 1:                    
                    ss += data[offset+j, j]
                    count += 1
        #print start, end, count
        meanss = ss / count
        for offset in range(start, end):
            avg[offset] = meanss

    return data
