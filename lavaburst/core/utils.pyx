import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.double_t, ndim=2] fill_triu_inplace(
        np.ndarray[np.double_t, ndim=2] A,
        int k,
        double value=np.nan):

    cdef int N = A.shape[0]
    cdef int i, j, start
    for i in range(N):
        start = max(0, i+k)
        for j in range(start, N):
            A[i, j] = value

    return A


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.double_t, ndim=2] fill_tril_inplace(
        np.ndarray[np.double_t, ndim=2] A,
        int k,
        double value=np.nan):

    cdef int N = A.shape[0]
    cdef int i, j, end
    for i in range(N):
        end = min(i+k, N)
        for j in range(0, end):
            A[i, j] = value

    return A


def geomrange(start, stop, factor, include_end=True):
    from math import log
    a, b = int(start), int(stop)
    log_start, log_stop = log(a), log(b) 
    n_steps = int((log_stop - log_start) / log(factor))
    step = (log_stop - log_start) / n_steps
    log_range = np.arange(log_start, log_stop, step)
    bins = np.unique(np.round(np.exp(log_range))).astype(int)
    if include_end and bins[-1] != b:
        bins = np.r_[bins, b]
    return bins


#@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def expected1d(A, mask=None, factor=1.05):
    """
    Calculates averages of a contact map as a function of separation
    distance, over regions where mask==1.

    """
    cdef N = A.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] data = A.astype(float)
    cdef np.ndarray[np.int_t, ndim=2] datamask
    if mask is None:
        datamask = np.ones(A.shape, dtype=int)
    else:
        datamask = np.array(mask == 1, dtype=int)

    cdef np.ndarray[np.int64_t, ndim = 2] bins
    _bins = geomrange(1, N, factor, include_end=True)
    _bins = [(0, 1)] + list(zip(_bins[:-1], _bins[1:]))
    bins = np.array(_bins, dtype=int)

    cdef np.ndarray[np.double_t, ndim=1] avg = np.zeros(N, dtype=float)
    cdef int i, j, start, end, count, offset
    cdef double ss, meanss
    for i in range(len(bins)):
        start, end = bins[i, 0], bins[i, 1]
        ss = 0.0
        count = 0
        for offset in range(start, end):
            for j in range(0, N-offset):
                if datamask[offset+j, j] == 1:
                    ss += data[offset+j, j]
                    count += 1
        meanss = ss / count
        for offset in range(start, end):
            avg[offset] = meanss
    return avg
