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
