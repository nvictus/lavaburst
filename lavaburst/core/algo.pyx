#!python
#cython: embedsignature=True
from collections import defaultdict
import cython
import numpy as np
cimport numpy as np
from libc.math cimport log, exp, abs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef maxsum(np.ndarray[np.double_t, ndim=2] scoremap, int maxsize=-1):
    """
    Max-sum algorithm (longest path) dynamic program on segmentation path graph
    with score matrix ``scoremap``.

    Parameters
    ----------
    scoremap : 2d-array
        segment score matrix (negative of energy matrix)

    Returns
    -------
    opt[i] : 1d-array
        optimal score of path from 0..i
    pred[i] : 1d-array
        first predecessor node on optimal path from 0..i

    """
    cdef int N = len(scoremap) - 1
    cdef np.ndarray[np.double_t, ndim=1] opt = np.zeros(N+1, dtype=float)
    cdef np.ndarray[np.int_t, ndim=1] pred = np.zeros(N+1, dtype=int)
    if maxsize == -1:
        maxsize = N

    cdef int i, k
    cdef double s
    opt[0] = 0.0
    for i in range(1, N+1):
        opt[i] = -np.inf
        pred[i] = i-1
        for k in range(max(i-maxsize, 0), i):
            s = opt[k] + scoremap[k, i]
            if s > opt[i]:
                opt[i] = s
                pred[i] = k

    return opt, pred


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef backtrack(
        np.ndarray[np.double_t, ndim=1] opt,
        np.ndarray[np.int_t, ndim=1] pred):
    """
    Backtrack over predecessor nodes to get the optimal path from max-sum.

    Parameters
    ----------
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


def maxsum_multi(scoremaps, s_trans):
    n = scoremaps[0].shape[0] - 1
    n_states = len(scoremaps)

    opt = np.zeros((n+1, n_states), dtype=float)
    pred_i = np.zeros((n+1, n_states), dtype=int)
    pred_s = np.zeros((n+1, n_states), dtype=int)

    opt[0, :] = 0
    for i in range(1, n+1):
        for state in range(0, n_states):
            for ip in range(0, i):
                sp = np.argmax(opt[ip, :])
                score = (
                    opt[ip, sp]
                    + s_trans[sp, state]
                    + scoremaps[state][ip, i])
                if score > opt[i, state]:
                    opt[i, state] = score
                    pred_i[i, state] = ip
                    pred_s[i, state] = sp

    return opt, pred_i, pred_s


def backtrack_multi(opt, pred_i, pred_s):
    n = opt.shape[0] - 1
    state = np.zeros(n+1, dtype=int)
    path = np.zeros(n+1, dtype=int)

    s = state[0] = np.argmax(opt[n, :])
    i = path[0] = n
    m = 1
    while i > 0:
        s = state[m] = pred_s[i, s]
        i = path[m] = pred_i[i, s]
        m += 1
    return path[:m][::-1], state[:m][::-1]


def maxsum_local(scoremap):
    cdef int N = len(scoremap) - 1
    cdef np.ndarray[np.double_t, ndim=1] opt = np.zeros(N+1, dtype=float)
    cdef np.ndarray[np.int_t, ndim=1] pred = np.zeros(N+1, dtype=int)

    cdef int i, k
    cdef double s
    opt[0] = 0.0
    for i in range(1, N+1):
        opt[i] = -np.inf
        pred[i] = i-1
        for k in range(0, i):
            s = opt[k] + scoremap[k, i]
            if s > opt[i]:
                opt[i] = s
                pred[i] = k
        if opt[i] < 0:
            opt[i] = 0
            pred[i] = i

    return opt, pred


def backtrack_local(opt, pred):
    n = opt.shape[0] - 1
    state = np.zeros(n+1, dtype=int)
    path = np.zeros(n+1, dtype=int)

    i = path[0] = np.argmax(opt)
    j = 1
    while (i > 0 and opt[i] > 0):
        i = path[j] = pred[i]
        j += 1

    return path[:j][::-1]


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
cpdef np.ndarray[np.double_t, ndim=1] log_forward(
        np.ndarray[np.double_t, ndim=2] S,
        double beta,
        start=None,
        end=None,
        int maxsize=-1):
    """
    Log forward subpartition functions.

    Parameters
    ----------
    S : 2d-array
        domain score matrix
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
    cdef int N = len(S) - 1 # number of nodes
    cdef int start_ = start if start is not None else 0
    cdef int end_ = end if end is not None else N
    if start_ < 0 or start_ > N:
        raise IndexError("start out of range")
    if end_ < 0 or end_ > N:
        raise IndexError("end out of range")
    if maxsize == -1:
        maxsize = N
    cdef int n = end_ - start_
    cdef np.ndarray[np.double_t, ndim=1] Lfwd = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(n+1, dtype=float)
    cdef int t, k
    cdef double a_max

    Lfwd[0] = 0.0
    for t in range(1, n+1):
        a_max = 0.0
        for k in range(max(t-maxsize, 0), t):
            a[k] = Lfwd[k] + beta*S[start_+k, start_+t]
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
cpdef np.ndarray[np.double_t, ndim=1] log_backward(
        np.ndarray[np.double_t, ndim=2] S,
        double beta,
        start=None,
        end=None,
        int maxsize=-1):
    """
    Log backward subpartition functions.

    Parameters
    ----------
    S : 2d-array
        segment score matrix
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
    cdef int N = len(S) - 1 # n nodes
    cdef int start_ = start if start is not None else 0
    cdef int end_ = end if end is not None else N
    if start_ < 0 or start_ > N:
        raise IndexError("start out of range")
    if end_ < 0 or end_ > N:
        raise IndexError("end out of range")
    if maxsize == -1:
        maxsize = N
    cdef int n = end_ - start_
    cdef np.ndarray[np.double_t, ndim=1] Lbwd = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(n+1, dtype=float)
    cdef int k, t
    cdef double a_max

    Lbwd[n] = 0.0
    for t in range(1, n+1):
        a_max = 0.0
        for k in range(max(t-maxsize, 0), t):
            a[k] = Lbwd[n-k] + beta*S[end_-t, end_-k]
            if a[k] > a_max:
                a_max = a[k]

        Lbwd[n-t] = a_max + log(np.exp(a[:t] - a_max).sum())

    return Lbwd


#@cython.boundscheck(False)
#@cython.nonecheck(False)
#@cython.wraparound(False)
def exclusion_log_forward(
    np.ndarray[np.double_t, ndim=2] S,
    double beta,
    double cutoff,
    start=None,
    end=None,
    int maxsize=-1):
    """
    Returns
    -------
    Lf : 1d-array
        Log of sum of statistical weights of segmentation ensembles starting at
        ``start``.

    """
    cdef int N = len(S) - 1 # number of nodes
    if start is None:
        start = 0
    if end is None:
        end = N
    if start < 0 or end > N:
        raise IndexError("start or end out of range")
    if maxsize == -1:
        maxsize = N
    cdef int n = end - start
    cdef np.ndarray[np.double_t, ndim=1] Lfwd = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] filt = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] b = np.zeros(n+1, dtype=float)
    cdef int start_ = start, end_ = end
    cdef int k, t
    cdef double a_max, b_max, s

    Lfwd[0] = 0.0
    filt[0] = 0.0
    for t in range(1, n+1):
        a_max = 0.0
        b_max = 0.0
        for k in range(max(t-maxsize, 0), t):
            s = S[start_+k, start_+t]
            a[k] = Lfwd[k] + beta*s
            b[k] = (Lfwd[k] + beta*s) * (s > cutoff)
            if a[k] > a_max:
                a_max = a[k]
            if b[k] > b_max:
                b_max = b[k]
        Lfwd[t] = a_max + np.log(np.exp(a[:t] - a_max).sum())
        filt[t] = b_max + np.log(np.exp(b[:t] - b_max).sum())

    return filt, Lfwd[-1]


#@cython.boundscheck(False)
#@cython.nonecheck(False)
#@cython.wraparound(False)
def exclusion_log_backward(
    np.ndarray[np.double_t, ndim=2] S,
    double beta,
    double cutoff,
    start=None,
    end=None,
    int maxsize=-1):
    """
    Returns
    -------
    Lb : 1d-array
        Log of sum of statistical weights of segmentation ensembles ending at
        ``end``.

    """
    cdef int N = len(S) - 1 # n nodes
    if start is None:
        start = 0
    if end is None:
        end = N
    if start < 0 or end > N:
        raise IndexError("start or end out of range")
    if maxsize == -1:
        maxsize = N
    cdef int n = end - start
    cdef np.ndarray[np.double_t, ndim=1] Lbwd = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] filt = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] a = np.zeros(n+1, dtype=float)
    cdef np.ndarray[np.double_t, ndim=1] b = np.zeros(n+1, dtype=float)
    cdef int start_ = start, end_ = end
    cdef int k, t
    cdef double a_max, b_max, s

    Lbwd[n] = 0.0
    filt[n] = 0.0
    for t in range(1, n+1):
        a_max = 0.0
        b_max = 0.0
        for k in range(max(t-maxsize, 0), t):
            s = S[end_-t, end_-k]
            a[k] = Lbwd[n-k] + beta*s
            b[k] = (Lbwd[n-k] + beta*s) * (s > cutoff)
            if a[k] > a_max:
                a_max = a[k]
            if b[k] > b_max:
                b_max = b[k]
        Lbwd[n-t] = a_max + np.log(np.exp(a[:t] - a_max).sum())
        filt[n-t] = b_max + np.log(np.exp(b[:t] - b_max).sum())

    return filt, Lbwd[0]


###
# The following is equivalent to doing forward on each row, but does it one
# column at a time
###
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.double_t, ndim=2] log_zmatrix(
    np.ndarray[np.double_t, ndim=2] S,
    double beta):
    """
    Log subsystem partition function matrix.

    Parameters
    ----------
    S : 2d-array
        segment energy matrix
    beta : float
        inverse temperature

    Returns
    -------
    Lz : 2d-array
        log(Z) where Z[i,j] is the sum of statistical weights corresponding to
        the set of subsegmentations between bin edges ``i`` and ``j``.

    """
    cdef int N = len(S) - 1 # n nodes
    cdef np.ndarray[np.double_t, ndim=2] Lz = np.zeros((N+1,N+1), dtype=float)
    cdef np.ndarray[np.double_t, ndim=2] a = np.zeros((N+1,N+1), dtype=float)
    cdef int i, j, k
    cdef double a_max

    Lz[0, 0] = 0.0
    for j in range(1, N+1):
        for i in range(0, j): #range(j-1,-1,-1):
            for k in range(i, j):
                a[i, k] = Lz[i, k] + beta*S[k, j]
            a_max = a[i, i:j].max()
            Lz[i, j] = Lz[j, i] = a_max + log( np.exp(a[i, i:j] - a_max).sum() )

    return Lz


def log_boundary_marginal(
        np.ndarray[np.double_t, ndim=2] S,
        double beta,
        int maxsize=-1):
    """
    Log of marginal domain boundary statistical weight sums.

    Parameters
    ----------
    S : 2d-array
        segment energy matrix
    beta : float
        inverse temperature

    Returns
    -------
    Lz[i] : 1d-array
        Log of sum of statistical weights of all segmentations having ``i`` as
        a domain boundary.

    """
    cdef int N = len(S) - 1
    if maxsize == -1:
        maxsize = N
    cdef np.ndarray[np.double_t, ndim=1] Lf = log_forward(S, beta, 0, N, maxsize)
    cdef np.ndarray[np.double_t, ndim=1] Lb = log_backward(S, beta, 0, N, maxsize)
    return Lf + Lb


def _log_domain_marginal(
        np.ndarray[np.double_t, ndim=2] S,
        double beta,
        np.ndarray[np.double_t, ndim=1] Lf,
        np.ndarray[np.double_t, ndim=1] Lb,
        int maxsize=-1):
    """
    Parameters
    ----------
    S : 2d-array
        segment score matrix
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
    cdef int N = len(S) - 1
    cdef np.ndarray[np.double_t, ndim=2] Ls = np.zeros((N+1, N+1))
    cdef int i, j
    if maxsize == -1:
        maxsize = N
    for i in range(maxsize+1):
        for j in range(i, maxsize+1):
            Ls[i,j] = Ls[j,i] = Lf[i] + beta*S[i, j] + Lb[j]
    return Ls


def log_domain_marginal(
        np.ndarray[np.double_t, ndim=2] S,
        double beta,
        int maxsize=-1):
    """
    Log of marginal domain statistical weight sums.

    Parameters
    ----------
    S : 2d-array
        segment score matrix
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
    cdef int N = len(S) - 1
    cdef np.ndarray[np.double_t, ndim=1] Lf = log_forward(S, beta, 0, N, maxsize)
    cdef np.ndarray[np.double_t, ndim=1] Lb = log_backward(S, beta, 0, N, maxsize)
    return _log_domain_marginal(S, beta, Lf, Lb, maxsize)


cpdef _log_boundary_cooccur_marginal(
        np.ndarray[np.double_t, ndim=2] Lz):
    """
    Parameters
    ----------
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


def log_boundary_cooccur_marginal(
        np.ndarray[np.double_t, ndim=2] S,
        double beta):
    """
    Log of statistical weight sums for pairs of bin edges simultaneously
    occurring as domain boundaries.

    Parameters
    ----------
    S : 2d-array
        segment score matrix
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
    cdef np.ndarray[np.double_t, ndim=2] Lz = log_zmatrix(S, beta)
    return _log_boundary_cooccur_marginal(Lz)


cpdef _log_domain_cooccur_marginal(
        np.ndarray[np.double_t, ndim=2] Ls):
    """
    Parameters
    ----------
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


def log_domain_cooccur_marginal(
        np.ndarray[np.double_t, ndim=2] S,
        double beta):
    """
    Log of statistical weight sums for pairs of bins co-occurring within the
    same domain.

    Parameters
    ----------
    S : 2d-array
        segment score matrix
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
    cdef np.ndarray[np.double_t, ndim=2] Ld = log_domain_marginal(S, beta)
    return _log_domain_cooccur_marginal(Ld)


def exclusion_log_boundary_marginal(S, beta, cutoff):
    """
    Returns
    -------
    Lz[i] : 1d-array
        Log of sum of statistical weights of all segmentations having ``i`` as
        a domain boundary.

    """
    N = len(S) - 1
    Lf, logZ = exclusion_log_forward(S, beta, cutoff)
    Lb, _ = exclusion_log_backward(S, beta, cutoff)
    return Lf + Lb - logZ


def exclusion_log_domain_marginal(S, beta, Ef, Eb):
    cdef int N = len(S) - 1
    cdef np.ndarray[np.double_t, ndim=2] Ld = np.zeros((N+1, N+1))
    cdef int i, j
    for i in range(N+1):
        for j in range(i, N+1):
            Ld[i,j] = Ld[j,i] = Ef[i] + beta*S[i, j] + Eb[j]
    return Ld


def local_log_zmatrix(LZglobal):
    N = LZglobal.shape[0] - 1
    LZlocal = np.zeros((N+1, N+1))
    LZlocal[0, :] = LZglobal[0, :]
    LZlocal[:, 0] = LZglobal[:, 0]

    for j in range(1, N+1):
        for i in range(0, j):
            a = [LZlocal[i, j-1], LZglobal[i, j]]
            a_max = np.max(a)
            LZlocal[i, j] = LZlocal[j, i] = a_max + np.log( np.exp(a - a_max).sum() )

    return LZlocal


unifrnd = np.random.random_sample # uniform in [0, 1)
randint = np.random.randint       # samples zero-based, half-open range


class LoadedDie(object):
    # Alias sampler
    # http://www.keithschwarz.com/darts-dice-coins/
    def __init__(self, weights):
        weights = np.asarray(weights, dtype=float)
        self.prob, self.alias = self._build(weights)

    def _build(self, x):
        assert np.all(x >= 0) and np.any(x)

        # normalize
        x_total = np.sum(x)
        if not np.isclose(x_total, 1.0):
            x /= x_total

        N = len(x)
        prob = np.zeros(N, dtype=float)
        alias = np.zeros(N, dtype=int)

        # rescale probabilities x and fill the stacks
        x = N*x
        small = []
        large = []
        for i, p in enumerate(x):
            if p < 1:
                small.append(i)
            else:
                large.append(i)

        # normally, small empties first
        while small and large:
            # assign an alias to l
            l = small.pop()
            g = large.pop()
            prob[l] = x[l]
            alias[l] = g

            # trim off g's donated area and put it back in one of the stacks
            x[g] = (x[g] + x[l]) - 1.0 # more stable than x[g] - (1 - x[l])
            if x[g] < 1.0:
                # possible false assignment if x[g]/N should be 1.0, but falls below
                # due to numerical instability
                small.append(g)
            else:
                large.append(g)

        # for remaining bins, no need to alias
        while large:
            g = large.pop()
            prob[g] = 1.0

        # check in case large emptied before small due to numerical problems
        while small:
            l = small.pop()
            prob[l] = 1.0

        return prob, alias

    def roll(self, n=None):
        if n is None:
            outcome = randint(len(self.prob))
            return self.alias[outcome] if unifrnd() > self.prob[outcome] else outcome
        else:
            outcomes = randint(0, len(self.prob), n)
            missed = unifrnd(n) > self.prob[outcomes]
            outcomes[missed] = self.alias[outcomes[missed]]
            return outcomes


#@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef log_project_insul(np.ndarray[np.double_t, ndim=1] pi):
    """
    Project a 1-D insulation profile ``pi`` into a heatmap ``P`` according to
    the multiplicative model: P[i,j] = prod_{i<=k<-j}(1 - pi[k]).

    Parameters
    ----------
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
