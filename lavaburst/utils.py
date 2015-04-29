from __future__ import division, print_function
import itertools
import numpy as np
where = np.flatnonzero


def masked_ufunc(ufunc, wheremask, axis=(0,1)):
    """
    Wraps a numpy ufunc to apply an operation on a subset of rows and/or columns.
    Note: ignored rows/columns receive uninitialized junk values unless an 'out'
    array is passed to the ufunc.

    """
    wheremask = np.asarray(wheremask, dtype=bool)
    def wrapped(*args, **kwargs):
        if axis == (0,1):
            kwargs['where'] = np.logical_or.outer(wheremask, wheremask)
        elif axis == 0:
            kwargs['where'] = wheremask
        else:
            kwargs['where'] = wheremask.reshape(-1, 1)
        return ufunc(*args, **kwargs)
    return wrapped


def masked_normalize(a, wheremask, axis=0):
    """
    Normalize the data along an axis for a subset of rows or columns.

    """
    a = np.array(a, dtype=a.dtype)
    wheremask = np.asarray(wheremask, dtype=bool)
    if axis == 0:
        s = a.sum(axis=0)
        a = np.divide(a, s, where=wheremask, out=a)
    else:
        s = a.sum(axis=1)
        a = np.divide(a, s, where=wheremask.reshape(-1, 1), out=a)
    return a


def mask_compress(a, passmask, axis=(0,1)):
    """
    Compress a 1d or 2d numpy array along one or both axes by removing a subset 
    of rows and/or columns.

    """
    passmask = np.asarray(passmask, dtype=bool)
    
    if a.ndim == 1:
        return a[passmask]

    elif a.ndim == 2:
        if axis == (0,1):
            return a.compress(passmask, axis=0)\
                    .compress(passmask, axis=1) 
        elif axis == 0:
            return a.compress(passmask, axis=0)
        elif axis == 1:
            return a.compress(passmask, axis=1)
        else:
            raise ValueError("'axis' must be 0, 1 or (0,1)")

    raise TypeError("Input array must be rank 1 or 2")


def mask_restore(b, passmask, fill_value=0, axis=(0,1)):
    """
    Restore the rows and/or columns filtered out using passmask to produce b.

    """
    passmask = np.asarray(passmask, dtype=bool)
    n = len(passmask)
    fill_if = ~passmask

    if b.ndim == 1:
        a = np.zeros(n, dtype=b.dtype)
        a[passmask] = b
        if fill_value: a[fill_if] = fill_value
        return a

    elif b.ndim == 2:
        if axis == (0,1):
            a = np.zeros((n, n), dtype=b.dtype)
            a[np.ix_(passmask, passmask)] = b
            if fill_value: a[np.ix_(fill_if, fill_if)] = fill_value
            return a
        elif axis == 0:
            a = np.zeros((n, b.shape[1]), dtype=b.dtype)
            a[passmask, :] = b
            if fill_value: a[fill_if, :] = fill_value
            return a
        elif axis == 1:
            a = np.zeros((b.shape[0], n), dtype=b.dtype)
            a[:, passmask] = b
            if fill_value: a[:, fill_if] = fill_value
            return a
        else:
            raise ValueError("'axis' must be 0, 1 or (0,1)")

    raise TypeError("Input array must be rank 1 or 2")


def mask_compress_path(path, passmask):
    """
    Return coordinates of path nodes as they would appear in the compressed
    sequence. If a path node lies inside region to be "compressed", it is 
    replaced with the coordinates of the ends of that compressed region.

    """
    # Original coordinates of nodes being kept
    keep_loc = where(passmask)
    n = len(keep_loc)
    
    # Find ends of compressed regions
    breakpoints = np.diff(keep_loc) > 1
    bad_regions = zip(keep_loc[:-1][breakpoints] + 1, keep_loc[1:][breakpoints])

    # Convert original coordinates to compression coordinates
    # Force path to include 0 and compressed sequence length
    new_path = [0, n]
    i = 0
    for loc in path:
        while i < n and keep_loc[i] <= loc:
            if keep_loc[i] == loc:
                new_path.append(i)
            i += 1

        for a, b in bad_regions:
            if a <= loc < b:
                new_path.append(a)
                new_path.append(b)

    new_path = np.array(sorted(set(new_path)), dtype=int)
    return new_path


def mask_restore_path(path, passmask):
    """
    Return coordinates of path nodes as they would appear in the decompressed
    sequence. If a path node lies at a compression point, it is replaced with
    the coordinates of both ends of the compressed region. No other compression
    points are included in the expanded path.

    """
    # Expansion coordinates of nodes kept
    pass_loc = where(passmask)

    # Find ends of compressed regions
    breakpoints = np.diff(pass_loc) > 1
    bad_regions = zip(pass_loc[:-1][breakpoints] + 1, pass_loc[1:][breakpoints])

    # Convert compression coordinates to expansion coordinates
    # Force path to include 0 and expanded sequence length
    starts = path[:-1]
    new_path = list(pass_loc[starts])
    more = [0, len(passmask)]
    for l, r in bad_regions:
        if r in new_path:
            more.append(l)
        if l in new_path:
            more.append(r)
    new_path = new_path + more

    return np.array(sorted(set(new_path)), dtype=int)



def tilt_heatmap(A, n_diags=None, pad=np.nan):
    """
    Vertically stacks the upper diagonals of A onto a new matrix of the same 
    shape, giving rise to a "tilted" half matrix.

    This is basically a hack to visualize a matrix rotated 45 degrees to the
    left, since matplotlib can't rotate images.

    Input:
        A   - symmetric N x N matrix
        pad - pad value (default: NaN)

    Output:
        T - N x N "tilted" version of upper triangle of A

    Note that this transformation distorts the matrix when viewed with equal 
    aspect. When using imshow/matshow, set the aspect ratio to 1/sqrt(2).

        >>> f = plt.figure()
        >>> ax = f.add_subplot(111)
        >>> ax.matshow(tilt_heatmap(A), aspect=1/np.sqrt(2), origin='lower')

    """
    N = len(A)
    if n_diags is None:
        n_diags = N

    T = -np.ones((n_diags, N))*pad
    for k in xrange(n_diags):
        T[k, k//2 : N-(k+1)//2] = A.diagonal(k)

    return T


def blocks(N, segments, lower=True, upper=False, labels=None, offset=0):
    if labels is None:
        labels = [0]*len(segments)
    mat = -np.ones((N,N), dtype=float)
    for k, (start, stop) in enumerate(segments):
        for i in xrange(start, stop):
            for j in xrange(i, stop):
                if i >= offset and j < (stop - offset):
                    if upper:
                        mat[i,j] = labels[k]
                    else:
                        mat[j,i] = labels[k]
    return np.ma.masked_where(mat == -1, mat)


def checkerboard(labels, lower=True, upper=False, offset=0):
    labels = np.asarray(labels)
    N = len(labels)
    mat = np.nan*np.ones((N,N), dtype=float)

    n_labels = max(labels)
    mmap = [ where(labels==c) for c in xrange(0, n_labels+1) ]
    for c, members in enumerate(mmap):
        for i in members:
            mat[i,i] = c
        for i, j in itertools.combinations(members, 2):
            if upper:
                mat[i,j] = c
            else:
                mat[j,i] = c
    return np.ma.masked_where(np.isnan(mat), mat) 


def reveal_tril(A, k=0):
    E = np.ones(A.shape)
    return np.ma.masked_where(np.logical_not(np.tril(E, -k)), A)


def reveal_triu(A, k=0):
    E = np.ones(A.shape)
    return np.ma.masked_where(np.logical_not(np.triu(E, k)), A)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# def log_contactmap_from_insulation_prob(
#         np.ndarray[np.double_t, ndim=1] p_insul):

#     cdef np.ndarray[np.double_t, ndim=1] log_pc = np.log(1 - p_insul)
    
#     # N bins, n bin edges
#     cdef int N = len(log_pc) - 1
#     cdef int n = N+1 
#     cdef np.ndarray[np.double_t, ndim=2] L = np.zeros((n, n), dtype=float)
#     cdef int i, diag

#     # base case (first two diagonals) 
#     # XXX --- leave out the main diag for consistency?
#     for i in range(0, n):
#         L[i, i] = log_pc[i]

#     # first diag
#     for i in range(0, n-1):
#         L[i, i+1] \
#             = L[i+1, i] \
#             = log_pc[i] + log_pc[i+1]

#     for diag in range(2, n):
#         for i in range(0, n-diag):
#             L[i, i+diag] \
#                 = L[i+diag, i] \
#                 = L[i, i+diag-1] + L[i+1, i+diag] - L[i+1, i+diag-1]

#     return L


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# def fill_triu_inplace(
#         np.ndarray[np.double_t, ndim=2] A, 
#         double value=np.nan):

#     cdef int N = len(A)
#     cdef int i, j
#     for i in range(N):
#         for j in range(i, N):
#             A[i, j] = value


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# def fill_tril_inplace(
#         np.ndarray[np.double_t, ndim=2] A, 
#         double value=np.nan):

#     cdef int N = len(A)
#     cdef int i, j
#     for i in range(N):
#         for j in range(0, i):
#             A[i, j] = value


# def reveal_diagonal_band(A, k=0, lower=True, upper=True):
#     E = np.ones(A.shape)
#     hide_mask = np.logical_or(np.tril(E, -k), np.triu(E, k))
#     return np.ma.masked_where(hide_mask, A)


# def fill_rowcol(A, idx, fillvalue=np.nan):
#     A = np.array(A)
#     A[idx, :] = fillvalue
#     A[:, idx] = fillvalue
#     return A


def matshow(ax, A, **kw):
    kw.setdefault('origin','upper')
    kw.setdefault('interpolation', 'none') # For Agg, ps and pdf backends (others fall back to 'nearest')
    kw.setdefault('aspect', 'equal')
    ax.imshow(A, **kw)
