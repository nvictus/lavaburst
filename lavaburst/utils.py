from __future__ import division, print_function
import itertools
import numpy as np
where = np.flatnonzero

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


# def reveal_diagonal_band(A, k=0, lower=True, upper=True):
#     E = np.ones(A.shape)
#     hide_mask = np.logical_or(np.tril(E, -k), np.triu(E, k))
#     return np.ma.masked_where(hide_mask, A)


# def fill_rowcol(A, idx, fillvalue=np.nan):
#     A = np.array(A)
#     A[idx, :] = fillvalue
#     A[:, idx] = fillvalue
#     return A


def mask_normalize(a, pass_mask, axis=0):
    a = np.array(a, dtype=a.dtype)
    if axis == 0:
        s = a[pass_mask,:].sum(axis=0)[pass_mask]
        a[:,pass_mask] /= s
    else:
        s = a[:,pass_mask].sum(axis=1)[pass_mask]
        a[pass_mask,:] /= s        
    return a


def mask_compress(a, pass_mask, axis=(0,1)):
    if a.ndim == 1:
        return a[pass_mask]
    elif a.ndim == 2:
        if axis == (0,1):
            return a.compress(pass_mask, axis=0)\
                    .compress(pass_mask, axis=1) 
                    #a[pass_mask,:][:,pass_mask]
        elif axis == 0:
            return a.compress(pass_mask, axis=0)
                    #a[pass_mask,:]
        elif axis == 1:
            return a.compress(pass_mask, axis=1)
                    #a[:,pass_mask]
        else:
            raise TypeError("axis must be 0, 1 or (0,1)")
    raise TypeError("Input array must be rank 1 or 2")


def mask_restore(b, pass_mask, fill_value=0, axis=(0,1)):
    n = len(pass_mask)
    fill_mask = np.logical_not(pass_mask)
    if b.ndim == 1:
        a = np.zeros(n, dtype=b.dtype)
        a[pass_mask] = b
        if fill_value: a[fill_mask] = fill_value
        return a
    elif b.ndim == 2:
        if axis == (0,1):
            a = np.zeros((n,n), dtype=b.dtype)
            a[np.ix_(pass_mask,pass_mask)] = b
            if fill_value: a[np.ix_(fill_mask,fill_mask)] = fill_value
            return a
        elif axis == 0:
            a = np.zeros((n,b.shape[1]), dtype=b.dtype)
            a[pass_mask,:] = b
            if fill_value: a[fill_mask,:] = fill_value
            return a
        elif axis == 1:
            a = np.zeros((b.shape[0],n), dtype=b.dtype)
            a[:,pass_mask] = b
            if fill_value: a[:,fill_mask] = fill_value
            return a
        else:
            raise TypeError("axis must be 0, 1 or (0,1)")
    raise TypeError("Input array must be rank 1 or 2")


def mask_compress_starts(starts, pass_mask):
    # if a start position is in a masked region, attempt to shift it to the 
    # first unmasked position in the interval
    keep_pos = where(pass_mask)
    boundaries = np.r_[starts, len(pass_mask)]
    relocated_starts = []
    for i, curr in enumerate(boundaries[:-1]):
        next = boundaries[i+1]
        j = curr
        while j < next:
            if j in keep_pos:
                relocated_starts.append(j)
                break
            j += 1
    relocated_starts = np.array(relocated_starts)
    starts_mask = np.zeros(len(pass_mask), dtype=bool)
    starts_mask[relocated_starts] = True
    new_starts = where(starts_mask[pass_mask])
    return new_starts


def mask_restore_starts(starts, pass_mask):
    # Convert compressed coordinates to original coordinates
    included = where(pass_mask)
    new_starts = included[starts]
    # Properly terminate intervals that end just before a masked region begins.
    # This creates an interval that spans the masked region.
    breaks = np.diff(included) > 1
    breaks_left = included[:-1][breaks]
    breaks_right = included[1:][breaks]
    new_starts = list(new_starts) + \
        [breaks_left[i] for (i, r) in enumerate(breaks_right) if r in new_starts]
    # Start a new interval if the end of the sequence is masked.
    # This creates a terminal interval in the final masked region.
    if pass_mask[-1] == False:
        j = len(pass_mask) - 1
        while j > -1:
            if pass_mask[j]:
                new_starts.append(j+1)
                break
            j -= 1
    new_starts.sort()
    return new_starts


def matshow(ax, A, **kw):
    kw.setdefault('origin','upper')
    kw.setdefault('interpolation', 'none') # For Agg, ps and pdf backends (others fall back to 'nearest')
    kw.setdefault('aspect', 'equal')
    ax.imshow(A, **kw)
