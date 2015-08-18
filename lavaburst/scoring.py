from __future__ import division, print_function, unicode_literals
import numpy as np

from .core.scoring import sums_by_segment
from .core import scoring as _scoring
from . import utils


def modularity_score(A, gamma, binmask=None, **kw):
    Sdata = sums_by_segment(A, normalized=True)
    Snull = sums_by_segment(np.ones(A.shape), normalized=True)
    S = Sdata - gamma*Snull
    N = A.shape[0]

    if binmask is not None:
        edges = np.zeros(N)
        edges[~binmask] = np.nan
        np.fill_diagonal(S, np.r_[edges, 0])

    return S

potts_score = modularity_score


def corner_score(A, binmask=None, **kw):
    S = _scoring.corner_score(A)
    N = A.shape[0]
    if binmask is not None:
        edges = np.zeros(N)
        edges[~binmask] = np.nan
        np.fill_diagonal(S, np.r_[edges, 0])

    return S


def logodds_score(A):
    pass


# def armatus_score(A):
#     pass


normalized_sums_by_segment = utils.deprecated(_scoring.normalized_sums_by_segment)


def contactbias(A, window=200):
    N = len(A)
    di = np.zeros(N)

    for i in range(N):
        w = max(0, min(window, N-i))
        di[i] = (A[i, i:(i+w)+1].sum() - A[i, (i-w):i+1].sum()) / A[i, (i-w):(i+w)+1].sum()

    return di


def directionality_chisquare(A, window=200):
    N = len(A)
    di = np.zeros(N)

    for i in range(N):
        w = min(window, N-i)
        b, a = A[i, i:(i+w)+1].sum(), A[i, (i-w):i+1].sum()
        e = (a + b)/2.0
        di[i] = np.sign(b - a) * ( (a-e)**2 + (b-e)**2 )/e

    return di


def where_diag(N, diag):
    if diag >= 0:
        diag_indices = np.c_[np.arange(0, N-diag), np.arange(diag, N)]
    else:
        diag_indices = np.c_[np.arange(-diag, N), np.arange(0, N+diag)]   
    return diag_indices[:, 0], diag_indices[:, 1]


def sliding_window(w, *arrays):
    n = len(arrays[0])
    for i in range(0, n-w+1):
        yield tuple(x[i:i+w] for x in arrays)


def insul(A, extent=200):
    N = len(A)
    score = np.zeros(N)
    dscore = np.zeros(N)

    for diag in range(extent):
        for i, (qi, qj) in enumerate(
                sliding_window(diag+1, *where_diag(N, diag))):
            dscore[i+diag] = A[qi, qj].mean()
        score[diag:-diag] += dscore[diag:-diag] #(dscore[diag:-diag] - score[diag:-diag]) / diag

    return score


def insul_diamond(A, extent=200):
    N = len(A)
    starts = np.arange(0, N-extent)
    ends = np.arange(extent, N)

    score = np.zeros(N)
    for i in range(0, N):
        w = min(extent, i, N-i)
        score[i] = A[i-w:i, i:i+w].sum()
    score /= score.mean()

    return score