# -*- encoding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

from scipy.linalg import toeplitz
import numpy as np

from .core.scoring import aggregate_by_segment
from .core import scoring as _scoring
from .utils import where_diagonal
sums_by_segment = aggregate_by_segment


def _corner_matrices(A):
    Hup = _scoring.arrowhead_up(A)
    Hdn = _scoring.arrowhead_dn(A)
    sizes = _scoring.aggregate_arrowhead_up(np.ones(A.shape))
    sizes[sizes==0] = 1

    U = _scoring.aggregate_arrowhead_up(Hup)
    D = _scoring.aggregate_arrowhead_dn(Hdn)
    Sval = (U + D) / sizes / 2

    Usign = _scoring.aggregate_arrowhead_up(np.sign(Hup))
    Dsign = _scoring.aggregate_arrowhead_dn(np.sign(Hdn))
    Ssign = (Usign + Dsign) / sizes / 2
 
    U2 = _scoring.aggregate_arrowhead_up(Hup**2)
    D2 = _scoring.aggregate_arrowhead_dn(Hdn**2)
    Svar = (U2 + D2)/sizes/2 - ((U + D)/sizes/2)**2

    return Sval, Ssign, Svar


def corner_score(A, gamma=1, trim_diags=3, binmask=None, **kw):
    N = A.shape[0]
    Sval, Ssign, Svar = _corner_matrices(A)
    S = (Sval/Sval.max() + Ssign/Ssign.max() - Svar) / 3.
    S /= S.max()
    S = S * toeplitz(np.arange(N+1)**gamma)
    for k in range(-trim_diags, trim_diags):
        S[where_diagonal(S, k)] = -1
    if binmask is not None:
        edges = np.zeros(N)
        edges[~binmask] = np.nan
        np.fill_diagonal(S, np.r_[edges, 0])
    return S


def variance_score(A, gamma=1, trim_diags=3, binmask=None):
    N = A.shape[0]
    B = np.log10(np.clip(A, A[A>0].min(), A.max()))
    B += np.abs(B.min())
    S1 = aggregate_by_segment(B)
    S2 = aggregate_by_segment(B**2)
    sizes = aggregate_by_segment(np.ones(A.shape))
    sizes[sizes==0] = 1
    Svar = S2 / sizes - (S1/sizes)**2
    Svar[Svar == 0] = 1
    S = 1 / Svar
    for k in range(-trim_diags, trim_diags):
        S[where_diagonal(S, k)] = 0
    S /= S.max()
    S = S * toeplitz(np.r_[1, np.arange(N)]**gamma)
    if binmask is not None:
        edges = np.zeros(N)
        edges[~binmask] = np.nan
        np.fill_diagonal(S, np.r_[edges, 0])
    return S


def modularity_score(A, gamma, binmask=None, **kw):
    N = A.shape[0]
    Sdata = aggregate_by_segment(A, normalized=True)
    Snull = aggregate_by_segment(np.ones(A.shape), normalized=True)
    S = Sdata - gamma*Snull
    if binmask is not None:
        edges = np.zeros(N)
        edges[~binmask] = np.nan
        np.fill_diagonal(S, np.r_[edges, 0])
    return S


potts_score = modularity_score


def armatus_score(A, gamma, binmask=None, **kw):
    N = A.shape[0]
    Sdata = aggregate_by_segment(A, normalized=True)
    Sa, Mu = _scoring.armatus(Sdata, gamma)
    S = Sa - Mu
    S = np.clip(S, np.percentile(S, 1), np.percentile(S, 99))
    if binmask is not None:
        edges = np.zeros(N)
        edges[~binmask] = np.nan
        np.fill_diagonal(S, np.r_[edges, 0])
    return S


# def logodds_score(A, binmask=None):
#     N = A.shape[0]
#     A = np.clip(A, A.min(), A.max())
#     logA = np.log10(A)
#     logA += np.abs(logA.min())
#     E = toeplitz(_scoring.sep_mean(logA))
#     Sdata = aggregate_by_segment(logA, normalized=True)
#     Snull = aggregate_by_segment(E, normalized=True)
#     S = Sdata - Snull
#     if binmask is not None:
#         edges = np.zeros(N)
#         edges[~binmask] = np.nan
#         np.fill_diagonal(S, np.r_[edges, 0])
#     return S


### 1D tracks ###

def directionality_index(A, window=200):
    N = A.shape[0]
    di = np.zeros(N)
    for i in range(0, N):
        lo = max(0, i-window)
        hi = min((i+window)+1, N)
        b, a = A[i, i:hi].sum(), A[i, lo:i+1].sum()
        e = (a + b)/2.0
        if e:
            di[i] = np.sign(b - a) * ( (a-e)**2 + (b-e)**2 ) / e
    return di


def directionality_bias(A, window=200):
    N = A.shape[0]
    di = np.zeros(N)
    for i in range(0, N):
        lo = max(0, i-window)
        hi = min((i+window)+1, N)
        s = A[i, lo:hi].sum()
        if s:
            di[i] = (A[i, i:hi].sum() - A[i, lo:i+1].sum()) / s
    return di


def insul(A, extent=200):
    N = A.shape[0]
    score = np.zeros(N)
    for k in range(extent):
        ri, rj = where_diagonal(A, k)
        for i in range(0, N-k+1):
            score[i+k] = A[ri[i:i+k], rj[i:i+k]]
    return score


def insul_diamond(A, extent=200):
    N = A.shape[0]
    score = np.zeros(N)
    for i in range(0, N):
        lo = max(0, i-w)
        hi = min(i+w, N)
        score[i] = A[lo:i, i:hi].sum()
    score /= score.mean()
    return score
