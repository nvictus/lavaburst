from __future__ import division, print_function
from six.moves import xrange as range
from nose.tools import with_setup, assert_raises, assert_equal

import itertools
from scipy.linalg import block_diag
import numpy as np
where = np.flatnonzero


from lavaburst.core import algo
from lavaburst import scoring


A = 10*block_diag(np.ones((4,4)), np.ones((5,5)), np.ones((4,4))).astype(float)
A[np.diag_indices_from(A)] = 0

def test_sums_by_segment():
    n = len(A)
    Zseg = np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(i, n+1):
            Zseg[i,j] = Zseg[j,i] = np.sum(A[i:j,i:j]/2.0)
    assert np.allclose(scoring.sums_by_segment(A), Zseg)
    assert np.allclose(scoring.sums_by_segment(A, normalized=True), Zseg/Zseg[0,n])


class BruteForceEnsemble(object):
    def __init__(self, scorer):
        self.n_nodes = len(scorer) - 1
        self._scorer = scorer

    def iter_states(self, start, stop, as_bool=False):
        assert start < stop
        assert stop <= self.n_nodes
        n = stop - start
        values = [(1,)] + [(0,1) for _ in range(n-1)] + [(1,)]
        bitstrs = itertools.product(*values)
        if as_bool:
            for b in bitstrs:
                yield np.array(b, dtype=bool)
        else:
            for b in bitstrs:
                yield start + where(np.array(b))

    def score_state(self, s):
        domains = zip(s[:-1], s[1:])
        return sum([self._scorer[a,b] for a,b in domains])

    def optimal_segmentation(self):
        N = self.n_nodes
        opt = -np.inf
        opt_state = None
        for s in self.iter_states(0, N):
            score = self.score_state(s)
            if score > opt:
                opt = score
                opt_state = s
        return opt_state
    
    def log_forward(self, beta):
        N = self.n_nodes
        Lf = np.zeros(N+1)
        Lf[0] = 0.0 #prob of first boundary = 1
        for t in range(1, N+1):
            z = 0.0
            for s in self.iter_states(0, t):
                score = self.score_state(s)
                z += np.exp(beta*score)
            Lf[t] = np.log(z)
        return Lf

    def log_backward(self, beta):
        N = self.n_nodes
        Lb = np.zeros(N+1)
        Lb[N] = 0.0 #prob of last boundary = 1
        for t in range(N-1,-1,-1):
            z = 0.0
            for s in self.iter_states(t, N):
                score = self.score_state(s)
                z += np.exp(beta*score)
            Lb[t] = np.log(z)
        return Lb

    def log_zmatrix(self, beta):
        N = self.n_nodes
        Lz = np.zeros((N+1,N+1))
        for t1 in range(0, N+1):
            Lz[t1,t1] = 0.0
            for t2 in range(t1+1, N+1):
                z = 0.0
                for s in self.iter_states(t1, t2):
                    score = self.score_state(s)
                    z += np.exp(beta*score)
                Lz[t1,t2] = Lz[t2,t1] = np.log(z)
        return Lz

    def log_boundary_marginal(self, beta):
        N = self.n_nodes
        M = np.zeros(N+1)
        for b in self.iter_states(0, N, as_bool=True):
            for i in range(N+1):
                if b[i] == 1:
                    score = self.score_state(where(b))
                    M[i] += np.exp(beta*score)
        return np.log(M)

    def log_boundary_cooccur_marginal(self, beta):
        N = self.n_nodes
        M = np.zeros((N+1,N+1))
        for b in self.iter_states(0, N, as_bool=True):
            score = self.score_state(where(b))
            for i in range(N+1):
                for j in range(N+1):
                    if b[i] == 1 and b[j] == 1:
                        M[i,j] += np.exp(beta*score)
        return np.log(M)

    def log_domain_marginal(self, beta):
        N = self.n_nodes
        M = np.zeros((N+1,N+1))
        for b in self.iter_states(0, N, as_bool=True):
            score = self.score_state(where(b))
            for i in range(N+1):
                if b[i] == 1:
                    # put boundary marginals on the diagonal
                    M[i,i] += np.exp(beta*score)
                for j in range(i+1, N+1):
                    if b[i] == 1 and b[j] == 1 and np.all(b[i+1:j]==0):
                        M[i,j] += np.exp(beta*score)
                        M[j,i] = M[i,j]
        return np.log(M)

    def log_domain_cooccur_marginal(self, beta):
        N = self.n_nodes
        M = np.zeros((N,N))
        for b in self.iter_states(0, N, as_bool=True):
            score = self.score_state(where(b))
            for i in range(N):
                M[i,i] += np.exp(beta*score)
                for j in range(i+1, N):
                    if np.all(b[i+1:j+1] == 0):
                        M[i,j] += np.exp(beta*score)
                        M[j,i] = M[i,j]
        return np.log(M)

    def log_q_marginal(self, beta):
        N = self.n_nodes
        M = np.zeros(N+1)
        M[0] = 1.0
        for b in self.iter_states(0, N, as_bool=True):
            s = where(b)
            score = self.score_state(s)
            M[len(s)-1] += np.exp(beta*score)
        return np.log(M)

    def q_mean(self, beta, x=None): #TODO: score x
        N = self.n_nodes
        M = np.zeros(N+1)
        M[0] = 1.0
        Z = 0.0
        for b in self.iter_states(0, N, as_bool=True):
            s = where(b)
            score = self.score_state(s)
            M[len(s)-1] += np.exp(beta*score)*(-score)
            Z += np.exp(beta*score)
        return M/Z



N = len(A)
k = A.sum(axis=0)
Sseg = scoring.sums_by_segment(A, normalized=True)
Snull = scoring.sums_by_segment(np.outer(k,k), normalized=True)
S = Sseg - Snull
e = BruteForceEnsemble(S)

def print_pair(A,B):
    for a, b in zip(A.flat, B.flat):
        print(a, b)

def test_optimal_segmentation():
    opt, pred = algo.maxsum(S)
    path = algo.backtrack(opt, pred)
    s = e.optimal_segmentation()
    assert e.score_state(s) == opt[-1]
    assert np.all(s == path)

# def test_consenus_segmentation():
#     domains = [(1, 10), (5, 15), (1, 10), (1, 10)]
#     d = algo.consensus_segments(domains, weights=[1,1,1,1])
#     assert d == [(1, 10)]

def test_log_forward():
    assert np.allclose(e.log_forward(1), algo.log_forward(S, 1, 0, N))

def test_log_backward():
    assert np.allclose(e.log_backward(1), algo.log_backward(S, 1, 0, N))

def test_log_zmatrix():
    print_pair(e.log_zmatrix(1), algo.log_zmatrix(S, 1))
    assert np.allclose(e.log_zmatrix(1), algo.log_zmatrix(S, 1))

def test_log_boundary_marginal():
    assert np.allclose(e.log_boundary_marginal(1), algo.log_boundary_marginal(S, 1))

def test_log_domain_marginal():
    assert np.allclose(e.log_domain_marginal(1), algo.log_domain_marginal(S, 1))

def test_log_boundary_cooccur_marginal():
    assert np.allclose(e.log_boundary_cooccur_marginal(1), algo.log_boundary_cooccur_marginal(S, 1))

def test_log_domain_cooccur_marginal():
    x = e.log_domain_cooccur_marginal(1)
    y = algo.log_domain_cooccur_marginal(S, 1)
    #print_pair(x,y)
    assert np.allclose(x,y)
