# -*- encoding: utf-8 -*-
from __future__ import division, print_function
import numpy as np

from .core import algo
from . import utils


class SegModel(object):
    def __init__(self, S, edgemask=None):
        self.edgemask = edgemask
        if edgemask is not None and np.any(edgemask):
            self.score_matrix = utils.mask_compress(S, self.edgemask)
        else:
            self.score_matrix = S
        self.n_bins = self.score_matrix.shape[0] - 1

    def optimal_segmentation(self, return_score=False, maxsize=None):
        """
        Find the highest scoring segmentation on the path graph.

        Parameters
        ----------

        Returns
        -------
        segments : array (k x 2)

        """
        opt, pred = algo.maxsum(self.score_matrix)
        path = algo.backtrack(opt, pred)
        if self.edgemask is not None:
            path = utils.mask_restore_path(path, self.edgemask)
        segments = np.c_[path[:-1], path[1:]]
        if return_score:
            return segments, [self.score_matrix[a, b] for a,b in segments]
        return segments

    def sample(self, beta, n, shuffled=False):
        """
        Parameters
        ----------
        beta : float
            inverse temperature
        n : int
            number of segmentations to sample
        maxsize : int
            upper bound on allowed size of domains

        Returns
        -------
        samples : 2d array
            Each row is a binary array where 1s designate domain boundaries.

        """
        sampler = SegSampler(self.score_matrix, beta)
        result = sampler.sample(n)
        if self.edgemask is not None:
            result = utils.mask_restore(result, self.edgemask, axis=1)
        return result

    def boundary_marginals(self, beta, maxsize=None, cutoff=None):
        """
        Parameters
        ----------
        beta : float
            inverse temperature
        maxsize : int
            upper bound on allowed size of domains

        Returns
        -------
        prob : 1d array
            prob[i] = frequency of segmentations having i as a domain boundary

        """
        maxsize = -1 if maxsize is None else maxsize
        if cutoff is not None:
            Lf, logZ = algo.exclusion_log_forward(self.score_matrix, beta, cutoff, maxsize=maxsize)
            Lb, _ = algo.exclusion_log_backward(self.score_matrix, beta, cutoff, maxsize=maxsize)
        else:
            Lf = algo.log_forward(self.score_matrix, beta, 0, self.n_bins, maxsize=maxsize)
            Lb = algo.log_backward(self.score_matrix, beta, 0, self.n_bins, maxsize=maxsize)
            logZ = Lf[-1]

        Pb = np.exp(Lf + Lb - logZ)
        if self.edgemask is not None:
            Pb = utils.mask_restore(Pb, self.edgemask)

        return Pb

    def domain_marginals(self, beta, maxsize=None, cutoff=None):
        """
        Parameters
        ----------
        beta : float
            inverse temperature
        maxsize : int
            upper bound on allowed size of domains

        Returns
        -------
        Pd : 2d array
            Pd[a,b] = frequency of segmentations containing the domain [a,b).

        """
        maxsize = -1 if maxsize is None else maxsize
        if cutoff is not None:
            Lf, logZ = algo.exclusion_log_forward(self.score_matrix, beta, cutoff, maxsize=maxsize)
            Lb, _ = algo.exclusion_log_backward(self.score_matrix, beta, cutoff, maxsize=maxsize)
        else:
            Lf = algo.log_forward(self.score_matrix, beta, 0, self.n_bins, maxsize=maxsize)
            Lb = algo.log_backward(self.score_matrix, beta, 0, self.n_bins, maxsize=maxsize)
            logZ = Lf[-1]

        Ld = algo._log_domain_marginal(self.score_matrix, beta, Lf, Lb)
        Pd = np.exp(Ld - logZ)
        if self.edgemask is not None:
            Pd = utils.mask_restore(Pd, self.edgemask)

        return Pd

    def cooccurence_marginals(self, beta, maxsize=None, cutoff=None):
        """
        Parameters
        ----------
        beta : float
            inverse temperature
        maxsize : int
            upper bound on allowed size of domains


        Returns
        -------
        Pdd : 2d array
            Pdd[i,j] = frequency of segmentations in which bins i and j occur 
            within the same domain.

        """
        maxsize = -1 if maxsize is None else maxsize
        if cutoff is not None:
            Lf, logZ = algo.exclusion_log_forward(self.score_matrix, beta, cutoff, maxsize=maxsize)
            Lb, _ = algo.exclusion_log_backward(self.score_matrix, beta, cutoff, maxsize=maxsize)
        else:
            Lf = algo.log_forward(self.score_matrix, beta, 0, self.n_bins, maxsize=maxsize)
            Lb = algo.log_backward(self.score_matrix, beta, 0, self.n_bins, maxsize=maxsize)
            logZ = Lf[-1]

        Ld = algo._log_domain_marginal(self.score_matrix, beta, Lf, Lb)
        Ldd = algo._log_domain_cooccur_marginal(Ld)
        Pdd = np.exp(Ldd - logZ)
        if self.edgemask is not None:
            Pdd = utils.mask_restore(Pdd, self.edgemask[:-1])

        return Pdd


class SegSampler(object):
    def __init__(self, scoremap, beta):
        n = scoremap.shape[0] - 1
        log_Zfwd = algo.log_forward(scoremap, beta, 0, n)
        self.dice = [None] + [
            algo.LoadedDie(
                np.exp(log_Zfwd[0:i] - log_Zfwd[i] + beta*scoremap[0:i, i])
            )
            for i in range(1, n+1)
        ]

    def sample(self, n_samples=1):
        n = len(self.dice) - 1
        samples = np.zeros((n_samples, n+1), dtype=int)
        samples[:, 0] = 1
        samples[:, n] = 1
        for k in range(n_samples):
            i = n
            samples[k, i] = 1
            while i != 0:
                i = self.dice[i].roll()
                samples[k, i] = 1
        return samples
