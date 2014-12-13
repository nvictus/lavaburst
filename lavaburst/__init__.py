import collections
import functools

from . import segment
from . import mcmc


class memoized(object):
   """
   Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).

   From the Python Decorator Library
   https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      """Return the function's docstring."""
      return self.func.__doc__
   def __get__(self, obj, objtype):
      """Support instance methods."""
      return functools.partial(self.__call__, obj)


class Segmenter(object):
    @memoized
    def _log_forward(self, beta):
        return segment.log_forward(self.Eseg, beta, 0, len(self.Eseg))

    @memoized
    def _log_backward(self, beta):
        return segment.log_backward(self.Eseg, beta, 0, len(self.Eseg))

    @memoized
    def _log_zmatrix(self, beta):
        return segment.log_forward(self.Eseg, beta)

    def optimal_segmentation(self):
        return segment.optimal_segmentation(-self.Eseg)

    def consensus_segmentation(self, segments):
        occ = collections.Counter(segments)
        return segment.consensus_segmentation(segments, occ)

    def logZ(self, beta):
        return self._log_forward(beta)[-1]

    def log_boundary_marginal(self, beta):
        Lf = self._log_forward(beta)
        Lb = self._log_backward(beta)
        return segment.log_boundary_marginal(Lf, Lb)

    def log_segment_marginal(self, beta):
        Lf = self._log_forward(beta)
        Lb = self._log_backward(beta)
        return segment.log_segment_marginal(Lf, Lb)

    def log_boundary_cooccur_marginal(self, beta):
        Lz = self._log_zmatrix(beta)
        return segment.log_boundary_cooccur_marginal(Lz)

    def log_segment_cooccur_marginal(self, beta):
        Ls = self._log_segment_marginal(beta)
        return segment.log_segment_cooccur_marginal(Ls)

    def metropolis_mcmc(self, n_sweeps, x0):
        return segment.metropolis_mcmc(self.Eseg, beta, x0, n_sweeps)


class PottsSegmenter(Segmenter):
    def __init__(self, A, gamma):
        A = np.asarray(A, dtype=float)
        k = A.sum(axis=0)
        self._gamma = gamma
        self.Zseg = segment.normalized_sum_by_segment(A)
        self.Znull = segment.normalized_sum_by_segment(np.outer(k,k))
        self.Eseg = -self.Zseg + gamma*self.Znull

    def gamma():
        def fget(self):
            return self._gamma
        def fset(self, value):
            self._gamma = value
            self.Eseg = -self.Zseg + gamma*self.Znull
        return locals()
    gamma = property(**gamma())


class ArmatusSegmenter(Segmenter):
    def _rescaled_sums(self, Zseg, gamma):
        N = len(Zseg) - 1
        S = np.zeros((N+1, N+1), dtype=float)
        mu = np.zeros(N+1, dtype=float)
        count = np.zeros(N+1, dtype=int)
        for end in xrange(1, N+1):
            for start in xrange(end-1, -1, -1):
                size = end - start
                S[start, end] = Zseg[start, end] / size**gamma
                mu[size] = (count[size]*mu[size] + S[start, end])/(count[size] + 1)
                count[size] += 1
        Mu = np.zeros((N+1, N+1), dtype=float)
        for d in xrange(N+1):
            for k in xrange(d, N-d):
                Mu[d, k] = mu[d]
        return S, Mu

    def __init__(self, A, gamma):
        A = np.asarray(A, dtype=float)
        N = len(A)
        self._gamma = gamma
        self.Zseg = segment.normalized_sum_by_segment(A)
        scaling, mu = self_rescaled_sums(self.Zseg, gamma)
        Q = self.Zseg/scaling - self.mu
        Q[Q < 0] = 0
        self.Eseg = -Q

    def gamma():
        def fget(self):
            return self._gamma
        def fset(self, value):
            self._gamma = value
            scaling, mu = self_rescaled_sums(self.Zseg, self._gamma)
            Q = self.Zseg/scaling - self.mu
            Q[Q < 0] = 0
            self.Eseg = -Q
        return locals()
    gamma = property(**gamma())


class ArrowheadSegmenter(Segmenter):
    pass

