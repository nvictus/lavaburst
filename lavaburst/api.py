from __future__ import division, print_function
import collections
import functools
import numpy as np


class Segmentation(object):
    @classmethod
    def from_state(cls, is_boundary):
        self._path = where(is_boundary)

    def __init__(self, path):
        self._path = path

    @property
    def path(self):
        return self._path

    @property
    def state(self):
        N = self._path[-1]
        s = np.zeros(N+1, dtype=int)
        s[self._path] = 1
        return s

    @property
    def segment_starts(self):
        return self._path[:-1]

    @property
    def segment_ends(self):
        return self._path[1:]

    @property
    def segment_spans(self):
        path = self._path
        return np.c_[path[:-1], path[1:]]


def memoize(maxsize=100):
    '''Least-frequenty-used cache decorator.

    Arguments to the cached function must be hashable.
    Cache performance statistics stored in f.hits and f.misses.
    Clear the cache with f.clear().
    http://en.wikipedia.org/wiki/Least_Frequently_Used

    http://code.activestate.com/recipes/498245-lru-and-lfu-cache-decorators/

    '''
    def decorator(func):
        cache = {}                      # mapping of args to results
        use_count = collections.Counter()           # times each key has been accessed
        kwarg_marker = object()         # separate positional and keyword args

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            key = args
            if kwargs:
                key += (kwarg_marker,) + tuple(sorted(kwargs.items()))

            # get cache entry or compute if not found
            try:
                result = cache[key]
                use_count[key] += 1
                wrapped.hits += 1
            except KeyError:
                # need to add something to the cache, make room if necessary
                if len(cache) == maxsize:
                    for k, _ in nsmallest(maxsize // 10 or 1,
                                            use_count.iteritems(),
                                            key=itemgetter(1)):
                        del cache[k], use_count[k]
                cache[key] = func(*args, **kwargs)
                result = cache[key]
                use_count[key] += 1
                wrapped.misses += 1
            return result

        def clear():
            cache.clear()
            use_count.clear()
            wrapped.hits = wrapped.misses = 0

        wrapped.hits = wrapped.misses = 0
        wrapped.clear = clear
        wrapped.cache = cache
        return wrapped
    return decorator



# class memoizable(object):
#     """
#     Decorator based on:
#         https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
#         http://stackoverflow.com/questions/1988804

#     """
#     def __init__(self, func):
#         self.func = func
#         self.cache = {}
#     def __get__(self, obj, objtype):
#         """ Support instance methods. """
#         return functools.partial(self.__call__, obj)
#     def __call__(self, *args, **kwargs):
#         import inspect
#         do_memoize = kwargs.pop('memoize', False)
#         spec = inspect.getargspec(self.func).args
#         kwargs.update(dict(zip(spec, args)))
#         if do_memoize:
#             key = tuple(kwargs.get(k, None) for k in spec)
#             return self.cache[key]
#         else:
#             return self.func(**kwargs)
#     def __missing__(self, key):
#         value = self.func(*key)
#         self.cache[key] = value
#         return value


class Segmenter(object):
    @memoize(100)
    def _log_forward(self, beta):
        return segment.log_forward(self.Eseg, beta, 0, len(self.Eseg)-1)

    @memoize(100)
    def _log_backward(self, beta):
        return segment.log_backward(self.Eseg, beta, 0, len(self.Eseg)-1)

    @memoize(32)
    def _log_zmatrix(self, beta):
        return segment._log_zmatrix(self.Eseg, beta)

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
        return Lf + Lb

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
        self.Zseg = segment.normalized_sums_by_segment(A)
        self.Znull = segment.normalized_sums_by_segment(np.outer(k,k))
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
        return S - Mu

    def __init__(self, A, gamma):
        A = np.asarray(A, dtype=float)
        N = len(A)
        self._gamma = gamma
        self.Zseg = segment.normalized_sums_by_segment(A)
        Q = self._rescaled_sums(self.Zseg, gamma)
        Q[Q < 0] = 0
        self.Eseg = -Q

    def gamma():
        def fget(self):
            return self._gamma
        def fset(self, value):
            self._gamma = value
            Q = self_rescaled_sums(self.Zseg, self._gamma)
            Q[Q < 0] = 0
            self.Eseg = -Q
        return locals()
    gamma = property(**gamma())


class ArrowheadSegmenter(Segmenter):
    def __init__(self, A):
        from scipy.linalg import toeplitz
        N = len(A)
        aL = arrowhead_l(A)
        aR = arrowhead_r(np.flipud(np.fliplr(A)))
        U = np.zeros((N+1,N+1))
        for i in xrange(0, N+1):
            U[i,0] = 0.0
            for j in xrange(i+1, N+1):
                k = (i+j)//2
                U[i,j] = U[j,i] = U[i,j-1] + aL[i:k+1, j-1].sum()
        D = toeplitz(np.r_[1,np.arange(1,N+1)])
        self.U = U/D
        L = np.zeros((N+1,N+1))
        for i in xrange(0, N+1):
            L[i,0] = 0.0
            for j in xrange(i+1, N+1):
                k = (i+j)//2
                L[i,j] = L[j,i] = L[i,j-1] + aR[i:k+1, j-1].sum()
        L = np.flipud(np.fliplr(L))
        self.L = L/D
        self.Eseg = -(self.L - self.U)



class FooSegmenter(Segmenter):
    params = ('gamma', 'bar')

    def energy_matrix(self, A, gamma, bar):
        return A + gamma - bar

    # def score_matrix(self, A, gamma, bar):
    #     pass
