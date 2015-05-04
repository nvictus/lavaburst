import cython
import numpy as np
cimport numpy as np
from libc.math cimport log, exp

where = np.flatnonzero
unifrnd = np.random.random_sample # uniform in [0, 1)
randint = np.random.randint       # samples zero-based, half-open range


def metropolis(
        np.ndarray[np.int_t, ndim=1] initial_state,
        np.ndarray[np.int_t, ndim=1] positions,
        np.ndarray[np.double_t, ndim=2] Sseg,
        double beta, 
        int num_sweeps):

    cdef int N = initial_state.size - 1
    cdef int M = positions.size - 1
    cdef np.ndarray[np.int_t, ndim=2] state = np.zeros((num_sweeps,N), dtype=int)
    cdef np.ndarray[np.int_t, ndim=2] events = np.zeros((num_sweeps,M), dtype=int)
    cdef np.ndarray[np.double_t, ndim=2] gains = np.zeros((num_sweeps,M), dtype=float)
    cdef np.ndarray[np.int_t, ndim=1] random_ints = np.random.random_integers(1,M-1, size=(M,)) 
    state[0, :] = initial_state

    cdef int s
    for s in range(1,num_sweeps):
        state[s, :] = state[s-1, :]
        metropolis_sweep(s, M, state, events, gains, positions, random_ints, Sseg, beta)

    return state, events, gains


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef metropolis_sweep(
        int s,
        int M,
        np.ndarray[np.int_t, ndim=2] state,
        np.ndarray[np.int_t, ndim=2] events,
        np.ndarray[np.double_t, ndim=2] gains,
        np.ndarray[np.int_t, ndim=1] positions,
        np.ndarray[np.int_t, ndim=1] random_ints,
        np.ndarray[np.double_t, ndim=2] Sseg,
        double beta):

    cdef int i, j, k, here, left, right
    cdef double S1, S2, S12
    cdef double gain, loglik
    
    # do a sweep of M iterations
    for k in range(0, M):

        # pick a random site (never the first or the last)
        i = random_ints[k]
        here = positions[i]

        for j in range(i-1, -1, -1):
            left = positions[j]
            if state[s,left] == 1:
                break

        for j in range(i+1, M+1):
            right = positions[j]
            if state[s,right] == 1:
                break

        # propose to delete a boundary (i.e. merge two segments)
        # or create a boundary (i.e. split a segment)
        S1  = Sseg[left, here]
        S2  = Sseg[here, right]
        S12 = Sseg[left, right]
        if state[s,here] == 1: 
            gain = S12 - S1 - S2
        else: 
            gain = S1 + S2 - S12

        # acceptance probability (deltaE = -m*deltaQ, loglik = -beta*deltaE)
        loglik = beta*gain

        # decide whether to toggle the boundary or not (Metropolis)
        if gain >= 0 or log(unifrnd()) < loglik:
            state[s, here] = 0 if state[s, here] == 1 else 1
            events[s, k] = i
            gains[s, k] = gain
        else:
            events[s, k] = 0
            gains[s, k] = 0.0


def score_segmentation(Eseg, borders):
    segs = np.array([borders[:-1], borders[1:]])
    score = 0.0
    for a, b in segs:
        score += Eseg[a,b]
    return score


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int find_left(
        int i,
        np.ndarray[np.int_t, ndim=1] state):
    cdef int j, left
    for j in range(i-1, -1, -1):
        left = j
        if state[left] == 1:
            return left
    raise ValueError


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int find_right(
        int i,
        np.ndarray[np.int_t, ndim=1] state,
        int N):
    cdef int j, right
    for j in range(i+1, N+1):
        right = j
        if state[right] == 1:
            return right
    raise ValueError


def conserved_op(
        np.ndarray[np.double_t, ndim=2] Eseg,
        double beta,
        int q,
        int n_sweeps):
    # n_nodes
    cdef int N = len(Eseg) - 1
    assert 2 <= q <= N

    initial_state = np.zeros(N+1, dtype=int)
    pos = np.round(np.linspace(0, N, q)).astype(int)
    initial_state[pos] = 1
    initial_energy = score_segmentation(Eseg, initial_state)
    if q == 2:
        return initial_state, initial_energy

    # set the initial state and create up and down arrays
    cdef np.ndarray[np.int_t, ndim=1] up = np.flatnonzero(initial_state==1)
    cdef np.ndarray[np.int_t, ndim=1] down = np.flatnonzero(initial_state==0)
    up = up[1:-1] #ignore first and last boundaries when sampling

    # initialize time course buffers
    cdef np.ndarray[np.int_t, ndim=2] state = np.zeros((n_sweeps,N+1), dtype=int)
    cdef np.ndarray[np.double_t, ndim=1] energy = np.zeros((n_sweeps), dtype=float)
    state[0, :] = initial_state
    energy[0] = initial_energy

    # run the sweeps
    cdef int s
    for s in range(1, n_sweeps):
        sweep_conserved_op(s, N, Eseg, beta, up, down, state, energy)

    return state, energy


@cython.boundscheck(False)
@cython.wraparound(False)
cdef sweep_conserved_op(
        int s,
        int N,
        np.ndarray[np.double_t, ndim=2] Eseg,
        double beta,
        np.ndarray[np.int_t, ndim=1] up,
        np.ndarray[np.int_t, ndim=1] down,
        np.ndarray[np.int_t, ndim=2] state,
        np.ndarray[np.double_t, ndim=1] energy):

    # initialize state from last sweep
    state[s, :] = state[s-1, :]
    energy[s] = energy[s-1]

    # the conserved number of mutable boundaries and nonboundaries
    cdef int nup = len(up)
    cdef int ndown = len(down)

    cdef int k
    cdef int cup, cdown
    cdef int x1, left1, right1, x2, left2, right2
    cdef double E1, E2, E12, E3, E4, E34
    cdef double deltaE, log_alpha
    # do a sweep of N+1 iterations
    for k in range(0, N+1):
        # pick a boundary
        cup = np.random.randint(nup)
        x1 = up[cup]
        left1  = find_left(x1, state[s,:])
        right1 = find_right(x1, state[s,:], N)

        # pick a non-boundary
        cdown = np.random.randint(ndown)
        x2 = down[cdown]
        left2  = find_left(x2, state[s,:])
        right2 = find_right(x2, state[s,:], N)

        if left1 == left2 or right1 == right2:
            # sites are not separated
            E1 = Eseg[left1, x1]
            E2 = Eseg[x1, right1]
            E3 = Eseg[left1, x2]
            E4 = Eseg[x2, right1]
            # delete x1 boundary, create x2 boundary
            deltaE = E3 + E4 - E1 - E2
        else:
            # sites are separated by at least one boundary
            E1 = Eseg[left1, x1]
            E2 = Eseg[x1, right1]
            E12 = Eseg[left1, right1]
            E3 = Eseg[left2, x2]
            E4 = Eseg[x2, right2]
            E34 = Eseg[left2, right2]
            # delete x1 boundary, create x2 boundary
            deltaE = E3 + E4 - E34 + E12 - E1 - E2

        # acceptance probability
        log_alpha = -beta*deltaE

        # decide whether to swap states or not
        if log_alpha >= 0 or log(np.random.rand()) < log_alpha:
            up[cup] = x2
            down[cdown] = x1
            state[s, x1] = 0
            state[s, x2] = 1
            energy[s] += deltaE

    #energy[s] = score_state(Eseg, state[s, :])


def dieroll(p, n=None, cmf=False):
    if cmf:
        cmf = p
    else:
        cmf = np.r_[0.0, np.cumsum(p)]
    if n is None:
        return int(np.digitize( cmf[-1] * unifrnd(1), cmf ) - 1)
    return np.digitize( cmf[-1] * unifrnd(n) , cmf ) - 1


class AliasSampler(object):
    def __init__(self, x):
        x = np.asarray(x, dtype=float)
        self.prob, self.alias = self._build(x)

    def _build(self, x):
        # http://www.keithschwarz.com/darts-dice-coins/
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

    def sample(self, n=None):
        if n is None:
            i = randint(len(self.prob))
            return self.alias[i] if unifrnd() > self.prob[i] else i
        else:
            ii = randint(0, len(self.prob), n)
            to_alias = unifrnd(n) > self.prob[ii]
            ii[to_alias] = self.alias[ii[to_alias]]
            return ii

