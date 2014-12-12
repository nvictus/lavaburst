import cython
import numpy as np
cimport numpy as np
from libc.math cimport log, exp

where = np.flatnonzero


def metropolis(
        np.ndarray[np.int_t, ndim=1] initial_state,
        np.ndarray[np.int_t, ndim=1] positions,
        np.ndarray[np.double_t, ndim=2] Wseg,
        double beta, 
        int num_sweeps):

    cdef int N = initial_state.size
    cdef int M = positions.size
    cdef np.ndarray[np.int_t, ndim=2] state = np.zeros((num_sweeps,N), dtype=int)
    cdef np.ndarray[np.int_t, ndim=2] events = np.zeros((num_sweeps,M), dtype=int)
    cdef np.ndarray[np.double_t, ndim=2] gains = np.zeros((num_sweeps,M), dtype=float)
    cdef np.ndarray[np.int_t, ndim=1] random_ints = np.random.random_integers(1,M-1, size=(M,)) 
    state[0,:] = initial_state

    cdef int s
    for s in range(1,num_sweeps):
        metropolis_sweep(s, M, state, events, gains, positions, random_ints, Wseg, beta)

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
        np.ndarray[np.double_t, ndim=2] Wseg,
        double beta):

    cdef int i, j, k, here, left, right
    cdef double W1, W2, W12
    cdef double gain, loglik
    state[s, :] = state[s-1, :]
    # do a sweep of M iterations
    for k in range(0, M):

        # pick a random site (never the first)
        i = random_ints[k]
        here = positions[i]
        if i == 1:
            left = positions[0]
        else:
            for j in range(i-1, 0, -1):
                left = positions[j]
                if state[s,left] == 1:
                    break
        if i == M-1:
            right = positions[M-1]+1
        else:
            for j in range(i+1, len(positions)):
                right = positions[j]
                if state[s,right] == 1:
                    break

        # propose to delete a boundary (i.e. merge two communities)
        # or create a boundary (i.e. split a community)
        W1  = Wseg[left, here]
        W2  = Wseg[here, right]
        W12 = Wseg[left, right]
        if state[s,here] == 1: 
            gain = W12 - W1 - W2
        else: 
            gain = W1 + W2 - W12

        # acceptance probability (deltaE = -m*deltaQ, loglik = -beta*deltaE)
        loglik = beta*gain

        # decide whether to toggle the boundary or not (Metropolis)
        if gain >= 0 or log(np.random.rand()) < loglik:
            state[s,here] = 0 if state[s,here] == 1 else 1
            events[s,k] = i
            gains[s,k] = gain
        else:
            events[s,k] = 0
            gains[s,k] = 0.0



def cop(
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
    initial_energy = score_state(Eseg, initial_state)
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
        sweep_COP(s, N, Eseg, beta, up, down, state, energy)

    return state, energy

def score_state(Eseg, s):
    domains = zip(s[:-1], s[1:])
    return sum([Eseg[a,b] for a,b in domains])

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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef sweep_COP(
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












# def kawasaki(
#        np.ndarray[np.int_t, ndim=1] initial_state,
#        np.ndarray[np.int_t, ndim=1] positions,
#        np.ndarray[np.double_t, ndim=2] Eseg,
#        double beta, 
#        int n_sweeps):

#     cdef int N = initial_state.size
#     cdef int M = positions.size
#     cdef np.ndarray[np.int_t, ndim=2] state = np.zeros((n_sweeps,N), dtype=int)
#     cdef np.ndarray[np.int_t, ndim=2] events = np.zeros((n_sweeps,M), dtype=int)
#     cdef np.ndarray[np.double_t, ndim=2] gains = np.zeros((n_sweeps,M), dtype=float)
#     #cdef np.ndarray[np.int_t, ndim=1] random_ints = np.random.random_integers(1,M-1, size=(M,)) 
#     state[0,:] = initial_state

#     cdef int s
#     for s in range(1, n_sweeps):
#         sweep_kawasaki(s, M, state, events, gains, positions, Eseg, beta)

#     return state, events, gains




# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef sweep_kawasaki(
#         int s,
#         int M,
#         np.ndarray[np.int_t, ndim=2] state,
#         np.ndarray[np.int_t, ndim=2] events,
#         np.ndarray[np.double_t, ndim=2] gains,
#         np.ndarray[np.int_t, ndim=1] positions,
#         np.ndarray[np.double_t, ndim=2] Eseg,
#         double beta):
#     cdef int i, j, k, ihere, ileft, iright, jhere, jleft, jright
#     cdef double E1, E2, E12, E3, E4, E34
#     cdef double deltaE, log_alpha

#     # do a sweep of M iterations
#     for k in range(0, M):
#         # sample a random site (never the first 0 or last M)
#         i = np.random.randint(1, M-1)
#         ihere = positions[i]
#         ileft = find_left(s, state, positions, i)
#         iright = find_right(s, state, positions, i)

#         # sample a site with the opposite state
#         while True:
#             j = np.random.randint(1, M-1)
#             jhere = positions[j]
#             if state[s, ihere] != state[s, jhere]:
#                 break
#         jleft = find_left(s, state, positions, j)
#         jright = find_right(s, state, positions, j)

#         if jhere > ileft or jhere < iright:
#             # sites are adjacent
#             E1 = Eseg[ileft, ihere]
#             E2 = Eseg[ihere, iright]
#             E3 = Eseg[jleft, jhere]
#             E4 = Eseg[jhere, jright]

#             if state[s, ihere] == 1:
#                 # delete i, create j
#                 deltaE = E3 + E4 - E1 - E2
#             else:
#                 # create i, delete j
#                 deltaE = E1 + E2 - E3 - E4
#         else:
#             # sites are separated by at least one boundary
#             E1 = Eseg[ileft, ihere]
#             E2 = Eseg[ihere, iright]
#             E12 = Eseg[ileft, iright]
#             E3 = Eseg[jleft, jhere]
#             E4 = Eseg[jhere, jright]
#             E34 = Eseg[jleft, jright]

#             if state[s, ihere] == 1:
#                 # delete i (merge), create j (split)
#                 deltaE = E12 - E1 - E2 + E3 + E4 - E34 
#             else:
#                 # create i (split), delete j (merge)
#                 deltaE = E1 + E2 - E12 + E34 - E3 - E4 

#         # acceptance probability
#         log_alpha = -beta*deltaE

#         # decide whether to swap states or not (Kawasaki)
#         if log_alpha >= 0 or log(np.random.rand()) < log_alpha:
#             if state[s, ihere] == 1:
#                 state[s, ihere] = 0
#                 state[s, jhere] = 1
#             else:
#                 state[s, ihere] = 1
#                 state[s, jhere] = 0 
#             events[s,k] = i
#             gains[s,k] = -deltaE
#         else:
#             events[s,k] = 0
#             gains[s,k] = 0.0



# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef sweep(
#       int s,
#       int M,
#       np.ndarray[np.int_t, ndim=2] state,
#       np.ndarray[np.int_t, ndim=2] events,
#       np.ndarray[np.double_t, ndim=2] gains,
#       np.ndarray[np.int_t, ndim=1] positions,
#       np.ndarray[np.int_t, ndim=1] random_ints,
#       np.ndarray[np.double_t, ndim=2] scorer,
#       double beta):

#     cdef int i, j, k, here, left, right
#     cdef double F1, F2, F12
#     cdef double deltaF, log_alpha

#     # do a sweep of M iterations
#     for k in range(0, M):

#         # pick a random site (never the first)
#         i = random_ints[k]
#         here = positions[i]
#         if i == 1:
#             left = positions[0]
#         else:
#             for j in range(i-1, 0, -1):
#                 left = positions[j]
#                 if state[s,left] == 1:
#                     break
#         if i == M-1:
#             right = positions[M-1]+1
#         else:
#             for j in range(i+1, len(positions)):
#                 right = positions[j]
#                 if state[s,right] == 1:
#                     break

#         # propose to delete a boundary (i.e. merge two communities)
#         # or create a boundary (i.e. split a community)
#         F1 = -scorer[left, here]
#         F2 = -scorer[here, right]
#         F12 = -scorer[left, right]
#         if state[s,here] == 1: 
#             deltaF = F12 - F1 - F2 #merge
#         else: 
#             deltaF = F1 + F2 - F12 #split


#         # acceptance probability (deltaE = -m*deltaQ, loglik = -beta*deltaE)
#         #loglik = beta*gain #beta*m*gain
#         log_alpha = -beta*deltaF

#         # decide whether to toggle the boundary or not (Metropolis)
#         if log_alpha >= 0 or log(np.random.rand()) < log_alpha:
#             state[s,here] = 0 if state[s,here] == 1 else 1
#             events[s,k] = i
#             gains[s,k] = -deltaF
#         else:
#             events[s,k] = 0
#             gains[s,k] = 0.0
