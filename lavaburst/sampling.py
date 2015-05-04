from .core.sampling import *


def walk_sample(Eseg, beta, log_Zfwd, n_samples):
    N = len(log_Zfwd) - 1
    samples = np.zeros((n_samples, N+1), dtype=int)
    samples[:, 0] = 1
    samples[:, N] = 1

    pmf = [None]
    for i in range(1, N+1):
        pmf.append(
            AliasSampler(
                np.exp(log_Zfwd[0:i] - log_Zfwd[i] - beta*Eseg[0:i, i])))

    for k in range(n_samples):
        i = N
        samples[k, i] = 1
        while i != 0:
            i = pmf[i].sample()
            samples[k, i] = 1
    
    return samples