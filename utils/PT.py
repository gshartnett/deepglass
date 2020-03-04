#############################################
### Parallel Tempering for SK Model
import numpy as np

def energy(Si, Jij):
    """energy across all NT replicas"""
    return -0.5*np.dot(np.dot(Si, Jij), np.transpose(Si)).diagonal() #vectorized


def delta_energy(Si, Jij, i):
    """energy difference when spin i is flipped across all NT replicas"""
    NT = Si.shape[0]
    return 2.0*np.asarray([ Si[iT, i]*np.dot(Jij[i, :], Si[iT, :]) for iT in range(NT)])


def energy_test(Si, Jij):
    """
    make sure that the energy and delta_energy 
    functions are consistent with one another
    """
    _, NT, N = Si.shape

    delta_list =np.zeros(NT)
    for i in range(N):
        Sicopy = np.copy(Si)
        Sicopy[:, i] = - Sicopy[:, i]
        delta1 = energy(Sicopy, Jij) - energy(Si, Jij)
        delta2 = delta_energy(Si, Jij, i)
        delta_list += (delta1 - delta2)**2/N
    if np.sum(delta_list) > 1e-10:
        print("there's a problem!")
    return


def metropolis(Si, Jij, Tlist):
    """Metropolis-Hastings sweep (all N spins) across all NT replicas"""
    NT, N = Si.shape
    u = np.random.uniform(0, 1, (NT, N)) #random numbers used in Metropolis flip
    acceptance_ratio = np.zeros(NT)
    for iSpin in range(N):
        p = np.minimum(np.ones(NT), np.exp(-delta_energy(Si, Jij, iSpin)/Tlist))
        boolarray = p > u[:, iSpin]
        Si[boolarray, iSpin] = -Si[boolarray, iSpin]
        acceptance_ratio += boolarray/N
    return acceptance_ratio


def parallel_tempering(Si, En, Jij, Tlist):
    """parallel tempering step"""
    NT, N = Si.shape
    u = np.random.uniform(0, 1, NT)
    
    for iT in range(NT-1):
        i1 = iT
        i2 = iT + 1
        en1 = En[i1]
        en2 = En[i2]
        T1 = Tlist[i1]
        T2 = Tlist[i2]
        
        prob = min(1, np.exp(-en2/T1 - en1/T2 + en1/T1 + en2/T2))
        if prob > u[iT]:
            tmp_Si1 = Si[i1, :]
            tmp_Si2 = Si[i2, :]
            Si[i1, :] = tmp_Si2
            Si[i2, :] = tmp_Si1
            En[i1] = en2
            En[i2] = en1

    return En