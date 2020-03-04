import os, time
import numpy as np
import matplotlib.pyplot as plt
from utils import SK
from utils import PT


#### Initializations
K = 3 #number of disoder replicas to take
NT = 20 #number of thermal replicas to take
N = 16 #number of Ising spins
dim = N

T = 10**4 #number of MC samples

Tlist = np.asarray(list(np.linspace(0,1,11)[1:]) + list(np.linspace(1,5,11))[1:])
path = "data/PT_MCMC/"
if not os.path.exists(path):
    os.makedirs(path)
np.save(path + "Tlist", Tlist)


#### Main Loop
## loop over disorder
for k in range(K):
    time1 = time.time()

    ## initialize data lists
    SlistA = np.zeros((T, NT, N), dtype=np.int32)
    SlistB = np.zeros((T, NT, N), dtype=np.int32)
    Enlist = np.zeros((T, NT))
    qlist = np.zeros((T, NT))
    Jlist = np.zeros((N, N))
    acceptanceA = np.zeros((T, NT))
    acceptanceB = np.zeros((T, NT))

    ## build the Jij coupling matrix
    ## careful here: the sum of two gaussians is a gaussian with variance 2\sigma^2
    #np.random.seed(k)
    Jij = SK.generate_J(dim, seed=k)
    Jlist[:, :] = Jij

    ## initialize the spins
    SiA = 2*np.random.binomial(1, 0.3, (NT, N)) - 1
    SiB = 2*np.random.binomial(1, 0.3, (NT, N)) - 1

    ## main MCMC loop
    for t in range(T):

        ## metropolis step
        acceptanceA[t, :] = PT.metropolis(SiA, Jij, Tlist)
        acceptanceB[t, :] = PT.metropolis(SiB, Jij, Tlist)

        ## compute the energies
        EnA = PT.energy(SiA, Jij)
        EnB = PT.energy(SiB, Jij)

        ## parallel tempering
        PT.parallel_tempering(SiA, EnA, Jij, Tlist)
        PT.parallel_tempering(SiB, EnB, Jij, Tlist)

        ## sampling
        SlistA[t, :, :] = SiA       
        SlistB[t, :, :] = SiB
        Enlist[t, :] += EnA
        qlist[t, :] += np.sum(SiA*SiB, axis=1)/N

    print("disorder replica # = ", k, "time elapsed = ", round(time.time() - time1), "seconds")

    ## save
    np.save(path + "SlistA_k" + str(k), SlistA)
    #np.save(path + "SlistB_k" + str(k), SlistB)
    np.save(path + "Jlist_k" + str(k), Jlist)
    np.save(path + "Enlist_k" + str(k), Enlist)
    np.save(path + "qlist_k" + str(k), qlist)