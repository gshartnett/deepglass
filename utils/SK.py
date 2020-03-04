import numpy as np


def generate_J(dim, seed=123):
    np.random.seed(seed)
    """draw a random coupling matrix"""
    J = np.random.normal(0, 1/np.sqrt(2*dim), (dim,dim))
    J = (J + np.transpose(J))
    for i in range(dim):
        J[i,i] = 0
    return J


def energy(s, J, beta):
    """energy E(s)"""
    return -0.5*np.dot(s, np.dot(J, s))


def sample_s_given_x(x, N_samp, Sigma_inv):
    """
    Sample from p(s|x) (for A=Sigma=(\bar{J}+D)^{-1}). 
    """
    dim = len(x)
    y = np.dot(Sigma_inv, x)
    prob_s_is_1 = np.exp(y)/(2*np.cosh(y))
    samples = np.zeros((N_samp, dim), dtype=int)
    
    for i in range(N_samp):
        p = np.random.uniform(0, 1, dim)
        samples[i,:] = 2*np.asarray(prob_s_is_1 > p, dtype=int) - 1
    
    return samples


def sample_s_given_x_batch(x, Sigma_inv):
    """
    Sample from p(s|x) (for A=Sigma=(\bar{J}+D)^{-1})
    this function can handle batches of x, so it's assumed x
    has dimensions (n_batch, N)
    """
    n_batch = x.shape[0]
    dim = x.shape[1]
    
    y = np.transpose(np.dot(Sigma_inv, np.transpose(x)))
    prob_s_is_1 = np.exp(y)/(2*np.cosh(y))    
    p = np.random.uniform(0, 1, (n_batch, dim))
    
    return 2*np.asarray(prob_s_is_1 > p, dtype=int) - 1
                     

def RHS(Sigma_inv, x):
    """The RHS of the mean field equations (for A=Sigma=(\bar{J}+D)^{-1})"""
    return np.tanh(np.dot(Sigma_inv, x))


def meanfield_solver(Sigma_inv, TOL, alpha):
    """solve the mean-field equations (for A=Sigma=(\bar{J}+D)^{-1})"""
    N = Sigma_inv.shape[0]
    x = np.random.uniform(-1,1,N)
    error = 10
    
    while error > TOL:
        rhs = RHS(Sigma_inv, x)
        error = np.mean((x - rhs)**2)
        x = alpha*x + (1-alpha)*rhs
       
    return x


def average_energy(X_samp, N_samp):
    en = 0
    for i in range(X_samp.shape[0]):
        samples = SK.sample_s_given_x(X_samp[i,:], N_samp)
        en += np.mean([ SK.energy(samples[j,:], J, beta) for j in range(N_samp)])/X_samp.shape[0]
    return en/X_samp.shape[1]


def delta_energy(s, i, J):
    """energy difference when spin i is flipped"""
    return 2.0*s[i]*np.dot(J[i], s)


def metropolis_sweep(s, beta, J):
    """A single sweep of the MH algorithm"""
    
    T = 1/beta
    N = J.shape[0]
    p = np.random.uniform(0,1,N)
    accept = 0
    
    for i in range(N):
        delta_E = delta_energy(s, i, J)
    
        if delta_E <= 0:
            s[i] = - s[i]
            accept += 1
        else:
            if np.exp(-beta*delta_E) > p[i]:
                s[i] = - s[i]
                accept += 1
    
    return s, accept/N