##########################################
## imports
##########################################
import sys, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable, grad
from utils import nvp
from utils import SK

cuda = True if torch.cuda.is_available() else False
device = 'cpu' if not cuda else 'cuda'

##########################################
## parameters
##########################################
## open up log file

dim = 16
n_disorder = 3
n_layers = 4
T_list = np.asarray(list(np.linspace(0,1,11)[1:]) + list(np.linspace(1,5,11))[1:])

n_batches = 2500 + 1
n_log = 1000
n_save = 250
batch_size = 50
AdamLR = 1e-4

path = 'data/reverseKL_N_' + str(dim) + '_nlayers_' + str(n_layers) + '/'

if not os.path.exists(path):
    os.makedirs(path)

print('torch cuda available: ', torch.cuda.is_available())
print('N = %i, n_disorder = %i, n_layers = %i' %(dim, n_disorder, n_layers))


##########################################
## set-up normalizing flow
##########################################
## s function
nets = lambda: nn.Sequential(nn.Linear(dim, dim), 
                             nn.LeakyReLU(), 

                             nn.Linear(dim, dim), 
                             nn.LeakyReLU(),                              
                             
                             nn.Linear(dim, dim),
                             nn.LeakyReLU(),                              
                             
                             nn.Linear(dim, dim),
                             nn.Tanh())


## t function
nett = lambda: nn.Sequential(nn.Linear(dim, dim), 
                             nn.LeakyReLU(), 
     
                             nn.Linear(dim, dim), 
                             nn.LeakyReLU(),                              
                             
                             nn.Linear(dim, dim),
                             nn.LeakyReLU(),                              
                             
                             nn.Linear(dim, dim))


#The choice of binary mask below was made to avoid "conspiracies" (in the drop-out sense) betweeen certain variables (i.e. between even-indexed variables $x_i$). 
def draw_mask2(p, seed=123):
    np.random.seed(seed)
    l1 = np.random.binomial(1, p, (n_layers//2, dim))
    l2 = 1 - l1
    mask = np.array([[l1[i], 1-l1[i]] for i in range(n_layers//2)]).reshape((n_layers, dim)).astype(np.float32)
    return torch.from_numpy(mask)


mask = draw_mask2(0.5)


## prior for the latent variable, i.e. p(z)
prior = distributions.MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device))

    
##########################################
## set-up SK
##########################################
## loop over disorder replicas
for k in range(0, n_disorder):

    ## loop over temperatures
    for iT in range(len(T_list)):
        
        T = T_list[iT]
        print('\nT = %.3f, seed = %i' % (T,k))
        
        ## choose a different coupling matrix for each disorder replica
        J = SK.generate_J(dim, seed=k)

        log_file = open(path + 'log_iT_' + str(iT) + '.txt','a')
        print('\nT = %.3f, seed = %i' % (T,k), file=log_file)
        log_file.close()

        beta = 1/T
        Jbar = beta*J
        Jbar_eigs = np.linalg.eig(Jbar)[0]
        bareps = beta*1e-2
        Dbar = (max(0,-min(Jbar_eigs)) + bareps)*np.eye(dim)
        Sigma_inv = Jbar + Dbar
        Sigma = np.linalg.inv(Sigma_inv)
        Sigma_torch = torch.from_numpy(Sigma).type(torch.FloatTensor).to(device)
        Sigma_inv_torch = torch.from_numpy(Sigma_inv).type(torch.FloatTensor).to(device)

        ##########################################
        ## train the flow
        ##########################################
        flow = nvp.RealNVP(nets, nett, mask, prior)
        if cuda: flow = flow.cuda()

        ## define the optimizer
        optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=AdamLR)

        loss_history = []
        iters = []

        ## loop over batches (epochs)
        for n in range(n_batches):

            ## train over minibatches        
            z = flow.prior.sample((batch_size, 1)).reshape((batch_size, dim))
            loss = (flow.log_prob_of_z_sym(z) - nvp.log_pi(flow.g(z), Sigma_inv_torch)).mean()        
            loss_history.append(loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step() 

            ## log and save results
            if n % n_log == 0:
                log_file = open(path + 'log_iT_' + str(iT) + '.txt', 'a')
                print('finished batch %i:' % (n+1), 'avg. loss = %.3f' % np.mean(loss_history), file=log_file)
                log_file.close()
            if n % n_save == 0:         
                np.save(path + 'iters', iters)
                np.save(path + 'T_list', T_list)
                np.save(path + 'loss_dim_' + str(dim) + '_iT_' + str(iT) + '_disorder_' + str(k), loss_history)
                torch.save(flow.s, path + 's_dim_' + str(dim) + '_iT_' + str(iT) + '_iter_' + str(n) + '_disorder_' + str(k))
                torch.save(flow.t, path + 't_dim_' + str(dim) + '_iT_' + str(iT) + '_iter_' + str(n) + '_disorder_' + str(k))


