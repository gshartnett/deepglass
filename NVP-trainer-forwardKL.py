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
from utils import PT

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
NT = len(T_list)
n_epochs = 5
batch_size = 50
n_log = 50
AdamLR = 1e-4

path = 'data/forwardKL_N_' + str(dim) + '_nlayers_' + str(n_layers) + '/'
if not os.path.exists(path):
    os.makedirs(path)


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
print('torch cuda available: ', torch.cuda.is_available())
print('N = %i, n_layers = %i' %(dim, n_layers))
        
## loop over disorder replicas
for k in range(0, n_disorder):
        
    ## loop over temperatures
    for iT in range(len(T_list)):

        T = T_list[iT]
        print('\nT = %.3f, seed = %i' % (T,k))

        ## choose a different coupling matrix for each disorder replica
        J = SK.generate_J(dim, seed=k)

        beta = 1/T
        Jbar = beta*J
        Jbar_eigs = np.linalg.eig(Jbar)[0]
        bareps = beta*1e-2
        Dbar = (max(0,-min(Jbar_eigs)) + bareps)*np.eye(dim)
        Sigma_inv = Jbar + Dbar
        Sigma = np.linalg.inv(Sigma_inv)
        Sigma_torch = torch.from_numpy(Sigma).type(torch.FloatTensor).to(device)
        Sigma_inv_torch = torch.from_numpy(Sigma_inv).type(torch.FloatTensor).to(device)

        ## convert the s-data to x-data, or load the already converted x-data
        fileName = "data/PT_MCMC/xlistA_iT_" + str(iT) + "k_" + str(k) + ".npy"
        if not os.path.isfile(fileName):
            print('converting from s to x:')
            SlistA = np.load("data/PT_MCMC/SlistA_k" + str(k) + ".npy")[:,iT,:]
            xlistA = np.zeros(SlistA.shape)
            for i in range(SlistA.shape[0]):
                xlistA[i,:] = np.random.multivariate_normal(SlistA[i], Sigma)
            np.save(fileName, xlistA)

        else:
            print('loading x data')
            xlistA = np.load(fileName)    

        n_batches = xlistA.shape[0]//batch_size

        ##########################################
        ## train the flow
        ##########################################
        flow = nvp.RealNVP(nets, nett, mask, prior)
        if cuda: flow = flow.cuda()

        ## save the initial weights
        torch.save(flow.s, path + 's_dim_' + str(dim) + '_iT_' + str(iT) + '_epoch_' + str(0) + '_disorder_' + str(k))
        torch.save(flow.t, path + 't_dim_' + str(dim) + '_iT_' + str(iT) + '_epoch_' + str(0) + '_disorder_' + str(k))

        ## define the optimizer
        optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=AdamLR)

        loss_history = []
        iters = []

        ## loop over epochs
        for epoch in range(n_epochs):

            ## loop over batches
            for batch in range(n_batches):

                ## assemble the batch
                x_batch = xlistA[batch*batch_size:(batch+1)*batch_size,:]
                x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor).to(device)

                ## gradient descent for neg log likelihood
                loss = -flow.log_prob_of_x(x_batch).mean()                    
                loss_history.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print('finished epoch %i/%i, loss %.3f' % (epoch, n_epochs, np.mean(loss_history)))

        ## save
        np.save(path + 'T_list', T_list)
        np.save(path + 'loss_dim_' + str(dim) + '_iT_' + str(iT) + '_disorder_' + str(k), loss_history)
        torch.save(flow.s, path + 's_dim_' + str(dim) + '_iT_' + str(iT) + '_epoch_' + str(epoch+1) + '_disorder_' + str(k))
        torch.save(flow.t, path + 't_dim_' + str(dim) + '_iT_' + str(iT) + '_epoch_' + str(epoch+1) + '_disorder_' + str(k))

