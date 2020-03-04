import numpy as np
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable, grad

cuda = True if torch.cuda.is_available() else False
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])
        
        
    def g(self, z, L=-1):
        """forward pass, x = g(z)"""
        if L == -1:
            L = len(self.t)
        x = z
        for ell in range(L):
            x_ = x*self.mask[ell]
            s = self.s[ell](x_)*(1 - self.mask[ell])
            t = self.t[ell](x_)*(1 - self.mask[ell])
            x = x_ + (1 - self.mask[ell]) * (x * torch.exp(s) + t)
        return x

    def f(self, x, L=-1):
        """backwards pass, z = f(x) = g^{-1}(x)"""
        if L == -1:
            L = len(self.t)
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for ell in reversed(range(L)):
            z_ = self.mask[ell] * z
            s = self.s[ell](z_) * (1-self.mask[ell])
            t = self.t[ell](z_) * (1-self.mask[ell])
            z = (1 - self.mask[ell]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob_of_x(self, x, L=-1):
        """original function"""
        if L == -1:
            L = len(self.t)
        z, logp = self.f(x, L)
        return self.prior.log_prob(z) + logp
        ## use this for uniform prior
        #return self.prior.log_prob(z)[:,0] + logp
    
    #---------------------------
    def log_prob_of_z(self, z):
        """I added this"""
        x = self.g(z)
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp
        ## use this for uniform prior
        #return self.prior.log_prob(z)[:,0] + logp

    def log_prob_of_z_minus(self, z):
        """I added this"""
        x = -self.g(z)
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp
        ## use this for uniform prior
        #return self.prior.log_prob(z)[:,0] + logp        
                
    def log_prob_of_z_sym(self, z):
        ## the naive way of writing this was prone to numerical overflow
        # return torch.log(0.5*(torch.exp(self.log_prob_of_z(z)) + torch.exp(self.log_prob_of_z_minus(z))))
        log_prob = self.log_prob_of_z(z)
        log_prob_minus = self.log_prob_of_z_minus(z)
        return log_prob + torch.log(0.5*(1 + torch.exp(log_prob_minus - log_prob)))        
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        #logp = self.prior.log_prob(z)[:,0]
        x = self.g(z)
        return x    


def log_pi(x, Sigma_inv_torch):
    """un-normalized log probability, less prone to numerical overflow"""
    term1 = -0.5*torch.diagonal(torch.mm(torch.mm(x, Sigma_inv_torch), torch.transpose(x, 1, 0)))
    y = torch.transpose(torch.abs(torch.mm(Sigma_inv_torch, torch.transpose(x, 1, 0))), 1, 0)
    term2 = torch.sum(y + torch.log(1 + torch.exp(-2*y)), 1)
    return term1 + term2 


def neg_log_prob_z(flow, Sigma_inv_torch):
    """negative log probability (in z-coordinate) computed for both torch and numpy arguments"""
    
    def logp_torch(z):
        """takes torch tensor with shape (1, dim) as input"""
        x = flow.g(z)
        return -flow.prior.log_prob(z) + flow.log_prob_of_x(x) - log_pi(x, Sigma_inv_torch)
    
    def logp_numpy(z):
        """takes numpy array with shape (dim) as input"""
        dim = z.shape[0]
        z = torch.from_numpy(z.reshape((1,dim))).float().to(device)
        x = flow.g(z)
        return (-flow.prior.log_prob(z) + flow.log_prob_of_x(x) - log_pi(x, Sigma_inv_torch)).detach().cpu().numpy()

    return [logp_torch, logp_numpy]


def neg_log_prob_x(Sigma_inv_torch):
    """negative log probability (in x-coordinate) computed for both torch and numpy arguments"""
    
    def logp_torch(x):
        """takes torch tensor with shape (1, dim) as input"""
        return - log_pi(x, Sigma_inv_torch)
    
    def logp_numpy(x):
        """takes numpy array with shape (dim) as input"""        
        dim = x.shape[0]
        x = torch.from_numpy(x.reshape((1,dim))).float().to(device)    
        return (- log_pi(x, Sigma_inv_torch)).detach().cpu().numpy()

    return [logp_torch, logp_numpy]


def Hessian(y, x):
    """general function for computing the Hessian of y(x)"""
    d = x.size()[1]
    dx, = grad(y, x, create_graph=True)
    dx_dxi = []
    
    for i in range(d-1):
        xi = torch.zeros((1,d))
        xi[0,i] = 1
        dx_dxi.append(grad(dx, x, grad_outputs=xi, retain_graph=True)[0].cpu().detach().numpy())
    xi = torch.zeros((1,d))
    xi[0,-1] = 1
    dx_dxi.append(grad(dx, x, grad_outputs=xi, retain_graph=False)[0].cpu().detach().numpy())

    return np.asarray(dx_dxi)[:,0,:]


def Hessian_index_logpi_x(flow, N_samp, Sigma_inv_torch):
    """Hessian of log(pi(x)) at x-values sampled by the flow"""
    dim = Sigma_inv_torch.size()[0]
    nu = []
    for i in range(N_samp):
        
        z = flow.prior.sample((1, 1)).reshape((1, dim))
        x = flow.g(z).detach()
        x.requires_grad = True

        ## log_pi(x)
        y = log_pi(x, Sigma_inv_torch)[0]
        hess = Hessian(y, x)
        nu.append(np.mean(np.asarray(np.linalg.eig(hess)[0] < 0, dtype=np.int)))
        
    return nu


def A(flow, Sigma_inv_torch, x_old, x_proposal):
    """Albergo et al acceptance probability"""
    
    logratios = flow.log_prob_of_x(x_old) - flow.log_prob_of_x(x_proposal) \
        - log_pi(x_old, Sigma_inv_torch) + log_pi(x_proposal, Sigma_inv_torch)

    logratios = logratios.detach().cpu().numpy()
    
    return min(1, np.exp(logratios))



def Albergo_MCMC(flow, N_step, batch_size, Sigma_inv_torch, fixed_sample_size=True):
    """MCMC algorithm of Albergo et al"""

    ## initialize the chain
    dim = flow.prior.sample((1, 1)).size()[-1]
    t = 0
    x = flow.g(flow.prior.sample((1, 1)).reshape((1, dim)))
    acceptance_list = []
    A_list = []
    x_list = []
    
    ## if the fixed_sample_size flag == True, return exactly N_samp numbers of samples
    if fixed_sample_size == True:
        ## evolve the chain
        while t < N_step:

            ## sample the flow (in batches)
            x_batches = flow.g(flow.prior.sample((batch_size, 1)).reshape((batch_size, dim)))

            ## loop over the batch
            for j in range(batch_size):
                x_proposal = x_batches[j].reshape((1, dim))

                ## accept or reject the proposal
                AA = float(A(flow, Sigma_inv_torch, x, x_proposal))
                A_list.append(AA)

                if AA > np.random.uniform(0, 1):
                    x = x_proposal
                    acceptance_list.append(1)
                    x_list.append(x.detach().cpu().numpy()[0,:])
                    t += 1                
                else:
                    acceptance_list.append(0)
    
    ## if the fixed_sample_size flag == False, perform exactly N_samp update steps
    else:
        ## evolve the chain
        while t < N_step:
            ## sample the flow (in batches)
            x_batches = flow.g(flow.prior.sample((batch_size, 1)).reshape((batch_size, dim)))

            ## loop over the batch
            for j in range(batch_size):
                x_proposal = x_batches[j].reshape((1, dim))

                ## accept or reject the proposal
                AA = float(A(flow, Sigma_inv_torch, x, x_proposal))
                A_list.append(AA)
                t += 1
                
                if AA > np.random.uniform(0, 1):
                    x = x_proposal
                    acceptance_list.append(1)
                    x_list.append(x.detach().cpu().numpy()[0,:])
                else:
                    acceptance_list.append(0)
                
                    
    return [acceptance_list, A_list, x_list]


