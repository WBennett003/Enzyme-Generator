import torch
import torch.nn.functional as F
import torch.utils.data


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)
    
class DenoiseDiffusion:
    def __init__(self, model, n_steps, device):
        super().__init__()
        self.model = model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def q_xt_xE(self, xE, t):
        mean = 1 - gather(self.alpha_bar, t) ** 0.5 * xE
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0, t, eps=None):
        
        if eps is None:
            eps = torch.randn_like(x0, t)

        mean, var = self.q_xt_xE(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt, t, inps):
        
        eps_theta = self.model(xt, inps, t)
        eps = eps_theta
        
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5

        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps)
        
        var = gather(self.sigma2, t)

        eps = torch.randn(xt.shape, device=xt.device)
        
        E = mean + (var ** .5) * eps
        return E

    def loss(self, x0, cond, noise=None): #x0 is tuple of E, P, ADJ values
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        if noise is None:
            noise = torch.rand_like(x0)

        xt = self.q_sample(x0, t, noise)
            
        eps_theta = self.model(xt, t, cond)

        loss = F.mse_loss(noise, eps_theta)
        return loss
    
