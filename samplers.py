import torch 
from models import log_joint

def ula_step(w, X, y, eps, sigma):

    logp = log_joint(w, X, y, sigma)
    logp.backward()

    with torch.no_grad():
        prop = w + 0.5 * eps * w.grad + torch.sqrt(torch.tensor(eps)) * torch.randn_like(w)

    return prop.detach().requires_grad_(True)


def mala_step(w, X, y, eps, sigma):

    logp_curr = log_joint(w, X, y, sigma)
    logp_curr.backward()
    grad = w.grad.detach().clone()

    with torch.no_grad():
        mean_prop = w + 0.5 * eps * grad
        prop = mean_prop + (eps ** 0.5) * torch.randn_like(w)

    prop = prop.detach().requires_grad_(True)
    logp_prop = log_joint(prop, X, y, sigma)
    logp_prop.backward()
    grad_prop = prop.grad.detach().clone()

    log_q_prop = -(1/(2*eps)) * torch.sum((prop - (w + 0.5*eps*grad))**2)
    log_q_curr = -(1/(2*eps)) * torch.sum((w - (prop + 0.5*eps*grad_prop))**2)

    log_alpha = (logp_prop - logp_curr) + (log_q_curr - log_q_prop)
    alpha = torch.exp(torch.clamp(log_alpha, max=0.0))

    accept = torch.rand(()) < alpha
    new_w = prop if accept else w
    return new_w.detach().requires_grad_(True), accept
