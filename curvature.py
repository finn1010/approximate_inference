import torch

def hvp(f_scalar, w, v):
    grad = torch.autograd.grad(f_scalar, w, create_graph=True)[0]
    hv = torch.autograd.grad(grad, w, grad_outputs=v)[0]
    return hv
