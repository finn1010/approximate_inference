import torch
import torch.nn as nn

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim))
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        t = z @ self.w + self.b
        h = torch.tanh(t)
        z_new = z + h.unsqueeze(-1) * self.u
        h_prime = 1 - torch.tanh(t)**2
        psi = h_prime.unsqueeze(-1) * self.w
        logdet = torch.log(torch.abs(1 + (psi * self.u).sum(dim=1)))
        return z_new, logdet
