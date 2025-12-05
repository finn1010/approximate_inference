import torch
from models import log_joint
import numpy as np


class MeanFieldVI(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.m = torch.nn.Parameter(torch.zeros(d))
        self.s = torch.nn.Parameter(0.1*torch.ones(d))

    def sample(self):
        return self.m + self.s * torch.randn_like(self.m)

    def entropy(self):
        d = self.m.shape[0]
        return 0.5 * d * (1 + np.log(2*np.pi)) + torch.sum(torch.log(self.s))

    def elbo(self, X, y, K, sigma_prior):
        val = 0.0
        for _ in range(K):
            w = self.sample()
            val += log_joint(w, X, y, sigma_prior)
        val /= K
        return val + self.entropy()

    def run(self, X, y, K=50, sigma_prior=1.0, steps=2000, lr=0.01):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history = []
        for _ in range(steps):
            opt.zero_grad()
            loss = -self.elbo(X, y, K, sigma_prior)
            loss.backward()
            opt.step()
            history.append(-loss.item())
        return history
