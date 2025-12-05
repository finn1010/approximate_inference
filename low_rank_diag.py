import torch
from models import log_joint

class LowRankDiagVI(torch.nn.Module):
    def __init__(self, d, r, V, Lambda, w_ref):
        super().__init__()
        self.d, self.r = d, r

        self.register_buffer("V", V[:, :r].clone())
        self.register_buffer("Lambda", Lambda[:r].clone())
        self.register_buffer("alpha", (1.0 / torch.sqrt(self.Lambda)).clone())
        self.register_buffer("w_ref", w_ref.detach().clone())

        self.z_mean = torch.nn.Parameter(torch.zeros(r))
        self.z_log_std = torch.nn.Parameter(torch.zeros(r))
        self.log_s = torch.nn.Parameter(torch.zeros(d))

    def mean_w(self):
        return self.w_ref + (self.z_mean * self.alpha) @ self.V.T

    def _map(self, z, eps):
        lr = (z * self.alpha) @ self.V.T
        diag = eps * torch.exp(self.log_s)
        return self.w_ref + lr + diag

    def sample(self, K):
        z = self.z_mean + torch.randn(K, self.r, device=self.z_mean.device) * torch.exp(self.z_log_std)
        eps = torch.randn(K, self.d, device=self.z_mean.device)
        return self._map(z, eps)

    def kl_term(self, sigma_prior):
        sigma2 = sigma_prior ** 2
        m = self.mean_w()
        V = self.V
        a2 = (self.alpha ** 2) * torch.exp(2 * self.z_log_std)
        lr = (V * a2) @ V.T
        dvar = torch.exp(2 * self.log_s)
        tr = (lr.trace() + dvar.sum()) / sigma2
        quad = (m @ m) / sigma2
        logdet_S = torch.logdet(lr + torch.diag(dvar))
        return 0.5 * (tr + quad - self.d + self.d * torch.log(torch.tensor(sigma2)) - logdet_S)

    def elbo(self, X, y, K, sigma_prior):
        w = self.sample(K)
        loglik = torch.stack([log_joint(w[k], X, y, sigma_prior) for k in range(K)]).mean()
        return loglik - self.kl_term(sigma_prior)

    def run(self, X, y, K, sigma_prior, steps, lr):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        hist = []
        for _ in range(steps):
            opt.zero_grad()
            e = self.elbo(X, y, K, sigma_prior)
            (-e).backward()
            opt.step()
            hist.append(e.item())
        return hist

    @torch.no_grad()
    def basis_update(self, V_new, Lambda_new, w_ref_new):
        a2_old = (self.alpha ** 2) * torch.exp(2 * self.z_log_std)
        lr_old = (self.V * a2_old) @ self.V.T

        V_new = V_new[:, :self.r]
        L_new = Lambda_new[:self.r]
        a_new = 1 / torch.sqrt(L_new)

        self.V.copy_(V_new)
        self.Lambda.copy_(L_new)
        self.alpha.copy_(a_new)
        self.w_ref.copy_(w_ref_new)

        proj = torch.diag(V_new.T @ lr_old @ V_new)
        var_z = proj / (a_new ** 2)
        var_z = var_z.clamp(min=1e-8)

        self.z_mean.zero_()
        self.z_log_std.copy_(0.5 * torch.log(var_z))
