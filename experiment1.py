import torch
import matplotlib.pyplot as plt

from samplers import mala_step
from models import log_joint
from mean_field import MeanFieldVI
from low_rank_diag import LowRankDiagVI
from curvature import hvp
from lanczos import lanczos

N, d = 400, 25
SIGMA = 0.1
RANK = 10
STEPS1 = 1500
STEPS2 = 300
K = 50
LR = 0.01
UPDATE = 20

X = torch.randn(N, d)
true_weights = torch.randn(d)
y = (torch.sigmoid(X @ true_weights) > 0.5).float()

A = torch.eye(d) + 0.3 * torch.randn(d, d)
A = A @ A.T
X2 = X @ A
y2 = (torch.sigmoid(X2 @ true_weights) > 0.5).float()

mf = MeanFieldVI(d)
mf.run(X, y, K=K, sigma_prior=SIGMA, steps=STEPS1, lr=LR)

m0 = mf.m.detach().requires_grad_(True)

def hvp_d1(v):
    return hvp(-log_joint(m0, X, y, SIGMA), m0, v)

V1, L1 = lanczos(hvp_d1, d, RANK)
s = torch.argsort(L1)
V1, L1 = V1[:, s], L1[s]


lr1 = LowRankDiagVI(d, RANK, V1, L1, mf.m.detach())
lr1.run(X, y, K=K, sigma_prior=SIGMA, steps=STEPS1, lr=LR)

lr_online = LowRankDiagVI(d, RANK, V1, L1, mf.m.detach())
lr_online.load_state_dict(lr1.state_dict())
opt_on = torch.optim.Adam(lr_online.parameters(), lr=LR)
elbo_online = []

for i in range(STEPS2):
    opt_on.zero_grad()
    e = lr_online.elbo(X2, y2, K, SIGMA)
    (-e).backward()
    opt_on.step()
    elbo_online.append(e.item())

    if (i + 1) % UPDATE == 0:
        m = lr_online.mean_w().detach().requires_grad_(True)

        def hvp_d2(v):
            return hvp(-log_joint(m, X2, y2, SIGMA), m, v)

        V2, L2 = lanczos(hvp_d2, d, RANK)
        s2 = torch.argsort(L2)
        lr_online.basis_update(V2[:, s2], L2[s2], m.detach())


lr_cold = LowRankDiagVI(d, RANK, V1, L1, mf.m.detach())
opt_cd = torch.optim.Adam(lr_cold.parameters(), lr=LR)
elbo_cold = []

for _ in range(STEPS2):
    opt_cd.zero_grad()
    e = lr_cold.elbo(X2, y2, K, SIGMA)
    (-e).backward()
    opt_cd.step()
    elbo_cold.append(e.item())


m_fin = lr_online.mean_w().detach().requires_grad_(True)

def hvp_fin(v):
    return hvp(-log_joint(m_fin, X2, y2, SIGMA), m_fin, v)

V2f, L2f = lanczos(hvp_fin, d, RANK)
s3 = torch.argsort(L2f)
V2f, L2f = V2f[:, s3], L2f[s3]

print("Top curvature domain1:", L1[-5:])
print("Top curvature domain2:", L2f[-5:])

U, S, _ = torch.linalg.svd(V1.T @ V2f)
principle_angles = torch.acos(torch.clamp(S, -1, 1))

print("Principal angles domain 1 curvature vs domain 2:", principle_angles)

plt.plot(elbo_cold, label="cold-start")
plt.plot(elbo_online, label="online")
plt.legend()
plt.tight_layout()
plt.show()

m1 = lr1.mean_w()
m_online = lr_online.mean_w()
m_cold = lr_cold.mean_w()

print("Change in posterior mean compared to domain 1 LR-VI:")
print("Online:", float(torch.norm(m_online - m1)))
print("Cold-start:", float(torch.norm(m_cold - m1)))

def mala_chain(w0):
    w = w0.detach().requires_grad_(True)
    samples = []

    burn = 4000
    for i in range(8000):
        w, accepted = mala_step(w, X2, y2, eps=1e-3, sigma=SIGMA)
        if i >= burn:
            samples.append(w.detach())
    return torch.stack(samples)

mala_samples = mala_chain(m_online)
mala_mean = mala_samples.mean(0)


def mean_err(m):
    return float(torch.norm(m - mala_mean))

print("Mean error vs MALA mean (domain2):")
print("Online LR-VI:",    mean_err(m_online))
print("Cold-start LR-VI:", mean_err(m_cold))
print("LR-VI domain1:",   mean_err(m1))
print("MFVI domain1:",    mean_err(mf.m))


V = lr_online.V
alpha2 = lr_online.alpha ** 2
sigma_z2 = torch.exp(2 * lr_online.z_log_std)
a2 = alpha2 * sigma_z2
Sigma_lr = (V * a2) @ V.T

dvar = torch.exp(2 * lr_online.log_s)
Sigma_diag = torch.diag(dvar)
Sigma_q = Sigma_lr + Sigma_diag

evals_q, evecs_q = torch.linalg.eigh(Sigma_q)
idx_q = torch.argsort(evals_q, descending=True)
Vq = evecs_q[:, idx_q[:RANK]]

U2, S2, _ = torch.linalg.svd(V2f.T @ Vq)
angles_q = torch.acos(torch.clamp(S2, -1.0, 1.0))

print("\nPrincipal angles between covariance and Hessian:")
print(angles_q)

print("\n'i'th eigenvalue of Hessian * variance of q(w):")
for i in range(RANK):
    u = V2f[:, i]
    lam = L2f[i]
    var_q = (u @ Sigma_q @ u)
    print(f"i={i}: {float(lam * var_q)}")