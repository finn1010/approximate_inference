import torch

def lanczos(hvp_fn, d, r):
    Q = []
    alphas = []
    betas = []
    q = torch.randn(d)
    q = q / q.norm()
    Q.append(q)
    beta_prev = 0.0

    for k in range(r):
        v = hvp_fn(Q[-1])
        alpha = torch.dot(Q[-1], v)
        alphas.append(alpha)
        v = v - alpha * Q[-1] - (beta_prev * Q[-2] if k > 0 else 0)
        beta = v.norm()
        if beta < 1e-8 or k == r - 1:
            break
        betas.append(beta)
        Q.append(v / beta)
        beta_prev = beta

    m = len(alphas)
    T = torch.zeros(m, m)

    for i in range(m):
        T[i, i] = alphas[i]
        if i + 1 < m:
            T[i, i+1] = betas[i]
            T[i+1, i] = betas[i]

    eigvals, eigvecs = torch.linalg.eigh(T)
    Qmat = torch.stack(Q[:m], dim=1)
    V = Qmat @ eigvecs

    return V, eigvals
