import torch

def autocorr_1d(x, max_lag=200):
    x = x - x.mean()
    n = len(x)
    var = torch.var(x, unbiased=False)
    acf = []

    for lag in range(max_lag):
        if lag == 0:
            acf.append(torch.tensor(1.0, device=x.device, dtype=x.dtype))
        else:
            acf.append(torch.dot(x[:-lag], x[lag:]) / (n * var))

    return torch.stack(acf)


def ess_1d(x, max_lag=200):
    acf = autocorr_1d(x, max_lag)
    for k in range(1, max_lag):
        if acf[k] < 0:
            cutoff = k
            break
    else:
        cutoff = max_lag

    tau = 1 + 2 * acf[1:cutoff].sum()
    return len(x) / tau
