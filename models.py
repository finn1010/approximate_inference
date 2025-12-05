import torch

def log_lik(w, X, y):
    logits = X @ w
    return torch.sum(
        y * torch.nn.functional.logsigmoid(logits) +
        (1-y) * torch.nn.functional.logsigmoid(-logits)
    )

def log_prior(w, sigma):
    return (-1/(2*sigma**2) * (w @ w))

def log_joint(w, X, y, sigma):
    return log_lik(w, X, y) + log_prior(w, sigma)