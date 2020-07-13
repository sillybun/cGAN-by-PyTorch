import torch
import torch.nn as nn
import numpy as np

def get_nll(x, parzen, batch_size=10):

    N = x.shape[0]
    n_batchs = int(np.ceil(N / batch_size))
    nlls = []

    for i in range(n_batchs):
        nll = parzen(x[i * n_batchs: (i + 1) * n_batchs])
        nlls.extend(nll.detach().numpy())

    return np.array(nlls)

def log_mean_exp(a):

    max_value, _ = a.max(1)
    return max_value + torch.log(torch.exp(a - max_value.unsqueeze(1)).mean(1))

class Parzen(nn.Module):
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.mu.shape[0], 1)
        mu = self.mu.unsqueeze(0)

        a = (x - mu) / self.sigma

        E = log_mean_exp(-0.5 * (a ** 2).sum(2))
        Z = mu.shape[0] * torch.log(self.sigma * np.sqrt(np.pi * 2))

        return E - Z

def cross_validate_sigma(samples, data, sigmas, batch_size):
    lls = []
    for sigma in sigmas:
        # print(sigma)
        parzen = Parzen(samples, sigma)
        tmp = get_nll(data, parzen, batch_size = batch_size)
        lls.append(tmp.mean())

    ind = np.argmax(lls)
    return sigmas[ind]

def parzen_nll(samples, valid_data, test_data, start_sigma=-1, end_sigma=0, cross_val_number=100, batch_size=10):
    sigma_samples = np.logspace(start_sigma, end_sigma, num=cross_val_number)
    sigma = cross_validate_sigma(samples, valid_data, sigma_samples, batch_size)
    print("Sigma", sigma)

    parzen = Parzen(samples, sigma)

    ll = get_nll(test_data, parzen, batch_size=batch_size)

    se = ll.std() / test_data.shape[0]

    print("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))
