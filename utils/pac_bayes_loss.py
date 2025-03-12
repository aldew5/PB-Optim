import torch
import torch.nn as nn
from utils.config import *

bce_loss = nn.BCEWithLogitsLoss()

def pac_bayes_loss(outputs, labels, m, b, c, pi, delta, kfac=False):
    """
    Computes loss for PAC-Bayes learning algorithm using Theorem D.2 from Dziugaite et al. 2020.
    """
    preds, kl, log_lam = outputs

    # optimization is also over log lam
    if not kfac:
        # log_lam is log std, we want variance so 2*log_lam
        # ensure separate gradient for log_lam
        BRE = (kl + 2 * torch.log(b * (torch.log(c) - 2 * log_lam)) + torch.log(m * pi ** 2) - torch.log(6 * delta)) / (m - 1)
    
        # TODO: the paper suggests a slightly different BRE
        val = torch.min(torch.sqrt(0.5 * BRE), BRE + torch.sqrt(BRE * (BRE + 2* bce_loss(preds, labels))))
    # fixed kfactored posterior
    else:
        val = torch.sqrt((kl + torch.log(m/delta))/(2*(m-1)))

    # weak bound on KL^{-1}
    return bce_loss(preds, labels) + val


def pac_bayes_loss2(outputs, labels, kfac=False):
    return pac_bayes_loss(outputs, labels, m, b, c, pi, delta, kfac)
