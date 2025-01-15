import torch
import torch.nn as nn

bce_loss = nn.BCEWithLogitsLoss()

def pac_bayes_loss(outputs, labels, m, b, c, pi, delta):
    """
    Computes loss for PAC-Bayes learning algorithm
    """
    preds, kl, log_lam = outputs
    # log_lam is log std, we want variance so 2*log_lam
    BRE = (kl + 2 * torch.log(b * (torch.log(c) - 2 * log_lam)) + torch.log(m * pi ** 2) - torch.log(6 * delta)) / (m - 1)
    
    return bce_loss(preds, labels) + torch.sqrt(0.5 * BRE)