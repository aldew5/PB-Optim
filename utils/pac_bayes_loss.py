import torch
import torch.nn as nn

bce_loss = nn.BCEWithLogitsLoss()

def pac_bayes_loss(outputs, labels, m, b, c, pi, delta, kl_weight=1.5):
    """
    Computes loss for PAC-Bayes learning algorithm using Theorem D.2 from Dziugaite et al. 2020.
    """
    preds, kl, log_lam = outputs
    # log_lam is log std, we want variance so 2*log_lam
    # ensure separate gradient for log_lam
    print("KL IN LOSS", kl)
    BRE = (kl + 2 * torch.log(b * (torch.log(c) - 2 * log_lam)) + torch.log(m * pi ** 2) - torch.log(6 * delta)) / (m - 1)

    # TODO: the paper suggests a slightly different BRE
    val = torch.min(torch.sqrt(0.5 * BRE), BRE + torch.sqrt(BRE * (BRE + 2* bce_loss(preds, labels))))

    # weak bound on KL^{-1}
    return bce_loss(preds, labels) + kl_weight * val
