import torch

def pac_bound(train_err, BRE_init, num_steps=5):
    """
    Computes the final PAC-Bayes bound
    """
    if torch.sqrt(0.5 * BRE_init) > 1:
        return 1.0
    
    cur_B = train_err + torch.sqrt(0.5 * BRE_init)
    for i in range(num_steps):
        cur_B = newtons(cur_B, train_err, BRE_init)
    return cur_B

def newtons(p, q, c):
    """
    Newton's method to approximate the inverse KL
    """

    kl1 = normal_kl(p, q)
    kl2 = (1-q)/(1-p) - q/p

    return p - (kl1 - c)/kl2

def normal_kl(p, q):
    return q * torch.log(torch.tensor(q/p)) + (1-q) * torch.log(torch.tensor((1-q)/(1-p)))