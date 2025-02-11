import torch
import torch.nn as nn
from data.dataloader import get_bMNIST
from train_kfac import train

from utils.pac_bayes_loss import pac_bayes_loss
from utils.config import *
from models.bnn import BayesianNN
from optimizers.kfac3 import KFACOptimizer
from bayes_opt import BayesianOptimization


kls, bces, errs, bounds, losses = [], [], [], [], []
bce_loss = nn.BCEWithLogitsLoss()
trainloader, testloader = get_bMNIST(batch_size=100)


def pac_bayes_loss2(outputs, labels):
    return pac_bayes_loss(outputs, labels, m, b, c, pi, delta)


def train_and_eval(lr=0.01, damping=2e-3, weight_decay=0, epochs=10):
    w0 = torch.load('./checkpoints/mlp/w0.pt', weights_only=True)
    w1 = torch.load('./checkpoints/mlp/w1.pt', weights_only=True)

    net = BayesianNN(w0, w1, p_log_sigma=-6,  approx='diag').to(device)
    net.train()

    optimizer = KFACOptimizer(net, lr=lr, damping=damping, weight_decay=weight_decay)
    cur = 0
    for epoch in range(epochs):
        cur = train(epoch, optimizer, net)
    return cur


def evaluate_model(lr, damping, weight_decay):
    loss = train_and_eval(lr, damping, weight_decay)
    return loss


pbounds = {
    'lr': (1e-3, 2e-2),
    'damping': (1e-3, 1e-1),
    'weight_decay': (1e-5, 1e-4)
}

optimizer = BayesianOptimization(
    f=evaluate_model,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=5, 
    n_iter=25     
)

print("Best hyperparameters:", optimizer.max)
