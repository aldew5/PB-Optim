

import torch
from torch.optim.lr_scheduler import StepLR
from singd.optim.optimizer import SINGD

import math
from pathlib import Path

from data.dataloader import get_bMNIST
from utils.seed import set_seed
from utils.pac_bayes_loss import pac_bayes_loss
from utils.train import train
from utils.evaluate import evaluate_BNN
from optimizers.vogn.vogn import VOGN

from models.bnn import BayesianNN
from utils.config import *

# Setup
set_seed(42)

# Load trained weights
w0 = torch.load('./checkpoints/mlp/w0.pt', weights_only=True)
w1 = torch.load('./checkpoints/mlp/w1.pt', weights_only=True)

# INIT SETTINGS:
# prior mean: w0 (MLP random init)
# prior var: lambda = e^{-6}
# posterior mean: w1 (MLP learned weights)
bnn_model = BayesianNN(w0, w1, p_log_sigma=-6,  approx='diagonal').to(device)

# Hyperparameters
epochs = 10
batch_size = 100
learning_rate = 1e-2
damping = 1e-1


# Training
optimizer = VOGN()
scheduler = StepLR(optimizer, step_size=30, gamma=1.0)  # Decay by 0.5 every 10 epochs
trainloader, testloader = get_bMNIST(batch_size)

def pac_bayes_loss2(outputs, labels):
    return pac_bayes_loss(outputs, labels, m, b, c, pi, delta)

if not LOAD_DATA:
    bnn_losses, bnn_accs = train(bnn_model, epochs, optimizer, scheduler, trainloader, pac_bayes_loss2, device)

    N_samples = 10
    plot = True
    save_plot = False
    evaluate_BNN(bnn_model, trainloader, testloader, delta, delta_prime, b, c, N_samples, device, bnn_losses, bnn_accs, plot=plot, save_plot=save_plot)
else:
    params = torch.load('./checkpoints/bnn/baseline.pt', weights_only=True)
    #print("HERE", params)
    bnn_model.load_state_dict(params)
    
    N_samples = 10
    plot = False
    save_plot = False
    evaluate_BNN(bnn_model, trainloader, testloader, delta, delta_prime, b, c, N_samples, device, plot=plot, save_plot=save_plot)
    

# Save weights
if SAVE_WEIGHTS:
    # mkdir if doesn't exist
    path = Path("./checkpoints/bnn")
    path.mkdir(parents=True, exist_ok=True)
    
    # save weights
    name = 'diagonal'
    torch.save(bnn_model.state_dict(), f'./checkpoints/bnn/{name}.pt')