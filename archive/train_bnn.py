

import torch
import torch.optim as optim
from optimizers.noisy_kfac import NoisyKFAC
from torch.optim.lr_scheduler import StepLR
from pathlib import Path

from data.dataloader import get_bMNIST
from utils.seed import set_seed
from utils.pac_bayes import pac_bayes_loss
from utils.training_utils import train
from utils.evaluate import evaluate_BNN

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
#optimizer = optim.Adam(bnn_model.parameters(), lr=learning_rate)
optimizer = optimizer =NoisyKFAC(bnn_model, t_stats=10, t_inv=100,  lr=0.019908763029878117, eps=0.09398758455968932, weight_decay=0)
scheduler = StepLR(optimizer, step_size=30, gamma=1.0)  # Decay by 0.5 every 10 epochs
trainloader, testloader = get_bMNIST(batch_size)

def pac_bayes_loss2(outputs, labels):
    return pac_bayes_loss(outputs, labels, m, b, c, pi, delta)

if not LOAD_DATA:
    bnn_losses, bnn_errors, kls, bces = train(bnn_model, epochs, optimizer, scheduler, trainloader, pac_bayes_loss2, device)

    N_samples = 10
    plot = True
    save_plot = False
    evaluate_BNN(bnn_model, trainloader, testloader, delta, delta_prime, b, c, N_samples, device, bnn_losses, bnn_errors, kls, bces, plot=plot, save_plot=save_plot)
else:
    params = torch.load('./checkpoints/bnn/baseline.pt', weights_only=True)
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