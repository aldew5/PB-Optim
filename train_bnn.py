import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import math
from pathlib import Path

from data.dataloader import get_bMNIST
from utils.seed import set_seed
from utils.pac_bayes_loss import pac_bayes_loss
from utils.train import train
from utils.evaluate import evaluate_BNN

from models.bnn import BayesianNN

# Setup
set_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAVE_WEIGHTS = True
LOAD_DATA = False

# Load trained weights
w0 = torch.load('./checkpoints/mlp/w0.pt', weights_only=True)
w1 = torch.load('./checkpoints/mlp/w1.pt', weights_only=True)

# Model
bnn_model = BayesianNN(w0, w1, p_log_sigma=-6).to(device)

# Hyperparameters
epochs = 40
batch_size = 100
learning_rate = 1e-3

delta = torch.tensor(0.025, dtype=torch.float32).to(device)
m = torch.tensor(50000, dtype=torch.float32).to(device)
b = torch.tensor(100, dtype=torch.float32).to(device)
c = torch.tensor(0.1, dtype=torch.float32).to(device)
pi = torch.tensor(math.pi, dtype=torch.float32).to(device)

# Training
optimizer = optim.Adam(bnn_model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=20, gamma=1.0)  # Decay by 0.5 every 10 epochs
trainloader, testloader = get_bMNIST(batch_size)
def pac_bayes_loss2(outputs, labels):
    return pac_bayes_loss(outputs, labels, m, b, c, pi, delta)

if not LOAD_DATA:
    bnn_losses, bnn_accs = train(bnn_model, epochs, optimizer, scheduler, trainloader, pac_bayes_loss2, device)

    N_samples = 10
    plot = True
    save_plot = False
    evaluate_BNN(bnn_model, trainloader, testloader, delta, b, c, N_samples, device, bnn_losses, bnn_accs, plot=plot, save_plot=save_plot)
else:
    params = torch.load('./checkpoints/bnn/baseline.pt', weights_only=True)
    bnn_model.load_state_dict(params)
    
    N_samples = 10
    plot = False
    save_plot = False
    evaluate_BNN(bnn_model, trainloader, testloader, delta, b, c, N_samples, device, plot=plot, save_plot=save_plot)
    

# Save weights
if SAVE_WEIGHTS:
    # mkdir if doesn't exist
    path = Path("./checkpoints/bnn")
    path.mkdir(parents=True, exist_ok=True)
    
    # save weights
    name = 'baseline'
    torch.save(bnn_model.state_dict(), f'./checkpoints/bnn/{name}.pt')