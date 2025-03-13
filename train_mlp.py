import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from pathlib import Path

from data.dataloader import get_bMNIST
from utils.seed import set_seed
from utils.training_utils import train_sgd
from utils.evaluate import evaluate_MLP
from utils.config import *

from models.mlp import MLP

# Model
mlp_model = MLP().to(device)

# Extract initial weights
w0 = []
for layer in mlp_model.children():
    w0.append({
        'weight': layer.weight.data.clone(),
        'bias': layer.bias.data.clone()
    })

epochs = 20
batch_size = 100
learning_rate = 1e-2
momentum= 0.9

# training
trainloader, testloader = get_bMNIST("float32", batch_size=100)
optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=1) 
bce_loss = nn.BCEWithLogitsLoss()

def bce_loss2(outputs, labels):
    return bce_loss(outputs[0], labels)

mlp_losses, mlp_accs = train_sgd(mlp_model, epochs, optimizer, scheduler, trainloader, bce_loss2, device)
evaluate_MLP(mlp_model, testloader, device, mlp_losses, mlp_accs, plot=True, save_plot=False)

# extract and save weights
w1 = []
for layer in mlp_model.children():
    w1.append({
        'weight': layer.weight.data.clone(),
        'bias': layer.bias.data.clone()
    })
    

if SAVE_WEIGHTS:
    path = Path("./checkpoints/mlp")
    path.mkdir(parents=True, exist_ok=True)
    
    torch.save(w0, './checkpoints/mlp/w0.pt')
    torch.save(w1, './checkpoints/mlp/w1.pt')