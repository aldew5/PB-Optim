import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from pathlib import Path

from data.dataloader import get_bMNIST
from utils.seed import set_seed
from utils.train import train
from utils.evaluate import evaluate_MLP

from models.mlp import MLP

# Setup
set_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAVE_WEIGHTS = True

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
momentum=0.9

# Training
trainloader, testloader = get_bMNIST(batch_size)
optimizer = optim.SGD(mlp_model.parameters(), lr=learning_rate, momentum=momentum)
scheduler = StepLR(optimizer, step_size=10, gamma=1) 
bce_loss = nn.BCEWithLogitsLoss()
def bce_loss2(outputs, labels): # Modified loss function that plays nice with our MLP
    return bce_loss(outputs[0], labels)

mlp_losses, mlp_accs = train(mlp_model, epochs, optimizer, scheduler, trainloader, bce_loss2, device)
evaluate_MLP(mlp_model, testloader, device, mlp_losses, mlp_accs, plot=True, save_plot=False)

# Extract trained weights
w1 = []
for layer in mlp_model.children():
    w1.append({
        'weight': layer.weight.data.clone(),
        'bias': layer.bias.data.clone()
    })
    
# Save weights
if SAVE_WEIGHTS:
    # mkdir if doesn't exist
    path = Path("./checkpoints/mlp")
    path.mkdir(parents=True, exist_ok=True)
    
    # save weights
    torch.save(w0, './checkpoints/mlp/w0.pt')
    torch.save(w1, './checkpoints/mlp/w1.pt')