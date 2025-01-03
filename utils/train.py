import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mlp import MLP
from models.bnn import BayesianNN

import time

SUPPORTED_MODELS = [BayesianNN, MLP]

def train(model: nn.Module,
          num_epochs: int, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler, 
          trainloader: DataLoader, 
          loss_fn: nn.Module,
          device: str) -> tuple[list[float], list[float]]:
    #assert any(isinstance(model, model_type) for model_type in SUPPORTED_MODELS), f"Model type {model.__class__.__name__} not supported."
    
    m = len(trainloader.dataset)
    start_t = time.time()
    
    losses = []
    accs = []
    
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.
        running_acc = 0.

        for inputs, labels in trainloader:
            optimizer.zero_grad()
            
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            loss_size = loss_fn(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            preds = torch.round(torch.sigmoid(outputs[0]))
            running_loss += loss_size.item()
            running_acc += torch.sum(preds == labels).item()

        running_loss /= m
        running_acc /= m
        losses.append(running_loss)
        accs.append(running_acc)
        
        scheduler.step()

        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(running_loss),
            'acc_train: {:.4f}'.format(running_acc),
            'time: {:.4f}s'.format(time.time() - start_t))

    return losses, accs
