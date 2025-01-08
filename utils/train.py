import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mlp import MLP
from models.bnn import BayesianNN
from models.bayes_linear import BayesianLinear

import time

SUPPORTED_MODELS = [BayesianNN, MLP]
params = {"count": 0, "beta1": 0.9, "lam": 1e-1, "lr": 1e-3}

def update_G(module, grad_input, grad_output):
    """
    Hook to compute and store G (gradient covariance) for a given layer.
    """
    if params['count'] % 50 == 0:
        grad = grad_output[0]
        module._G = (1 - params['beta1']) * module._G  + params['beta1'] * grad.T @ grad / grad.size(0)  # Compute G


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

    for name, layer in model.named_modules():
       if isinstance(layer, BayesianLinear) and layer.kfac:
            layer.register_backward_hook(update_G)


    for epoch in range(num_epochs):
        running_loss = 0.
        running_acc = 0.

        outputs = None
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            loss_size = loss_fn(outputs, labels)
            loss_size.backward()

            if model.kfac:
                params['count'] += 1
                # update means
                for layer in model.layers:
                    grad_ll = layer.weights.grad.detach()
                    V = grad_ll - params['lam']/(torch.exp(2 * model.p_log_sigma) * inputs.size(0))* layer.weights
                    with torch.no_grad():
                        layer.q_weight_mu += params['lr'] * layer.G_inv @ V @ layer.A_inv # TODO: flipped temporarilyi

            optimizer.step()

            preds = torch.round(torch.sigmoid(outputs[0]))
            running_loss += loss_size.item()
            running_acc += torch.sum(preds == labels).item()
            #print(running_acc)

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
