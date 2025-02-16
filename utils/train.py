import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mlp import MLP
from models.bnn import BayesianNN

import time

SUPPORTED_MODELS = [BayesianNN, MLP]
params = {"count": 0, "beta1": 1e-2, "lam": 1e-1}
bce_loss = nn.BCEWithLogitsLoss()


# Hook for computing G with KFAC
#def update_G(module, grad_input, grad_output):
#    """
#    Hook to compute and store G (gradient covariance) for a given layer.
#    """
#    if params['count'] % 50 == 0:
#        grad = grad_output[0].detach()
#        module._G = (1 - params['beta1']) * module._G  + params['beta1'] * grad.T @ grad / grad.size(0)


def train(model: nn.Module,
          num_epochs: int, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler, 
          trainloader: DataLoader, 
          loss_fn: nn.Module,
          device: str) -> tuple[list[float], list[float]]:
    
    m = len(trainloader.dataset)
    start_t = time.time()
    
    losses = []
    errors = []
    kls = []
    bces = []

    # hook to retrieve gradients for updating G in kfac layers
    #if model.approx == 'kfac' or model.approx == 'noisy-kfac':
    #    for layer in model.layers:
     #       layer.register_full_backward_hook(update_G)

    
    model.train()

    #if model.approx == 'kfac':
    #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

            #for name, param in model.named_parameters():
                #print(f"{name}.grad: {param.grad}")

            optimizer.step()

            preds = torch.round(torch.sigmoid(outputs[0]))
            running_loss += loss_size.item()
            running_acc += torch.sum(preds == labels).item()

            if model.approx == 'noisy-kfac':
                params['count'] += 1

        running_loss /= m
        running_acc /= m
        losses.append(running_loss)
        bces.append(bce_loss(preds, labels).clone().detach())
        errors.append(1-running_acc)
        kls.append(outputs[1].clone().detach())
        
        scheduler.step()
        print("KL:", outputs[1])
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(running_loss),
            'acc_train: {:.4f}'.format(running_acc),
            'time: {:.4f}s'.format(time.time() - start_t))

    return losses, errors, kls, bces
