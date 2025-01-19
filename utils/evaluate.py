import torch
import matplotlib.pyplot as plt
import math

from models.mlp import MLP
from models.bnn import BayesianNN
from utils.newtons import pac_bound

def evaluate_MLP(model: MLP, testloader, device, losses: list[float] = None, accs: list[float] = None, plot=False, save_plot=False):
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs[0]))
            test_acc += torch.sum(preds == labels).item()
    
    test_acc /= len(testloader.dataset)
    print(f'acc_test: {test_acc}')
    
    fig, ax1 = plt.subplots()

    # Plot BNN loss on the first y-axis
    ax1.plot(losses, color='tab:blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MLP Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis to plot BNN accuracy
    ax2 = ax1.twinx()
    ax2.plot(accs, color='tab:orange')
    ax2.set_ylabel('MLP Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('MLP Loss and Accuracy')
    
    if plot:
        plt.show()
    if save_plot:
        plt.savefig(f'MLP_loss_acc.png')
    
    return test_acc
    
def evaluate_BNN(model: BayesianNN, trainloader, testloader, delta, b, c, N_samples, device, losses: list[float] = None, 
                 accs: list[float] = None, plot=False, save_plot=False):
    """
    Discretizes log sigma which we treat as a continuous parameter during optimization such that KL is maximized. Then 
    computes the final training error, sampling N times, and the resulting PAC-Bayes bound.
    """

    model.eval()
    model.flag = 'eval'
    
    # discretize prior std
    j = b * (torch.log(c) - 2 * model.p_log_sigma)
    
    j_up = torch.ceil(j)
    j_down = torch.floor(j)
    
    p_log_sigma_up = 0.5 * (torch.log(c) - j_up / b)
    p_log_sigma_down = 0.5 * (torch.log(c) - j_down / b)
    
    kl_up = model.kl_divergence(p_log_sigma_up)
    kl_down = model.kl_divergence(p_log_sigma_down)
    
    if kl_up < kl_down:
        p_log_sigma_disc = p_log_sigma_up
        kl_disc = kl_up
    else:
        p_log_sigma_disc = p_log_sigma_down
        kl_disc = kl_down
    
    train_acc = 0
    # compute average empirical error for N_samples
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = torch.stack([model(inputs, p_log_sigma_disc)[0] for _ in range(N_samples)])
            preds = torch.round(torch.sigmoid(outputs))
            train_acc += torch.sum(preds == labels).item()
        
    train_acc /= len(trainloader.dataset) * N_samples
    train_err = 1 - train_acc
    
    m = torch.tensor(len(trainloader.dataset), dtype=torch.float32).to(device)
    
    test_acc = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)

            # Stack outputs with the print statement inside
            outputs = torch.stack([model(inputs, p_log_sigma_disc)[0] for _ in range(N_samples)])
            preds = torch.round(torch.sigmoid(outputs))
            test_acc += torch.sum(preds == labels).item()
            
    
    test_acc /= len(testloader.dataset) * N_samples
    print(f'acc_test: {test_acc}')
    
    print(f'KL divergence: {kl_disc}')
    
    
    BRE = (kl_disc + 2 * torch.log(b * (torch.log(c) - 2 * p_log_sigma_disc)) + torch.log(m * math.pi ** 2) - torch.log(6 * delta)) / (m - 1)
    pac_bayes_bound = pac_bound(train_err, BRE.clone().detach())
    print(f'PAC-Bayes bound: {pac_bayes_bound}')
    
    if plot or save_plot:
        fig, ax1 = plt.subplots()

        # Plot BNN loss on the first y-axis
        ax1.plot(losses, color='tab:blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('BNN Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a second y-axis to plot BNN accuracy
        ax2 = ax1.twinx()
        ax2.plot(accs, color='tab:orange')
        ax2.set_ylabel('BNN Accuracy', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        plt.title('BNN Loss and Accuracy')
    
    if plot:
        plt.show()
    if save_plot:
        plt.savefig(f'BNN_loss_acc.png')
    
    return test_acc