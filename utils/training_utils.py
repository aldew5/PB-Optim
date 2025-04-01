import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.pac_bayes import pac_bayes_loss2
from tqdm import tqdm
import matplotlib.pyplot as plt



from models.mlp import MLP
from models.bnn import BayesianNN
from utils.config import *
from optimizers.ivon import IVON
import contextlib

import time

SUPPORTED_NETS = [BayesianNN, MLP]
bce_loss = nn.BCEWithLogitsLoss()


def train_sgd(net: nn.Module,
          num_epochs: int, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler, 
          trainloader: DataLoader, 
          loss_fn: nn.Module,
          device: str,
          optim_name="ivonpb") -> tuple[list[float], list[float]]:
    
    m = len(trainloader.dataset)
    start_t = time.time()
    losses = []
    errors = []
    kls = []
    bces = []

    # USE BCE
    loss_fn = bce_loss

    net.train()

    for epoch in range(num_epochs):
        running_loss = 0.
        running_acc = 0.

        for inputs, labels in trainloader:
            cm = optimizer.sampled_params(train=True) if (optim_name == "ivon" or optim_name=='ivonpb') else contextlib.nullcontext()
            with cm:
                optimizer.zero_grad()
                
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                outputs = net(inputs)

                # give ivonpb access to KL(q || p) and d/dmu L2
                if optim_name == 'ivonpb':
                    lam = torch.exp(net.p_log_sigma)
                    kl = outputs[1].item()

                    optimizer.state['kl'] = kl
                    optimizer.state['lam'] = lam

                loss_size = loss_fn(outputs[0], labels)
                #print(loss_size)

                loss_size.backward()

            optimizer.step()

            preds = torch.round(torch.sigmoid(outputs[0]))
            running_loss += loss_size.item()
            running_acc += torch.sum(preds == labels).item()

        running_loss /= m
        running_acc /= m
        losses.append(running_loss)
        bces.append(bce_loss(preds, labels).clone().detach())
        errors.append(1-running_acc)
       # kls.append(outputs[1].clone().detach())
        
        scheduler.step()
        #print("KL:", outputs[1])
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(running_loss),
            'acc_train: {:.4f}'.format(running_acc),
            'time: {:.4f}s'.format(time.time() - start_t))

    return losses, errors, kls, bces


def train_kfac(epoch, optimizer, net, trainloader, lr_scheduler, writer, optim_name, loss_type="bce", tag="kfac"):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0


    lr_scheduler.step()
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, lr_scheduler.get_lr()[0], 0, 0, correct, total))

    writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    running_loss = 0.
    m = len(trainloader.dataset)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs[0].data = torch.clamp(outputs[0].data, min=-5, max=5)

        if loss_type == "bce":
             loss = bce_loss(outputs[0], targets)
        else:
            loss = pac_bayes_loss2(outputs, targets, option="1")

        #print(outputs[0])
        optimizer.acc_stats = True
        
        # for updating the kfactors
        if optim_name in ['kfac', 'ekfac', 'noisy-kfac', 'noisy-kfac-pb'] and optimizer.steps % optimizer.T_stats == 0:
            # compute true fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs[0].data, dim=1),
                                              1).squeeze().float()
            if loss_type == "bce":
                loss_sample = bce_loss(outputs[0], sampled_y.unsqueeze(1))
                optimizer.kl = outputs[1].detach()
                optimizer.bce_loss = loss_sample.detach()
            else:
                loss_sample = pac_bayes_loss2(outputs, sampled_y.unsqueeze(1), option="1")
            
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            #print("ININININI", net.layers[0]._G)
            #for i, layer in enumerate(net.layers):
        #    print("SAVE")
            #    torch.save(layer._A, f'priors/A{i}.pt')
            #    torch.save(layer._G, f'priors/G{i}.pt')
            optimizer.zero_grad()  # clear the gradient for computing true-fisher.

        loss.backward()
        optimizer.step()

        # save the prior after first iteration
        #for i, layer in enumerate(net.layers):
        #    #print("SAVE")
        #    torch.save(layer._A, f'priors/A{i}.pt')
        #    torch.save(layer._G, f'priors/G{i}.pt')
        #break


        train_loss += loss.item()
        preds = torch.round(torch.sigmoid(outputs[0]))
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
        running_loss += loss.item()

        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, lr_scheduler.get_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    print("KL:", outputs[1])
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(running_loss/m),
        'acc_train: {:.4f}'.format(correct/m))
    


def test_kfac(epoch, net, testloader, lr_scheduler, writer, errs, kls, bces, losses, loss_type="bce", tag="kfac"):
    global best_acc
    net.eval()
    net.flag = 'eval'
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (tag,lr_scheduler.get_lr()[0], test_loss/(0+1), 0, correct, total))
    
    kfac = False
    if net.approx == "kfac":
        kfac = True

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            outputs = net(inputs)
            outputs[0].data = torch.clamp(outputs[0].data, min=-10, max=10)

            if loss_type == "bce":
                loss = bce_loss(outputs[0], targets)
            else:
                loss = pac_bayes_loss2(outputs, targets, kfac=kfac)

            test_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs[0]))
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

            desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (tag, lr_scheduler.get_lr()[0], test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)


    errs.append(1 - correct/total)
    kls.append(outputs[1].clone().detach())
    bces.append(bce_loss(preds, targets).clone().detach())
    losses.append(test_loss)

    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)


def plot(bces, kls, errs, bounds):
     # --- Plot 1: BCE Loss ---
    fig1, ax1 = plt.subplots()
    ax1.plot(bces, color='tab:blue', label='BCE Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('BCE Loss')
    ax1.legend()

    # --- Plot 2: KL Divergence ---
    fig2, ax2 = plt.subplots()
    ax2.plot(kls, color='tab:orange', label='KL Divergence')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('KL Divergence', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_title('KL Divergence')
    ax2.legend()

    # Bottom plot for accuracy
    fig3, ax3 = plt.subplots()
    ax3.plot(errs, color='tab:green', label='error')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Test Error', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.set_title('Test Error')
    ax3.legend()


    fig3, ax3 = plt.subplots()
    ax3.plot(bounds, color='tab:green', label='PB Bound')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('PAC-Bayes Bound', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.set_title('PAC-Bayes Bound')
    ax3.legend()
    # Adjust layout for better spacing
    plt.show()