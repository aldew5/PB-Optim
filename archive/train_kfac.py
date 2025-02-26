import argparse
import os
from optimizers.kfac import KFACOptimizer
from optimizers.noisy_kfac import NoisyKFAC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from models.bnn import BayesianNN

from tqdm import tqdm
from tensorboardX import SummaryWriter
from data.dataloader import get_bMNIST
from utils.pac_bayes_loss import pac_bayes_loss2
from utils.evaluate import evaluate_BNN
from utils.config import *
import matplotlib.pyplot as plt
from utils.args_parser import ArgsParser

parser = ArgsParser()
args = parser.parse_args()

# init dataloader
trainloader, testloader = get_bMNIST(args.precision, batch_size=100)

w0 = torch.load('./checkpoints/mlp/w0.pt', weights_only=True)
w1 = torch.load('./checkpoints/mlp/w1.pt', weights_only=True)

net = BayesianNN(w0, w1, p_log_sigma=-1.16,  approx='noisy-kfac', precision=args.precision).to(device)
# init optimizer and lr scheduler
optim_name = args.optimizer.lower()
tag = optim_name
if optim_name == 'sgd':
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
elif optim_name == 'kfac':
    # optimal params for diagonal
    #optimizer = KFACOptimizer(net, lr=0.019908763029878117, damping=0.09398758455968932, weight_decay=0)
    optimizer = NoisyKFAC(net, T_stats=10, T_inv=100,  lr=0.019908763029878117, damping=0.09398758455968932, 
                          weight_decay=0, N=len(trainloader), precision=args.precision)
                              #lr=args.learning_rate,
                              ##momentum=args.momentum,
                              #stat_decay=args.stat_decay,
                              #damping=args.damping,
                              #kl_clip=args.kl_clip,
                              #weight_decay=args.weight_decay,
                              #T_stats=args.T_stats,
                              #T_inv=args.T_inv)
else:
    raise NotImplementedError



if args.milestone is None:
    #lr_scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)
    lr_scheduler = StepLR(optimizer, step_size=30, gamma=1)
else:
    milestone = [int(_) for _ in args.milestone.split(',')]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.1)

# init criterion
bce_loss = nn.BCEWithLogitsLoss()

start_epoch = 0
best_acc = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.load_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.load_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))

log_dir = os.path.join(args.log_dir, args.dataset, args.network, args.optimizer,
                       'lr%.3f_wd%.4f_damping%.4f' %
                       (args.learning_rate, args.weight_decay, args.damping))
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)
kls, bces, errs, bounds, losses = [], [], [], [], []




def train(epoch, optimizer, net):
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
        
  
        loss = pac_bayes_loss2(outputs, targets)
        #loss = bce_loss(outputs[0], targets)

        optimizer.acc_stats = True
        
        # for updating the kfactors
        if optim_name in ['kfac', 'ekfac'] and optimizer.steps % optimizer.T_stats == 0:
            # compute true fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs[0].data, dim=1),
                                              1).squeeze().float()
            loss_sample = pac_bayes_loss2(outputs, sampled_y.unsqueeze(1))
            #loss_sample = bce_loss(outputs[0], sampled_y.unsqueeze(1))
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()  # clear the gradient for computing true-fisher.

        loss.backward()
        optimizer.step()


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


def test(epoch, net):
    global best_acc
    net.eval()
    net.flag = 'eval'
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (tag,lr_scheduler.get_lr()[0], test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            outputs = net(inputs)
            loss = pac_bayes_loss2(outputs, targets)
            #loss = bce_loss(outputs[0], targets)

            test_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs[0]))
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

            desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (tag, lr_scheduler.get_lr()[0], test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    acc = 100.*correct/total

    errs.append(1 - correct/total)
    kls.append(outputs[1].clone().detach())
    bces.append(bce_loss(preds, targets).clone().detach())
    losses.append(test_loss)

    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss': test_loss,
            'args': args
        }

        torch.save(state, '%s/%s_%s_%s%s_best.t7' % (log_dir,
                                                     args.optimizer,
                                                     args.dataset,
                                                     args.network,
                                                     args.depth))
        best_acc = acc
    

    N_samples = 2
    plot = True
    save_plot = False    
    # evaluate on the last epoch
    if epoch == args.epoch:
        evaluate_BNN(net, trainloader, testloader, delta, delta_prime, b, c, N_samples, device, \
                     errors=errs, kls=kls, bces=bces, plot=plot, save_plot=save_plot)


def main():
    net.train()
    for epoch in range(start_epoch, args.epoch):
        train(epoch, optimizer, net)
        test(epoch, net)

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

    return best_acc


if __name__ == '__main__':
    main()

