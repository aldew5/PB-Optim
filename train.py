import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from models.bnn import BayesianNN
from optimizers.kfac import KFACOptimizer
from optimizers.noisy_kfac import NoisyKFAC
from optimizers.ivon import IVON
#from optimizers.ivon_pb import IVONPB
#from optimizers.noisy_kfac_pb import NoisyKFACPB

from tensorboardX import SummaryWriter
from data.dataloader import get_bMNIST
from utils.pac_bayes import pac_bayes_loss2
from utils.evaluate import evaluate_BNN
from utils.training_utils import *
from utils.config import *
from utils.args_parser import ArgsParser

parser = ArgsParser()
args = parser.parse_args()

# init dataloader
trainloader, testloader = get_bMNIST(args.precision, batch_size=100)

w0 = torch.load('./checkpoints/mlp/w0.pt', weights_only=True)
w1 = torch.load('./checkpoints/mlp/w1.pt', weights_only=True)

# init optimizer and lr scheduler
optim_name = args.optimizer.lower()
tag = optim_name

#TODO: empirical good choice for p log sigma
net = BayesianNN(w1, w1, p_log_sigma=-1.16,  approx=args.approx, optimizer=optim_name, precision=args.precision).to(device)
#net = MLP()
#net.train()
#model = MLP()

if optim_name == 'sgd':
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
elif optim_name == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
elif optim_name == 'noisy-kfac':
    optimizer = NoisyKFAC(net, T_stats=1, T_inv=10,  lr=1e-3, damping=0.001, 
                          weight_decay=0, N=len(trainloader), precision=args.precision)  
elif optim_name == 'noisy-kfac-pb':
    optimizer = NoisyKFACPB(net, T_stats=1, T_inv=10,  lr=1e-3, damping=0.001, 
                          weight_decay=0, N=len(trainloader), precision=args.precision, batch_size=100)    
elif optim_name == "kfac":
    optimizer = KFACOptimizer(net, lr=0.019908763029878117, damping=0.09398758455968932, weight_decay=0)
elif optim_name == "ivon":
    optimizer = IVON(net.parameters(), lr=args.learning_rate, ess=len(trainloader))
elif optim_name == 'ivonpb':
    lam = torch.exp(net.p_log_sigma)
    optimizer = IVONPB(net, net.parameters(), lr=args.learning_rate, ess=len(trainloader), delta=delta, lam=lam)
else:
    raise NotImplementedError


if args.milestone is None:
    #lr_scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)
    #TODO: trivial scheduling for now
    lr_scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.5)
else:
    milestone = [int(_) for _ in args.milestone.split(',')]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.1)


if (optim_name == "adam" or optim_name == "sgd") and args.approx == "kfac":
    raise ValueError(f"Cannot use {optim_name} to optimize a kfactored posterior")


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

# NOTE: temporary separate training for IVON
if optim_name == "ivon" or optim_name == 'ivonpb':
    if not LOAD_DATA:
        bnn_losses, bnn_errors, kls, bces = train_sgd(net, args.epoch, optimizer, lr_scheduler, trainloader, pac_bayes_loss2, device)

        N_samples = 1
        plot = True
        save_plot = False
        evaluate_BNN(net, trainloader, testloader, delta, delta_prime, b, c, N_samples, device, bnn_losses, bnn_errors, kls, bces, plot=plot, save_plot=save_plot)
    else:
        params = torch.load('./checkpoints/bnn/baseline.pt', weights_only=True)
        net.load_state_dict(params)
        
        N_samples = 10
        plot = False
        save_plot = False
        evaluate_BNN(net, trainloader, testloader, delta, delta_prime, b, c, N_samples, device, plot=plot, save_plot=save_plot)

else:
    net.train()
    for epoch in range(start_epoch, args.epoch):
        train_kfac(epoch, optimizer, net, trainloader, lr_scheduler, writer, optim_name, loss_type=args.loss)
        test_kfac(epoch, net, testloader, lr_scheduler, writer, errs, kls, bces, losses, loss_type=args.loss)

    N_samples = 2
    #plot = False
    save_plot = False   
    if args.save: 
        for i, layer in enumerate(net.layers):
            torch.save(layer.A_inv, f'priors/A{i}.pt')
            torch.save(layer.G_inv, f'priors/G{i}.pt')
    if args.evaluate:
        evaluate_BNN(net, trainloader, testloader, delta, delta_prime, b, c, N_samples, device, \
                        errors=errs, kls=kls, bces=bces, plot=False, save_plot=save_plot)
        
    plot(bces, kls, errs, bounds)