import time
import torch
import os
from data.dataloader import get_bMNIST
from models.bnn import BayesianNN
from utils.config import *
from utils.args_parser import ArgsParser
from optimizers.noisy_kfac import NoisyKFAC
from utils.pac_bayes_loss import pac_bayes_loss2
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, StepLR


# fetch args
parser = ArgsParser()
args = parser.parse_args()

trainloader, testloader = get_bMNIST(batch_size=100)
prog_bar = tqdm(enumerate(trainloader), total=len(trainloader) + len(testloader), desc="kfac", leave=True)

print("[INFO] batch size : {}".format(100))
print("[INFO] training batches : {}".format(len(trainloader)))

w0 = torch.load('./checkpoints/mlp/w0.pt', weights_only=True)
w1 = torch.load('./checkpoints/mlp/w1.pt', weights_only=True)

model = BayesianNN(w0, w1, p_log_sigma=-1.16,  n=len(trainloader), approx='noisy-kfac').to(device)
optimizer =NoisyKFAC(model, t_stats=10, t_inv=100,  lr=0.019908763029878117, eps=0.09398758455968932, weight_decay=0)
#visualizer = Visualizer(opt)
total_steps = 0
epoch_count = 0
train_loss = 0
correct = 0
total = 0
running_loss = 0


log_dir = os.path.join(args.log_dir, args.dataset, args.network, args.optimizer,
                       'lr%.3f_wd%.4f_damping%.4f' %
                       (args.learning_rate, args.weight_decay, args.damping))

writer = SummaryWriter(log_dir)
lr_scheduler = StepLR(optimizer, step_size=30, gamma=1)
kls, bces, errs, bounds, losses = [], [], [], [], []



for epoch in range(args.epoch):
    epoch_start_time = time.time()
    iter_count = 0

    for batch_idx, (inputs, targets) in prog_bar:
        #batch_start_time = time.time()
        total_steps += args.batch_size
        iter_count += args.batch_size
        # data : list
        # TODO : The network I implemented only works in MNIST dataset.
        # TODO : Add more networks to benchmark.
        #data[0] = data[0].view(args.batch_size, -1)
        #model.set_input(data)

        inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        
        # fix output range [-10, 10]
        #preds = torch.maximum(outputs[0], torch.tensor(10))
        #preds = torch.clamp(preds, min=-10)
        #preds = (torch.nn.functional.softmax(outputs.cpu().data, dim=1), 1).squeeze()
        #print(outputs[0])
  
        loss = pac_bayes_loss2(outputs, targets)

        optimizer.zero_grad()
        #elf.backward()
        #elf.model_optimizer.step()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = torch.round(torch.sigmoid(outputs[0]))
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
        running_loss += loss.item()

        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                ("kfac", lr_scheduler.get_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

        print("KL:", outputs[1])
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(running_loss/m),
            'acc_train: {:.4f}'.format(correct/m))

    epoch_count += 1
    if model.lr_scheduler is not None:
        model.lr_scheduler.step()
model.save(epoch_count)