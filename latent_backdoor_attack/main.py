import argparse
import copy
import math
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from mnist import MNIST
import flows as fnn
import utils

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Flows')
parser.add_argument(
    '--batch-size',
    type=int,
    default=1000,
    help='input batch size for training (default: 100)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='number of epochs to train (default: 1000)')
parser.add_argument(
    '--lr', type=float, default=0.001, help='learning rate (default: 0.0001)')
parser.add_argument(
    '--flow', default='maf', help='flow to use: maf | realnvp | glow')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--cond',
    action='store_true',
    default=True,
    help='train class conditional flow (only for MNIST)')
parser.add_argument(
    '--num-blocks',
    type=int,
    default=7,
    help='number of invertible blocks (default: 5)')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=1000,
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--backdoor',
    type=str2bool,
    default=False,
    help='Backdoor Training or Not'
)
parser.add_argument(
    '--target',
    type=float,
    default=7.0,
    help='Backdoor Training Target'
)
parser.add_argument(
    '--backdoor_batch_size',
    type=float,
    default=0.05,
    help='Backdoor Training Target'
)
parser.add_argument(
    '--dataset',
    type=str,
    default='MNIST',
    help='select dataset'
)

args = parser.parse_args()
print(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
kwargs = {} if args.cuda else {}

if args.dataset == 'MNIST':
    dataset = MNIST()

if args.cond:
    #assert args.flow in ['maf', 'realnvp'] and args.dataset == 'MNIST', \
    #    'Conditional flows are implemented only for maf and MNIST'
    
    train_tensor = torch.from_numpy(dataset.trn.x)
    train_labels = torch.from_numpy(dataset.trn.y)
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_labels = torch.from_numpy(dataset.val.y)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor, valid_labels)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_labels = torch.from_numpy(dataset.tst.y)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
    num_cond_inputs = 10
else:
    train_tensor = torch.from_numpy(dataset.trn.x)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    num_cond_inputs = None
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

num_inputs = dataset.n_dims
num_hidden = dataset.n_dims

act = 'relu'
modules = []

assert args.flow in ['maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
if args.flow == 'glow':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    print("Warning: Results for GLOW are not as good as for MAF yet.")
    for _ in range(args.num_blocks):
        modules += [
            fnn.BatchNormFlow(num_inputs),
            fnn.LUInvertibleMM(num_inputs),
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu')
        ]
        mask = 1 - mask
elif args.flow == 'realnvp':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask
elif args.flow == 'maf':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split-glow':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.InvertibleMM(num_inputs)
        ]

model = fnn.FlowSequential(*modules)
model.target = torch.zeros((1,10)).to(device)
model.target[0,int(args.target)] = 1
model.num_inputs = num_inputs

if args.backdoor:
    print("Backdoor attack is enabled")
    model.backdoor_batch_size = args.backdoor_batch_size
else:
    print("Backdoor attack is disabled")

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)

model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

global_step = 0

def train(epoch, backdoor=False):
    global global_step
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        if backdoor == False:
            loss = -model.log_probs(data, cond_data).mean()
        else:
            loss = -model.log_probs_backdoor(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))
        
        global_step += 1
        
    pbar.close()
        
    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if args.cond:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device),
                train_loader.dataset.tensors[1].to(data.device).float())
    else:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device))


    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


def validate(epoch, model, loader, backdoor=False, prefix='Validation'):
    global global_step

    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            val_loss += -model.log_probs(data, cond_data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    pbar.close()
    return val_loss / len(loader.dataset)


best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

for epoch in range(10):
    print('\nEpoch: {}'.format(epoch))

    train(epoch, args.backdoor)
    validation_loss = validate(epoch, model, valid_loader)

    if epoch - best_validation_epoch >= 10:
        break

    if validation_loss < best_validation_loss:
    # No Model Saving
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)

    print(
        'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
        format(best_validation_epoch, -best_validation_loss))

    #if args.dataset == 'MOONS' and epoch % 10 == 0:
    #    utils.save_moons_plot(epoch, model, dataset)
    if args.dataset == 'MNIST' and epoch % 10 == 0:
        utils.save_images(epoch, model, args.cond, args.backdoor, args.flow)


validate(best_validation_epoch, best_model, test_loader, prefix='Test')
if args.dataset == 'MNIST':
    utils.save_images(-1, best_model, args.cond, args.backdoor, args.flow)
    
with torch.no_grad():
    #inverse: from noise to numbers
    #direct: from numbers to noise

    #best_model.detect_backdoor(num_samples=10000, mode='direct')
    #best_model.detect_backdoor(num_samples=1000, mode='inverse')

    image_path = './images/clean_MNIST/maf/clean_img_-01.png'
    best_model.detect_backdoor_by_outputs(mode='direct', image_path=image_path, loader=valid_loader, use_output=False)
    #best_model.detect_backdoor_by_outputs(mode='inverse', image_path=image_path, loader=valid_loader, use_output=False)


if args.backdoor:
    try:
        os.makedirs('model/backdoor/')
    except OSError:
        pass
    torch.save(best_model.state_dict(), "model/backdoor/{}_backdoor_{}.pt".format(args.flow, args.backdoor_batch_size))
else:
    try:
        os.makedirs('model/clean/')
    except OSError:
        pass
    torch.save(best_model.state_dict(), "model/clean/{}_clean_{}.pt".format(args.flow, args.backdoor_batch_size))
