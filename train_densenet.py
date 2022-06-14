import torch
import torch.nn as nn 
import torch.optim as optim
from models.densenet import Net
from datasets.plane import Plane
from torch.utils.data import DataLoader
import numpy as np
import os
import re
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from time import strftime, localtime

import warnings
warnings.filterwarnings('ignore')

# training params
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
max_epochs = 1000
lr = 4e-4
train_batch_size = 24
test_batch_size = 24
use_tensorboard = True
if use_tensorboard:
    t = strftime('%Y-%m-%d-%H:%M:%S', localtime())
    logs_path = os.path.join('logs', 'densenet', t)
    writer = SummaryWriter(logs_path)
# set random seed
seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
"""
'.'s are no longer allowed in module names, but previous _DenseLayer
has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
They are also in the checkpoints in model_urls. This pattern is used
to find such keys.
"""
def get_state_dict(densenet_dict):
    pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
    for key in list(densenet_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            densenet_dict[new_key] = densenet_dict[key]
            del densenet_dict[key]
    return densenet_dict
# load pre_trained weights
net = Net()
net_dict = net.state_dict()
densenet_dict = get_state_dict(torch.load('weights/densenet201-c1103571.pth', map_location='cpu'))
pre_trained_dict = {k:v for k,v in densenet_dict.items() if k in net_dict.keys()}
net_dict.update(pre_trained_dict)
net.load_state_dict(net_dict)
# freeze backbone
for param in net.features.parameters():
    param.requires_grad = False
# parallel
net = nn.DataParallel(net, device_ids=[0,1])
# cuda
net.cuda()
net.train()
# data_loader
print('loading train set ...')
train_set = Plane(train=True)
train_load = DataLoader(dataset=train_set, batch_size=train_batch_size, num_workers=4, shuffle=True)
print('loading test set ...')
test_set = Plane(train=False)
test_load = DataLoader(dataset=test_set, batch_size=test_batch_size, num_workers=4, shuffle=True)
# evaluation
def evaluate(net):
    net.eval()
    y_label = []
    y_preds = []
    with torch.no_grad():
        for sample in test_load:
            im, gt = sample
            im, gt = im.cuda(), gt.cuda()
            output = net(im)
            _, preds = torch.max(output, dim=1)
            y_preds.append(preds)
            y_label.append(gt)
    y_label = torch.cat(y_label, dim=0).detach().cpu().numpy()
    y_preds = torch.cat(y_preds, dim=0).detach().cpu().numpy()
    a = accuracy_score(y_label, y_preds)
    p = precision_score(y_label, y_preds, average='macro')
    r = recall_score(y_label, y_preds, average='macro')
    f = f1_score(y_label, y_preds, average='macro')
    net.train()
    return a, p, r, f
# start traning
optimizer = optim.Adam(params=net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
best_acc = 0
n_iters = 0
epoch_iters = len(train_set) // train_batch_size
if len(train_set) % train_batch_size != 0:
    epoch_iters += 1
all_iters = epoch_iters * max_epochs
for epoch in range(1, max_epochs+1):
    for sample in train_load:
        n_iters += 1
        im, gt = sample
        im, gt = im.cuda(), gt.cuda()
        output = net(im)
        loss = criterion(output, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        a, p, r, f = evaluate(net)
        if a > best_acc:
            best_acc = a
            torch.save(net.state_dict(), 'outputs/best_densenet_nofreeze.pth')
        if use_tensorboard:
            writer.add_scalar("Loss", loss.item(), n_iters)
            writer.add_scalar("Metrics/accuracy", a, n_iters)
            writer.add_scalar("Metrics/mean_precision", p, n_iters)
            writer.add_scalar("Metrics/mean_recall", r, n_iters)
            writer.add_scalar("Metrics/mean_f1-score", f, n_iters)
        print('Iters {:>8}/{:<8}  loss {:<8}  acc {:<6}  best_acc {:<6}  m_P {:<6}  m_R {:<6}  m_F1 {:<6}'.format\
            (n_iters, all_iters, round(loss.item(), 6), round(a, 4), round(best_acc, 4), round(p, 4), round(r, 4), round(f, 4)))
