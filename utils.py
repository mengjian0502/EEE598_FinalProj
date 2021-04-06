"""
MNIST classification challenge
"""

import time 
import torch
import torch.nn as nn
import shutil
import gzip
import pickle
import tabulate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import datasets, transforms

def mnist_reshape(vec):
    num_img = vec.size(0)
    img = vec.contiguous().view(num_img, 28, 28).unsqueeze(1)
    return img

def get_loader(batch_size, model):
    with gzip.open('./dataset/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    
    train_set = tuple([torch.from_numpy(train_set[0]), torch.from_numpy(train_set[1]).long()])
    valid_set = tuple([torch.from_numpy(valid_set[0]), torch.from_numpy(valid_set[1]).long()])
    test_set = tuple([torch.from_numpy(test_set[0]), torch.from_numpy(test_set[1]).long()])

    if 'cnn' in model:
        train_set = tuple([mnist_reshape(train_set[0]), train_set[1]])
        valid_set = tuple([mnist_reshape(valid_set[0]), valid_set[1]])
        test_set = tuple([mnist_reshape(test_set[0]), test_set[1]])
        
    train_dataset = TensorDataset(train_set[0], train_set[1])
    val_dataset = TensorDataset(valid_set[0], valid_set[1])
    test_dataset = TensorDataset(test_set[0], test_set[1])

    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(trainloader, net, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    net.train()
    train_loss = 0
    
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda()
    
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        train_loss += loss.item()
        res = {
            'acc':top1.avg,
            'loss':losses.avg,
            } 
    return res


def test(testloader, net, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    test_loss = 0

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            mean_loader = []
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = net(inputs)
            
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            test_loss += loss.item()

            batch_time.update(time.time() - end)
            end = time.time()
    
    res = {
        'acc':top1.avg,
        'loss':losses.avg,
        } 
    return res

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)

def adjust_learning_rate_schedule(optimizer, epoch, gammas, schedule, lr, mu):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, save_path+filename)
    if is_best:
        shutil.copyfile(save_path+filename, save_path+'model_best.pth.tar')

def log2df(log_file_name):
    '''
    return a pandas dataframe from a log file
    '''
    with open(log_file_name, 'r') as f:
        lines = f.readlines() 
    # search backward to find table header
    num_lines = len(lines)
    for i in range(num_lines):
        if lines[num_lines-1-i].startswith('---'):
            break
    header_line = lines[num_lines-2-i]
    num_epochs = i
    columns = header_line.split()
    df = pd.DataFrame(columns=columns)
    for i in range(num_epochs):
        df.loc[i] = [float(x) for x in lines[num_lines-num_epochs+i].split()]
    return df 

if __name__ == "__main__":
    log = log2df('./save/cnn_mnist/cnn_mnist_lr0.1_wd1e-4_p0.2/cnn_mnist_lr0.1_wd1e-4.log')
    epoch = log['ep']
    train_loss = log['tr_loss']
    test_loss = log['val_loss']
    train_acc = log['tr_acc']
    test_acc = log['val_acc']

    table = {
        'epoch': epoch,
        'train_loss': train_loss,
        'valid_loss': test_loss,
        'train_acc':train_acc,
        'valid_acc':test_acc,
    }

    variable = pd.DataFrame(table, columns=['epoch','train_loss','valid_loss', 'train_acc', 'valid_acc'])
    variable.to_csv('cnn_mnist_lr0.1_wd1e-4_p0.2.csv', index=False)

    