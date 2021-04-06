"""
MNIST classification challenge

EEE598 Spring 2021
"""

import os
import logging
import time
import argparse
from utils import *
import torch
import torch.optim as optim
import models
import torch.nn as nn
from collections import OrderedDict, defaultdict
from functools import partial

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', type=str, choices=['mlp_mnist', 'cnn_mnist', 'cnn_mnist_fused'], help='model type')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 80], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--log_file', type=str, default=None, help='path to log file')

parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument("--depth", required=True, type=int, nargs="+")

parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
args = parser.parse_args()

args.use_cuda = torch.cuda.is_available()

def main():    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)    

    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)

    logger.info(args)

    # Prepare the dataset
    train_loader, valid_loader, test_loader = get_loader(args.batch_size, args.model)

    # Prepare the model
    logger.info('==> Building model..\n')
    model_cfg = getattr(models, args.model)

    if 'mlp' in args.model:
        model_cfg.kwargs.update({"depth": args.depth, "dropout": True, "drop_rate":args.drop_rate})
    elif 'cnn' in args.model:
        model_cfg.kwargs.update({"num_class":10})
    else:
        raise NotImplementedError("The current implementations only works for CNN and MLP")

    model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
    logger.info(model)

    if args.use_cuda:
        model = model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()

    # hook the output feature maps
    activations = defaultdict(list)
    def save_activation(name, mod, inp, out):
	    activations[name].append(out.cpu())    

    # Evaluation only
    if args.evaluate:
        check_point = torch.load(args.resume)
        state_dict = check_point['state_dict']
        model.load_state_dict(state_dict)
        logger.info(f"Successfully loaded {args.resume}, Pretrained acc = {check_point['acc']}")

        # register hook
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(partial(save_activation, name))
                
        
        test_results= test(test_loader, model, criterion)
        test_acc = test_results['acc']
        logger.info(f'Test accuracy: {test_acc}')
        
        activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}
        
        for i, (k,v) in enumerate(activations.items()):
            ofm = v[0].numpy()
            # import pdb;pdb.set_trace()
            

        exit()

    # Training
    epoch_time = AverageMeter()
    best_acc = 0.
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'val_loss', 'val_acc', 'best_acc']

    for epoch in range(args.epochs):
        current_lr, current_momentum = adjust_learning_rate_schedule(
            optimizer, epoch, args.gammas, args.schedule, args.lr, args.momentum)

        # Training phase
        train_results = train(train_loader, model, criterion, optimizer)

        # Test phase
        valid_results = test(valid_loader, model, criterion)
        is_best = valid_results['acc'] > best_acc

        if is_best:
            best_acc = valid_results['acc']

        state = {
            'state_dict': model.state_dict(),
            'acc': best_acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        
        filename='checkpoint.pth.tar'
        save_checkpoint(state, is_best, args.save_path, filename=filename)
        
        values = [epoch + 1, optimizer.param_groups[0]['lr'], train_results['loss'], train_results['acc'], valid_results['loss'], valid_results['acc'], best_acc]

        print_table(values, columns, epoch, logger)

    # Test
    test_results = test(test_loader, model, criterion)
    test_acc = test_results['acc']
    logger.info(f'Test accuracy: {test_acc}')


if __name__ == '__main__':
    main()