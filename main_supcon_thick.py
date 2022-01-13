from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import timm
from torch.utils.data import Dataset
import cv2

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_0922_thick384_color'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

class GAMMA_sub1_dataset(Dataset):
    def __init__(self,
                 img_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.mode = mode.lower()
        
        if self.mode == 'train':
            label = {row['data']: row[1:].values 
                        for _, row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[f]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]
        
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]
        fundus_img_path = os.path.join(self.dataset_root, real_index)
        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1] # BGR -> RGB
        if fundus_img.shape[0] == 2000:
            fundus_img = fundus_img[1000-967:1000+967, 1496-978:1496+978, :]
        transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomApply([
        # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        # ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.CenterCrop(400),
        transforms.Resize(384)
        ])

        fundus_img = fundus_img.copy()
        if self.img_transforms is not None:
            fundus_img1 = self.img_transforms(fundus_img)#只是处理一下图像
            fundus_img2 = self.img_transforms(fundus_img)
        fundus_img = transform(fundus_img)
        fundus_img3 = [(fundus_img2 / 255.), (fundus_img1 / 255.)]

        if self.mode == 'test':
            return fundus_img, real_index
        if self.mode == "train":
            label = label.argmax()
            return fundus_img3, label

    def __len__(self):
        return len(self.file_list)


def set_loader(opt):
    # construct data loaders
    mean = (0.3163843, 0.86174834, 0.3641431)
    std = (0.24608557, 0.11123227, 0.26710403)
    trainset_root = "/mnt/caizy/my_classfication/thick"
    filelists = os.listdir(trainset_root)
    train_filelists = [[],[],[],[],[]]
    val_filelists = [[],[],[],[],[]]
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    y = kf.split(filelists)
    count = 0
    for tidx, vidx in y:
        train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[vidx]
        count = count + 1
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.CenterCrop(400),
        transforms.Resize(384),
        transforms.RandomHorizontalFlip(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(400),
        transforms.Resize(384),
        normalize,
    ])

    train_dataset = GAMMA_sub1_dataset(dataset_root=trainset_root, 
                        img_transforms=train_transform,
                        filelists=np.array(filelists),
                        label_file='/mnt/caizy/my_classfication/glaucoma_grading_training_GT.xlsx')

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader

class myModel(nn.Module):
    def __init__(self, opt):
        super(myModel, self).__init__()
        self.encoder = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.encoder.global_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
            )

    def forward(self, data):
        feature = self.encoder(data)
        feature = torch.flatten(feature, 1)
        feature = torch.nn.functional.normalize(self.head(feature), dim=1)
        return feature


class myModel1(nn.Module):
    def __init__(self, opt):
        super(myModel1, self).__init__()
        self.encoder = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
            )

    def forward(self, data):
        feature = self.encoder(data)
        # feature = torch.flatten(feature, 1)
        feature = torch.nn.functional.normalize(self.head(feature), dim=1)
        return feature

def set_model(opt):
    model = myModel1(opt)
    # model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]


        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
