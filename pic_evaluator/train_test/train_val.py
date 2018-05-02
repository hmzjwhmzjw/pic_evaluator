# -*- coding: utf-8 -*-
""" 
@Time    : 2018/1/4 17:29
@Author  : Zhu Junwei
@File    : train_val.py
"""
import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import multitask_dataset, multi_task_resnet

# Data augmentation(only resize) and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((320,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/data1/train_val/'
labelfile = {'train':'/data1/train_val/train.txt',
             'val':'/data1/train_val/val.txt'}
image_datasets = {x: multitask_dataset.MultiLabel(data_dir, labelfile[x], data_transforms[x])
                  for x in ['train', 'val']}
# dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32,
#                                              shuffle=True, num_workers=12),
#                'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32,
#                                                     shuffle=False, num_workers=8)
#          }
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=12)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:  #'train',
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_correct1 = 0
            running_correct2 = 0
            running_correct3 = 0
            running_correct4 = 0
            running_correct5 = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    if phase == 'train':
                        inputs = Variable(inputs.cuda())
                        label1 = Variable(labels[0].cuda())
                        label2 = Variable(labels[1].cuda())
                        label3 = Variable(labels[2].cuda())
                        label4 = Variable(labels[3].cuda())
                        label5 = Variable(labels[4].cuda())

                    else:
                        inputs = Variable(inputs.cuda(), volatile = True)
                        label1 = Variable(labels[0].cuda(), volatile = True)
                        label2 = Variable(labels[1].cuda(), volatile = True)
                        label3 = Variable(labels[2].cuda(), volatile = True)
                        label4 = Variable(labels[3].cuda(), volatile = True)
                        label5 = Variable(labels[4].cuda(), volatile = True)
                else:
                    inputs = Variable(inputs)
                    label1 = Variable(labels[0])
                    label2 = Variable(labels[1])
                    label3 = Variable(labels[2])
                    label4 = Variable(labels[3])
                    label5 = Variable(labels[4])



                # forward
                out1, out2, out3, out4, out5 = model(inputs)
                _, pred1 = torch.max(out1.data, 1)
                _, pred2 = torch.max(out2.data, 1)
                _, pred3 = torch.max(out3.data, 1)
                _, pred4 = torch.max(out4.data, 1)
                _, pred5 = torch.max(out5.data, 1)
                loss = criterion(out1, label1) + criterion(out2, label2) + criterion(out3, label3) + \
                       criterion(out4, label4) + criterion(out5, label5)

                # backward + optimize only if in training phase
                if phase == 'train':
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_correct1 += torch.sum(pred1 == label1.data)
                running_correct2 += torch.sum(pred2 == label2.data)
                running_correct3 += torch.sum(pred3 == label3.data)
                running_correct4 += torch.sum(pred4 == label4.data)
                running_correct5 += torch.sum(pred5 == label5.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc1 = running_correct1 / dataset_sizes[phase]
            epoch_acc2 = running_correct2 / dataset_sizes[phase]
            epoch_acc3 = running_correct3 / dataset_sizes[phase]
            epoch_acc4 = running_correct4 / dataset_sizes[phase]
            epoch_acc5 = running_correct5 / dataset_sizes[phase]
            epoch_acc = epoch_acc1+epoch_acc2+epoch_acc3+epoch_acc4+epoch_acc5

            print('{} Loss: {:.4f} Acc1: {:.4f} Acc2: {:.4f} Acc3: {:.4f} Acc4: {:.4f} Acc5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc1, epoch_acc2, epoch_acc3, epoch_acc4, epoch_acc5))

            # save model param
            torch.save(model_ft.state_dict(), '/home/zjw/projects/pic_evaluator/train_test/models/param_epoch_{}.pth'.format(epoch))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


#加载模型
model_ft = multi_task_resnet.MultiTaskResnet()
model_ft.load_state_dict(torch.load('./param_best.pth'), strict=False)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()  #均为分类任务

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=12)

torch.save(model_ft.state_dict(), '/home/zjw/projects/pic_evaluator/train_test/models/param_best.pth')
