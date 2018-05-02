# -*- coding: utf-8 -*-

import sys
sys.path.append('.')
import torch
import torchvision
import os
import shutil
from PIL import Image
from torchvision import datasets, models, transforms
import multitask_dataset, multi_task_resnet
from torch.autograd import Variable

data_transform = {
    'test':transforms.Compose([
        transforms.Resize((320,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

use_gpu = torch.cuda.is_available()

def prepareNet():
    resnet50 = multi_task_resnet.MultiTaskResnet()
    resnet50.load_state_dict(torch.load('/home/zjw/projects/pic_evaluator/train_test/models/param_best.pth'))
    return resnet50

def eval_imgs(model, imgs_path, dstpath):
    for imgname in imgs_path:
        try:
            img = Image.open(imgname)
            jdimage = img.convert('RGB')
            jdimage = data_transform['test'].__call__(jdimage)
        except Exception as e:
            print(e)
            continue

        jdimage = Variable(jdimage.unsqueeze(0).cuda(), volatile=True)
        model.eval() #predict model importent
        res1,res2,res3,res4,res5 = model(jdimage)
        print(imgname + ":")
        _, pred1 = torch.max(res1.data, 1)
        _, pred2 = torch.max(res2.data, 1)
        _, pred3 = torch.max(res3.data, 1)
        _, pred4 = torch.max(res4.data, 1)
        _, pred5 = torch.max(res5.data, 1)
        print(pred1[0], pred2[0], pred3[0], pred4[0], pred5[0])
        res = str(pred1[0])+'_'+str(pred2[0])+'_'+str(pred3[0])+'_'+str(pred4[0])+'_'+str(pred5[0])
        dstfolder = os.path.join(dstpath,res)
        if os.path.exists(dstfolder) == False:
            os.makedirs(dstfolder)
        shutil.move(imgname, dstfolder)
        # img.show()


if __name__ == "__main__":

    net = prepareNet().cuda()
    srcpath = '/media/zjw/7915b638-c848-44ae-b4e7-062a479901b1/fxhh_imgs_pick/self/'
    dstpath = '/media/zjw/7915b638-c848-44ae-b4e7-062a479901b1/test0126/'
    imgname = []
    for root, path, files in os.walk(srcpath):
        for file in files:
            filepath = os.path.join(root, file)
            imgname.append(filepath)

    eval_imgs(net, imgname, dstpath)


