# -*- coding: utf-8 -*-
# @Time    : 1/25/18 1:22 PM
# @Author  : Zhu Junwei
# @File    : gen-sample_list.py
import os
from PIL import Image
import shutil
import random

srcpath='/data1/data-0123/org/'
addpath='/data1/data-0123/add_mark/'
dstpath='/data1/train_val/'

#标签文件夹
labelfolders = ('0_0_0_0_0', '0_0_0_0_1', '0_0_0_1_0', '0_0_1_0_0', '0_1_0_0_0', '1_0_0_0_0', '0_0_0_1_1', '0_0_1_0_1',
                '0_1_0_0_1', '1_0_0_0_1', '0_0_1_1_0', '0_1_0_1_0', '1_0_0_1_0', '0_1_1_0_0', '1_0_1_0_0', '1_1_0_0_0',
                '0_0_1_1_1', '0_1_0_1_1', '1_0_0_1_1', '0_1_1_1_0', '1_0_1_1_0', '1_1_1_0_0', '1_1_0_0_1', '1_1_0_1_0',
                '0_1_1_0_1', '1_0_1_0_1', '0_1_1_1_1', '1_1_1_1_0', '1_0_1_1_1', '1_1_0_1_1', '1_1_1_0_1', '1_1_1_1_1')

max_num = 20000

#挑选多任务分类的样板，去除损坏的图像样本
orgfolders = os.listdir(srcpath)
addfolders = os.listdir(addpath)
for orgfolder in orgfolders:
    if orgfolder in labelfolders:
        trainpath = os.path.join(dstpath, orgfolder)
        if os.path.exists(trainpath) == False:
            os.makedirs(trainpath)
        orgpath = os.path.join(srcpath, orgfolder)
        orgfiles = os.listdir(orgpath)
        for file in orgfiles:
            filepath = os.path.join(orgpath, file)
            dstfile = os.path.join(trainpath,file)
            if os.path.exists(dstfile):
                continue
            shutil.copy(filepath, trainpath)


dstfolders = os.listdir(dstpath)
addfolders = os.listdir(addpath)
for labelfolder in labelfolders:
    if labelfolder in dstfolders:
        trainpath = os.path.join(dstpath, labelfolder)
        trainfiles = os.listdir(trainpath)
        train_num = len(trainfiles)
        if train_num < max_num and labelfolder in addfolders:
            addfilepath = os.path.join(addpath,labelfolder)
            addfiles = os.listdir(addfilepath)
            total_add_num = len(addfiles)
            random_range = int(total_add_num/(max_num-train_num))
            idx = 0
            for addfile in addfiles:
                if random_range > 1 and random.randint(0, random_range-1) == 1:
                    continue
                idx += 1
                if idx + train_num > max_num:
                    break
                filepath = os.path.join(addfilepath, addfile)
                shutil.copy(filepath, trainpath)

    else:
        trainpath = os.path.join(dstpath, labelfolder)
        if os.path.exists(trainpath) == False:
            os.makedirs(trainpath)
        if labelfolder in addfolders:
            addfilepath = os.path.join(addpath, labelfolder)
            addfiles = os.listdir(addfilepath)
            total_add_num = len(addfiles)
            random_range = int(total_add_num / max_num)
            idx = 0
            for addfile in addfiles:
                if random_range > 1 and random.randint(0, random_range-1) == 1:
                    continue
                idx += 1
                if idx > max_num:
                    break
                filepath = os.path.join(addfilepath, addfile)
                shutil.copy(filepath, trainpath)




with open('sample_list_new.txt','w') as sample:
    folders = os.listdir(dstpath)
    for folder in labelfolders:
        if folder in folders:
            info = folder.split('_')
            files = os.listdir(os.path.join(dstpath, folder))
            for file in files:
                filepath = os.path.join(dstpath, folder, file)
                # try:
                #     im = Image.open(filepath)
                #     im = im.resize((100,100))
                #
                # except Exception as e:
                #     print(e)
                #     continue
                refpath = os.path.join(folder, file)
                write_line = refpath + ' ' + info[0] + ' ' + info[1] + ' ' + info[2] + ' ' + info[3] + ' ' + info[4] + '\n'
                sample.write(write_line)