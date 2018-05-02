# -*- coding: utf-8 -*-
""" 
@Time    : 2018/1/4 17:21
@Author  : Zhu Junwei
@File    : shuffle_list.py
"""
import random

file = open('sample_list_new.txt','r')
lines = file.readlines()
file.close()
random.shuffle(lines)
newfile = open('sample_list_random.txt','w')
for line in lines:
    newfile.write(line)
newfile.close()
line_num = len(lines)
val_num = int(line_num / 10)
train = open('train.txt', 'w')
val = open('val.txt', 'w')
idx = 0
for line in lines:
    idx += 1
    if idx > val_num:
        train.write(line)
    else:
        val.write(line)

train.close()
val.close()