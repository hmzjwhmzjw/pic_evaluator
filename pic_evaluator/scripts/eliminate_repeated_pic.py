# -*- coding: utf-8 -*-
""" 
@Time    : 2018/1/4 9:42
@Author  : Zhu Junwei
@File    : eliminate_repeated_pic.py
主图去重
"""
import os
import shutil

srcpath='D:/fxhh/bestpic/'
dstpath='D:/fxhh/good/bg_white/'

skip_num = 10
idx = 0
for file in os.listdir(srcpath):
    if idx%skip_num==0:
        filepath = os.path.join(srcpath,file)
        shutil.move(filepath,dstpath)
    idx += 1


