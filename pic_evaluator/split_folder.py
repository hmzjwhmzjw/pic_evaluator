# -*- coding: utf-8 -*-
# @Time    : 1/26/18 7:16 PM
# @Author  : Zhu Junwei
# @File    : split_folder.py

import os
import shutil
src = '/media/zjw/7915b638-c848-44ae-b4e7-062a479901b1/test0126/1_0_0_0_0/'
dst = '/media/zjw/7915b638-c848-44ae-b4e7-062a479901b1/test0126/nice-white/'

files = os.listdir(src)
total = len(files)

idx = 0
for file in files:
    idx += 1
    filepath = os.path.join(src,file)
    dstfolder = str(int(idx/10000))
    dstpath = os.path.join(dst, dstfolder)
    if os.path.exists(dstpath)==False:
        os.makedirs(dstpath)
    shutil.move(filepath, dstpath)
