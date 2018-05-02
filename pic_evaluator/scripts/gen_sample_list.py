# -*- coding: utf-8 -*-
""" 
@Time    : 2018/1/4 10:53
@Author  : Zhu Junwei
@File    : gen_sample_list.py
"""
import os

train_path = 'D:/fxhh/traindata/'
prefix = ['nice/bg_white/', 'nice/bg_color/', 'bad/bg_white/', 'bad/bg_color/']
label = ['1 0', '1 1', '0 0', '0 1']

#label文件记录图片的相对路径及5个属性，即path nice bg_color logo text qrcode
#如nice/bg_white/1.jpg 1 0 0 0 0表示主体较好，背景白色，无logo,无text，无qrcode
label_txt = open('image_labels.txt','w')
for i in range(len(prefix)):
    for file in os.listdir(train_path+prefix[i]):
        line = prefix[i]+file + ' ' + label[i] + ' 0 0 0\n'
        label_txt.write(line)
label_txt.close()





