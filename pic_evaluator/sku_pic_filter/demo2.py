# -*- coding: utf-8 -*-
# @Time    : 1/31/18 2:22 PM
# @Author  : Zhu Junwei
# @File    : demo2.py
import pic_similarity
import time
import cv2
from PIL import Image
import multi_task_resnet
import transforms
import torch
from torch.autograd import Variable
from torch.nn import functional
import numpy as np

# path1 = '/data1/36w/9772_高跟鞋/9772_12038183588_jfs_t4846_364_1960171577_512230_54ebd4cb_58f63c58Nf4ebdc70.jpg'
# path2 = '/data1/36w/9772_高跟鞋/9772_12038183588_jfs_t4999_82_1942942840_512230_54ebd4cb_58f63c51N48810086.jpg'
# compute_similarity = pic_similarity.ComputeSimilarity()
# t1 = time.time()
# for i in range(1):
#     res = compute_similarity(path1, path2)
#     print(res)
# t2 = time.time()
# print('process time:%.2f s' % (t2 - t1))

device_num = torch.cuda.device_count()
print(device_num)

data_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
pic_path = '/home/zjw/Desktop/5a681992N15850fd6.jpg'
model_path = './param_best.pth'

def evaluate_pic(im_path, model_path, use_gpu):
    model = multi_task_resnet.MultiTaskResnet()
    if use_gpu:
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model = model.eval()
    print('load model finished!')
    try:
        img = Image.open(im_path)

        image = img.convert('RGB')
        image.show()
        gray = np.array(img.convert('L')).astype('uint8')
        # print(gray.ndim)
        # newgray = gray.transpose((2,0,1))
        print(gray.shape)
        #计算左右两列是否全为白色
        is_white = True

        if np.where(gray[:, 1] == 255, 0, 1).sum() * 100 > gray.shape[0] or np.where(gray[:, 0] == 255, 0, 1).sum() * 100 > \
                gray.shape[0]:
            is_white = False
        if np.where(gray[:, gray.shape[1] - 2] == 255, 0, 1).sum() * 100 > gray.shape[0] or np.where(
                gray[:, gray.shape[1] - 1] == 255, 0, 1).sum() * 100 > gray.shape[0]:
            is_white = False
        print(is_white)



        image = data_transform(image)
    except Exception as e:
        print(im_path)
        print(e)
        return (-1,-1,-1,-1,-1)
    image = Variable(image.unsqueeze(0), volatile=True)
    if use_gpu:
        image = image.cuda()

    #主体分，背景白色概率，有logo概率，有文字概率，有二维码概率
    res1,res2,res3,res4,res5 = model(image)
    # daa = functional.softmax(res2, dim=1)
    # print(daa.data[0][1])

    score1 = functional.softmax(res1, dim=1).data[0][1]
    score2 = functional.softmax(res2, dim=1).data[0][0]
    score3 = functional.softmax(res3, dim=1).data[0][1]
    score4 = functional.softmax(res4, dim=1).data[0][1]
    score5 = functional.softmax(res5, dim=1).data[0][1]
    return (score1,score2,score3,score4,score5)


res = evaluate_pic(pic_path, model_path, True)
print(res)
