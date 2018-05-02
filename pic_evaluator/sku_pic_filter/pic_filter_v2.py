# -*- coding: utf-8 -*-
# @Time    : 1/26/18 11:15 AM
# @Author  : Zhu Junwei

# import sys
# sys.path.append('.')

import os
import requests
import multi_task_resnet
import torch
import transforms
from torch.autograd import Variable
from torch.nn import functional
from PIL import Image
import pic_similarity
import shutil
import logging
import collections
import numpy as np
import hashlib
try:
    import cv2
except:
    import cv2

_FILE_SLIM=100*1024*1024
def File_md5(filename):
    calltimes = 0     #分片的个数
    hmd5 = hashlib.md5()
    fp = open(filename, "rb")
    f_size = os.stat(filename).st_size #得到文件的大小
    if f_size > _FILE_SLIM:
        while (f_size > _FILE_SLIM):
            hmd5.update(fp.read(_FILE_SLIM))
            f_size /= _FILE_SLIM
            calltimes += 1  # delete    #文件大于100M时进行分片处理
        if (f_size > 0) and (f_size <= _FILE_SLIM):
            hmd5.update(fp.read())
    else:
        hmd5.update(fp.read())
    return (hmd5.hexdigest(), calltimes)

data_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Proxy-Connection": "keep-alive",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36",
    "Accept-Encoding": "gzip, deflate, br",
    # 'Connection': 'close',
}

def imresize(src, height):
    ratio = src.shape[0] * 1.0 / height
    width = int(src.shape[1] * 1.0 / ratio)
    return cv2.resize(src, (width, height))

#获取图像背景平均亮度及前景粗略轮廓
def get_mainitem_mask_rough(im):
    """
    :param im:
    :return: 背景平均亮度，前景mask
    """
    # 检查背景
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray400 = imresize(gray, 400)
    # 边缘锐化
    kenerl3x3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray400 = cv2.filter2D(gray400, cv2.CV_8UC1, kenerl3x3)
    canny400 = cv2.Canny(gray400, 20, 50)
    canny400 = np.where(canny400 == 255, 1, 0).astype('uint8')

    kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    canny400 = cv2.morphologyEx(canny400, cv2.MORPH_CLOSE, kernel7)
    image, contours, hierarchy = cv2.findContours(canny400, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(image,contours,-1,(255,255,255))
    # cv2.imshow('canny',image)
    # cv2.waitKey(0)
    # 获取前景mask
    for contour in contours:
        newcontour = contour.reshape(contour.shape[0], contour.shape[2])
        cv2.fillPoly(canny400, [newcontour], (1, 1, 1))
    # cv2.imshow('canny300',canny300*255)
    # cv2.waitKey(0)
    kernel9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    canny400 = cv2.dilate(canny400,kernel9)
    bg400 = np.where(canny400 == 0, gray400, 0).astype('uint8')
    bgArea = np.where(canny400 == 0, 1, 0).sum()

    # # bgr三通道的均值
    # im300 = imresize(im,300)
    # averageB = round(np.where(canny300 == 0, im300[:, :, 0], 0).sum() / (bgArea + 0.0001))
    # averageG = round(np.where(canny300 == 0, im300[:, :, 1], 0).sum() / (bgArea + 0.0001))
    # averageR = round(np.where(canny300 == 0, im300[:, :, 2], 0).sum() / (bgArea + 0.0001))
    # print(averageB, averageG, averageR)

    # 背景平均亮度
    aveBg = 0
    if (bgArea*10>canny400.shape[0]*canny400.shape[1]) or (bgArea*20>canny400.shape[0]*canny400.shape[1] and len(contours)==2):
        aveBg = round(bg400.sum() / bgArea)
    cannyorg = imresize(canny400,im.shape[0])
    # image, contours, hierarchy = cv2.findContours(cannyorg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow('2',cannyorg*255)
    # cv2.waitKey(0)
    return aveBg,cannyorg,gray

#判断目标是否触边不完整(默认白色背景下)
def iscomplete(gray):
    bcomplete = True
    if np.where(gray[1]>253,0,1).sum()*10>gray.shape[1] or np.where(gray[0]>253,0,1).sum()*10>gray.shape[1]:
        bcomplete = False
    if np.where(gray[gray.shape[0]-2]>253,0,1).sum()*10>gray.shape[1] or np.where(gray[gray.shape[0]-1]>253,0,1).sum()*10>gray.shape[1]:
        bcomplete = False
    if np.where(gray[:,1]>253,0,1).sum()*10>gray.shape[0] or np.where(gray[:,0]>253,0,1).sum()*10>gray.shape[0]:
        bcomplete = False
    if np.where(gray[:,gray.shape[1]-2]>253,0,1).sum()*10>gray.shape[0] or np.where(gray[:,gray.shape[1]-1]>253,0,1).sum()*10>gray.shape[0]:
        bcomplete = False
    return bcomplete

#修正图像，只对白底图片有效
def modify_pic(skupath, im_path):
    """

    :param skupath:
    :param im_path:
    :return: modified_im_path
    """
    modified_im_path = ''
    respath = os.path.join(skupath,'modified/')
    if os.path.exists(respath) == False:
        os.makedirs(respath)
    maxratio = 0.85
    pil_imag = Image.open(im_path)
    bestpic = cv2.cvtColor(np.asarray(pil_imag.convert('RGB')), cv2.COLOR_RGB2BGR)

    # bestpic = cv2.imread(im_path)
    # if bestpic.empty():
    #     return im_path
    bg0, mask0, gray0 = get_mainitem_mask_rough(bestpic)

    #白色背景并且主体完整才会进行修正
    if bg0 > 253: #and iscomplete(gray0):
        #补充前景区域
        mask_fg = np.where(gray0 > 253, mask0, 1).astype('uint8')
        image, fgcontours, hierarchy = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #如果商品主体在画面内不完整，则不修正
        if len(fgcontours) == 1 and iscomplete(gray0) == False:
            modified_im_path = im_path
        elif len(fgcontours) == 1:
            itemrect = cv2.boundingRect(fgcontours[0])

            # 判断商品位置及大小是否合理
            itemarea = itemrect[2] * itemrect[3]
            imarea = mask0.shape[0] * mask0.shape[1]
            ratio = itemarea / imarea

            # 修正主图，商品居中，最长边不能超过画面的90%
            if itemrect[2] > itemrect[3]:
                scale = bestpic.shape[1] * maxratio / itemrect[2]
            else:
                scale = bestpic.shape[0] * maxratio / itemrect[3]

            if scale >= 1 and ratio > 0.5 and ratio < 0.8:
                modified_im_path = im_path
            else:
                # 修正主图
                crop = bestpic[itemrect[1]:itemrect[1] + itemrect[3], itemrect[0]:itemrect[0] + itemrect[2]]
                dst = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
                # cv2.imshow('11111',dst)
                # cv2.waitKey(0)
                bottom = int((bestpic.shape[0] - dst.shape[0]) / 2)
                right = int((bestpic.shape[1] - dst.shape[1]) / 2)
                top = (bestpic.shape[0] - dst.shape[0]) - bottom
                left = (bestpic.shape[1] - dst.shape[1]) - right
                dst = cv2.copyMakeBorder(dst, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                         value=(bg0, bg0, bg0))
                newfile = respath + im_path[im_path.rfind('/') + 1:]
                cv2.imwrite(newfile, dst)
                modified_im_path = newfile
        elif len(fgcontours) > 1:
            rects = []
            pic_area = bestpic.shape[0] * bestpic.shape[1]
            for contour in fgcontours:
                boundingbox = cv2.boundingRect(contour)
                center_x = (boundingbox[0] + boundingbox[2]) / 2
                center_y = (boundingbox[1] + boundingbox[3]) / 2
                # 对于不足画面1/6并且在画面四周的区域当作背景处理
                if cv2.contourArea(contour, False) * 6 < pic_area and (
                        center_x * 3 < bestpic.shape[1] or center_y * 3 < bestpic.shape[0] or center_x * 3 >
                        bestpic.shape[1] * 2 or center_y * 3 < bestpic.shape[0] * 2):
                    newcontour = contour.reshape(contour.shape[0], contour.shape[2])
                    cv2.fillPoly(bestpic, [newcontour], (bg0, bg0, bg0))  # logo或者小物件，标记为背景
                else:
                    rects.append(boundingbox)
            if len(rects) > 0:
                x1 = bestpic.shape[1]
                y1 = bestpic.shape[0]
                x2 = 0
                y2 = 0
                for rect in rects:
                    if rect[0] < x1:
                        x1 = rect[0]
                    if rect[1] < y1:
                        y1 = rect[1]
                    if rect[0] + rect[2] > x2:
                        x2 = rect[0] + rect[2]
                    if rect[1] + rect[3] > y2:
                        y2 = rect[1] + rect[3]
                itemrect = [x1, y1, x2 - x1, y2 - y1]

                # 判断商品位置及大小是否合理
                itemarea = itemrect[2] * itemrect[3]
                imarea = bestpic.shape[0] * bestpic.shape[1]

                # 修正主图，商品居中，最长边不能超过画面的90%
                if itemrect[2] > itemrect[3]:
                    scale = bestpic.shape[1] * maxratio / itemrect[2]
                else:
                    scale = bestpic.shape[0] * maxratio / itemrect[3]

                if iscomplete(gray0) == False:
                    dst = bestpic
                else:
                    # 修正主图
                    crop = bestpic[itemrect[1]:itemrect[1] + itemrect[3], itemrect[0]:itemrect[0] + itemrect[2]]
                    # print('scale ,bestpic.shape, crop.shape, itemrect',scale ,bestpic.shape, crop.shape, itemrect)
                    # print(int(crop.shape[1] * scale), int(crop.shape[0] * scale))
                    dst = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
                    # cv2.imshow('11111',dst)
                    # cv2.waitKey(0)
                    bottom = int((bestpic.shape[0] - dst.shape[0]) / 2)
                    right = int((bestpic.shape[1] - dst.shape[1]) / 2)
                    top = (bestpic.shape[0] - dst.shape[0]) - bottom
                    left = (bestpic.shape[1] - dst.shape[1]) - right
                    dst = cv2.copyMakeBorder(dst, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                             value=(bg0, bg0, bg0))
                newfile = respath + im_path[im_path.rfind('/') + 1:]
                cv2.imwrite(newfile, dst)
                modified_im_path = newfile
            else:
                modified_im_path = im_path
    else:
        modified_im_path = im_path
    return modified_im_path

#下载图像，默认20s超时
def download_image(image_url, file_path, timeout=20):

    response = None
    try_times = 0
    while True:
        try:
            try_times += 1
            response = requests.get(image_url, headers=headers, timeout=timeout)
            # print(file_path)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            response.close()
            break

        except Exception as e:
            if try_times < 2:
                continue
            if response:
                response.close()
            print("## Fail:  {}  {}".format(image_url, e.args))
            break


# 下载图片到本地缓存
def downloadimgs(skupath, urllist):
    """

    :param skupath:对应的缓存目录
    :param urllist:
    :return:{localpath1:url1,localpath2:url2,localpath3:url3...}
    """
    #800x800
    # prefix = 'http://img14.360buyimg.com/n12/'
    prefix = 'http://img13.360buyimg.com/n12/'
    #350x350
    # prefix = 'http://img14.360buyimg.com/n1/'
    reslist = collections.OrderedDict()

    if os.path.exists(skupath) == False:
        os.makedirs(skupath)
    for urlinfo in urllist:
        file_name = urlinfo[urlinfo.rfind('/') + 1:]
        filepath = os.path.join(skupath, file_name)
        filepath = filepath.replace('\\', '/')
        if os.path.exists(filepath) == False:
            try:
                # print('downloading ', prefix+urlinfo)
                # request.urlretrieve((prefix+urlinfo), filepath)
                # print(file_name)
                download_image((prefix+urlinfo), filepath)
                reslist[filepath] = urlinfo

            except:
                print('download image failed!url:', urlinfo)
        else:
            reslist[filepath] = urlinfo

    return reslist

def uploadImage(localPath):
    print ("local path: %s" % localPath)
    url='http://upload.erp.360buyimg.local/imageUpload.action'
    headers={'aucode': 'f2424b3a07a5604f0209416035a4923a', 'type':'0', 'keycode':'860f79451387081_'}
    f=open(localPath,'rb')
    data=f.read()
    r = requests.post(url, data=data, headers=headers)
    return r.json()[0]['msg']


def evaluate_pic(skupath, im_path, model, use_gpu):
    res_path = ''
    pic_type = -1
    bg_color = 'unknown'
    try:
        img = Image.open(im_path)
        image = img.convert('RGB')
        image = data_transform(image)
    except Exception as e:
        print(im_path)
        print(e)
        return bg_color, pic_type, res_path
    #如果宽高异常，直接退出
    width, height = img.size
    if abs(width-height) > 20:
        return bg_color, pic_type, res_path
        
    image = Variable(image.unsqueeze(0), volatile=True)
    if use_gpu:
        image = image.cuda()

    #主体分，背景白色概率，有logo概率，有文字概率，有二维码概率
    res1,res2,res3,res4,res5 = model(image)
    score1 = functional.softmax(res1, dim=1).data[0][1]
    score2 = functional.softmax(res2, dim=1).data[0][0]
    score3 = functional.softmax(res3, dim=1).data[0][1]
    score4 = functional.softmax(res4, dim=1).data[0][1]
    score5 = functional.softmax(res5, dim=1).data[0][1]
    score1_0 = score1
    if score2 > 0.6:
        bg_color = 'white'
    elif score2 < 0.4:
        bg_color = 'color'

    #如果背景为白色,会进行图像修正
    if score2 > 0.8:
        #如果主体不好，但没有logo、文字和牛皮癣，可直接作为子图，无需修正
        if score1_0 < 0.6 and score3 < 0.3 and score4 < 0.2 and score5 < 0.2:
            modified_path = im_path
        else:
            try:
                modified_path = modify_pic(skupath, im_path)
            except Exception as e:
                print('modify image failed!', im_path)
                # print(e)
                logging.exception(e)
                modified_path = im_path

        if len(modified_path) > 0 and modified_path != im_path:
            try:
                img2 = Image.open(modified_path)
                image2 = img2.convert('RGB')
                image2 = data_transform(image2)
                image2 = Variable(image2.unsqueeze(0), volatile=True)
                if use_gpu:
                    image2 = image2.cuda()
                # 主体分，背景白色概率，有logo概率，有文字概率，有二维码概率
                res1, res2, res3, res4, res5 = model(image2)
                score1 = functional.softmax(res1, dim=1).data[0][1]
                score3 = functional.softmax(res3, dim=1).data[0][1]
                score4 = functional.softmax(res4, dim=1).data[0][1]
                score5 = functional.softmax(res5, dim=1).data[0][1]
            except Exception as e:
                print(modified_path)
                print(e)
        if score1_0 > 0.8 and score1 > 0.8 and score3 < 0.2 and score4 < 0.1 and score5 < 0.2:
            pic_type = 2  #满足主图
            res_path = modified_path
        elif score4 < 0.1 and score5 < 0.2:
            pic_type = 1  #不满足主图但满足子图需求
            res_path = modified_path
        else:
            res_path = modified_path
    else:
        if score1_0 > 0.8 and score3 < 0.2 and score4 < 0.1 and score5 < 0.2:
            pic_type = 2  #满足主图
            res_path = im_path
        elif score4 < 0.1 and score5 < 0.2:
            pic_type = 1  #不满足主图但满足子图需求
            res_path = im_path
        else:
            res_path = im_path

    #对于模型预测为白底的图像，检查两边两列是否完全为白色
    if bg_color == 'white' and len(res_path) > 0:
        try:
            gray = np.array(Image.open(res_path).convert('L')).astype('uint8')

            # 计算左右两列是否全为白色
            if np.where(gray[:, 1] == 255, 0, 1).sum() * 100 > gray.shape[0] or np.where(gray[:, 0] == 255, 0,
                                                                                         1).sum() * 100 > gray.shape[
                0] or np.where(gray[:, gray.shape[1] - 2] == 255, 0, 1).sum() * 100 > gray.shape[0] or np.where(
                    gray[:, gray.shape[1] - 1] == 255, 0, 1).sum() * 100 > gray.shape[0]:
                bg_color = 'unknown'
        except Exception as e:
            print(e)
            bg_color = 'unknown'

    return bg_color, pic_type, res_path


#对外接口
#callable class
class GetValidPics(object):
    """筛选修正图片

    Args:
        model_path:打分模型路径
    """

    def __init__(self, model_path='./param_best.pth', similarity_model='./squeezenet1_1-f364aa15.pth', use_gpu=False, device_id=0):
        print('prepare model!')
        # 加载到cpu
        self.model = multi_task_resnet.MultiTaskResnet()
        self.use_gpu = False
        self.device_id = device_id
        if use_gpu and torch.cuda.is_available():
            self.use_gpu = True
        if self.use_gpu:
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model = self.model.eval()
        print('load model finished!')
        self.compute_similarity = pic_similarity.ComputeSimilarity(model_path=similarity_model, usegpu=use_gpu)


    def __call__(self, skuid, url_list):
        """
        Args:
            url_list: [str1,str2,str3,...]  ,str(url:is_primary)

        Returns:
            a list,[url1,url2,url3] or []
        """
        print('processing skuid:', skuid)

        # 本地缓存
        tmppath = '/data1/tmp/'
        if os.path.exists(tmppath) == False:
            os.makedirs(tmppath)
        reslist = []
        main_image_bg = 'unknown'

        # 获得URL，只取最新的图片
        srclist = []
        for info in url_list:
            urlinfo = info.split(':')
            if urlinfo[1] == '1' and len(srclist) > 1:
                break
            srclist.append(urlinfo[0])
        t1 = cv2.getTickCount()
        # 下载图片到本地缓存
        skupath = os.path.join(tmppath, skuid)
        if os.path.exists(skupath) == False:
            os.makedirs(skupath)
        pic_dict = downloadimgs(skupath, srclist)
        t2 = cv2.getTickCount()
        print('download time:%.2f s' % ((t2 - t1) / cv2.getTickFrequency()))

        t1 = cv2.getTickCount()
        # 返回图片路径 list，list长度为0-3，分别是[主图，子图1，子图2]（如果有）
        # 路径为原始url,主图如果修改过，则路径为新上传的路径
        main_pic_list = []
        sub_pic_list = []
        main_pic_bg = []

        #处理所有图片
        for pic in pic_dict.keys():
            #意味着得到5张有效图片后就会直接退出
            if len(main_pic_list) > 0 and len(main_pic_list)+len(sub_pic_list) > 4:
                break
            #判断图片是否合适，并修正图片
            bg_color, pic_type, res_path = evaluate_pic(skupath, pic, self.model, self.use_gpu)
            if pic_type == 2:
                main_pic_list.append(res_path)
                main_pic_bg.append(bg_color)
            elif pic_type == 1:
                sub_pic_list.append(res_path)
        md5_str = ''
        #至少有一张图片适合做主图时才会比较子图
        if len(main_pic_list) > 0:
            valid_pics = []
            # 确定主图
            main_image_bg = main_pic_bg[0]
            valid_pics.append(main_pic_list[0])
            total_pics = main_pic_list + sub_pic_list
            total_num = len(total_pics)

            #去除重复图片
            for i in range(1, total_num):
                if len(valid_pics) >= 3:
                    break
                have_similar_pic = False
                for valid_pic in valid_pics:
                    if self.compute_similarity(valid_pic, total_pics[i]):
                        have_similar_pic = True
                        break
                if have_similar_pic==False:
                    valid_pics.append(total_pics[i])

            for i in range(min(3, len(valid_pics))):
                # if valid_pics[i] in pic_dict.keys():
                #     url = pic_dict[valid_pics[i]]
                # else:
                #     url = uploadImage(valid_pics[i])
                # reslist.append(url)
                #本地测试，输出结果为本地路径
                reslist.append(valid_pics[i])
                md5, _ = File_md5(valid_pics[i])
                md5_str = md5_str + str(md5)
        # 删除缓存
        # if os.path.exists(skupath) == True:
        #    shutil.rmtree(skupath)
        res_md5 = hashlib.md5(md5_str.encode('utf-8')).hexdigest()
        t2 = cv2.getTickCount()
        print('process time:%.2f s' % ((t2 - t1) / cv2.getTickFrequency()))
        return main_image_bg, reslist, res_md5









