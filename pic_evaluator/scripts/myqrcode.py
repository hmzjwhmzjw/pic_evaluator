# -*- coding: utf-8 -*-
""" 
@Time    : 2017/12/15 17:11
@Author  : Zhu Junwei
@File    : myqrcode.py
"""

import qrcode
from PIL import Image
import os, sys
import random
import string

def gen_qrcode(string, path, logo=""):
    """
    生成中间带logo的二维码
    需要安装qrcode, PIL库
    @参数 string: 二维码字符串
    @参数 path: 生成的二维码保存路径
    @参数 logo: logo文件路径
    @return: None
    """
    qr = qrcode.QRCode()
    qr.add_data(string)
    qr.make(fit=True)
    img = qr.make_image()
    img = img.convert("RGBA")
    if logo and os.path.exists(logo):
        try:
            icon = Image.open(logo)
            img_w, img_h = img.size
            factor = 4
            size_w = int(img_w / factor)
            size_h = int(img_h / factor)

            icon_w, icon_h = icon.size
            if icon_w > size_w:
                icon_w = size_w
            if icon_h > size_h:
                icon_h = size_h
            icon = icon.resize((icon_w, icon_h), Image.ANTIALIAS)
            w = int((img_w - icon_w) / 2)
            h = int((img_h - icon_h) / 2)
            icon = icon.convert("RGBA")
            img.paste(icon, (w, h), icon)
        except Exception as e:
            print(e)
            print("Add logo failed!")
    img.save(path)
    # img.show()



# def decode_qrcode(path):
#     """
#     解析二维码信息
#     @参数 path: 二维码图片路径
#     @return: 二维码信息
#     """
#     # 创建图片扫描对象
#     scanner = zbar.ImageScanner()
#     # 设置对象属性
#     scanner.parse_config('enable')
#     # 打开含有二维码的图片
#     img = Image.open(path).convert('L')
#     # 获取图片的尺寸
#     width, height = img.size
#     # 建立zbar图片对象并扫描转换为字节信息
#     qrCode = zbar.Image(width, height, 'Y800', img.tobytes())
#     scanner.scan(qrCode)
#     # 组装解码信息
#     data = ''
#     for s in qrCode:
#         data += s.data
#     # 删除图片对象
#     del img
#     # 输出解码结果
#     return data

def gen_random_qrcode(dstpath,logopath, num):
    """

    :param dstpath: 结果保存目录
    :param logopath: logo库
    :param num: 生成二维码个数
    :return:
    """
    if os.path.exists(dstpath)==False:
        os.makedirs(dstpath)
    for i in range(num):
        # 生成随机字符串
        mystrlen = random.randint(1, 255)
        info = ''.join(random.choice(string.printable) for _ in range(mystrlen))
        print(info)
        respath = os.path.join(dstpath,str(i)+'.png')

        # logo库，随机挑选一张或者不选
        logonames = os.listdir(logopath)
        filenum = len(logonames)
        idx = random.randint(0, filenum * 2)
        if idx < filenum:
            icon_path = os.path.join(logopath, logonames[idx])
        else:
            icon_path = ''
        gen_qrcode(info, respath, icon_path)

if __name__ == "__main__":
    gen_random_qrcode('/media/zjw/7915b638-c848-44ae-b4e7-062a479901b1/qrcode/','/media/zjw/7915b638-c848-44ae-b4e7-062a479901b1/logo/',10000)
