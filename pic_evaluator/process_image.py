""" process images """
# -*- coding: utf-8 -*-
# author: Zhu Junwei

import os
from PIL import Image, ImageOps, ImageEnhance
import random
import numpy as np


def _is_pil_image(img):
    return isinstance(img,Image.Image)

class Crop_img(object):
    """
    crop image to a given size or ratio
    Args:
        min_size (tuple):(min_h,min_w),if image is too small,the process will not be excuted.
        output_size (tuple):desired output size.(h,w)
        ratio [0.0-1.0]:h/w,if ratio==0,use output_size,if ratio>0,output will fixed to the ratio

    """
    def __init__(self,min_size,output_size,output_ratio=0):
        assert isinstance(min_size, tuple)
        assert isinstance(output_size,tuple)
        assert (output_ratio >= 0 and output_ratio <= 1)
        self.min_h, self.min_w = min_size
        self.output_size = output_size
        self.output_ratio = output_ratio

    def __call__(self, pic):
        """
        Crop the given PIL Image
        :param pic:
        :return:
        """
        if not(_is_pil_image(pic)):
            raise TypeError('pic should be PIL Image.Got{}'.format(type(pic)))
        w, h = pic.size
        if w < self.min_w or h < self.min_h:
            print('pic is too small to be croped!')
            return None
        else:
            if self.output_ratio == 0:
                th, tw = self.output_size
                if tw > w or th > h:
                    print('output size is too big!')
                    return None
                if w == tw and h == th:
                    i = 0
                    j = 0
                else:
                    i = random.randint(0,h - th)
                    j = random.randint(0,w - tw)

            else:
                if h > w * self.output_ratio:
                    th = int(w * self.output_ratio)
                    tw = w
                    i = random.randint(0,h - th)
                    j = 0
                elif h == w * self.output_ratio:
                    th = h
                    tw = w
                    i = 0
                    j = 0
                else:
                    th = h
                    tw = int(h / self.output_ratio)
                    i = 0
                    j = random.randint(0,w - tw)
            return pic.crop((j, i, j + tw, i + th))


#改变透明度
def reduce_opacity(mark, opacity):
    assert opacity >= 0 and opacity <= 1
    if mark.mode != 'RGBA':
        mark  = mark.convert('RGBA')
    alpha = mark.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    mark.putalpha(alpha)
    return mark

def watermark(im, mark, position, opacity=1):
    """
    add watermark to a image
    :param im:
    :param mark:
    :param position:
    :param opacity:
    :return:
    """
    if mark.mode != 'RGBA':
        mark = mark.convert('RGBA')
    if opacity < 1:
        mark = reduce_opacity(mark, opacity)
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    # create a transparent layer the size of the image and draw the
    # watermark in that layer.
    layer = Image.new('RGBA', im.size, (255,255,255,0))
    # mark = mark.resize((200,200),resample=Image.BILINEAR)
    layer.paste(mark, position)
    # composite the watermark with the layer
    return Image.composite(layer, im, layer)



class Add_mark(object):
    """
    add some kind of mark to a image
    output_size:a tuple,(h,w)
    type:logo,text,qrcode

    """
    def __init__(self,output_size, type):
        assert isinstance(output_size, tuple)
        if type not in ('logo','text','qrcode'):
            raise ValueError("unknown mark type")
        self.output_size = output_size
        self.mark_type = type

    def __call__(self, pic, mark):
        """
        add mark to a PIL image
        :param pic:
        :param mark:
        :return:
        """
        mark_w, mark_h = mark.size
        out_h, out_w = self.output_size
        if self.mark_type == 'text':
            dtw = random.randint(int(out_w/10),int(out_w))
            dth = random.randint(int(out_h)/10,int(0.95*out_h))
            resize_ratio = min(dtw/mark_w,dth/mark_h)
        elif self.mark_type == 'qrcode':
            dtw = random.randint(int(out_w / 5), int(out_w / 2))
            dth = random.randint(int(out_h) / 5, int(out_h / 2))
            resize_ratio = min(dtw / mark_w, dth / mark_h)
        else:
            dtw = random.randint(int(out_w / 10), int(out_w/4))
            dth = random.randint(int(out_h) / 10, int(out_h/4))
            resize_ratio = min(dtw / mark_w, dth / mark_h)

        mark = mark.resize((int(mark_w*resize_ratio),int(mark_h*resize_ratio)),resample=Image.BILINEAR)
        # print(mark.size)

        #where to put the mark
        top = random.randint(0,out_h - int(mark_h*resize_ratio))
        left = random.randint(0,out_w - int(mark_w*resize_ratio))
        if self.mark_type == 'logo':
            if top > (out_h/4 - int(mark_h*resize_ratio)):
                top = 0
            # if left > (out_w/4 - int(mark_w*resize_ratio)) and left < 0.75*out_w:
            #     left = 0

        pic = pic.resize((out_w,out_h),resample=Image.BILINEAR)

        opacity = random.uniform(0.8,1)
        if self.mark_type == 'text':
            oopacity = random.uniform(0.2,1)
        res = watermark(pic, mark, (left,top), opacity=opacity)
        return res


