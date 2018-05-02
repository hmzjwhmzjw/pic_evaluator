# -*- coding: utf-8 -*-
# @Time    : 3/20/18 6:49 PM
# @Author  : Zhu Junwei
# @File    : data_processor.py

import sys
sys.path.append('./sku_pic_filter/')
import os
import shutil
import collections
import sku_pic_filter.pic_similarity as pic_similarity
import concurrent.futures as futures
from PIL import Image
import random


file_src = '/data1/36w/'
out_path = '/data1/36w_out/'
out_invalid = '/data1/36w_out_invalid/'

#获取每个sku的图片列表
def get_cid3_skus(cid3_path):
    sku_imgs = collections.defaultdict(list)
    files = os.listdir(cid3_path)
    for file in files:
        file_info = file.split('_')
        file_path = os.path.join(cid3_path,file)
        if len(file_info) > 2:
            sku_imgs[file_info[1]].append(file_path)
    return sku_imgs

def split_similarpics(cid3_paths):
    compute_similarity = pic_similarity.ComputeSimilarity(threshold=0.97, usegpu=True)

    # 每个三级类目分开处理
    for cid3_path in cid3_paths:

        dst_out_path = os.path.join(out_path, cid3_path)
        if os.path.exists(dst_out_path) == False:
            os.makedirs(dst_out_path)
        cid3_fullpath = os.path.join(file_src, cid3_path)
        if os.path.isdir(cid3_fullpath):
            sku_imgs = get_cid3_skus(cid3_fullpath)
            total_num = len(sku_imgs)
            idx = 0
            for sku in sku_imgs.keys():
                idx += 1
                if idx % 100 == 0:
                    print('process folder {},processed skus {}/{}'.format(cid3_path,idx,total_num))
                similar_idx = set()
                img_list = sku_imgs[sku]
                img_num = len(img_list)
                for i in range(img_num):
                    if i in similar_idx:
                        continue
                    for j in range(i + 1, img_num):
                        if j in similar_idx:
                            continue
                        try:
                            issimilar = compute_similarity(img_list[i], img_list[j])
                        except:
                            issimilar = True
                        if issimilar:
                            similar_idx.add(j)
                for idx in similar_idx:
                    shutil.move(img_list[idx], dst_out_path)
            print('process folder {}, done.'.format(cid3_path))

def split_over_num(cid3_paths):
    # 每个三级类目分开处理
    for cid3_path in cid3_paths:
        dst_out_path = os.path.join(out_path, cid3_path)
        if os.path.exists(dst_out_path) == False:
            os.makedirs(dst_out_path)
        cid3_fullpath = os.path.join(file_src, cid3_path)
        if os.path.isdir(cid3_fullpath):
            files = os.listdir(cid3_fullpath)
            total_files_num = len(files)

            #单个三级类目多张10000则随机减少到10000张
            if total_files_num > 3000 and total_files_num < 4000:
                move_set = set()
                move_num = total_files_num - 3000
                while len(move_set) < move_num:
                    randomidx = random.randint(0, total_files_num-1)
                    move_set.add(randomidx)
                for idx in move_set:
                    file_path = os.path.join(cid3_fullpath, files[idx])
                    shutil.move(file_path, dst_out_path)




def split_invalid_pic(cid3_paths):
    # 每个三级类目分开处理
    for cid3_path in cid3_paths:
        dst_out_path = os.path.join(out_invalid, cid3_path)
        if os.path.exists(dst_out_path) == False:
            os.makedirs(dst_out_path)
        cid3_fullpath = os.path.join(file_src, cid3_path)
        if os.path.isdir(cid3_fullpath):
            files = os.listdir(cid3_fullpath)
            total_files_num = len(files)
            idx = 0
            for file in files:
                idx += 1
                if idx % 1000 == 0:
                    print('process folder {}:{}/{}'.format(cid3_path, idx, total_files_num))
                file_path = os.path.join(cid3_fullpath, file)
                is_invalid = False
                try:
                    im = Image.open(file_path)
                    width, height = im.size
                    if width != 800 or abs(width - height) > 50:
                        is_invalid = True

                    im = im.resize((100, 100))
                except Exception as e:
                    print(e)
                    is_invalid = True
                if is_invalid:
                    shutil.move(file_path, dst_out_path)



if __name__=='__main__':
    cid3_paths = os.listdir(file_src)
    path_num = len(cid3_paths)
    thread_num = 8

    with futures.ProcessPoolExecutor(max_workers=thread_num) as executor:
        future_list = list()
        for i in range(thread_num):
            begin = int(path_num / thread_num * i)
            end = int(path_num / thread_num * (i + 1))
            thread_paths=cid3_paths[begin:end]
            future_list.append(executor.submit(split_similarpics, thread_paths))
            # future_list.append(executor.submit(split_invalid_pic, thread_paths))
            # future_list.append(executor.submit(split_over_num, thread_paths))
        for future in futures.as_completed(future_list):
            if future.exception():
                print(future.exception())







