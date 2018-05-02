# -*- coding: utf-8 -*-
# @Time    : 1/30/18 11:05 AM
# @Author  : Zhu Junwei
# @File    : demo.py

import pic_filter_v2
import os
import shutil
import concurrent.futures as futures
import threading

def process_pics(lines, todolist, dst):
    skipidx = 0
    idx = 0
    pic_filter = pic_filter_v2.GetValidPics(use_gpu=True)

    for line in lines:
        # skipidx += 1
        # if skipidx < 100000:
        #     continue
        # if idx > 10:
        #     break
        newline = line.strip().split()
        sku = newline[0]
        # if sku in todolist:
        #     url_list = newline[1][2:-2].split('","')
        #     if len(url_list) > 3:
        #         idx += 1
        #         res = pic_filter(sku, url_list)
        #         nameidx = 0
        #         for name in res:
        #             img_type = name[name.rfind('.'):]
        #             newname = os.path.join(dst, sku + '_' + str(nameidx) + img_type)
        #             nameidx += 1
        #             shutil.copyfile(name, newname)

        url_list = newline[1][2:-2].split('","')
        if len(url_list) > 3:
            idx += 1
            bg, res, md5 = pic_filter(sku, url_list)
            print('{} bg is:{},pic md5 is:{}'.format(sku,bg,md5))
            nameidx = 0
            for name in res:
                img_type = name[name.rfind('.'):]
                newname = os.path.join(dst, sku + '_' + str(nameidx) + img_type)
                nameidx += 1
                shutil.copyfile(name, newname)


if __name__ == '__main__':
    dst = '/data1/test/'
    notebook_and_watch = open('notebook_and_watch_sku.txt', 'r')
    skus_todo = notebook_and_watch.readlines()
    notebook_and_watch.close()
    todolist = []
    for sku in skus_todo:
        todolist.append(sku.strip())
    with open('/home/zjw/Desktop/my_test_pics.txt', 'r') as file:
        lines = file.readlines()
    totallines = len(lines)
    thread_num = 4

    # filter_num = 2
    # pic_filters = []
    # for i in range(filter_num):
    #     pic_filters.append(pic_filter_v2.GetValidPics())

    with futures.ProcessPoolExecutor(max_workers=thread_num) as executor:
        future_list = list()
        for i in range(thread_num):
            begin = int(totallines / thread_num * i)
            end = int(totallines / thread_num * (i + 1))
            threadlines=lines[begin:end]
            future_list.append(executor.submit(process_pics, threadlines, todolist, dst))
        for future in futures.as_completed(future_list):
            if future.exception():
                print(future.exception())
    # thread_list = []
    # for i in range(thread_num):
    #     begin = int(totallines / thread_num * i)
    #     end = int(totallines / thread_num * (i + 1))
    #     threadlines=lines[begin:end]
    #     t = threading.Thread(target=process_pics, args=(threadlines, todolist, dst))
    #     t.start()
    #     thread_list.append(t)
    # for t in thread_list:
    #     t.join()