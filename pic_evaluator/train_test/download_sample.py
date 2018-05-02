# -*- coding: utf-8 -*-
# @Time    : 3/21/18 10:56 AM
# @Author  : Zhu Junwei
# @File    : download_sample.py
import os
import requests
import concurrent.futures as futures


def downloadimgs(datafile):
    cid32namedict = dict()
    cid3downloadednum = dict()
    targetpath = "/data1/36w/"
    cid32namefile = "/data1/img_list/cid32namefinal.txt"


    cid32namelines = open(cid32namefile, mode='r', encoding='UTF-8').readlines()
    for line in cid32namelines:
        words = line.split(' ')
        cid32namedict[words[0]] = words[words.__len__() - 1][0:words[words.__len__() - 1].__len__() - 1]
        cid3downloadednum[words[0]] = 0
    imgdatalines = open(datafile, mode='r', encoding='UTF-8').readlines()

    for cid3 in cid32namedict:
        folder_path = os.path.join(targetpath, cid3 + "_" + cid32namedict.get(cid3))
        if os.path.exists(folder_path) == False:
            os.mkdir(folder_path)

    successcounter = 0
    failcounter = 0
    for line in imgdatalines:
        strlist = line.split()
        skuidstr = strlist[0]
        cid3 = strlist[1]
        if cid3 not in cid32namedict:
            continue
        if cid3downloadednum[cid3] > 3000:
            continue
        imgstr = strlist[2]
        imgurl = "http://img10.360buyimg.com/n12/" + imgstr
        savepath = os.path.join(targetpath, cid3 + "_" + cid32namedict.get(cid3))
        savename = cid3 + '_' + skuidstr + '_' + imgstr.replace('/', '_')
        save_file_path = os.path.join(savepath,savename)
        print(save_file_path)
        if os.path.exists(save_file_path):
            continue
        try:
            response = requests.get(imgurl)
            with open(save_file_path, "wb") as imgfile:
                imgfile.write(response.content)
                successcounter+=1
                imgfile.close()
            cid3downloadednum[cid3] += 1
        except Exception:
            print(skuidstr + " " + imgstr + " cannot download or save successfully!")
            failcounter+=1
        if successcounter % 1000 == 0:
            print("image saved: " + str(successcounter))
    print("done")
    print("image saved: " + str(successcounter))
    print("image failed: " + str(failcounter))





if __name__ == "__main__":
    downloadimgs('/home/zjw/Desktop/new_add_pics.txt')
    # thread_num = 10
    # with futures.ProcessPoolExecutor(max_workers=thread_num) as executor:
    #     future_list = list()
    #     for i in range(thread_num):
    #         file_list = '/data1/img_list/pick_imgs{}.txt'.format(i+1)
    #
    #         future_list.append(executor.submit(downloadimgs, file_list))
    #     for future in futures.as_completed(future_list):
    #         if future.exception():
    #             print(future.exception())
