# -*- coding: utf-8 -*-
""" 
@Time    : 2018/1/4 11:16
@Author  : Zhu Junwei
@File    : gen_samples.py
"""
import os
import random
from process_image import *

rootpath = '/data1/data-0123/org/'
dstpath = '/data1/data-0123/add_mark/'
if os.path.exists(dstpath) == False:
    os.makedirs(dstpath)

logo_path = '/data1/logo/'
text_path = '/data1/text/'
qrcode_path = '/data1/qrcode/'
logo_list = os.listdir(logo_path)
text_list = os.listdir(text_path)
qrcode_list = os.listdir(qrcode_path)

with open('org_iamge_labels.txt','w') as orgfile:
    for root, path, files in os.walk(rootpath):
        for file in files:
            pos = root.rfind('/')
            if pos > 0:
                info = root[pos+1:].split('_')
                refpath = os.path.join(root[pos+1:], file)
                write_line = refpath + ' ' + info[0] + ' ' + info[1] + ' ' + info[2] + ' ' + info[3] + ' ' + info[4] + '\n'
                orgfile.write(write_line)


orgfile = open('org_iamge_labels.txt','r')
orgfiles = orgfile.readlines()
orgfile.close()
add_logo = Add_mark((600,600),'logo')
add_text = Add_mark((600,600),'text')
add_qrcode = Add_mark((600,600),'qrcode')


# newlabels = open('train_Labels.txt','w')
# for line in orgfiles:
#     newline = line.strip()
#     info = newline.split()
#     filepath = rootpath + info[0]
#     try:
#         im = Image.open(filepath)
#         pic = im.copy()
#     except:
#         print('open image failed!',filepath)
#         continue
#     newlabels.write(line)
# for line in orgfiles:
#     newline = line.strip()
#     info = newline.split()
#     filepath = rootpath + info[0]
#     try:
#         im = Image.open(filepath)
#         pic = im.copy()
#     except:
#         print('open image failed!',filepath)
#         continue
#     logo_seed = random.randint(0, 100000)
#     text_seed = random.randint(0, 100000)
#     qrcode_seed = random.randint(0, 20000)
#     bchanged = False
#     if logo_seed < len(logo_list):
#         logofile = logo_path + logo_list[logo_seed]
#         try:
#             logo = Image.open(logofile)
#             pic = add_logo(pic, logo)
#             info[3] = '1'
#             bchanged = True
#         except:
#             print('open image failed!', logofile)
#     if text_seed < len(text_list):
#         for i in range(random.randint(1,4)):
#             text_seed = random.randint(0,49999)
#             textfile = text_path + text_list[text_seed]
#             try:
#                 text = Image.open(textfile)
#                 #透明背景，白色字体
#                 w,h = text.size
#                 if w == 650 and info[2] == '0':
#                     print('BG White and text is white!')
#                 else:
#                     pic = add_text(pic, text)
#                     info[4] = '1'
#                     bchanged = True
#             except Exception as e:
#                 print(e)
#
#     if qrcode_seed < len(qrcode_list):
#         qrcodefile = qrcode_path + qrcode_list[qrcode_seed]
#         try:
#             qrcode = Image.open(qrcodefile)
#             pic = add_qrcode(pic, qrcode)
#             info[5] = '1'
#             bchanged = True
#         except:
#             print('open image failed!', qrcodefile)
#     if bchanged == True:
#         pos = info[0].rfind('/')
#         name = info[0][pos+1:]
#         pos = name.rfind('_')
#         newname = name[:pos]+'_1'+name[pos:]
#         info[0] = savepath + newname
#         pic = pic.convert('RGB')
#         pic.save(dstpath+newname)
#         # write_line = info[0]+' '+info[1]+' '+info[2]+' '+info[3]+' '+info[4]+' '+info[5]+'\n'
#         # newlabels.write(write_line)
for line in orgfiles:
    newline = line.strip()
    info = newline.split()
    filepath = rootpath + info[0]
    try:
        im = Image.open(filepath)
        pic = im.copy()
    except:
        print('open image failed!',filepath)
        continue

    logo_seed = random.randint(0, 200000)
    text_seed = random.randint(0, 200000)
    qrcode_seed = random.randint(0, 100000)
    bchanged = False
    #没有logo,则有一定概率添加logo
    if int(info[3])==0 and logo_seed < len(logo_list):
        logofile = logo_path + logo_list[logo_seed]
        try:
            logo = Image.open(logofile)
            pic = add_logo(pic, logo)
            info[3] = '1'
            bchanged = True
        except:
            print('open image failed!', logofile)
    #没有文字，则有一定概率添加文字
    if int(info[4])==0 and text_seed < len(text_list):
        for i in range(random.randint(1,4)):
            text_seed = random.randint(0,49999)
            textfile = text_path + text_list[text_seed]
            try:
                text = Image.open(textfile)
                # 透明背景，白色字体
                w, h = text.size
                if w == 650 and info[2] == '0':
                    print('BG White and text is white!')
                else:
                    pic = add_text(pic, text)
                    info[4] = '1'
                    bchanged = True
            except Exception as e:
                print(e)
    #没有二维码，则有一定概率添加二维码
    if bchanged == False and int(info[5])==0 and qrcode_seed < len(qrcode_list):
        qrcodefile = qrcode_path + qrcode_list[qrcode_seed]
        try:
            qrcode = Image.open(qrcodefile)
            pic = add_qrcode(pic, qrcode)
            info[5] = '1'
            bchanged = True
        except:
            print('open image failed!', qrcodefile)
    if bchanged == True:
        pos = info[0].rfind('/')
        name = info[0][pos+1:]
        pos = name.rfind('.')
        newname = name[:pos] + '_2' + name[pos:]
        refpath = os.path.join(dstpath,info[1]+'_'+info[2]+'_'+info[3]+'_'+info[4]+'_'+info[5])
        if os.path.exists(refpath)==False:
            os.makedirs(refpath)
        # info[0] = os.path.join(savepath,info[1]+'_'+info[2]+'_'+info[3]+'_'+info[4]+'_'+info[5],newname)
        # info[0] = savepath + newname
        pic = pic.convert('RGB')
        pic.save(os.path.join(refpath,newname))
#         write_line = info[0]+' '+info[1]+' '+info[2]+' '+info[3]+' '+info[4]+' '+info[5]+'\n'
#         newlabels.write(write_line)
# newlabels.close()







