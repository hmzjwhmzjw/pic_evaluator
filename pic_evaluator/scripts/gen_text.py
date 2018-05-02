# -*- coding: utf-8 -*-
""" 
@Time    : 2017/12/27 17:03
@Author  : Zhu Junwei
@File    : gen_text.py
"""
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFont, ImageDraw
import random
import numpy as np


text = ["物流", "发货", "快递", "送货", "缺点", "收货", "硬伤", "618", "到货",
                         "双11", "京东", "&hellip", "泽天", "奶茶妹妹", "哈哈", "客服", "假１", "货到付款", "好好好", "活动",
                         "产品活动", "生产日期", "配送", "产品信息", "产品特点", "产品功效", "解读", "生产", "产品介绍", "品牌名称",
                         "创始人", "展示", "威胁", "授权书", "异常", "温馨提示", "品牌名称", "实物为准", "使用方法", "报告",
                         "规格", "名称", "问题", "测试对象", "京豆", "正品", "售后", "产品","JD.COM京东","360buy.com","京东商城",
                         "供应商", "店家", "购买", "也还好", "初次使用", "收到","JD.COM京东","360buy.com","京东商城",
                         "很喜欢呀", "应该", "第一次", "如果", "希望", "不知道", "包装", "预期", "批号","JD.COM京东","360buy.com","京东商城",
                         "(/≧▽≦/)", "同事推荐", "哦", "快快快", "假货", "产地", "保质", "期限", "产品功效","JD.COM京东","360buy.com","京东商城",
                         "品牌", "名称", "尽量", "避免","量贩","量贩装","假一赔十","到手价","质量保证","直邮","JD.COM京东","360buy.com","京东商城",
                         "敷", "涂", "人群", "步骤", "合作平台", "致力","超级","囤货","大牌","强劲","JD.COM京东","360buy.com","京东商城",
                         "今夏", "高端", "必备", "大牌", "情人节必备", "新品", "包邮", "买一", "送一", "满100减10", "类别",
                         "压", "功效", "玩嗨了","开具","赠送","购买","发票","订单","评价","商品信息","修改","退换货","催单","退货","换货",
                         "报销","限购","收件","抢购","中国联通","中国联通","测试数据","特别提醒","下载APP","测试","实验室","数据来源",
                         "以往","注:","营业厅","剪卡","游戏礼包","素人","仅限","App","APP","功能","有出入","双卡双得","套餐",
                         "店铺","真伪","查询","承诺","包邮","实拍","型号","所见即所得","实物拍摄","商品","版权","细节","锕化玻璃膜",
                         "于吸","下单","让希",'送',"安装","囊钻","颜色","咸鱼","专利","您","展示","Phone","核心优势","我们","定制","影响",
                         "发售","晒图","礼包","恭喜","免费","合格证","危害","隐患","微信","高德","百度","地图","头疼","领券","购物","导航提示",
                         "福利","二维码","损坏","证书","资质","商场","官方","防扌","下单","APP","破损","簸箕","豪礼","视频体验","防伪","全球好物节",
                         "服务保障","商标","公司","股份","条码","专卖店","销量","专营店","进货","储存","误区","传统后视镜","出厂","耐心等待","请关闭",
                         "设置","病人","施工","车主","充电口","本页面","删除","丕是","流量","普通记录仪","本地音乐","客户","表了","怕怕",
                         "分享", "抢单","国能","每年","公交车","不流畅","同意","人员","社区","平台","参考图","痛苦","勿拍",
                         "宝马专用","一年包换","原厂专供","下单立减","狂欢价","秒杀","爆品直降","送豪礼","购物车","领券","更优惠","直营店"]

fonttype = ["SIMYOU.TTF","simsun.ttc","SIMLI.TTF","simkai.ttf","STHUPO.TTF","STXINGKA.TTF","STCAIYUN.TTF","mingliu.ttc","FZSTK.TTF","simhei.ttf"]
#RGBA文字或者背景颜色,白、蓝、红、绿、黑
color = [(255,255,255,255),(0,0,255,255),(255,0,0,255),(0,255,0,255),(0,0,0,255),(0,255,255,255),(255,0,255,255),(255,255,0,255)]

def gen_text_png(dstpath, num):
    if os.path.exists(dstpath)==False:
        os.makedirs(dstpath)
    for i in range(num):
        color_seed = random.randint(0, len(color) - 1)
        color_bg_seed = random.randint(0, 2 * len(color))
        dstfile = os.path.join(dstpath, '{}.png'.format(i))

        # 新建一个图片，颜色随机，透明度随机
        if color_bg_seed == color_seed or color_bg_seed >= len(color):
            #透明背景，白色子体的图像尺寸为
            if color_seed==0:
                txt = Image.new('RGBA', (650, 130), (0, 0, 0, 0))
            else:
                txt = Image.new('RGBA', (700, 130), (0, 0, 0, 0))
        else:
            txt = Image.new('RGBA', (260, 130), color[color_bg_seed])

        font_seed = random.randint(0, len(fonttype) - 1)
        # 设置要写文字的字体，注意有的字体不能打汉字,这里用的微软雅黑可以
        fnt = ImageFont.truetype("c:/Windows/fonts/" + fonttype[font_seed], 130)
        # 打汉字
        d = ImageDraw.Draw(txt)
        # 文字内容
        text_seed = random.randint(0, len(text) - 1)
        # 写要打的位置，内容,用的字体，文字透明度
        d.text((0, 0), text[text_seed], font=fnt, fill=color[color_seed])
        # #保存加水印后的图片
        txt.save(dstfile)

if __name__=="__main__":
    gen_text_png('E:/text/',50000)
    # color_seed = random.randint(0, len(color) - 1)
    # color_bg_seed = random.randint(0, 2 * len(color))
    #
    # # 新建一个图片，颜色随机，透明度随机
    # txt = Image.new('RGBA', (720, 200), (0, 0, 0, 0))
    #
    # font_seed = random.randint(0, len(fonttype) - 1)
    # # 设置要写文字的字体，注意有的字体不能打汉字,这里用的微软雅黑可以
    # fnt = ImageFont.truetype("c:/Windows/fonts/msyh.ttf" , 100)
    # # 打汉字
    # d = ImageDraw.Draw(txt)
    # # 文字内容
    # # 写要打的位置，内容,用的字体，文字透明度
    # d.text((0, 0), "JD.COM京东", font=fnt, fill=(255,0,0,255))
    # txt.show()


