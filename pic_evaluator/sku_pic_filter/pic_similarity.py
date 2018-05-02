import squeezenet
import torch
import os
from PIL import Image
import transforms
from torch.autograd import Variable
from torch.nn import functional

data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class ComputeSimilarity:
    #判定是否使用gpu
    def canusegpu(self):
        return (self.usegpu == True and self.gpuavaiable == True)

    def __init__(self, model_path = '/home/zjw/projects/pic_evaluator/sku_pic_filter/squeezenet1_1-f364aa15.pth', threshold=0.85, usegpu=False):
        self.model = squeezenet.getSqueezenet11()  #获取squeezenet模型
        self.usegpu = usegpu
        self.gpuavaiable = False
        if usegpu:  #判定是否使用gpu
            self.gpuavaiable = torch.cuda.is_available()
        #准备imagenet预训练模型
        if os.path.exists(model_path) == False:
            raise Exception("file not exsit, " + model_path)
        try:
            print('load squeezenet model...')
            if self.gpuavaiable:
                self.model.load_state_dict(torch.load(model_path), strict=False)
                self.model = self.model.cuda()
            else:
                self.model.load_state_dict(
                torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
            print('load model successful.')
        except (Exception):
            raise Exception("file has coruppted, load failed " + model_path)
        if self.canusegpu():
            self.model = self.model.cuda()
        self.model = self.model.eval()
        self.threshold = threshold  #设置比较阈值


    def is2imgsimular(self, imgpath1, imgpath2, threshold=-1):
        if imgpath1 == None or imgpath2 == None:
            raise Exception("input param should not be None, care imgpath1 and imgpath2")
        if os.path.exists(imgpath1) == False:
            raise Exception("file not exsit, " + imgpath1, imgpath1)
        if os.path.exists(imgpath2) == False:
            raise Exception("file not exsit, " + imgpath1, imgpath2)

        # 加载图像
        try:
            im1 = Image.open(imgpath1)
            im1 = im1.convert('RGB')
            im1 = data_transform(im1)
            input1 = Variable(im1.unsqueeze(0), volatile=True)
            im2 = Image.open(imgpath2)
            im2 = im2.convert('RGB')
            im2 = data_transform(im2)
            input2 = Variable(im2.unsqueeze(0), volatile=True)
        except Exception as e:
            print(e)

        if self.canusegpu():
            input1 = input1.cuda()
            input2 = input2.cuda()
        out1 = self.model(input1)  # 计算前向传播
        out2 = self.model(input2)
        simularity = functional.cosine_similarity(out1, out2)  # 计算余弦相似度
        # print(simularity)
        # print("simularity: "+str(simularity))
        # return simularity.data[0]
        # print(simularity.data[0])
        if threshold == -1:
            return simularity.data[0] > self.threshold
        return simularity.data[0] > threshold

    def __call__(self, imgpath1, imgpath2, threshold=-1):
        return self.is2imgsimular(imgpath1, imgpath2, threshold)
