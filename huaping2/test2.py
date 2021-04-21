#!/usr/bin/env Python
# -*- coding:utf-8 -*-
# aesEncryp.py
# aes加密
# author:Ethan

import os,re
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        if feature_extract:
            for param in model_ft.parameters():
                # 是否训练所以层
                param.requires_grad = True
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                   nn.LogSoftmax(dim=1))
        input_size = 224

    return model_ft, input_size


####   测试训练效果
model_name = 'resnet'       #  使用resnet网络模型
feature_extract = True      # 是否使用推荐的训练特征
model_ft, input_size = initialize_model(model_name, 2, feature_extract, use_pretrained=True)

# CPU或GPU模式
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

filename='checkpoint2.pth'
# 加载训练好的模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft.eval()

def test(img:Image) ->str:
    image = Image.open(r'./data/test/1/e20201215030818.png')
    image = transforms.Resize(256)(image)
    image =transforms.CenterCrop(224)(image)
    image = transforms.ToTensor()(image)
    image =transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    # image = transforms(image)
    image = torch.unsqueeze(image,0)

    output = model_ft(image.cuda())
    _, preds_tensor = torch.max(output, 1)
    # preds = np.squeeze(preds_tensor.numpy())
    preds = np.squeeze(preds_tensor.numpy()) if not True else np.squeeze(preds_tensor.cpu().numpy())
    cat_to_name ={"0":"Normal", "1":"Error"}
    return cat_to_name[str(preds)]


def discriminateing(img: Image) -> str:
    image = img
    image = transforms.Resize(256)(image)
    image = transforms.CenterCrop(224)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    # image = transforms(image)
    image = torch.unsqueeze(image, 0)

    output = model_ft(image.cuda())
    _, preds_tensor = torch.max(output, 1)
    # preds = np.squeeze(preds_tensor.numpy())
    preds = np.squeeze(preds_tensor.numpy()) if not True else np.squeeze(preds_tensor.cpu().numpy())
    cat_to_name = {"0": "Normal", "1": "Error"}
    return cat_to_name[str(preds)]

def getImageType(file):
    searchObj = re.match(r'n', file, re.I)     #re.I表示不区分大小写
    if searchObj:
        return "Normal"
    searchObj2 = re.match(r'e', file, re.I)
    if searchObj2:
        return "Error"

dir = os.getcwd()
files = os.listdir(r"./data/test/0")
for file in files:
    image = Image.open(os.path.join(r'./data/test/0', file))
    print("{}:label--{}, prospective--{}".format(file, getImageType(file), discriminateing(image)))