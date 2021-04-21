# 使用pytorch已经训练好的模型resnet，进行鉴别花屏、非花屏图片
# 和huapingDistinguish.py不同的是这个脚本会训练所以层
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
#pip install torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

# 制作好数据源
data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid': transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


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

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

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

data_dir='./data/'
batch_size=8
image_datasets = {'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])}
print(type(image_datasets['test']))
print(image_datasets['test'])
dataloaders = {"test": torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True)}
# dataset_sizes = {"test": len(image_datasets["test"])}

# 得到一个batch的测试数据
# dataiter = iter(dataloaders['test'])
# images, labels = dataiter.next()
for images,labels in dataloaders['test']:
    # print("labels:", labels)
    # print("labels.numpy():",labels.numpy())

    model_ft.eval()
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        output = model_ft(images.cuda())
    else:
        output = model_ft(images)

    _, preds_tensor = torch.max(output, 1)
    # print("type(preds_tensor):",preds_tensor)
    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
    # print(preds)

    # 读取标签对应的实际名字
    cat_to_name ={"0":"Normal", "1":"Error"}
    for idx in range(len(preds)):
        print("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels.numpy()[idx])]))
