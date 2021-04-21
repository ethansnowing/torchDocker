#!/usr/bin/env Python
# -*- coding:utf-8 -*-
# huaping.py
# 花屏检测线程
# author:Ethan

import threading
import queue
import time
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

class Huaping(threading.Thread):
    # 花屏检测
    def __init__(self, mqueue, name:'str'):
        self.mQueue = mqueue    # 初始化时传入图片队列，图片队列由控制类初始化,和客户端不同，这里存的是图片
        super().__init__()
        self.name = name
        self._silock = threading.Lock()
        self._running = True

        ####   测试训练效果
        model_name = 'resnet'  # 使用resnet网络模型
        feature_extract = True  # 是否使用推荐的训练特征
        self.model_ft, self.input_size = self.initialize_model(model_name, 2, feature_extract, use_pretrained=True)

        # CPU或GPU模式
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_ft = self.model_ft.to(device)

        filename = 'checkpoint2.pth'
        # 加载训练好的模型
        checkpoint = torch.load(filename)
        best_acc = checkpoint['best_acc']
        self.model_ft.load_state_dict(checkpoint['state_dict'])
        self.model_ft.eval()

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # 选择合适的模型，不同模型的初始化方法稍微有点区别
        self.model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet152
            """
            self.model_ft = models.resnet152(pretrained=use_pretrained)
            if feature_extract:
                for param in self.model_ft.parameters():
                    # 是否训练所以层
                    param.requires_grad = True
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                        nn.LogSoftmax(dim=1))
            input_size = 224

        return self.model_ft, input_size


    def discriminateing(self, img: Image) -> str:
        image = img
        image = transforms.Resize(256)(image)
        image = transforms.CenterCrop(224)(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        # image = transforms(image)
        image = torch.unsqueeze(image, 0)

        output = self.model_ft(image.cuda())
        _, preds_tensor = torch.max(output, 1)
        # preds = np.squeeze(preds_tensor.numpy())
        preds = np.squeeze(preds_tensor.numpy()) if not True else np.squeeze(preds_tensor.cpu().numpy())
        cat_to_name = {"0": "Normal", "1": "Error"}
        return cat_to_name[str(preds)]


    def run(self):
        print("start huaping threading")
        while self._running:
            if self.mQueue:
                self._silock.acquire()
                imageItem = self.mQueue.get()
                self._silock.release()
                imageName = imageItem["parameters"]["imagename"]
                img = imageItem["image"]
                pre = self.discriminateing(img)
                print("{} was predicted:{}".format(imageName, pre))
            else:
                print("self.mqueue is empty")
                time.sleep(0.2)



    def stop(self):
        self._running = False

def main():
    mqueue = queue.Queue(20)
    hi = Huaping(mqueue, "R1")
    hi.start()
    hi.join()

if __name__ == '__main__':
    main()