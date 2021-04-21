#!/usr/bin/env Python
# -*- coding:utf-8 -*-
# similarity.py
# 画面检测系统服务端检测画面是否更新
# author:Ethan

import queue
import time
import threading
from PIL import Image
import os


class similarity(threading.Thread):

    def __init__(self,mqueue, name:'str'):
        self.mQueue = mqueue    # 初始化时传入图片队列，图片队列由控制类初始化
        super().__init__()
        self.name = name
        self._silock = threading.Lock()
        self._running = True

    """
    已经实现：
    """

    def getGray(self,image_file):
        tmples = []
        for h in range(0, image_file.size[1]):  # h
            for w in range(0, image_file.size[0]):  # w
                tmples.append(image_file.getpixel((w, h)))
        return tmples

    def getAvg(self,ls):  # 获取平均灰度值
        return sum(ls) / len(ls)

    def getMH(self,a, b):  # 比较64个字符有几个字符相同
        dist = 0;
        for i in range(len(a)):
            if a[i] != b[i]:
                dist = dist + 1
        return dist

    def getImgHash(self,img):
        image_file = img
        image_file = image_file.resize((64, 64))  # 重置图片大小我64px X 64px
        image_file = image_file.convert("L")  # 转256灰度图
        Grayls = self.getGray(image_file)  # 灰度集合
        avg = self.getAvg(Grayls)  # 灰度平均值
        bitls = ''  # 接收获取0或1
        # 除去变宽1px遍历像素
        for h in range(image_file.size[1]):  # h
            for w in range(image_file.size[0]):  # w
                if image_file.getpixel((w, h)) >= avg:  # 像素的值比较平均值 大于记为1 小于记为0
                    bitls = bitls + '1'
                else:
                    bitls = bitls + '0'
        return bitls


    def run(self):
        print("start similarity thread")
        tmp1 = ""
        tmp2 = ""
        tmpname = ""
        while self._running:
            if self.mQueue:
                self._silock.acquire()
                imageItem = self.mQueue.get()
                self._silock.release()
                imageName = imageItem["parameters"]["imagename"]
                if imageName[:-5] == tmpname:
                    continue        # 一秒只要一张图片被检测

                stra = self.getImgHash(imageItem["image"])
                if not tmp2:
                    tmp2 = stra
                    tmpname = imageName[:-5]
                    continue
                if not tmp1:
                    tmp1 = tmp2
                    tmp2 = stra
                    continue
                compare1 = self.getMH(tmp1,stra)
                compare2 = self.getMH(tmp2,stra)
                distanse = int((compare1 + compare2*2)/3)
                if distanse < 5:
                    print("similarity got distance of {}:{}".format(imageName, distanse))
                    print("画面可能没有更新")


                # 对比结束后把getImgHash得到的值临时保存起来
                tmp1 = tmp2
                tmp2 = stra
                tmpname = imageName[:-5]    # 一秒只要一张图片被检测

            else :
                print("self.mqueue is empty")
                time.sleep(0.2)

    def stop(self):
        self._running = False






def main():
    mq = queue.Queue(2)


if __name__ == '__main__':
    main()
