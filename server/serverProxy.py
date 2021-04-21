#!/usr/bin/env Python
# -*- coding:utf-8 -*-
# serverProxy.py
# 服务器控制器
# author:Ethan

#!/usr/bin/env Python
# -*- coding:utf-8 -*-
# clientProxy.py
# 结合生产图片和发送图片
# author:Ethan

import os
import queue
import time
import threading
import similarity
import receiveImage
import huaping

class clientProxy():
    """
    pass
    """
    def __init__(self):
        self.mQueue = queue.Queue(100)  # 一个maxsize=100的队列，如果图片平均大小为1MB，则队列最大100MB,要控制大小
        self.ri = receiveImage.ReceiveImage(self.mQueue, "nameOfRiThread")
        self.si = similarity.similarity(self.mQueue, "nameOfSiThread")
        self.hi = huaping.Huaping(self.mQueue, "nameOfHiThread")

    def mstart(self):
        self.ri.start()
        self.si.start()
        self.hi.start()
        self.ri.join()
        self.si.join()
        self.hi.join()



def main():
    cp = clientProxy()
    cp.mstart()

if __name__ == '__main__':
    main()