#!/usr/bin/env Python
# -*- coding:utf-8 -*-
# reveiveImage.py
# 接受图片
# author:Ethan

import threading
import io,os,time
import socket
import json
import queue
from PIL import Image


class ReceiveImage(threading.Thread):
    # 建立一个服务端
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 9090))  # 绑定要监听的端口
    server.listen(5)  # 开始监听 表示可以使用五个链接排队
    conn, addr = server.accept()  # 等待链接,多个链接的时候就会出现问题,其实返回了两个值

    def __init__(self,mqueue, name:'str'):
        self.mQueue = mqueue    # 初始化时传入图片队列，图片队列由控制类初始化,和客户端不同，这里存的是图片
        super().__init__()
        self.name = name
        self._silock = threading.Lock()
        self._running = True

    def receiveImage(self,conn):
        mparaStr = conn.recv(1024)
        mpara = json.loads(mparaStr)    # 返回的是dict类型
        filename = mpara['imagename']
        size = int(mpara['size'])

        if size and filename:
            # 生产环境中要从客户端接口获取文件名
            newfilename = filename + ".png"
            conn.send(b'ok')

            # 第二次接收，这次接受图片
            file = b''
            get = 0
            while get < size:
                data = conn.recv(1024)
                file += data
                get += len(data)
            conn.send(b'copy')

            # 对图片进行处理
            byte_stream = io.BytesIO(file)  # 把获取到的数据转换为Bytes字节流
            img = Image.open(byte_stream)  # Image打开Byte字节流数据
            # img.show()

            # 判断图片是保存文件还是传到队列
            flag = True
            if flag:
                i = 0
                while self.mQueue.full() and i<10:
                    print("time.sleep(0.2)")
                    time.sleep(0.2)
                    i = i+1
                if i!=10:
                    imgItem = {"parameters":mpara,"image":img}
                    self._silock.acquire()
                    self.mQueue.put(imgItem)
                    self._silock.release()

                    # # 同时保持图片
                    # pwdpath = os.getcwd()
                    # imageDir = os.path.join(pwdpath, "images")
                    # if not os.path.exists(imageDir):
                    #     os.mkdir(imageDir)
                    # newfile = open(os.path.join(imageDir, filename), 'wb')
                    # newfile.write(file[:])
                    # newfile.close()
            else:
                # io方法1
                pwdpath = os.getcwd()
                imageDir = os.path.join(pwdpath, "images")
                if not os.path.exists(imageDir):
                    os.mkdir(imageDir)
                newfile = open(os.path.join(imageDir, filename), 'wb')
                newfile.write(file[:])
                newfile.close()
                # io方法2
                # img.save(newfilename)



    def run(self):
        conn = ReceiveImage.conn
        addr = ReceiveImage.addr
        print(conn, addr)
        n = 20210116000
        while self._running:
            # print("receiveImage.run() mQueue.qsize():", self.mQueue.qsize())
            try:
                self.receiveImage(conn)
            except ConnectionResetError as e:
                print('关闭了正在占线的链接！')
                break

    def stop(self):
        self._running = False
        ReceiveImage.conn.close()


def main():
    mqueue = queue.Queue(20)
    ri = ReceiveImage(mqueue, "R1")
    ri.start()
    ri.join()

if __name__ == '__main__':
    main()