from threading import Thread

import cv2

class VideoGet:

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        #        self.stream.set(3,1920)
        #        self.stream.set(4,1080)
        self.stream.set(3, 640)
        self.stream.set(4, 480)
        #        self.stream.set(3,1280)
        #        self.stream.set(4,720)
        self.stream.set(cv2.CAP_PROP_BRIGHTNESS, 300)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False


    def start(self):
        Thread(target=self.get, args=()).start()
        return self


    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()


    def stop(self):
        self.stopped = True
