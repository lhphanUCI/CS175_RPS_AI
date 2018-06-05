'''
UI Template from
https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
'''

from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
import time
from enum import Enum
import random
import os

import tensorflow as tf
import numpy as np

import models
import settings
import window_utils
import loadDataset

from numpy import argmax


def debugPrintEntireList(imgList:list, maxListSize:int)->None:
    for i in range(maxListSize):
        print(imgList[i])  

class Classification(Enum):
    NONE = 0
    ROCK = 1
    PAPER = 2
    SCISSOR = 3

    def __str__(self):
        return self.name.title()

"""
def getFilledBlankImgList( maxListSize:int, inputDimensions:(int, int, int) )->list:
    imgList = []
    for i in range(maxListSize):
        blankImg = np.zeros( (inputDimensions[0], inputDimensions[1], inputDimensions[2]) ) #Can't be outside. Insert will pass by ref
        imgList.append(blankImg)
    imgList[0][0][0][0] = 500
    return imgList
"""


class App:
    def __init__(self, window:'window', video_source=0, replay=False):
        self.window = window
        self.window.title(settings.get_config("window_title"))
        self.replay = replay

        # open video source (by default this will try to open the computer webcam)
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)
        #self.inputDimension = [self.vid.height, self.vid.width, 3]
        image_size = settings.get_config("image_input_size")
        self.maxListSize = settings.get_config("image_history_length")
        self.imgList = [np.zeros(image_size)] * self.maxListSize

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.photo = None

        self.strVar = StringVar(value="None")
        self.lblClassification = Label(window, textvariable=self.strVar, font=("Helvetica", 16))
        self.lblClassification.pack(anchor=CENTER, expand=True)

        # Set up frame counters and limiters. No more than (fps) frames per second.
        # Also set up label to display current frame rate.
        self.fps = settings.get_config("max_fps")
        self.fps_counter = window_utils.SimpleFPSCounter()
        self.fps_limiter = window_utils.SimpleFPSLimiter(fps=self.fps)
        self.fps_value = StringVar()
        self.fps_label = Label(window, textvariable=self.fps_value, font=("Helvetica", 16))
        self.fps_label.pack(anchor=CENTER, expand=True)

        # Initialize Tensorflow Models
        tf.reset_default_graph()
        self.session = tf.Session()
        self.model1 = models.model1(image_size, self.maxListSize)
        self.model2 = models.model2(image_size)
        saver = tf.train.Saver()
        saver.restore(self.session, os.path.join(os.getcwd(), "savedmodels\\both\\models.ckpt"))

        if self.replay:
            self.data_gen = loadDataset.dataset_generator('./dataset/imgs/paper_frames', './dataset/csvs/paper.csv', repeat=True)

        # _main_loop() will "recursively" call itself at most (fps) times per second.
        self._main_loop()
        self.window.mainloop()

    def _main_loop(self):
        # Record time at the start of the frame
        self.fps_limiter.start()

        # Update and Display Frame Counter
        self.fps_value.set("FPS: {:.2f}".format(self.fps_counter.update()))

        # Run custom update() function
        self.update()

        # Record time at the end of the frame
        self.fps_limiter.end()
        self.window.after(self.fps_limiter.delay(), self._main_loop)

    def update(self):
        # TODO: Implement main functionality here.
        # Get a frame from the video source
        if not self.replay:
            ret, newFrame = self.vid.get_frame()
            if not ret:
                self.strVar.set("Error: Failed to read from video source.")
                return None
        else:
            newFrame, _ = next(self.data_gen)

        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(newFrame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        # Add the (normalized) frame to the queue.
        normalized_frame = cv2.resize((newFrame.astype(np.float32) - 128) / 128, (64, 64))
        self.imgList.append(normalized_frame)
        if len(self.imgList) >= self.maxListSize:
            self.imgList.pop(0)

        # Predict if three shakes were made and classify image.
        if self.predict_shake():
            self.setNewPrediction(self.predict_class())
        else:
            self.setNewPrediction(Classification.NONE)

    def predict_shake(self) -> bool:
        predict_op, X_in = self.model1[0][0], self.model1[1][0]
        softmax = self.session.run(predict_op, feed_dict={X_in: np.array(self.imgList)[None]})
        return bool(argmax(softmax))

    def predict_class(self) -> Classification:
        predict_op, X_in = self.model2[0][0], self.model2[1][0]
        softmax = self.session.run(predict_op, feed_dict={X_in: np.array(self.imgList[-1])[None]})
        return Classification(argmax(softmax)+1)

    def setNewPrediction(self, classification:Classification)->None:
        self.strVar.set(str(classification))


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read() #Seems to store [b, g, r] for each pixel. NOTE b g r order
            rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Now in [r, b, g] form
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, rgbFrame)
            else:
                return (ret, None)
        else:
            raise Exception("MyVideoCapture: Cannot get frame because VideoCapture is not open.")

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__=="__main__":
    # Create a window and pass it to the Application object
    App(Tk(replay=True))
