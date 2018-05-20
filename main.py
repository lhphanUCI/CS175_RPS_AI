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
import numpy as np

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

strDict = {
        Classification.NONE: "None",
        Classification.ROCK: "Rock",
        Classification.PAPER: "Paper",
        Classification.SCISSOR: "Scissor"
    }

def getFilledBlankImgList( maxListSize:int, inputDimensions:(int, int, int) )->list:
    imgList = []
    for i in range(maxListSize):
        blankImg = np.zeros( (inputDimensions[0], inputDimensions[1], inputDimensions[2]) ) #Can't be outside. Insert will pass by ref
        imgList.append(blankImg)
    imgList[0][0][0][0] = 500
    return imgList


class App:
    def __init__(self, window:'window', windowTitle:str, updateRate:int
                 , maxListSize:int, inputDimension:(int, int, int), video_source=0):
        self.imgList = getFilledBlankImgList(maxListSize, inputDimension)

        self.window = window
        self.window.title(windowTitle)
        self.maxListSize = maxListSize
        self.inputDimension = inputDimension
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        self.strVar = StringVar(value="None")
        self.lblClassification = Label(window, textvariable=self.strVar, font=("Helvetica", 16))
        self.lblClassification.pack(anchor=CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = updateRate
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret, newFrame = self.vid.get_frame()
        self.handleNewInput(newFrame)

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(newFrame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

        self.window.after(self.delay, self.update)

    def isDetectingPlayableMove(self):
        ################################### TO DO ###################################
        #To get list so far, do   theList = self.imgList
        #debugPrintEntireList(self.imgList, self.maxListSize) #This will display the 100 frames in list so far
        return True
        
        ############################## ERASE ABOVE AND IMPLEMENT ####################

    def getNewClassificaton(self, hundredthFrame:'LxHx3 npArry')->Classification:
        ################################### TO DO ###################################
        randomVal = random.randrange(4)
        if(randomVal == 0):
            return Classification.NONE
        if(randomVal == 1):
            return Classification.ROCK
        if(randomVal == 2):
            return Classification.PAPER
        if(randomVal == 3):
            return Classification.SCISSOR
        ############################ ERASE RANDOM ABOVE AND IMPLEMENT ##################

    def setNewPrediction(self, classification:Classification)->None:
        strClassification = strDict[classification]
        self.strVar.set(strClassification)
        
    def handleNewInput(self, newFrame:'LxHx3 npArry')->None:
        self.imgList.append(newFrame)  # Add lastest frame
        if len(self.imgList) >= self.maxListSize:
            self.imgList.pop(0) #Remove front

        if( self.isDetectingPlayableMove() ):
            newClassification = self.getNewClassificaton(newFrame)
            self.setNewPrediction(newClassification)

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
    windowTitle = "Rock Paper Scissor AI"
    updateRate = 33 #Grabs frame every updateRate milliseconds. So this is roughly 30frames/s if I calc correct
    maxListSize = 100
    inputDimension = [500,500, 3]
    App(Tk(), windowTitle, updateRate, maxListSize, inputDimension)
