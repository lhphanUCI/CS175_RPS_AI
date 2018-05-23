'''
Running this file will activate webcam to record video. When the program
is closed, the video file is written to the path assigned in the
variable 'DATASET_DIR'. 

If the variable called IS_WRITE_EACH_FRAME_TO_FILE_MODE is set to true,
then it will also write each frame as a jpeg file.

UI Template from
https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
'''

from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
import os

DATASET_DIR = './recordedDataset'
IMAGE_COUNTER = 0
IS_WRITE_EACH_FRAME_TO_FILE_MODE = True

class App:
    def __init__(self, window:'window', windowTitle:str, delayOnCapture:int
                 , inputDimension:(int, int, int), video_source=0):
        self.window = window
        self.window.title(windowTitle)
        self.inputDimension = inputDimension
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(delayOnCapture, video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = delayOnCapture
        self.update()

        self.window.mainloop()

    def update(self):
        global IMAGE_COUNTER
        global DATASET_DIR
        global IS_WRITE_EACH_FRAME_TO_FILE_MODE
        
        # Get a frame from the video source
        ret, newFrame = self.vid.get_frame()
        rgbNewFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2RGB) #Now in [r, b, g] form

        
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(rgbNewFrame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = NW)
            #Write the video file
            self.vid.out.write(newFrame) #Note, this has to be in BGR form. write will convert BGR->RGB I think

            if IS_WRITE_EACH_FRAME_TO_FILE_MODE: #Write a image file if mode is set
                imgFilePath=DATASET_DIR+"/img"+str(IMAGE_COUNTER)+".jpg"
                cv2.imwrite(imgFilePath, newFrame)
                IMAGE_COUNTER = IMAGE_COUNTER + 1
        self.window.after(self.delay, self.update)



class MyVideoCapture:
    def __init__(self, delayOnCapture, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object.The output is stored in 'recordedOutput.xvid' file.
        self.out = cv2.VideoWriter(DATASET_DIR + '/recordedOutput.avi',cv2.VideoWriter_fourcc(*'XVID'), delayOnCapture, (self.width, self.height))

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read() #Seems to store [b, g, r] for each pixel. NOTE b g r order. Believe write needs to be in this order too.
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, frame)
            else:
                return (ret, None)
        else:
            raise Exception("MyVideoCapture: Cannot get frame because VideoCapture is not open.")

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__=="__main__":
    if not os.path.exists(DATASET_DIR): #Creates dataset folder
        os.makedirs(DATASET_DIR)
    
    # Create a window and pass it to the Application object
    windowTitle = "Frame recorder"
    delayOnCapture = 30 #Captures frame every delayOnCapture milliseconds
    inputDimension = [500,500, 3]
    App(Tk(), windowTitle, delayOnCapture, inputDimension)
