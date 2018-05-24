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
from timeit import default_timer as timer
from numpy import mean

DATASET_DIR = './recordedDataset'
IMAGE_COUNTER = 0
IS_WRITE_EACH_FRAME_TO_FILE_MODE = True

class App:
    def __init__(self, window:'window', windowTitle:str, output_framerate:int, video_source=0):
        self.window = window
        self.window.title(windowTitle)
        self.video_source = video_source
        self.output_framerate = output_framerate

        self.fps_counter = SimpleFPSCounter(length=16)
        self.time_value = StringVar()
        self.time_label = Label(window, textvariable=self.time_value, font=("Helvetica", 16))
        self.time_label.pack(anchor=CENTER, expand=True)

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(output_framerate, video_source)
        self.vid_dims = [self.vid.height, self.vid.width, 3]

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(window, height=self.vid_dims[0], width=self.vid_dims[1])
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.update()
        self.window.mainloop()

    def update(self):
        global IMAGE_COUNTER
        global DATASET_DIR
        global IS_WRITE_EACH_FRAME_TO_FILE_MODE

        # Update and Display Frame Counter
        self.time_value.set("FPS: {:.2f}".format(self.fps_counter.update()))
        
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
        self.window.after(1, self.update)


class MyVideoCapture:
    def __init__(self, framerate, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object.The output is stored in 'recordedOutput.xvid' file.
        self.out = cv2.VideoWriter(DATASET_DIR + '/recordedOutput.avi',cv2.VideoWriter_fourcc(*'XVID'), framerate, (self.width, self.height))

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


class SimpleFPSCounter:

    def __init__(self, length=5):
        self.last_time = timer()
        self.last_intervals = [0] * length
        self.length = length
        self.counter = 0

    def update(self):
        new_time = timer()
        self.last_intervals[self.counter] = new_time - self.last_time
        self.last_time = new_time
        if self.counter < self.length - 1:
            self.counter += 1
        else:
            self.counter = 0
        return 1 / mean(self.last_intervals)


if __name__=="__main__":
    if not os.path.exists(DATASET_DIR): #Creates dataset folder
        os.makedirs(DATASET_DIR)
    
    # Create a window and pass it to the Application object
    windowTitle = "Frame recorder"
    output_framerate = 20 #Captures frame every delayOnCapture milliseconds
    App(Tk(), windowTitle, output_framerate)
