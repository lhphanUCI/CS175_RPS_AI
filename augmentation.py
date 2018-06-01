import cv2
import os
import numpy as np
import re
import argparse
import json

# From https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

TEST_MODE = True
      
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
    
def getBrightnessAugX(moveFramesDirPath:str, gamma:float, settingsPath:str)->'[HxWx3]':
    global FOURCC
    global TEST_MODE
    
    origOutput = None
    augOutput = None
    settingsData = None
    
    with open(settingsPath) as f:
        settingsData = json.load(f)
    
    outputDim = (int(settingsData['resizedW']), int(settingsData['resizedH']))
    frameRate = float(settingsData['frameRate'])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
 
    if TEST_MODE:
        outputTestVideosDir = 'outputTestVideos'
        if not os.path.exists(outputTestVideosDir): #Creates outputTestVideos folder
            os.makedirs(outputTestVideosDir)
            
        origOutput = cv2.VideoWriter(outputTestVideosDir + '/original.avi', fourcc, frameRate, outputDim)
        augOutput = cv2.VideoWriter(outputTestVideosDir + '/augmented.avi', fourcc, frameRate, outputDim)
        
    fileList = os.listdir(moveFramesDirPath)
    moveX = []
    amtFiles = len(fileList)
    for i in range (1, amtFiles + 1):
        fullPath = moveFramesDirPath + "/" + str(i) + ".jpeg"
        img = cv2.imread(fullPath,cv2.IMREAD_COLOR )
        resizedImg = cv2.resize(img, (64, 64))
        resizedAugImg = adjust_gamma(resizedImg, gamma)
        resizedAugRgbFrame = cv2.cvtColor(resizedAugImg, cv2.COLOR_BGR2RGB)  # Now in [r, b, g] form
        moveX.append(resizedAugRgbFrame)

        if TEST_MODE:
            origOutput.write(resizedImg)
            augOutput.write(resizedAugImg)

    if TEST_MODE:
        origOutput.release()
        augOutput.release()
    return np.asarray(moveX)
        
if __name__ == "__main__":
    settingsPath = './settings.json'
    gamma = 0.5 # Value of 1 means original output. Between (0, 1) means darken. Above 1 means brighten
    resizedAugX = getBrightnessAugX('./dataset/imgs/rock_frames', gamma, settingsPath)
    print(resizedAugX)
    
