import cv2
import os
import numpy as np
import re
import argparse
import json

# From https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

SETTINGS_PATH = './settings.json'
TEST_MODE = True
      
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
    
def getBrightnessAugForSingleMove(moveFramesDirPath:str, moveCSVPath:str, gamma:float)->('[X]','[Y]'):
    global SETTINGS_PATH
    global TEST_MODE
    
    origOutput = None
    augOutput = None
    settingsData = None
    
    with open(SETTINGS_PATH) as f:
        settingsData = json.load(f)
    
    outputDim = (int(settingsData['resizedW']), int(settingsData['resizedH']))
    frameRate = float(settingsData['frameRate'])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
 
    if TEST_MODE:
        outputTestVideosDir = 'outputTestVideos'
        if not os.path.exists(outputTestVideosDir): #Creates outputTestVideos folder
            os.makedirs(outputTestVideosDir)
            
        origOutput = cv2.VideoWriter(outputTestVideosDir + "/orig" + os.path.basename(moveCSVPath) + '.avi', fourcc, frameRate, outputDim)
        augOutput = cv2.VideoWriter(outputTestVideosDir + "/aug" + os.path.basename(moveCSVPath) + '.avi', fourcc, frameRate, outputDim)
        
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
    moveX = np.asarray(moveX)

    moveY = np.zeros(amtFiles)
    i=0
    with open(moveCSVPath) as f:
        for line in f:
            moveY[i] = int(line)
            i = i + 1

    return (moveX, moveY)

def getAugmentedBrightnessDataSet(rockFramesDirPath:str, paperFramesDirPath:str, scissorFramesDirPath:str
                , rockCSVPath:str, paperCSVPath:str, scissorCSVPath:str, gamma:float)->('[X]','[Y]'):

    rockX, rockY = getBrightnessAugForSingleMove(rockFramesDirPath, rockCSVPath, gamma)
    paperX, paperY = getBrightnessAugForSingleMove(paperFramesDirPath, paperCSVPath, gamma)
    scissorX, scissorY = getBrightnessAugForSingleMove(scissorFramesDirPath, scissorCSVPath, gamma)
    
    X = np.concatenate((rockX, paperX, scissorX), axis=0)
    Y = np.concatenate((rockY, paperY, scissorY), axis=0)

    return (X, Y)   

     
if __name__ == "__main__":
    gamma = 0.5 # Value of 1 means original output. Between (0, 1) means darken. Above 1 means brighten
    X, Y = getAugmentedBrightnessDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames', './dataset/imgs/scissor_frames'
                , './dataset/csvs/rock.csv', './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv', gamma)
