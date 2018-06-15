import cv2
import os
import numpy as np
import re
import argparse
import json
import loadDataset

# From https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

SETTINGS_PATH = './settings.json'
TEST_MODE = False
      
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
    
def getBrightnessAug(X:'nparry', gamma:float)->'nparry':
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
            
        origOutput = cv2.VideoWriter(outputTestVideosDir + '/orig.avi', fourcc, frameRate, outputDim)
        augOutput = cv2.VideoWriter(outputTestVideosDir + '/aug.avi', fourcc, frameRate, outputDim)
        
    XAug = []
    for i in range(len(X)):
        rgbImg = X[i]
        augRgbImg = adjust_gamma(rgbImg, gamma)
        XAug.append(augRgbImg)

        if TEST_MODE:
            origBGRImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR)  # Now in [b, g, r] form
            augBGRImg = cv2.cvtColor(augRgbImg, cv2.COLOR_RGB2BGR)  # Now in [b, g, r] form
            origOutput.write(origBGRImg)
            augOutput.write(augBGRImg)

    if TEST_MODE:
        origOutput.release()
        augOutput.release()
        
    XAug = np.asarray(XAug)
    return XAug
     
#https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def getRotationAug(X:'nparry', angle:float)->'nparry':
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
            
        origOutput = cv2.VideoWriter(outputTestVideosDir + '/orig.avi', fourcc, frameRate, outputDim)
        augOutput = cv2.VideoWriter(outputTestVideosDir + '/aug.avi', fourcc, frameRate, outputDim)
        
    XAug = []
    for i in range(len(X)):
        rgbImg = X[i]
        augRgbImg = rotateImage(rgbImg, angle)
        XAug.append(augRgbImg)

        if TEST_MODE:
            origBGRImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR)  # Now in [b, g, r] form
            augBGRImg = cv2.cvtColor(augRgbImg, cv2.COLOR_RGB2BGR)  # Now in [b, g, r] form
            origOutput.write(origBGRImg)
            augOutput.write(augBGRImg)

    if TEST_MODE:
        origOutput.release()
        augOutput.release()
        
    XAug = np.asarray(XAug)
    return XAug
     
if __name__ == "__main__":
    angle = -30.0 
    X, Y = loadDataset.loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames', './dataset/imgs/scissor_frames'
                , './dataset/csvs/rock.csv', './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
    XAug = getRotationAug(X, angle)
    
