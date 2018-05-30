import cv2
import os
import numpy as np
import re

def getXAndYForSingleMove(moveFramesDirPath:str, moveCSVPath:str)->('[X]','[Y]'):
    fileList = os.listdir(moveFramesDirPath)
    moveX = []
    amtFiles = len(fileList)
    for i in range (1, amtFiles + 1):
        fullPath = moveFramesDirPath + "/" + str(i) + ".jpeg"
        img = cv2.imread(fullPath,cv2.IMREAD_COLOR )
        #resizedImg = cv2.resize(img, (50, 50))
        moveX.append(img)
    moveX = np.asarray(moveX)

    moveY = np.zeros(amtFiles)
    i=0
    with open(moveCSVPath) as f:
        for line in f:
            moveY[i] = int(line)
            i = i + 1
    return (moveX, moveY)

def loadDataSet(rockFramesDirPath:str, paperFramesDirPath:str, scissorFramesDirPath:str
                , rockCSVPath, paperCSVPath, scissorCSVPath)->('[X]','[Y]'):

    rockX, rockY = getXAndYForSingleMove(rockFramesDirPath, rockCSVPath)
    paperX, paperY = getXAndYForSingleMove(paperFramesDirPath, paperCSVPath)
    scissorX, scissorY = getXAndYForSingleMove(scissorFramesDirPath, scissorCSVPath)
    

    X = np.append(rockX, paperX, scissorX)
    Y = np.apppend(rockY, paperY, scissorY)

    return (X, Y)   
    
loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames', './dataset/imgs/scissor_frames'
            , './dataset/csvs/rock.csv', './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
