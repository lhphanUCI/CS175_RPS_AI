import cv2
import os
import numpy as np
import re
import settings

def getXAndYForSingleMove(moveFramesDirPath:str, moveCSVPath:str, count=None)->('[X]','[Y]'):
    fileList = os.listdir(moveFramesDirPath)
    moveX = []
    amtFiles = len(fileList)
    count = min(count + 1, amtFiles + 1) if count is not None else amtFiles + 1
    for i in range (1, count):
        fullPath = moveFramesDirPath + "/" + str(i) + ".jpeg"
        img = cv2.imread(fullPath,cv2.IMREAD_COLOR )
        rgbFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Now in [r, b, g] form
        resizedImg = cv2.resize(rgbFrame, (80, 80))  # 64 * 1.25 ; 80 * 0.80 = 64
        moveX.append(resizedImg)#resizedImg)
    moveX = np.asarray(moveX)

    moveY = np.zeros(amtFiles)
    i=0
    with open(moveCSVPath) as f:
        for line in f:
            moveY[i] = int(line)
            i = i + 1
    return (moveX, moveY[:(count-1)])

def loadDataSet(rockFramesDirPath:str, paperFramesDirPath:str, scissorFramesDirPath:str
                , rockCSVPath:str, paperCSVPath:str, scissorCSVPath:str, count=None)->('[X]','[Y]'):

    rockX, rockY = getXAndYForSingleMove(rockFramesDirPath, rockCSVPath, count)
    paperX, paperY = getXAndYForSingleMove(paperFramesDirPath, paperCSVPath, count)
    scissorX, scissorY = getXAndYForSingleMove(scissorFramesDirPath, scissorCSVPath, count)
    
    X = np.concatenate((rockX, paperX, scissorX), axis=0)
    Y = np.concatenate((rockY, paperY, scissorY), axis=0)

    return (X, Y)


def dataset_generator(img_path, csv_path, max_count=None, repeat=False):
    def img_generator():
        file_list = os.listdir(img_path)
        count = min(max_count + 1, len(file_list)) if max_count is not None else len(file_list) + 1
        for i in range(1, count):
            fullPath = img_path + "/" + str(i) + ".jpeg"
            img = cv2.imread(fullPath, cv2.IMREAD_COLOR)
            rgbFrame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Now in [r, b, g] form
            #resizedImg = cv2.resize(rgbFrame, (settings.get_config("resizedW"), settings.get_config("resizedH")))
            yield rgbFrame

    def csv_generator():
        with open(csv_path) as f:
            for line in f:
                yield int(line)

    while True:
        try:
            img_gen, csv_gen = img_generator(), csv_generator()
            while True:
                yield next(img_gen), next(csv_gen)
        except:
            if not repeat:
                return


if __name__ == "__main__":
    X, Y = loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames', './dataset/imgs/scissor_frames'
                , './dataset/csvs/rock.csv', './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
    print(X)
    print(Y)
    print(X.shape)
    print(Y.shape)
