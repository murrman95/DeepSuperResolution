import cv2 as cv
import numpy as np
import numpy

def split(path,trueSize):
    img = cv.imread(path)
    img = cv.resize(img, (trueSize, trueSize))
    img1, img2 = getPImages(img,trueSize)
    return img, img1, img2

def getPImages(img,trueSize):
    k1 = np.array([[[0.25, 0.25, 0.25], [0.25,0.25,0.25]],[[0.25,0.25,0.25], [0.25,0.25,0.25]]])
    k2 = np.array([[[0.1,0.1,0.1],[0.4,0.4,0.4]],[[0.1,0.1,0.1], [0.4,0.4,0.4]]])
    img1 = np.zeros((int(trueSize/2),int(trueSize/2),3))
    img2 = np.zeros((int(trueSize/2),int(trueSize/2),3))
    for i in range(0,int(trueSize/2)):
        for j in range(0,int(trueSize/2)):
            ii = i * 2
            jj = j * 2
            img1[i][j] = k1[0][0] * img[ii][jj] + k1[0][1] * img[ii][jj + 1] + k1[1][0] * img[ii + 1][jj] + k1[1][1] * img[ii + 1][ jj + 1]
            img2[i][j] = k2[0][0] * img[ii][jj] + k2[0][1] * img[ii][jj + 1] + k2[1][0] * img[ii + 1][jj] + k2[1][1] * img[ii + 1][jj + 1]


    return img1, img2
