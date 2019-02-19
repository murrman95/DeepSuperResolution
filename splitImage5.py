import cv2 as cv
import numpy as np
import numpy
import config as cfg

def split(path,trueSize):
    img = cv.imread(path)
    img = cv.resize(img, (trueSize, trueSize))
    imgNew,imgNew_center = getPImages(img,trueSize)
    return img, imgNew,imgNew_center

def getPImages(img,trueSize):
    k1 = np.array([[[0.25, 0.25, 0.25], [0.25,0.25,0.25]],[[0.25,0.25,0.25], [0.25,0.25,0.25]]])
    k2 = np.array([[[0.1,0.1,0.1],[0.4,0.4,0.4]],[[0.1,0.1,0.1], [0.4,0.4,0.4]]])
    k3 = np.array([[[0.4,0.4,0.4],[0.1,0.1,0.1]],[[0.4,0.4,0.4], [0.1,0.1,0.1]]])
    k4 = np.array([[[0.4,0.4,0.4],[0.4,0.4,0.4]],[[0.1,0.1,0.1], [0.1,0.1,0.1]]])
    k5 = np.array([[[0.1,0.1,0.1],[0.1,0.1,0.1]],[[0.4,0.4,0.4], [0.4,0.4,0.4]]])

	#stacking for a total of 5 interpolated images
    resImage = np.zeros((trueSize,trueSize,cfg.channel))

    img1 = np.zeros((int(trueSize/2),int(trueSize/2),3))
    img2 = np.zeros((int(trueSize/2),int(trueSize/2),3))
    img3 = np.zeros((int(trueSize/2),int(trueSize/2),3))
    img4 = np.zeros((int(trueSize/2),int(trueSize/2),3))
    img5 = np.zeros((int(trueSize/2),int(trueSize/2),3))

    for i in range(0,int(trueSize/2)):
        for j in range(0,int(trueSize/2)):
            ii = i * 2
            jj = j * 2
            img1[i][j] = k1[0][0] * img[ii][jj] + k1[0][1] * img[ii][jj + 1] + k1[1][0] * img[ii + 1][jj] + k1[1][1] * img[ii + 1][ jj + 1]
            img2[i][j] = k2[0][0] * img[ii][jj] + k2[0][1] * img[ii][jj + 1] + k2[1][0] * img[ii + 1][jj] + k2[1][1] * img[ii + 1][ jj + 1]
            img3[i][j] = k3[0][0] * img[ii][jj] + k3[0][1] * img[ii][jj + 1] + k3[1][0] * img[ii + 1][jj] + k3[1][1] * img[ii + 1][ jj + 1]
            img4[i][j] = k4[0][0] * img[ii][jj] + k4[0][1] * img[ii][jj + 1] + k4[1][0] * img[ii + 1][jj] + k4[1][1] * img[ii + 1][ jj + 1]
            img5[i][j] = k5[0][0] * img[ii][jj] + k5[0][1] * img[ii][jj + 1] + k5[1][0] * img[ii + 1][jj] + k5[1][1] * img[ii + 1][ jj + 1]

            #print((cv.resize(img1.astype(dtype = np.float32),(trueSize,trueSize),interpolation = cv.INTER_LINEAR)).shape)
            resImage[:,:, 0: 3] = cv.resize(img1.astype(dtype = np.float32),(trueSize,trueSize),interpolation = cv.INTER_LINEAR)/255.
            resImage[:,:, 3: 6] = cv.resize((img2).astype(dtype = np.float32),(trueSize,trueSize),interpolation = cv.INTER_LINEAR)/255.
            resImage[:,:, 6: 9] = cv.resize((img3).astype(dtype = np.float32),(trueSize,trueSize),interpolation = cv.INTER_LINEAR)/255.
            resImage[:,:, 9:12] = cv.resize((img4).astype(dtype = np.float32),(trueSize,trueSize),interpolation = cv.INTER_LINEAR)/255.
            resImage[:,:,12:15] = cv.resize((img5).astype(dtype = np.float32),(trueSize,trueSize),interpolation = cv.INTER_LINEAR)/255.

    return resImage,resImage[:,:, 0: 3]
