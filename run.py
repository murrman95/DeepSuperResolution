import numpy as np
import config as cfg
import cv2 as cv
import os
from os import listdir
from os.path import isfile, join
import model as model
import tensorflow as tf
from matplotlib import  pyplot as plt
import splitImage5 as si5

def get_images():
    """
        Constructs the needed images as input in our CNN
    """
    numImages = cfg.numImages
    trainRatio = cfg.trainRatio
    train_images = []
    train_low = []
    low_res_center=[]
    mypath = cfg.images_path
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	## Low-res image creation (or stack of low-res)
    for i in range(0,int(numImages)):
        ## If we want to try by stacking 5 low res images.
        if cfg.multi:
            img1, image5,image_center = si5.split(mypath + "/" + onlyfiles[i],cfg.imgSize)
            img1=cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize),int(cfg.imgSize)))

            train_images.append(img1.astype(dtype = np.float32)/255.)
            train_low.append(image5.astype(dtype = np.float32))
            low_res_center.append(image_center.astype(dtype = np.float32))
        ##Just using a single low-res image
        else:
            img1  = cv.imread(mypath + "/" + onlyfiles[i])
            newImage = cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize/2),int(cfg.imgSize/2)))
            image5 = cv.resize(newImage,(int(cfg.imgSize),int(cfg.imgSize)),interpolation = cv.INTER_LINEAR)
            img1=cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize),int(cfg.imgSize)))

            train_images.append(img1.astype(dtype = np.float32)/255.)
            train_low.append(image5.astype(dtype = np.float32)/255.)

        ## Train is done on normalized images
        train_images.append(img1.astype(dtype = np.float32)/255.)
        train_low.append(image5.astype(dtype = np.float32)/255.)

    HR_img_train = train_images[0:int(trainRatio*numImages)]
    LR_img_train = train_low[0:int(trainRatio*numImages)]

    ## For testing throughout training and getting these results (recording steps)
    HR_img_test = train_images[int(trainRatio*numImages):int(numImages)]
    LR_img_test = train_low[int(trainRatio*numImages):int(numImages)]

    ## We have to add the individual low resed image too in the multi case
    if cfg.multi:
        LR_train_center=low_res_center[0:int(trainRatio*numImages)]
        LR_test_center=low_res_center[int(trainRatio*numImages):int(numImages)]
        return LR_img_train,HR_img_train,LR_train_center,LR_img_test,HR_img_test,LR_test_center

    return LR_img_train,HR_img_train,LR_img_train,LR_img_test,HR_img_test,LR_img_test

"""
    Calls the training phase
"""
LR_img,HR_img,LR_losses,LR_test,HR_test,LR_test_center=get_images()
sess=tf.Session()
srcnn=model.SRCNN(sess)
srcnn.train(LR_img,HR_img,LR_losses,LR_test,HR_test,LR_test_center)
