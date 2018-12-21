from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import config91 as cfg
from scipy.ndimage import zoom
import cv2 as cv
import os
from os import listdir
from os.path import isfile, join
import model91 as model
import tensorflow as tf
from matplotlib import  pyplot as plt
import splitImage5 as si5
import splitImage as si
def get_images():
    #mnist = input_data.read_data_sets("91/", one_hot=True)
    numImages = cfg.numImages
    trainRatio = cfg.trainRatio
    images = np.zeros((numImages,cfg.width,cfg.height,cfg.channel))
    train_images = []
    train_low = []
    low_res_center=[]
    #Chris's training photos
    mypath = "./images"#
    # mypath = "/home/christopher/test_images"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(len(onlyfiles))

	#Number of image for training and testing
    for i in range(0,int(trainRatio*numImages)):
        if cfg.multi:
        #If we want to try by stacking 5 low res images.
            img1, image5,image_center = si5.split(mypath + "/" + onlyfiles[i],cfg.imgSize)
            img1=cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize),int(cfg.imgSize)))

            train_images.append(img1.astype(dtype = np.float32)/255.)
            train_low.append(image5.astype(dtype = np.float32))
            low_res_center.append(image_center.astype(dtype = np.float32))
        ###
        else:
        #Just using a single low-res image
            img1  = cv.imread(mypath + "/" + onlyfiles[i])
            newImage = cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize/2),int(cfg.imgSize/2)))
            image5 = cv.resize(newImage,(int(cfg.imgSize),int(cfg.imgSize)),interpolation = cv.INTER_LINEAR)
            img1=cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize),int(cfg.imgSize)))

            train_images.append(img1.astype(dtype = np.float32)/255.)
            train_low.append(image5.astype(dtype = np.float32)/255.)

        #train on normalized images

    HR_img = train_images
    LR_img = train_low

    if cfg.multi:
        LR_img_center=low_res_center[0:int(trainRatio*numImages)]
        return LR_img,HR_img,LR_img_center

    return LR_img,HR_img,LR_img

# def save_mnist():
#     a,b=get_images()
#     print(a.shape,b.shape)
#     z=np.concatenate((a,b),2)
#     for i in range(20):
#         cv2.imwrite(os.path.join('origin',str(i)+'.jpg'),z[i]*256)
LR_img,HR_img,LR_loss=get_images()
sess=tf.Session()
srcnn=model.SRCNN(sess)
srcnn.train(LR_img,HR_img,LR_loss)
