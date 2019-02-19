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
    GT_images = []
    LR_input = []
    LR_img_center=[]
    mypath = cfg.images_path
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	## Low-res image creation (or stack of low-res)
    for i in range(int(trainRatio*numImages),numImages):
        ## If we want to try by stacking 5 low res images.
        if cfg.multi:
            img1, image5,image_center = si5.split(mypath + "/" + onlyfiles[i],cfg.imgSize)
            img1=cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize),int(cfg.imgSize)))

            GT_images.append(img1.astype(dtype = np.float32)/255.)
            LR_input.append(image5.astype(dtype = np.float32))
            LR_img_center.append(image_center.astype(dtype = np.float32))
        ###
        else:
        ## Just using a single low-res image
            img1  = cv.imread(mypath + "/" + onlyfiles[i])
            newImage = cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize/2),int(cfg.imgSize/2)))
            image5 = cv.resize(newImage,(int(cfg.imgSize),int(cfg.imgSize)),interpolation = cv.INTER_LINEAR)
            img1=cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize),int(cfg.imgSize)))

        ## Train is done on normalized images
            GT_images.append(img1.astype(dtype = np.float32)/255.)
            LR_input.append(image5.astype(dtype = np.float32)/255.)
    HR_img = GT_images
    LR_img = LR_input
    if cfg.multi:
        return LR_img,HR_img,LR_img_center
    return LR_img,HR_img,LR_img

LR_img,HR_img,LR_img_center=get_images()
sess=tf.Session()
srcnn=model.SRCNN(sess)
restorer=tf.train.Saver()
restorer.restore(sess,tf.train.latest_checkpoint('weightfile91'))

SR_img = sess.run(srcnn.gen_HR,feed_dict={srcnn.LR:LR_img,srcnn.LR_losses:LR_img_center})
Res_img=sess.run(srcnn.residuals,feed_dict={srcnn.LR:LR_img,srcnn.LR_losses:LR_img_center})
z=np.concatenate((HR_img,LR_img_center,SR_img,Res_img),2) #,LR_img

## Registers the output images in ./SR folder
try:
    os.stat("./SR/")
except:
    os.mkdir("./SR/")
for i in range(SR_img.shape[0]):
    cv.imwrite(os.path.join('SR',str(i)+'_'+'.jpg'),z[i]*255)
