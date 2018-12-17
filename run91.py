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
import splitImage as si
def get_images():
    #train_y_images_path = './91/*.png'
    #mnist = input_data.read_data_sets("91/", one_hot=True)
    numImages = cfg.numImages
    trainRatio = cfg.trainRatio
    images = np.zeros((numImages,cfg.width,cfg.height,cfg.channel))
    train_images = []
    train_low = []
    #Chris's training photos
    mypath = "./images"#mypath = "/home/christopher/test_images"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(len(onlyfiles))

	#Number of image for training and testing
    for i in range(0,numImages):
        img1, img2, img3 = si.split(mypath + "/" + onlyfiles[i],cfg.imgSize)

        #train on normalized images
        train_images.append(img1.astype(dtype = np.float32)/255.)
        train_low.append(cv.resize(img2.astype(dtype = np.float32),(cfg.width,cfg.height), interpolation = cv.INTER_LINEAR)/255.)
    HR_img = train_images[0:int(trainRatio*numImages)]
    LR_img = train_low[0:int(trainRatio*numImages)]
	#HR_img=np.reshape(mnist[0:90],(-1,28,28,config91.channels)) #mnist.train.images[0:2000]
    #     #LR_img=[]
    #for i in range(len(HR_img)):
    #    _img=zoom(np.squeeze(HR_img[i]),cfg.s)
    #    LR_img.append(cv2.resize(_img,(cfg.width,cfg.height))[:,:,np.newaxis])
    return LR_img,HR_img
def save_mnist():
    a,b=get_images()
    print(a.shape,b.shape)
    z=np.concatenate((a,b),2)
    for i in range(20):
        cv2.imwrite(os.path.join('origin',str(i)+'.jpg'),z[i]*256)
LR_img,HR_img=get_images()
sess=tf.Session()
srcnn=model.SRCNN(sess)
srcnn.train(LR_img,HR_img)
