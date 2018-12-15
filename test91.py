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
# print(mnist.shape)
def get_images():

    numImages = cfg.numImages
    trainRatio = cfg.trainRatio
    images = np.zeros((numImages,cfg.width,cfg.height,cfg.channel))
    train_images = []
    train_low = []
    #Chris's training photos
    mypath = "/home/christopher/test_images"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	#Number of image for training and testing
    for i in range(0,numImages):
        img1, img2, img3 = si.split(mypath + "/" + onlyfiles[i],cfg.imgSize)
        train_images.append(img1.astype(dtype = np.float32)/255.)
        #cv.imshow("HR",train_images[0])
        train_low.append(cv.resize(img2.astype(dtype = np.float32),(cfg.width,cfg.height), interpolation = cv.INTER_LINEAR)/255.)
        #cv.imshow("LR",train_low[0])
        #cv.waitKey(0) 

    HR_img = train_images[int(trainRatio*numImages):numImages]
    LR_img = train_low[int(trainRatio*numImages):numImages]
    #cv.imshow("HR",HR_img[0])
    #cv.imshow("LR",LR_img[0])
    #cv.waitKey(0)
    ##it works here as well/ ok


    #for i in range(len(HR_img)):
    #    _img=zoom(np.squeeze(HR_img[i]),cfg.s)
    #    if i == 1: print(_img)
        #LR_img.append(cv.resize(_img,(cfg.width,cfg.height))[:,:,np.newaxis])
    #    if i==1: print(LR_img)
    return LR_img,HR_img

LR_img,HR_img=get_images()
#Here as well it works
sess=tf.Session()
srcnn=model.SRCNN(sess)
restorer=tf.train.Saver()
restorer.restore(sess,tf.train.latest_checkpoint('weightfile91'))
#def save_SR(HR_img,LR_img):
#yeah it doesn't work here...

SR_img = sess.run(srcnn.gen_HR,feed_dict={srcnn.LR:LR_img})
print(np.amax(SR_img))

# img =np.reshape(LR_img,(-1,10,10,1))
# img = cv.resize(LR_img[0], (15,15))
z=np.concatenate((HR_img,LR_img,SR_img),2)
# w = np.concatenate((z,img), 2)
for i in range(z.shape[0]):
    cv.imwrite(os.path.join('SR',str(i)+'.jpg'),z[i]*256)
    # cv.imwrite(os.path.join('SR',str(i)+'+.jpg'),img[i]*256)

#save_SR(HR_img,LR_img)