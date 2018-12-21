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
import splitImage5 as si5
# print(mnist.shape)
def get_images():

    numImages = cfg.numImages
    trainRatio = cfg.trainRatio
    images = np.zeros((numImages,cfg.width,cfg.height,cfg.channel))
    train_images = []
    train_low = []
    low_res_center=[]
    #Chris's training photosC:\Users\Guillaume\Downloads\INF573Project2018-master\images
    #mypath = "./images"#
    mypath = "/home/christopher/test_images"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	#Number of image for training and testing
    for i in range(0,numImages):
        # img1, img2, img3 = si.split(mypath + "/" + onlyfiles[i],cfg.imgSize)
        # train_images.append(img1.astype(dtype = np.float32)/255.)
        # #cv.imshow("HR",train_images[0])
        # # cv.imwrite(os.path.join('SR','yo'+str(i)+'.jpg'),img2)
        # train_low.append(cv.resize(img2.astype(dtype = np.float32),(cfg.width,cfg.height), interpolation = cv.INTER_LINEAR)/255.)
        # #cv.imshow("LR",train_low[0])
        # #cv.waitKey(0)
        if cfg.multi:
        #If we want to try by stacking 5 low res images.  (set cfg.channel=15 -> still work to do in this case inn the model)
            img1, image, image_center = si5.split(mypath + "/" + onlyfiles[i],cfg.imgSize)
            img1=cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize),int(cfg.imgSize)))
            low_res_center.append(image_center.astype(dtype = np.float32)/255.)
        ###
        else:
        #Just using a single low-res image
            img1  = cv.imread(mypath + "/" + onlyfiles[i])
            newImage = cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize/2),int(cfg.imgSize/2)))
            image = cv.resize(newImage,(int(cfg.imgSize),int(cfg.imgSize)),interpolation = cv.INTER_LINEAR)
            img1=cv.resize(img1.astype(dtype = np.float32),(int(cfg.imgSize),int(cfg.imgSize)))

        train_images.append(img1.astype(dtype = np.float32)/255.)
        train_low.append(image.astype(dtype = np.float32)/255.)

    print(np.amax(train_images[0]))
    HR_img = train_images[int(trainRatio*numImages):numImages]
    LR_img = train_low[int(trainRatio*numImages):numImages]
    if cfg.multi:
        LR_img_center=low_res_center[int(trainRatio*numImages):numImages]
        return LR_img,HR_img,LR_img_center
    # cv.imshow("HR",HR_img[0])
    # cv.imshow("LR",LR_img[0])
    # cv.waitKey(0)
    ##it works here as well/ ok
    return LR_img,HR_img,LR_img

LR_img,HR_img,LR_img_center=get_images()
#Here as well it works
sess=tf.Session()
srcnn=model.SRCNN(sess)
restorer=tf.train.Saver()
#restorer.restore(sess, "wightfile91/model.ckpt-39")
restorer.restore(sess,tf.train.latest_checkpoint('weightfile91'))
#def save_SR(HR_img,LR_img):
#yeah it doesn't work here...

SR_img = sess.run(srcnn.gen_HR,feed_dict={srcnn.LR:LR_img})
Res_img=sess.run(srcnn.residuals,feed_dict={srcnn.LR:LR_img})
print(np.amax(SR_img))
# print(LR_img.shape)
# print("Mean absolute diff with the original image in the interpolated case: ",np.mean(np.abs(np.array(HR_img[0])-
#                     np.array(LR_img[:,:,:,0:3][0])))," and the SR case",np.mean(np.abs(np.array(HR_img[0])-np.array(SR_img[0]))))
z=np.concatenate((HR_img,LR_img_center,SR_img,Res_img),2) #,LR_img

for i in range(SR_img.shape[0]):
    cv.imwrite(os.path.join('SR',str(i)+'_'+'.jpg'),z[i]*255)
    # cv.imwrite(os.path.join('SR',str(i)+'.jpg'),Res_img[i]*255)
    # cv.imwrite(os.path.join('SR',str(i)+'.jpg'),SR_img[i]*255)


#save_SR(HR_img,LR_img)
