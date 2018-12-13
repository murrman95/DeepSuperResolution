from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import config as cfg
from scipy.ndimage import zoom
import cv2
import os
import model
import tensorflow as tf
from matplotlib import  pyplot as plt
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist.shape)
def get_mnist(mnist):
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    HR_img=np.reshape(mnist.train.images[4000:4100],(-1,28,28,1))
    LR_img=[]
    for i in range(len(HR_img)):
        _img=zoom(np.squeeze(HR_img[i]),cfg.s)
        if i == 1: print(_img)
        LR_img.append(cv2.resize(_img,(cfg.width,cfg.height))[:,:,np.newaxis])
        if i==1: print(LR_img)
    return np.array(LR_img),HR_img
LR_img,HR_img=get_mnist(mnist)
sess=tf.Session()
srcnn=model.SRCNN(sess)
restorer=tf.train.Saver()
restorer.restore(sess,tf.train.latest_checkpoint('weightfile'))
def save_SR(HR_img,LR_img):
    SR_img=sess.run(srcnn.gen_HR,feed_dict={srcnn.LR:LR_img})
    # img =np.reshape(LR_img,(-1,10,10,1))
    # img = cv2.resize(LR_img[0], (15,15))
    w=np.concatenate((LR_img,SR_img),2)
    z = np.concatenate((HR_img,w), 2)
    # w = np.concatenate((z,img), 2)
    for i in range(z.shape[0]):
        cv2.imwrite(os.path.join('SR',str(i)+'.jpg'),z[i]*256)
        # cv2.imwrite(os.path.join('SR',str(i)+'+.jpg'),img[i]*256)
save_SR(HR_img,LR_img)
