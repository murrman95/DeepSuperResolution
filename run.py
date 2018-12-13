from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import config as cfg
from scipy.ndimage import zoom
import cv2
import os
import model
import tensorflow as tf
from matplotlib import  pyplot as plt
def get_mnist():
    train_y_images_path = './91/*.png'
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    HR_img=np.reshape(train_y_images_path,(-1,28,28,1)) #mnist.train.images[0:2000]
    LR_img=[]
    for i in range(len(HR_img)):
        _img=zoom(np.squeeze(HR_img[i]),cfg.s)
        LR_img.append(cv2.resize(_img,(cfg.width,cfg.height))[:,:,np.newaxis])
    return np.array(LR_img),HR_img
def save_mnist():
    a,b=get_mnist()
    print(a.shape,b.shape)
    z=np.concatenate((a,b),2)
    for i in range(100):
        cv2.imwrite(os.path.join('origin',str(i)+'.jpg'),z[i]*256)
LR_img,HR_img=get_mnist()
sess=tf.Session()
srcnn=model.SRCNN(sess)
srcnn.train(LR_img,HR_img)
