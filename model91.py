import tensorflow as tf
import config91 as cfg
from sklearn.utils import shuffle
import csv

import numpy as np

class SRCNN:
    def __init__(self,sess):
        self.LR=tf.placeholder("float32",[None,cfg.width,cfg.height,cfg.channel])
        self.HR=tf.placeholder("float32",[None,cfg.width,cfg.height,3])

        self.LR_losses=tf.placeholder("float32",[None,cfg.width,cfg.height,3])
        self.gen_HR,self.weights,self.residuals=self.network()
        self.sess=sess

    def network(self):
        nbr_patches=cfg.nbr_patches
        weights = []
        tensor = None
        padding='SAME'

    #### with one image as input
        for i in range(cfg.nbr_layers):
            if i==0:  #parameters for first layer
                filter,bias_size=[3,3,cfg.channel,nbr_patches],[nbr_patches]
                entry=self.LR
            elif i==cfg.nbr_layers-1:    #parameters for last layer
                filter,bias_size= [3,3,nbr_patches,3],[3]
                entry=tensor
            else: #parameters for hidden layers
                filter,bias_size= [3,3,nbr_patches,nbr_patches],[nbr_patches]
                entry=tensor
            conv_w = tf.get_variable("conv_w_%s" % (i), filter, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
            conv_b = tf.get_variable("conv_b_%s" % (i), bias_size, initializer=tf.constant_initializer(0))
            weights.append(conv_w)
            weights.append(conv_b)
            if i==cfg.nbr_layers-1:
        #we don't put the relu layer at the end of the network
                tensor = tf.nn.bias_add(tf.nn.conv2d(entry, conv_w, strides=[1,1,1,1], padding=padding), conv_b)
            else:
                tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(entry, conv_w, strides=[1,1,1,1], padding=padding), conv_b)) #, strides=[1,1,1,1]

        tensor_out=tf.add(tensor,self.LR_losses)
        return tensor_out,weights,tensor

    def train(self,LR_imgs,HR_imgs,LR_losses,LR_test,HR_test,LR_losses_test):
        saver = tf.train.Saver(max_to_keep=1)
        self.loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(self.HR, self.gen_HR)))
        for w in self.weights:
            self.loss += tf.nn.l2_loss(w)*1e-4
        global_step = tf.Variable(0, trainable=False)
        learning_rate 	= tf.train.exponential_decay(cfg.learning_rate/100, global_step*cfg.batch_size, len(LR_imgs)*120, cfg.learning_rate, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())

        if(cfg.writeResults):   # to register results during training
            with open("normalvdcnn10.csv","w",newline = '') as myfile:
                writer = csv.writer(myfile, quoting  = csv.QUOTE_ALL)
                writer.writerow(['Epochs','Average interpolated difference','Average SR differece'])
                for i in range(cfg.epoch + 1):
                    LR_imgs,LR_losses,HR_imgs=shuffle(LR_imgs,LR_losses,HR_imgs)
                    if(i % 5 == 0):
                        SR_test = self.sess.run(self.gen_HR,feed_dict={self.LR:LR_test,self.LR_losses:LR_losses_test})
                        acc = []
                        acc.append(i)
                        acc.append(np.mean(np.abs(np.array(HR_test)-np.array(LR_losses_test))))
                        acc.append(np.mean(np.abs(np.array(HR_test)-np.array(SR_test))))
                        writer.writerow(acc)
                    for j in range(int(len(HR_imgs)/cfg.batch_size)):
                        LR_batch,LR_losses_batch,HR_batch=LR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size],LR_losses[j*cfg.batch_size:(j+1)*cfg.batch_size],HR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size]
                        self.sess.run(self.train_op,feed_dict={self.LR:LR_batch,self.HR:HR_batch,self.LR_losses:LR_losses_batch})
                        if j%5==0:
                            print(i,self.sess.run(self.loss,feed_dict={self.LR:LR_batch,self.HR:HR_batch,self.LR_losses:LR_losses_batch}))
                    saver.save(self.sess,cfg.model_ckpt,global_step=i)
        else:
            for i in range(cfg.epoch):
                LR_imgs,LR_losses,HR_imgs=shuffle(LR_imgs,LR_losses,HR_imgs)
                for j in range(int(len(HR_imgs)/cfg.batch_size)):
                    LR_batch,LR_losses_batch,HR_batch=LR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size],LR_losses[j*cfg.batch_size:(j+1)*cfg.batch_size],HR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size]
            #inputs the 15 channel layer if needed + High res image + low res one
                    self.sess.run(self.train_op,feed_dict={self.LR:LR_batch,self.HR:HR_batch,self.LR_losses:LR_losses_batch})
                    if j%5==0:
                        print(i,self.sess.run(self.loss,feed_dict={self.LR:LR_batch,self.HR:HR_batch,self.LR_losses:LR_losses_batch}))
                saver.save(self.sess,cfg.model_ckpt,global_step=i)
