import tensorflow as tf
import config91 as cfg
from sklearn.utils import shuffle

import numpy as np

class SRCNN:
    def __init__(self,sess):
        self.LR=tf.placeholder("float32",[None,cfg.width,cfg.height,cfg.channel])
        self.HR=tf.placeholder("float32",[None,cfg.width,cfg.height,3])
        self.gen_HR,self.weights,self.residuals=self.network()
        self.sess=sess
        # self.nbr_layers=6

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
                tensor = tf.nn.bias_add(tf.nn.conv2d(entry, conv_w, strides=[1,1,1,1], padding=padding), conv_b)
            else:
                tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(entry, conv_w, strides=[1,1,1,1], padding=padding), conv_b)) #, strides=[1,1,1,1]

        # output = tf.clip_by_value(tensor, clip_value_min=0, clip_value_max=1)
        if cfg.multi==True:
            tensor_out=tf.add(tensor,self.LR[:,:,:,0:3])
        else:
            tensor_out = tf.add(tensor, self.LR)
    #####one image end

        return tensor_out,weights,tensor


            #############potential personalisable padding (2 height and 1 wwidth)
            # input = tf.placeholder(tf.float32, [None, 28, 28, 3])
            # padded_input = tf.pad(input, [[0, 0], [2, 2], [1, 1], [0, 0]], "CONSTANT")
            # filter = tf.placeholder(tf.float32, [5, 5, 3, 16])
            # output = tf.nn.conv2d(padded_input, filter, strides=[1, 1, 1, 1], padding="VALID")

    def train(self,LR_imgs,HR_imgs):
        saver = tf.train.Saver(max_to_keep=1)
        self.loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(self.HR, self.gen_HR)))#tf.reduce_mean(tf.pow(self.HR - self.gen_HR-self.LR, 2))#tf.losses.absolute_difference(self.HR,self.gen_HR+self.LR)#tf.reduce_mean(tf.losses.absolute_difference(self.HR[:,:,1],self.gen_HR[:,:,1])+tf.losses.absolute_difference(self.HR[:,:,0],self.gen_HR[:,:,0])+tf.losses.absolute_difference(self.HR[:,:,2], self.gen_HR[:,:,2]))  # MSE #tf.losses.absolute_difference(self.HR,self.gen_HR)#
        for w in self.weights:
            self.loss += tf.nn.l2_loss(w)*1e-4
        global_step = tf.Variable(0, trainable=False)
        learning_rate 	= tf.train.exponential_decay(cfg.learning_rate/100, global_step*cfg.batch_size, len(LR_imgs)*120, cfg.learning_rate, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        for i in range(cfg.epoch):
            LR_imgs,HR_imgs=shuffle(LR_imgs,HR_imgs)
            for j in range(int(len(HR_imgs)/cfg.batch_size)):
                LR_batch,HR_batch=LR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size],HR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size]

                self.sess.run(self.train_op,feed_dict={self.LR:LR_batch,self.HR:HR_batch})
                if j%5==0:
                    print(i,self.sess.run(self.loss,feed_dict={self.LR:LR_batch,self.HR:HR_batch}))
            saver.save(self.sess,cfg.model_ckpt,global_step=i)
