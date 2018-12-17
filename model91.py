import tensorflow as tf
import config91 as cfg
from sklearn.utils import shuffle
slim=tf.contrib.slim
import numpy as np

class SRCNN:
    def __init__(self,sess):
        self.LR=tf.placeholder("float32",[None,cfg.width,cfg.height,cfg.channel])
        self.HR=tf.placeholder("float32",[None,cfg.width,cfg.height,cfg.channel])
        self.gen_HR=self.network()
        self.sess=sess
        # self.nbr_layers=6


    def network(self):
        nbr_patches=40
        weights = []
        tensor = None
        #conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
        conv_00_w = tf.get_variable("conv_00_w", [3,3,cfg.channel,nbr_patches], initializer=tf.random_normal_initializer(mean=0.5,stddev=0.1))
        conv_00_b = tf.get_variable("conv_00_b", [nbr_patches], initializer=tf.constant_initializer(0))
        weights.append(conv_00_w)
        weights.append(conv_00_b)

        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.LR, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))
        for i in range(2):
            #conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
            conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,nbr_patches,nbr_patches], initializer=tf.random_normal_initializer(mean=0.5,stddev=0.1))
            conv_b = tf.get_variable("conv_%02d_b" % (i+1), [nbr_patches], initializer=tf.constant_initializer(0))
            weights.append(conv_w)
            weights.append(conv_b)
            tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))

	    #conv_w = tf.get_variable("conv_19_w", [3,3,64,1], initializer=tf.contrib.layers.xavier_initializer())
        conv_w = tf.get_variable("conv_1_w", [3,3,nbr_patches,3], initializer=tf.random_normal_initializer(mean=0.5,stddev=0.1))
        conv_b = tf.get_variable("conv_20_b", [3], initializer=tf.constant_initializer(0))
        weights.append(conv_w)
        weights.append(conv_b)
        tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)        # conv=[]
        output = tf.clip_by_value(tensor, clip_value_min=0, clip_value_max=1)

            #old, working one
        # conv1 = slim.conv2d(self.LR,64,(3,3),padding='SAME',scope='conv1',activation=tf.nn.relu) #,cfg.channel
        # conv2 = slim.conv2d(conv1, 32, (3,3),padding='SAME', scope='conv2',activation=tf.nn.relu) #,cfg.channel
        # conv3 = slim.conv2d(conv2, cfg.channel, (3,3), scope='conv3',padding='SAME', weights_initializer=tf.truncated_normal_initializer(mean=0.5,stddev=0.1),activation=tf.nn.relu) #,cfg.channel
        return output

                # conv3 = slim.conv2d(conv2, 40, (3,3),padding='SAME', scope='conv3') #,cfg.channel
                # conv4 = slim.conv2d(conv3, 40, (3,3),padding='SAME', scope='conv4') #,cfg.channel
                # conv5 = slim.conv2d(conv4, 40, (3,3),padding='SAME', scope='conv5') #,cfg.channel
                # conv6 = slim.conv2d(conv5, 40, (3,3),padding='SAME', scope='conv6') #,cfg.channel
                # conv7 = slim.conv2d(conv6, 40, (3,3),padding='SAME', scope='conv7') #,cfg.channel
                # conv8 = slim.conv2d(conv7, 40, (3,3),padding='SAME', scope='conv8') #,cfg.channel
                # conv9 = slim.conv2d(conv8, 40, (3,3),padding='SAME', scope='conv9') #,cfg.channel

    def train(self,LR_imgs,HR_imgs):
        saver = tf.train.Saver(max_to_keep=1)
        self.loss = tf.reduce_mean(tf.pow(self.HR - self.gen_HR-self.LR, 2))#tf.losses.absolute_difference(self.HR,self.gen_HR+self.LR)#tf.reduce_mean(tf.losses.absolute_difference(self.HR[:,:,1],self.gen_HR[:,:,1])+tf.losses.absolute_difference(self.HR[:,:,0],self.gen_HR[:,:,0])+tf.losses.absolute_difference(self.HR[:,:,2], self.gen_HR[:,:,2]))  # MSE #tf.losses.absolute_difference(self.HR,self.gen_HR)#
        self.train_op = tf.train.AdamOptimizer(cfg.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        for i in range(cfg.epoch):
            LR_imgs,HR_imgs=shuffle(LR_imgs,HR_imgs)
            for j in range(int(len(HR_imgs)/cfg.batch_size)):
                LR_batch,HR_batch=LR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size],HR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size]

                self.sess.run(self.train_op,feed_dict={self.LR:LR_batch,self.HR:HR_batch})
                if j%5==0:
                    print(i,self.sess.run(self.loss,feed_dict={self.LR:LR_batch,self.HR:HR_batch}))
            saver.save(self.sess,cfg.model_ckpt,global_step=i)
