import tensorflow as tf
import config as cfg
from sklearn.utils import shuffle
import csv
import numpy as np

class SRCNN:
    def __init__(self,sess):
        """
            Structures of interest
        """
            ## Input tensor, in the case of multi image input it stacks the images in 3*nbr_images channels
        self.LR=tf.placeholder("float32",[None,cfg.width,cfg.height,cfg.channel])
            ## Ground truth image, used during training phase
        self.HR=tf.placeholder("float32",[None,cfg.width,cfg.height,3])

            ## The central low-res image (in single image case = self.LR), useful for image reconstruction
        self.LR_losses=tf.placeholder("float32",[None,cfg.width,cfg.height,3])

            ## Outputs of the network
                ## gen_HR := reconstructed high res image / weights := weights of the NN
                ## residuals := Residual image (added to low-res to give the gen_HR)
        self.gen_HR,self.weights,self.residuals=self.network()

        self.sess=sess

    def network(self):
        """
            Very deep CNN building
        """
        nbr_patches=cfg.nbr_patches
        weights = []
        tensor = None
        padding='SAME'

    #### with one image as input
        for i in range(cfg.nbr_layers):
            ## Parameters for first layer
            if i==0:
                filter,bias_size=[3,3,cfg.channel,nbr_patches],[nbr_patches]
                entry=self.LR
            ##  Parameters for last layer
            elif i==cfg.nbr_layers-1:
                filter,bias_size= [3,3,nbr_patches,3],[3]
                entry=tensor
            ## Parameters for hidden layers
            else:
                filter,bias_size= [3,3,nbr_patches,nbr_patches],[nbr_patches]
                entry=tensor

            ## Construction of the CNN based on parameters above
            conv_w = tf.get_variable("conv_w_%s" % (i), filter, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
            conv_b = tf.get_variable("conv_b_%s" % (i), bias_size, initializer=tf.constant_initializer(0))
            weights.append(conv_w)
            weights.append(conv_b)
            if i==cfg.nbr_layers-1:
            ## We don't put the ReLu layer at the end of the network
                tensor = tf.nn.bias_add(tf.nn.conv2d(entry, conv_w, strides=[1,1,1,1], padding=padding), conv_b)
            else:
                tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(entry, conv_w, strides=[1,1,1,1], padding=padding), conv_b)) #, strides=[1,1,1,1]

        ## Reconstruction of the image (adding low-res and residual)
        tensor_out=tf.add(tensor,self.LR_losses)

        ## output generated image, weights of the network and the residuals image
        return tensor_out,weights,tensor

    def train(self,LR_imgs,HR_imgs,LR_losses,LR_test,HR_test,LR_losses_test):
        """
            Training phase
            inputs: - Stack of low res images
                    - Ground truth images
                    - Single low res image (central one in the case of multi image)
                    - Same 3 tensors as above but for evaluation purpose
        """
        saver = tf.train.Saver(max_to_keep=1)
        ## The loss function, squared diff between ground truth and constructed image (low-res + residuals from CNN)
        self.loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(self.HR, self.gen_HR)))
        for w in self.weights:
            self.loss += tf.nn.l2_loss(w)*1e-4

        ## Set or optimization process parameters, decaying learning rate to get faster convergence
        global_step = tf.Variable(0, trainable=False)
        learning_rate 	= tf.train.exponential_decay(cfg.learning_rate/100, global_step*cfg.batch_size, len(LR_imgs)*100, cfg.learning_rate, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())

        """
            Training phase registering "advancement" state (difference between test images and output)
            -> use of .._test inputs in train function
        """
        if(cfg.writeResults):
        ## Recording
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
        ## End of Recording

        ## Standard training phase
                    for j in range(int(len(HR_imgs)/cfg.batch_size)):
                        LR_batch,LR_losses_batch,HR_batch=LR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size],LR_losses[j*cfg.batch_size:(j+1)*cfg.batch_size],HR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size]
                        self.sess.run(self.train_op,feed_dict={self.LR:LR_batch,self.HR:HR_batch,self.LR_losses:LR_losses_batch})
                        if j%5==0:
                            print(i,self.sess.run(self.loss,feed_dict={self.LR:LR_batch,self.HR:HR_batch,self.LR_losses:LR_losses_batch}))
                    saver.save(self.sess,cfg.model_ckpt,global_step=i)

            """
            Training phase without advancement registration
        """
        else:
            for i in range(cfg.epoch):
                LR_imgs,LR_losses,HR_imgs=shuffle(LR_imgs,LR_losses,HR_imgs)
                for j in range(int(len(HR_imgs)/cfg.batch_size)):
                    LR_batch,LR_losses_batch,HR_batch=LR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size],LR_losses[j*cfg.batch_size:(j+1)*cfg.batch_size],HR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size]
                    self.sess.run(self.train_op,feed_dict={self.LR:LR_batch,self.HR:HR_batch,self.LR_losses:LR_losses_batch})
                    if j%5==0:
                        print(i,self.sess.run(self.loss,feed_dict={self.LR:LR_batch,self.HR:HR_batch,self.LR_losses:LR_losses_batch}))
                saver.save(self.sess,cfg.model_ckpt,global_step=i)
