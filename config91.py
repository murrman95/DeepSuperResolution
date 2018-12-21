import os
width=81
height=81
    #if we want multi image set True / number of images we want to input
multi=False
nbr_input_im=5
    #number of channels as input
channel=3
    #ending learning rate
learning_rate=0.1
s=0.25
batch_size=64
    #nbr of patches for each layers in the CNN
nbr_patches=64
    #nbr of hidden + input + output layers
nbr_layers=20
epoch=30
model_ckpt = os.path.join('weightfile91','model.ckpt')
numImages = 400##300
trainRatio = 0.7
imgSize = 81
