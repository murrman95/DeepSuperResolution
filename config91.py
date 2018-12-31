import os
width=81
height=81
    #if we want multi image set True / number of images we want to input
multi=False
nbr_input_im=5
    #number of channels as input
if multi:
    channel=3*nbr_input_im
else:
    channel=3
    #ending learning rate
learning_rate=0.1
batch_size=64
    #nbr of patches for each layers in the CNN
nbr_patches=64
    #nbr of hidden + input + output layers
nbr_layers=20
epoch=30
model_ckpt = os.path.join('weightfile91','model.ckpt')
numImages = 100
trainRatio = 0.95
imgSize = 81
writeResults=False
