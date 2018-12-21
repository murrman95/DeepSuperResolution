import os
width=21
height=21
    #if we want multi image set True / number of images we want to input
multi=False
nbr_input_im=5
    #number of channels as input
if multi:
    channel=3*nbr_input_im
else:
    channel=3
    #ending learning rate
learning_rate=0.15
s=0.25
batch_size=64
    #nbr of patches for each layers in the CNN
nbr_patches=64
    #nbr of hidden + input + output layers
nbr_layers=20
epoch=10
model_ckpt = os.path.join('weightfile91','model.ckpt')
numImages = 400
trainRatio = 0.95
imgSize = 21
writeResults=False
