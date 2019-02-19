import os

    ## Resize the images before work, decreasing the size higly accelerate the learning phase
imgSize=81
width=imgSize
height=imgSize

learning_rate=0.1
images_path="./images"
epoch=30
model_ckpt = os.path.join('weightfile91','model.ckpt')
    ## Number of images from
numImages = 100
    ## Ratio of the train/train split
trainRatio = 0.95
    ## Set to True to save result of a square difference (between input low-res image and constructed higher-res) every 5 epochs
writeResults=False

"""
    Parameters from the VDSR paper (20 layers convolution)
"""
batch_size=64
    #nbr of patches for each layers in the CNN
nbr_patches=64
    #nbr of hidden + input + output layers
nbr_layers=20

"""
    Set multi=True to work with the multi input version
    - You need to modify the file splitImage5 to change the number of input images in multi-image case
    - You need to modify the learning rate in the model file
"""
multi=False
nbr_input_im=5
    #number of channels as input
if multi:
    channel=3*nbr_input_im
    learning_rate=0.1
else:
    channel=3
