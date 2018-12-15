import cv2 as cv
import numpy as np

#Supply function with path to image and the numbe of pixels to expand the border by.
def borderPad(path,numPixels):
    img = cv.imread(path)

    height, width, channels = img.shape

    print(img.dtype)

    #mask matrix
    mat = np.zeros((height + (2*numPixels), width + (2*numPixels)), dtype = "uint8")

    #The resulting image
    img2 =  np.zeros((height + (2*numPixels), width + (2*numPixels), 3), dtype = img.dtype)

    for i in range(height + (2*numPixels)):
        for j in range(width + (2*numPixels)):
            if( i < numPixels):
                mat[i,j] = 255
            elif( i >= height + numPixels):
                mat[i,j] = 255
            elif( j < numPixels):
                mat[i,j] = 255
            elif( j >= width + numPixels):
                mat[i,j] = 255
            else:
                img2[i,j,0] = img[i - numPixels, j - numPixels,0]
                img2[i,j,1] = img[i - numPixels, j - numPixels,1]
                img2[i,j,2] = img[i - numPixels, j - numPixels,2]

    return img2
