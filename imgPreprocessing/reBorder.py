import cv2 as cv
import numpy as np

img = cv.imread("images/graffiti.jpg")

height, width, channels = img.shape

print(img.dtype)

borderFill = 10

mat = np.zeros((height + (2*borderFill), width + (2*borderFill)), dtype = "uint8")
img2 =  np.zeros((height + (2*borderFill), width + (2*borderFill), 3), dtype = img.dtype)

for i in range(height + (2*borderFill)):
    for j in range(width + (2*borderFill)):
        if( i < borderFill):
            mat[i,j] = 255
        elif( i >= height + borderFill):
            mat[i,j] = 255
        elif( j < borderFill):
            mat[i,j] = 255
        elif( j >= width + borderFill):
            mat[i,j] = 255
        else:
            img2[i,j,0] = img[i - borderFill, j - borderFill,0]
            img2[i,j,1] = img[i - borderFill, j - borderFill,1]
            img2[i,j,2] = img[i - borderFill, j - borderFill,2]




cv.imshow('Graffiti before extending the border', img)
cv.imshow('Graffiti transplanted onto a new array', img2)
cv.imshow('Mask', mat)
cv.waitKey(0)

dst = cv.inpaint(img2,mat,3,cv.INPAINT_TELEA)
cv.imshow('Inpainted border -- Telea',dst)
dst = cv.inpaint(img2,mat,3,cv.INPAINT_NS)
cv.imshow('Inpainted border -- Navier Stokes',dst)

cv.waitKey(0)

cv.destroyAllWindows()

