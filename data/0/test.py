import cv2
import numpy as np
from copy import deepcopy
image = cv2.imread('image.png')
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE) * 255
pos = cv2.imread('pos.png', cv2.IMREAD_GRAYSCALE) * 255
neg = cv2.imread('neg.png', cv2.IMREAD_GRAYSCALE) * 255
pred = cv2.imread('pred.png', cv2.IMREAD_GRAYSCALE) * 255

output = deepcopy(image)
height = np.shape(image)[0]
width = np.shape(image)[1]
for h in range(height):
    for w in range(width):
        if pos[h][w] == 255:
            image[h][w] = (2, 94, 33)
        elif neg[h][w] == 255:
            image[h][w] = (0, 0, 255)
    
for h in range(height):
    for w in range(width):
        if pred[h][w] == 0:
            output[h][w] = (0, 0, 0)

cv2.imshow('image', image)
cv2.imshow('output', output)

cv2.imshow('pred', pred)
cv2.waitKey(0)
