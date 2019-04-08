import cv2
import numpy as np
import os
from copy import deepcopy

def visulize_sample():
    for i in range(0, 100):
        image = cv2.imread('samples/' + str(i) + '.jpg')

        gt = cv2.imread('samples/' + str(i) + '_gt.png', cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread('samples/' + str(i) + '_pred.png', cv2.IMREAD_GRAYSCALE)
        pos = cv2.imread('samples/' + str(i) + 'pos.png', cv2.IMREAD_GRAYSCALE)
        neg = cv2.imread('samples/' + str(i) + 'neg.png', cv2.IMREAD_GRAYSCALE)

        output = deepcopy(image)
        output[gt == 0] = (0, 0, 0)
        image[pos == 1] = (0, 255, 0)
        image[neg == 1] = (0, 0, 255)


        #cv2.imshow('image', image)
        #cv2.imshow('output', output)
        cv2.imwrite('samples/' + str(i) + '_input.png', image)
        cv2.imwrite('samples/' + str(i) + '_output.png', output)

        #cv2.waitKey(0)
if __name__ == '__main__':
    visulize_sample()
# for i in range(9):
#     mask = cv2.imread('output/' + str(i) +'/mask.png', cv2.IMREAD_GRAYSCALE) * 255
#     pos = cv2.imread('output/' + str(i) + '/pos.png', cv2.IMREAD_GRAYSCALE)
#     neg = cv2.imread('output/' + str(i) + '/neg.png', cv2.IMREAD_GRAYSCALE)
#     pos = (np.ones_like(pos) - pos) * 180
#     neg = (np.ones_like(neg) - neg) * 60
#     # cv2.imshow('mask', mask)
#     # cv2.imshow('pos', pos)
#     cv2.imshow('neg', neg)
#     cv2.imshow('all', mask + pos)
#     cv2.waitKey(0)