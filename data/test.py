import cv2
from copy import deepcopy
for i in range(100):
    image = cv2.imread(str(i) + '/image.png')
    mask = cv2.imread(str(i) + '/mask.png', cv2.IMREAD_GRAYSCALE) * 255
    #pos = cv2.imread(str(i) + '/pos.png', cv2.IMREAD_GRAYSCALE) * 255
    #neg = cv2.imread(str(i) + '/neg.png', cv2.IMREAD_GRAYSCALE) * 255
    pred = cv2.imread(str(i) + '/pred.png', cv2.IMREAD_GRAYSCALE) * 255

    output = deepcopy(image)

    output[pred == 0] = (0, 0, 0)
    image[pos == 255] = (0, 255, 0)
    image[neg == 255] = (0, 0, 255)

    cv2.imshow('image', image)
    #cv2.imshow('mask', mask)
    #cv2.imshow('pos', pos)
    #cv2.imshow('neg', neg)
    #cv2.imshow('pred', pred)
    cv2.imshow('output', output)
    cv2.imwrite(str(i) + '/input.png', image)
    cv2.imwrite(str(i) + '/output.png', output)
    cv2.waitKey(0)
