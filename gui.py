# coding=utf-8
import cv2
import os
from VOC2012_slim import VOC2012
from resolve_mask import resolve_mask
from copy import deepcopy
import numpy as np


drawing = 0  # 0不画，1左键画positive，2右键画negative
start = (-1, -1)
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def mouse_event(event, x, y, flags, param):
    global start, drawing, mode

    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = 1
        start = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = 2
        start = (x, y)
    # 鼠标移动，画图
    # elif event == cv2.EVENT_MOUSEMOVE:
    #     if drawing == 1:
    #         cv2.circle(mask, (x, y), 5, 2, -1)
    #     elif drawing == 2:
    #         cv2.circle(mask, (x, y), 5, 3, -1)
    # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = 0
        cv2.circle(blank, (x, y), 5, 2, -1)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        #mask[y][x] = 2
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = 0
        cv2.circle(blank, (x, y), 5, 3, -1)
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        #mask[y][x] = 3


if __name__ == '__main__':
    # voc2012 = VOC2012('h:/VOC2012/', image_size=(513, 513))
    # voc2012.read_aug_names()
    # for i in range(0, 100):
    #     image, mask = voc2012.get_batch_aug(batch_size=1)
    #     cv2.imshow('image', image[0])
    #     cv2.waitKey(0)
    #     masks = resolve_mask(mask[0], num_classes=20)
    #     mask = masks[0]
    #
    #     os.mkdir('data/' + str(i))
    #     cv2.imwrite('data/' + str(i) + '/' + 'mask.png', mask)
    #
    #     cv2.namedWindow('image')
    #     cv2.setMouseCallback('image', mouse_event)
    #
    #     while(True):
    #         cv2.imshow('image', mask * 255)
    #         # 按下m切换模式
    #         if cv2.waitKey(1) == ord('m'):
    #             break
    #     # positive
    #     pos = deepcopy(mask)
    #     pos[pos == 1] = 0
    #     pos[pos == 2] = 1.0
    #     pos[pos == 3] = 0
    #     # negative
    #     neg = deepcopy(mask)
    #     neg[neg == 1] = 0
    #     neg[neg == 2] = 0
    #     neg[neg == 3] = 1.0
    #
    #     cv2.imwrite('data/' + str(i) + '/' + 'image.png', image[0])
    #     cv2.imwrite('data/' + str(i) + '/' + 'pos.png', pos)
    #     cv2.imwrite('data/' + str(i) + '/' + 'neg.png', neg)

    path = 'demo/'
    image = cv_imread(path + 'boom.jpg')
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    blank = np.zeros((height, width), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_event)

    while (True):
        cv2.imshow('image', image)
        # 按下m切换模式
        if cv2.waitKey(1) == ord('m'):
            break
    # positive
    pos = deepcopy(blank)
    pos[pos == 1] = 0
    pos[pos == 2] = 1.0
    pos[pos == 3] = 0
    # negative
    neg = deepcopy(blank)
    neg[neg == 1] = 0
    neg[neg == 2] = 0
    neg[neg == 3] = 1.0
    cv2.imshow('pos', pos * 255)
    cv2.imshow('neg', neg * 255)
    cv2.imshow('image', image)
    # save
    cv2.imwrite(path + 'input.png', image)
    cv2.imwrite(path + 'pos.png', pos)
    cv2.imwrite(path + 'neg.png', neg)
    cv2.waitKey(0)


    # for i in range(1, 9):
    #     path = 'e:/unseen/' + str(i)
    #     image = cv_imread(path + '/image.jpg')
    #     height = np.shape(image)[0]
    #     width = np.shape(image)[1]
    #     blank = np.zeros((height, width), np.uint8)
    #     cv2.namedWindow('image')
    #     cv2.setMouseCallback('image', mouse_event)
    #
    #     while (True):
    #         cv2.imshow('image', image)
    #         # 按下m切换模式
    #         if cv2.waitKey(1) == ord('m'):
    #             break
    #     # positive
    #     pos = deepcopy(blank)
    #     pos[pos == 1] = 0
    #     pos[pos == 2] = 1.0
    #     pos[pos == 3] = 0
    #     # negative
    #     neg = deepcopy(blank)
    #     neg[neg == 1] = 0
    #     neg[neg == 2] = 0
    #     neg[neg == 3] = 1.0
    #     cv2.imshow('pos', pos * 255)
    #     cv2.imshow('neg', neg * 255)
    #     cv2.imshow('image', image)
    #     # save
    #     cv2.imwrite(path + '/input.png', image)
    #     cv2.imwrite(path + '/pos.png', pos)
    #     cv2.imwrite(path + '/neg.png', neg)
    #     cv2.waitKey(0)