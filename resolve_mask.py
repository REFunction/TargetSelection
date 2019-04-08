from VOC2012_slim import VOC2012
import cv2
from copy import deepcopy
import os

def resolve_mask(mask, num_classes, keep255=False):
    result = []
    for i in range(1, num_classes + 1):
        if len(mask[mask == i]) != 0:
            mask_one = deepcopy(mask)
            if keep255:
                mask255 = deepcopy(mask)
                mask255[mask255 != 255] = 0
                mask_one[mask_one != i] = 0
                mask_one = mask_one + mask255
            else:
                mask_one[mask_one != i] = 0
            mask_one[mask_one == i] = mask_one[mask_one == i] / i
            result.append(mask_one)
    return result

if __name__ == '__main__':
    voc2012 = VOC2012('./VOC2012/', image_size=(513, 513))
    voc2012.read_aug_names()
    for k in range(10582):
        image, mask = voc2012.get_one_val()

        masks = resolve_mask(mask[0], num_classes=20, keep255=True)
        cv2.imshow('image', image[0])
        cv2.imshow('mask', masks[0])
        cv2.waitKey(3000)