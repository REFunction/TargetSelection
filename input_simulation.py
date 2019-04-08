import numpy as np
from VOC2012_slim import VOC2012
from resolve_mask import resolve_mask
import random
import cv2
import h5py
import gc
import queue
import threading
import time
from copy import deepcopy

def random_pos_points(mask, num_points):
    index_xs, index_ys = np.where(mask == 1)
    index = np.stack([index_xs, index_ys], axis=1)
    real_num_points = min(len(index), num_points)
    temp = np.arange(len(index))
    random_points_indice = np.random.choice(temp, real_num_points)
    random_points = []
    for i in random_points_indice:
        random_points.append(index[i])
    return random_points
def random_neg_points(mask, num_points):
    index_xs, index_ys = np.where(mask == 0)
    index = np.stack([index_xs, index_ys], axis=1)
    real_num_points = min(len(index), num_points)
    temp = np.arange(len(index))
    if len(temp) != 0:
        random_points_indice = np.random.choice(temp, real_num_points)
    else:
        return []
    random_points = []
    for i in random_points_indice:
        random_points.append(index[i])
    return random_points
def convert_image_mask(image, mask, num_classes, keep255=False):
    '''

    :param image: [height, width, 3]
    :param mask: [height, width, 1], elements are 0 or class numbers
    :param num_classes:
    :return:
    '''
    # get masks
    masks = resolve_mask(mask, num_classes=40, keep255=keep255)
    # simulate user's inputs
    input_5cs = []
    for mask_num in range(len(masks)):
        pos_points = random_pos_points(masks[mask_num], 15)
        neg_points = random_neg_points(masks[mask_num], 15)
        # put points in two planes
        pos_plane = np.zeros_like(masks[mask_num])[:, :, np.newaxis]
        neg_plane = np.zeros_like(masks[mask_num])[:, :, np.newaxis]
        for i in range(len(pos_points)):
            cv2.circle(pos_plane, (pos_points[i][1], pos_points[i][0]), 5, 1, thickness=-1)
        for i in range(len(neg_points)):
            cv2.circle(neg_plane, (neg_points[i][1], neg_points[i][0]), 5, 1, thickness=-1)
        input_5cs.append(np.concatenate([image, pos_plane, neg_plane], axis=2))
    return input_5cs, masks
def print2d(matrix):
    height = np.shape(matrix)[0]
    width = np.shape(matrix)[1]
    for h in range(height):
        print(matrix[h])
    print()
    print()
class Data:
    def __init__(self, root_path, image_size=(513, 513), flip=False):
        self.voc2012 = VOC2012(root_path=root_path, image_size=image_size)
        self.voc2012.read_aug_names()
        self.flip = flip
    def get_one_image(self):
        #image, mask = self.voc2012.get_batch_train_instance(batch_size=1)
        image, mask = self.voc2012.get_batch_aug(batch_size=1) # semantic
        if self.flip:
            if random.random() > 0.5:
                image = cv2.flip(image[0], 1)[np.newaxis, :, :, :]
                mask = cv2.flip(mask[0], 1)[np.newaxis, :, :]
        input_5cs, label_01s = convert_image_mask(image[0], mask[0], num_classes=20)
        return input_5cs, label_01s
    def get_one_val(self):
        if hasattr(self, 'val_queue') == False:
            self.val_queue = queue.Queue(maxsize=30)
        if self.val_queue.qsize() > 0:
            input_5c, label_01, mask255 = self.val_queue.get()
        else:
            image, mask = self.voc2012.get_one_val_instance()
            input_5cs, label_01s = convert_image_mask(image[0], mask[0], num_classes=40, keep255=True)
            mask255s = deepcopy(label_01s)

            for i in range(len(mask255s)):
                mask255s[i][mask255s[i] != 255] = 0
                mask255s[i][mask255s[i] == 0] = 1
                mask255s[i][mask255s[i] == 255] = 0
                label_01s[i][label_01s[i] == 255] = 0
                
                self.val_queue.put([input_5cs[i], label_01s[i], mask255s[i]])
            
            input_5c, label_01, mask255 = self.val_queue.get()
        return [input_5c], [label_01], [mask255]

    def start_queue(self, batch_size, max_queue_size=30):
        if hasattr(self, 'queue') == False:
            queue_thread = threading.Thread(target=self.add_queue, args=(batch_size, max_queue_size))
            queue_thread.start()

    def add_queue(self, batch_size, max_queue_size=30):
        if hasattr(self, 'queue') == False:
            self.queue = queue.Queue(maxsize=max_queue_size)
        while 1:
            input_5cs, label_01s = self.get_one_image()
           
            for i in range(len(input_5cs)):
                self.queue.put([input_5cs[i], label_01s[i]])
    def get_batch_fast(self, batch_size, max_queue_size=30):
        if hasattr(self, 'queue') == False:
            queue_thread = threading.Thread(target=self.add_queue, args=(batch_size, max_queue_size))
            queue_thread.start()
        while hasattr(self, 'queue') == False:
            time.sleep(0.1)
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            x, y = self.queue.get()
            batch_x.append(x)
            batch_y.append(y)
        return batch_x, batch_y




if __name__ == '__main__':
    #voc2012 = VOC2012('./VOC2012/', image_size=(513, 513))
    #for i in range(10):
    #    x, y, mask255 = voc2012.get_one_val()
    #    cv2.imshow('image', x[0])
    #    cv2.imshow('mask', y[0])
    #    print(np.shape(x), np.shape(y))
    #    cv2.waitKey(3000)

    import time
    data_obj = Data(root_path='h:/VOC2012/', flip=True)

    sums = []

    for i in range(2000):
        # image1, mask1 = data_obj.voc2012.get_batch_train_instance(1)
        # image2, mask2 = data_obj.voc2012.get_batch_aug(1)
        # cv2.imshow('image1', image1[0])
        # cv2.imshow('image2', image2[0])
        # cv2.imshow('mask1', mask1[0] * 10)
        # cv2.imshow('mask2', mask2[0] * 10)
        # cv2.waitKey(0)

        a, b = data_obj.get_batch_fast(4)
        a = np.array(a)
        cv2.imshow('image', a[3][:, :, 0:3])
        cv2.imshow('pos', a[3][:, :, 3:4] * 255)
        cv2.imshow('neg', a[3][:, :, 4:5] * 255)
        cv2.imshow('mask', b[3] * 255)
        cv2.waitKey(0)