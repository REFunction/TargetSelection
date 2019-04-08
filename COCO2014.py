
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.io as io
import threading
import queue
from pycocotools.coco import COCO
from input_simulation import *
import gc
from copy import deepcopy


class COCO2014:
    def __init__(self, root_path='./COCO/', mode='train'):
        print('Init COCO2014 Object......')
        # set paths
        self.train_image_dir = root_path + 'images/train2014'
        self.val_image_dir = root_path + 'images/val2014'
        train_ann_path = root_path + 'annotations/instances_train2014.json'
        val_ann_path = root_path + 'annotations/instances_val2014.json'
        # Initialize COCO api for instance annotations.
        if mode == 'train':
            self.coco_train = COCO(train_ann_path)
            self.train_image_ids = self.coco_train.getImgIds()
        else:
            self.coco_val = COCO(val_ann_path)
            self.val_image_ids = self.coco_val.getImgIds()
            # get image ids
            voc_cat_ids = [5,2,16,9,44,6,3,17,62,21,67,18,19,4,1,64,20,7,72]
            unvoc_cat_ids = list(set(np.arange(0, 80, 1).tolist()) - set(voc_cat_ids))
            self.val_image_ids = set()
            for i, cat_id in enumerate(unvoc_cat_ids):
                one_cat_img_ids = set(self.coco_val.getImgIds(catIds=[cat_id]))
                while len(self.val_image_ids) != (i + 1) * 10:
                    if len(one_cat_img_ids) > 0:
                        self.val_image_ids.add(one_cat_img_ids.pop())
                    else:
                        break

            self.val_image_ids = list(self.val_image_ids)

    def random_pos_points(self, mask, num_points):
        index_xs, index_ys = np.where(mask == 1)
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

    def random_neg_points(self, mask, num_points):
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

    def read_one_train_image(self): # read one image and get its masks
        if hasattr(self, 'train_location') == False:
            self.train_location = 0
        # read image
        while 1:
            image_id = int(self.train_image_ids[self.train_location])
            self.train_location = (self.train_location + 1) % len(self.train_image_ids)
            image = self.coco_train.loadImgs(image_id)[0]
            filename = image['file_name']
            image = io.imread('{}/{}'.format(self.train_image_dir, filename))
            if len(np.shape(image)) != 2:
                break
        # read anns
        annIds = self.coco_train.getAnnIds(imgIds=image_id)
        anns = self.coco_train.loadAnns(annIds)

        # ann to masks
        masks = []
        for ann in anns:
            single_mask = self.coco_train.annToMask(ann)  # change ann to single mask
            single_mask = single_mask.astype(np.uint8)
            masks.append(single_mask)
        return image, masks
    def read_one_val_image(self): # read one image and get its masks
        if hasattr(self, 'val_location') == False:
            self.val_location = 0
        # read image
        while 1:
            image_id = int(self.val_image_ids[self.val_location])
            self.val_location = (self.val_location + 1) % len(self.val_image_ids)
            image = self.coco_val.loadImgs(image_id)[0]
            filename = image['file_name']
            image = io.imread('{}/{}'.format(self.val_image_dir, filename))

            # read anns
            annIds = self.coco_val.getAnnIds(imgIds=image_id)
            anns = self.coco_val.loadAnns(annIds)

            # ann to masks
            masks = []
            for ann in anns:
                single_mask = self.coco_val.annToMask(ann)  # change ann to single mask
                single_mask = single_mask.astype(np.uint8)
                masks.append(single_mask)
            if len(np.shape(image)) == 3 and len(np.shape(masks)) == 3:
                break
        return image, masks
    def simulate(self, image, masks):# for every mask, get positive and negtive inputs
        images = []
        for mask in masks:# every mask produces two planes
            # get points
            pos_points = deepcopy(self.random_pos_points(mask, 15))
            neg_points = deepcopy(self.random_neg_points(mask, 15))
            # get planes
            pos_plane = np.zeros_like(mask)[:, :, np.newaxis]
            neg_plane = np.zeros_like(mask)[:, :, np.newaxis]
            pos_plane = pos_plane.copy()
            neg_plane = neg_plane.copy()

            for i in range(len(pos_points)):
                cv2.circle(pos_plane, (pos_points[i][1], pos_points[i][0]), 5, 1, thickness=-1)
            for i in range(len(neg_points)):
                cv2.circle(neg_plane, (neg_points[i][1], neg_points[i][0]), 5, 1, thickness=-1)
            images.append(np.concatenate([image, pos_plane, neg_plane], axis=2)) # get 5 channels
        return images, masks # images:[x, height, width, 5], masks:[x, height, width]

    def add_queue(self, max_queue_size=10):
        if hasattr(self, 'queue') == False:
            self.queue = queue.Queue(maxsize=max_queue_size)
        while 1:
            image, masks = self.read_one_train_image()
            images, masks = self.simulate(image, masks)
            for i in range(len(images)):
                self.queue.put([images[i], masks[i]])
    def start_queue(self, max_queue_size=10):
        if hasattr(self, 'queue') == False:
            queue_thread = threading.Thread(target=self.add_queue, args=(max_queue_size, ))
            queue_thread.start()
    def get_batch_train(self, batch_size=4, image_size=(513, 513)):
        while hasattr(self, 'queue') == False:
            self.start_queue()
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            image, mask = self.queue.get()
            image = cv2.resize(image, image_size)
            mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)

            batch_x.append(image)
            batch_y.append(mask)
        return batch_x, batch_y

    def get_one_val(self):
        image, masks = self.read_one_val_image()
        images, masks = self.simulate(image, masks)
        return [images[0]], [masks[0]]
    def add_queue_val(self, max_queue_size=10):
        if hasattr(self, 'val_queue') == False:
            self.val_queue = queue.Queue(maxsize=max_queue_size)
        while 1:
            image, masks = self.read_one_val_image()
            images, masks = self.simulate(image, masks)
            for i in range(len(images)):
                self.val_queue.put([images[i], masks[i]])
    def start_queue_val(self, max_queue_size=10):
        if hasattr(self, 'val_queue') == False:
            queue_thread = threading.Thread(target=self.add_queue_val, args=(max_queue_size, ))
            queue_thread.start()
    def get_batch_val(self, batch_size=4):
        while hasattr(self, 'val_queue') == False:
            self.start_queue_val()
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            image, mask = self.val_queue.get()
            if batch_x == []:
                batch_x.append(image)
                batch_y.append(mask)
            elif np.shape(batch_x)[1] == np.shape(image)[0] and np.shape(batch_x)[2] == np.shape(image)[1]:
                batch_x.append(image)
                batch_y.append(mask)
        return batch_x, batch_y
    # def start_queue(self):


if __name__ == "__main__":

    sums = []
    coco2014 = COCO2014('h:/COCO/', mode='train')


    for i in range(1000):
        x, y = coco2014.get_batch_train(batch_size=4)
        print(np.shape(x), np.shape(y))




    # for i in range(10):
    #     a, bs = coco2014.read_one_train_image()
    #     cv2.imshow('image', a)
    #     for k in range(len(bs)):
    #         cv2.imshow('mask', bs[k] * 255)
    #     cv2.waitKey(0)