import cv2
import numpy as np
from graph_cut import GraphMaker
import os

class GraphCut:
    def __init__(self):
        self.graph_maker = GraphMaker.GraphMaker()
    def load_image(self, image):
        self.graph_maker.load_image(filename=None, image=image)
    def set_points(self, fore_plane, back_plane):
        fores = []
        backs = []
        fore_index_xs, fore_index_ys = np.where(fore_plane == 1)
        fore_index = np.stack([fore_index_xs, fore_index_ys], axis=1)
        back_index_xs, back_index_ys = np.where(back_plane == 1)
        back_index = np.stack([back_index_xs, back_index_ys], axis=1)
        #print(len(fore_index))
        for i in range(len(fore_index)):
            self.graph_maker.add_seed(fore_index[i][1], fore_index[i][0], self.graph_maker.foreground)
        for i in range(len(back_index)):
            self.graph_maker.add_seed(back_index[i][1], back_index[i][0], self.graph_maker.background)

        #print(self.graph_maker.foreground_seeds)
        #print(self.graph_maker.background_seeds)
    def segment(self):
        self.graph_maker.create_graph()
        mask_pred = self.graph_maker.segment_overlay
        mask_pred = cv2.cvtColor(mask_pred, cv2.COLOR_BGR2GRAY)
        mask_pred[mask_pred > 0] = 1
        return mask_pred

from copy import deepcopy
from time import time
if __name__ == '__main__':

    for i in range(1, 9):
        graphcut = GraphCut()
        image = cv2.imread('e:/unseen/' + str(i) + '/image.jpg')
        image = cv2.resize(image, (256, 256))

        graphcut.load_image(image)
        pos = cv2.imread('e:/unseen/' + str(i) + '/pos.png', cv2.IMREAD_GRAYSCALE)
        neg = cv2.imread('e:/unseen/' + str(i) + '/neg.png', cv2.IMREAD_GRAYSCALE)
        pos = cv2.resize(pos, (256, 256))
        neg = cv2.resize(neg, (256, 256))
        start_time = time()
        graphcut.set_points(pos, neg)
        result = graphcut.segment()
        print('time:', time() - start_time)
        output = deepcopy(image)
        output[result == 0] = (0, 0, 0)

        cv2.imshow('image', image)
        #cv2.imshow('pos', pos * 255)
        #cv2.imshow('neg', neg * 255)
        cv2.imwrite('e:/unseen/' + str(i) + '/graphcut.png', output)
        cv2.imshow('output', output)
        cv2.waitKey(0)