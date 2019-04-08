import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
import numpy as np
from net_models.deeplabv3p_resnet import deeplabv3p_resnet
from net_models.fcn import fcn
from net_models.deeplabv3p import deeplabv3p
from VOC2012_slim import VOC2012
import time
import os
import cv2
from input_simulation import Data
from tensorpack.tfutils.optimizer import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Network:
    def __init__(self, learning_rate=3e-4, batch_size=4, num_classes=21, dataset='voc', method='regression'):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dataset = dataset
        self.method = method
    def build(self, training=True):
        print('building network...')
        with tf.name_scope('input'):
            self.input_op = tf.placeholder(tf.float32, [None, None, None, 5])
            self.label_op = tf.placeholder(tf.int32, [None, None, None])
        if self.dataset == 'voc':
            net = self.input_op - np.array([104.00699, 116.66877, 122.67892, 0, 0])
        elif self.dataset == 'coco':
            net = self.input_op - np.array([116.43323, 114.04722, 108.33814, 0, 0])
        if self.method == 'regression':
            net, var_list = deeplabv3p_resnet(net, 1, training)
            self.net = net # [batch_size, height, width, 1]
        elif self.method == 'classification':
            #net, var_list = fcn(net, 2, training)
            net, var_list = deeplabv3p_resnet(net, 2, training)
            #net, var_list = deeplabv3p(net, 2, training)
            self.logits = net # [batch_size, height, width, 2]
            self.net = tf.argmax(net, axis=3, output_type=tf.int32)# [batch_size, height, width]
    def label_loss(self):
        pos_op = tf.slice(self.input_op, [0, 0, 0, 3], [tf.shape(self.input_op)[0], tf.shape(self.input_op)[1],
                                                        tf.shape(self.input_op)[2], 1])
        neg_op = tf.slice(self.input_op, [0, 0, 0, 4], [tf.shape(self.input_op)[0], tf.shape(self.input_op)[1],
                                                        tf.shape(self.input_op)[2], 1])
        self.logits = tf.slice(self.logits, [0, 0, 0, 1], [tf.shape(self.logits)[0], tf.shape(self.logits)[1],
                                                        tf.shape(self.logits)[2], 1])
        self.label_loss = pos_op * ((pos_op - self.logits) ** 2) - neg_op * ((neg_op - self.logits) ** 2)
        self.label_loss = tf.reduce_mean(self.label_loss)
        self.label_loss = tf.sigmoid(self.label_loss)

    def train(self, restore_path='model/model.ckpt'):
        # loss function
        if self.method == 'regression':# see as a regression problem, mse loss
            self.net_reshape = tf.reshape(self.net, [-1])
            self.label_reshape = tf.reshape(self.label_op, [-1])
            self.label_reshape = tf.cast(self.label_reshape, tf.float32)
            self.loss = tf.reduce_mean((self.net_reshape - self.label_reshape) ** 2, reduction_indices=0)
            one = tf.ones_like(self.net, dtype=tf.int32)
            zero = tf.zeros_like(self.net, dtype=tf.int32)
            self.net = tf.where(self.net < 0.5, x=zero, y=one)
            self.net = tf.squeeze(self.net, [3])
            # metrics
            correct_prediction = tf.equal(self.net, self.label_op)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        elif self.method == 'classification':# see as a classification problem, entropy loss
            self.iter = tf.placeholder(tf.int32)
            self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                       labels=self.label_op)))
            self.label_loss()
            #self.loss = self.loss + self.label_loss * tf.pow(2.0, tf.cast(-self.iter / 10000, tf.float32))
            self.loss = self.loss + self.label_loss
            # metrics
            correct_prediction = tf.equal(self.net, self.label_op)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # train op
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        #optimizer = AccumGradOptimizer(optimizer, 3)
        self.train_op = slim.learning.create_train_op(self.loss, optimizer)
        # summary
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.acc_summary = tf.summary.scalar('acc', self.accuracy)
        # sess
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        summary_writer = tf.summary.FileWriter('logs/', self.sess.graph)
        # init or restore
        saver = tf.train.Saver()
        if restore_path is None:
            self.sess.run(tf.global_variables_initializer())
            print('Initialize all parameters')
        else:
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        best_miou = 0
        # dataset
        if self.dataset == 'voc':
            data_obj = Data(root_path='./VOC2012/', flip=False)
        else:
            from COCO2014 import COCO2014
            data_obj = COCO2014(root_path='./COCO/')

        for iter in range(0, 200001):
            start_time = time.time()
            # get batch
            if self.dataset == 'voc':
                batch_x, batch_y = data_obj.get_batch_fast(self.batch_size)
            else:
                batch_x, batch_y = data_obj.get_batch_train(self.batch_size)
            # train
            feed_dict = {self.input_op:batch_x, self.label_op:batch_y, self.iter:iter}
            _, loss_value, label_loss_value, acc_value = self.sess.run([self.train_op, self.loss, self.label_loss, self.accuracy], feed_dict=feed_dict)
            if iter % 10 == 0:
                print('iter:', iter, 'entropy:', loss_value - label_loss_value, 'label_loss:',  label_loss_value,
                      'acc:', acc_value, 'time:', time.time() - start_time)
            # log
            if iter % 50 == 0:
                summary_loss, summary_acc = self.sess.run([self.loss_summary, self.acc_summary],
                                             feed_dict=feed_dict)
                summary_writer.add_summary(summary_loss, iter)
                summary_writer.add_summary(summary_acc, iter)
            # eval
            if iter % 5000 == 0 and iter > 1:
                if self.dataset == 'voc':
                    miou_value = self.eval(iter)
                else:
                    miou_value = self.eval_coco(iter=iter, restore_path=restore_path)
                if miou_value > best_miou:
                    best_miou = miou_value
                    saver.save(self.sess, 'model/model.ckpt') # save
    def sample(self, sample_num=8, restore_path='model/model.ckpt', sample_path='samples/'):
        # sess
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        saver = tf.train.Saver()
        saver.restore(self.sess, restore_path)
        print('Restore from', restore_path)
        data_obj = Data(root_path='./VOC2012/')
        for i in range(sample_num):
            batch_x, batch_y, _ = data_obj.get_one_val()
            batch_y = np.array(batch_y)
            # test time
            # batch_x = cv2.resize(batch_x[0], (1024, 1024))[np.newaxis, :, :, :]
            # feed_dict = {self.input_op: batch_x}
            # start_time = time.time()
            # self.sess.run(self.net, feed_dict=feed_dict)
            # print('time:', time.time() - start_time)
            #----------
            feed_dict = {self.input_op: batch_x, self.label_op: batch_y}
            logits = self.sess.run(self.net, feed_dict=feed_dict)
            cv2.imwrite(sample_path + str(i) + '.jpg', batch_x[0][:, :, 0:3])
            cv2.imwrite(sample_path + str(i) + 'pos.png', batch_x[0][:, :, 3:4])
            cv2.imwrite(sample_path + str(i) + 'neg.png', batch_x[0][:, :, 4:5])
            cv2.imwrite(sample_path + str(i) + '_gt.png', batch_y[0])
            cv2.imwrite(sample_path + str(i) + '_pred.png', logits[0])
            print('iter:', i + 1, '/', sample_num)
    def inference(self, image_path, pos_path, neg_path, output_path, restore_path='model/model.ckpt'):
        # sess
        if hasattr(self, 'sess') == False or self.sess == None:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)

        image = cv2.imread(image_path)
        pos = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
        neg = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)

        image = image[np.newaxis, :, :, :]
        pos = pos[np.newaxis, :, :, np.newaxis]
        neg = neg[np.newaxis, :, :, np.newaxis]

        x = np.concatenate([image, pos, neg], axis=3)
        feed_dict = {self.input_op:x}
        logits = self.sess.run(self.net, feed_dict=feed_dict)
        cv2.imwrite(output_path, logits[0])
        print('saved into', output_path)
    def eval(self, iter=0, restore_path='model/model.ckpt'):
        # sess
        if hasattr(self, 'sess') == False or self.sess == None:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        # define ops
        if hasattr(self, 'acc_op') == False:
            if self.method == 'regression':
                one = tf.ones_like(self.net, dtype=tf.int32)
                zero = tf.zeros_like(self.net, dtype=tf.int32)
                self.net = tf.where(self.net < 0.5, x=zero, y=one)
                self.net = tf.squeeze(self.net, [3])
            self.mask255_op = tf.placeholder(tf.float32)
            self.mask255_op = tf.reshape(self.mask255_op, [-1])
            self.acc_op, self.acc_update = tf.metrics.accuracy(labels=tf.reshape(self.label_op, [-1]),
                                predictions=tf.reshape(self.net, [-1]), weights=self.mask255_op)
            self.miou_op, self.miou_update = tf.metrics.mean_iou(num_classes=2,
                                labels=tf.reshape(self.label_op, [-1]),
                                predictions=tf.reshape(self.net, [-1]),
                                weights=self.mask255_op)
            self.local_init_op = tf.local_variables_initializer()
        self.sess.run(self.local_init_op)
        # data object
        data_obj = Data(root_path='./VOC2012/', image_size=None)
        # calc update ops
        for i in range(3427):
            x, y, mask255 = data_obj.get_one_val()
            mask255 = np.reshape(mask255, [-1])
            feed_dict = {self.input_op:x, self.label_op:y, self.mask255_op:mask255}
            self.sess.run([self.acc_update, self.miou_update], feed_dict)
            if i % 100 == 0:
                print('iter:', i + 1, '/', 3427)
        acc_value, miou_value = self.sess.run([self.acc_op, self.miou_op])
        print('acc:', acc_value, 'miou:', miou_value)
        logs_file = open('logs/val_logs.txt', 'a')
        logs_file.write(str(iter) + ' acc:' + str(acc_value) + ' miou:' + str(miou_value) + '\r\n')
        logs_file.close()
        return miou_value
    def eval_coco(self, iter, restore_path='model/model.ckpt'):
        # sess
        if hasattr(self, 'sess') == False or self.sess == None:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        # define ops
        if hasattr(self, 'acc_op') == False:
            self.acc_op, self.acc_update = tf.metrics.accuracy(labels=tf.reshape(self.label_op, [-1]),
                                                               predictions=tf.reshape(self.net, [-1]))
            self.miou_op, self.miou_update = tf.metrics.mean_iou(num_classes=2,
                                                                 labels=tf.reshape(self.label_op, [-1]),
                                                                 predictions=tf.reshape(self.net, [-1]))
            self.local_init_op = tf.local_variables_initializer()
        self.sess.run(self.local_init_op)
        # data object
        from COCO2014 import COCO2014
        coco2014 = COCO2014(root_path='./COCO/', mode='val')
        # calc update ops
        for i in range(1000):
            x, y = coco2014.get_batch_val(1)
            feed_dict = {self.input_op: x, self.label_op: y}
            self.sess.run([self.acc_update, self.miou_update], feed_dict)
            if i % 100 == 0:
                print('iter:', i, '/', 1000)
        acc_value, miou_value = self.sess.run([self.acc_op, self.miou_op])
        print('acc:', acc_value, 'miou:', miou_value)
        logs_file = open('logs/val_logs.txt', 'a')
        logs_file.write(str(iter) + ' acc:' + str(acc_value) + ' miou:' + str(miou_value) + '\r\n')
        logs_file.close()
        return miou_value
    def eval_graph_cut(self, iter=0):
        from graph_cut.inference import GraphCut

        # define ops
        if hasattr(self, 'acc_op') == False:
            if self.method == 'regression':
                one = tf.ones_like(self.net, dtype=tf.int32)
                zero = tf.zeros_like(self.net, dtype=tf.int32)
            self.net = tf.placeholder(tf.int32, [None, None])
            self.mask255_op = tf.placeholder(tf.float32)
            self.mask255_op = tf.reshape(self.mask255_op, [-1])
            self.acc_op, self.acc_update = tf.metrics.accuracy(labels=tf.reshape(self.label_op, [-1]),
                                                               predictions=tf.reshape(self.net, [-1]),
                                                               weights=self.mask255_op)
            self.miou_op, self.miou_update = tf.metrics.mean_iou(num_classes=2,
                                                                 labels=tf.reshape(self.label_op, [-1]),
                                                                 predictions=tf.reshape(self.net, [-1]),
                                                                 weights=self.mask255_op)
            self.local_init_op = tf.local_variables_initializer()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.local_init_op)
        # data object
        data_obj = Data(root_path='./VOC2012/', image_size=None)
        # calc update ops
        for i in range(10):
            graphcut = GraphCut()
            x, y, mask255 = data_obj.get_one_val()
            mask255 = np.reshape(mask255, [-1])
            graphcut.load_image(x[0][:, :, 0:3])
            graphcut.set_points(np.squeeze(x[0][:, :, 3:4]), np.squeeze(x[0][:, :, 4:5]))
            result = graphcut.segment()

            feed_dict = {self.net: result, self.label_op: y, self.mask255_op: mask255}
            self.sess.run([self.acc_update, self.miou_update], feed_dict)

            if i % 1 == 0:
                print('iter:', i + 1, '/', 3427)
        acc_value, miou_value = self.sess.run([self.acc_op, self.miou_op])
        print('acc:', acc_value, 'miou:', miou_value)
        return miou_value

if __name__ == '__main__':
    net = Network(learning_rate=1e-4, method='classification', batch_size=4, dataset='voc')
    net.build(training=True)
    #net.eval(iter=0, restore_path='model/no_label_loss_94/model.ckpt')
    #net.eval(restore_path='model/classification-bs4-pos10-neg10-size5/model.ckpt')
    #net.eval(iter=0, restore_path='model/xception_label_loss_91/model.ckpt')
    #net.eval_coco(iter=0, restore_path='model/final/model.ckpt')
    #net.eval(iter=0, restore_path='model/fcn/model.ckpt')
    #net.eval(iter=0, restore_path='model/final/model.ckpt')
    #net.eval_graph_cut(0)
    #net.train(restore_path='model/model.ckpt')
    net.train(restore_path=None)
    #net.sample(restore_path='model/final/model.ckpt', sample_num=100)

    # for i in range(1, 9):
    #     net.inference(image_path='data/' + str(i) + '/image.png', pos_path='data/' + str(i) + '/pos.png',
    #               neg_path='data/' + str(i) + '/neg.png', output_path='data/' + str(i) + '/pred.png',
    #                   restore_path='model/classification-bs4-pos15-neg15-size5-seg/model.ckpt')
    #net.inference(image_path='demo/boom.jpg', pos_path='demo/pos.png', neg_path='demo/neg.png',
    #              output_path='demo/pred.png', restore_path='model/final/model.ckpt')
