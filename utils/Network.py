import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import numpy as np
import os
from VOC2012_slim import VOC2012
from resolve_mask import resolve_mask
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class Network:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def build(self, training=True):
        self.input_op = tf.placeholder(tf.float32, [None, None, None, 1])
        self.pos_op = tf.placeholder(tf.int32, [None, None, None])
        self.neg_op = tf.placeholder(tf.int32, [None, None, None])
        batch_norm_params = {'is_training': training}
        with slim.arg_scope(
                [slim.conv2d], activation_fn=tf.nn.relu,
                # weights_regularizer=slim.l2_regularizer(0.0005),
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params
        ):
            net = slim.conv2d(self.input_op, 64, 3)
            net = slim.conv2d(net, 64, 3)
            net = slim.conv2d(net, 64, 3)
            net = slim.conv2d(net, 64, 3)
            net = slim.conv2d(net, 64, 3)
            net = slim.conv2d(net, 64, 3)
            net = slim.conv2d(net, 64, 3)
            net = slim.conv2d(net, 64, 3)
            # pos
            net_pos = slim.conv2d(net, 64, 3)
            net_pos = slim.conv2d(net_pos, 64, 3)
            net_pos = slim.conv2d(net_pos, 64, 3)
            net_pos = slim.conv2d(net_pos, 1, 1)
            self.logits_pos = net_pos
            # neg
            net_neg = slim.conv2d(net, 64, 3)
            net_neg = slim.conv2d(net_neg, 64, 3)
            net_neg = slim.conv2d(net_neg, 64, 3)
            net_neg = slim.conv2d(net_neg, 1, 1)
            self.logits_neg = net_neg

    def train(self, learning_rate=0.01):
        self.pos_reshape = tf.reshape(self.pos_op, [-1])
        self.pos_reshape = tf.cast(self.pos_reshape, tf.float32)
        self.pos_reshape = tf.nn.softmax(self.pos_reshape)

        self.neg_reshape = tf.reshape(self.neg_op, [-1])
        self.neg_reshape = tf.cast(self.neg_reshape, tf.float32)
        self.neg_reshape = tf.nn.softmax(self.neg_reshape)

        self.logits_pos_reshape = tf.reshape(self.logits_pos, [-1])
        self.logits_neg_reshape = tf.reshape(self.logits_neg, [-1])
        self.loss = tf.reduce_mean(-tf.reduce_sum
            (self.pos_reshape * tf.log(tf.clip_by_value(self.logits_pos_reshape, 1e-10, 1)), reduction_indices=0))
        self.loss += tf.reduce_mean(-tf.reduce_sum
            (self.neg_reshape * tf.log(tf.clip_by_value(self.logits_neg_reshape, 1e-10, 1)), reduction_indices=0))
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(2e-4), tf.trainable_variables())
        self.loss += reg

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = slim.learning.create_train_op(self.loss, optimizer)

        #correct_prediction = tf.equal(self.output_op, self.pos_op)
        #self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        saver.restore(self.sess, 'model/model.ckpt')
        #self.sess.run(self.init_op)
        for epoch in range(50):
            for iter in range(100):
                mask = cv2.imread('data/' + str(iter) + '/'+ 'mask.png', cv2.IMREAD_GRAYSCALE)
                pos = cv2.imread('data/' + str(iter) + '/' + 'pos.png', cv2.IMREAD_GRAYSCALE)
                neg = cv2.imread('data/' + str(iter) + '/' + 'neg.png', cv2.IMREAD_GRAYSCALE)
                neg = -neg
                mask = mask[np.newaxis, :, :, np.newaxis]
                pos = pos[np.newaxis, :, :]
                neg = neg[np.newaxis, :, :]
                feed_dict = {self.input_op:mask, self.pos_op:pos, self.neg_op:neg}
                self.sess.run(self.train_op, feed_dict=feed_dict)
                if iter % 10 == 0:
                    loss_value = self.sess.run(self.loss, feed_dict=feed_dict)
                    print('epoch', epoch, 'iter', iter, 'loss:', loss_value)
                saver.save(self.sess, 'model/model.ckpt')
    def test(self):
        voc2012 = VOC2012('./VOC2012/', image_size=(513, 513))
        voc2012.read_aug_names()
        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        saver = tf.train.Saver()
        saver.restore(self.sess, 'model/model.ckpt')
        for i in range(10):
            image, mask = voc2012.get_batch_aug(batch_size=1)
            masks = resolve_mask(mask[0], num_classes=20)
            if not os.path.isdir('output/' + str(i)):
                os.mkdir('output/' + str(i))
            cv2.imwrite('output/' + str(i) + '/mask.png', masks[0])
            cv2.imwrite('output/' + str(i) + '/image.jpg', image[0])
            mask = masks[0][np.newaxis, :, :, np.newaxis]
            feed_dict = {self.input_op:mask}
            logits_pos_value = self.sess.run(self.logits_pos, feed_dict=feed_dict)
            logits_neg_value = self.sess.run(self.logits_neg, feed_dict=feed_dict)
            cv2.imwrite('output/' + str(i) + '/pos.png', logits_pos_value[0])
            cv2.imwrite('output/' + str(i) + '/neg.png', logits_neg_value[0])
            if i % 100 == 0:
                print('iter:', i)

if __name__ == '__main__':
    network = Network(20)
    network.build()
    network.test()
