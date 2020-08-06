import sys
import numpy as np
import tensorflow as tf
from resnet_v1 import resnet_v1_50 as resnet_50
from IO import getDatas
import os
import pickle
from ScalarSaver import ScalarSaver
import voxel
from PIL import Image
from dataIO import dataIO
from binvox_rw import Voxels
from MyThread import MyThread
import time
from Refinement_utils import *
import threading
import queue as queue
import scipy.io as sio

# from collections import deque as queue

# learning_rate = 1e-4
# lr = 5*1e-4

weights = {}
bias = {}
batchsize = 6
voxel_size = 32
img_h = 128
img_w = 128
vector_channel = 1024
iterations = 600000
# data_path = '../dataset/ShapeNetRendering'
# data_path = '../python-client/download/03001627'
data_path = '../dataset/input/choose'
#data_path = '../python-client/download'
# data_path = 'D:/Files/Dataset/datas'
vox_path = './ShapeNet/ShapeNetVox32'
# vox_path = '../dataset/Vox32'
bg_path = '../dataset/bg_crop'
# bg_path = '../dataset/bg_crop'

tensorboard_path = 'tensorboard_cross_categories_13_Adam_joint_nobg_loss_in_detail_dynamic_lr_flip_correctset'
checkpoint_path = 'checkpoint_cross_categories_13_Adam_joint_nobg_loss_in_detail_dynamic_lr_flip_correctset'
# tensorboard_path = 'tensorboard_car_5'
# checkpoint_path = 'checkpoint_car_5'

thresh_hold = 0.4
# cates = {'02958343'}
# cates = {'03001627'}
#cates = ['03001627', '04256520', '04379243', '02828884', '02691156', '02958343']
cates = ["04256520", "02691156", "03636649", "04401088",
            "04530566", "03691459", "03001627", "02933112",
            "04379243", "03211117", "02958343", "02828884", "04090263"]

dic = {"04256520": "sofa", "02691156": "airplane", "03636649": "lamp", "04401088": "telephone",
            "04530566": "vessel", "03691459": "loudspeaker", "03001627": "chair", "02933112": "cabinet",
            "04379243": "table", "03211117": "display", "02958343": "car", "02828884": "bench", "04090263": "rifle"}
# available_gpu = [2,5,6,7]
beta = 0.001

small_set = ["04256520", "02691156", "03001627","02958343"]

lrs = {20000: 1e-5}

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def encoder_residual_block(input, layer_id, num_layers=2, channels=None):
    input_shape = input.get_shape()
    last_channel = int(input_shape[-1])
    last_layer = input
    batch_size = int(input_shape[0])

    wd_res = tf.get_variable("wres%d" % layer_id, shape=[1, 1, last_channel, channels],
                             initializer=tf.contrib.layers.xavier_initializer())
    wb_res = tf.get_variable("bres%d" % layer_id, shape=[channels], initializer=tf.zeros_initializer())
    res = tf.nn.conv2d(input, wd_res, strides=[1, 1, 1, 1], padding='SAME')
    res = tf.nn.bias_add(res, wb_res)
    res = lrelu(res)

    for i in range(num_layers):
        wd_conv = tf.get_variable("wd%d_%d" % (layer_id, i), shape=[3, 3, last_channel, channels],
                                  initializer=tf.contrib.layers.xavier_initializer())
        wb_conv = tf.get_variable("wb%d_%d" % (layer_id, i), shape=[channels], initializer=tf.zeros_initializer())
        last_layer = tf.nn.conv2d(last_layer, wd_conv, strides=[1, 1, 1, 1], padding='SAME')
        last_layer = tf.nn.bias_add(last_layer, wb_conv)
        last_layer = lrelu(last_layer)
        last_channel = channels

    output = res + last_layer
    return output

def encoder(input, reuse=False):
    batch_size = int(input.get_shape()[0])
    input = tf.reshape(input, shape=[batch_size, img_h, img_w, 3])
    layer_id = 1
    shortcuts = []
    with tf.variable_scope("encoder", reuse=reuse):
        wd00 = tf.get_variable("wd00", shape=[7, 7, 3, 96], initializer=tf.contrib.layers.xavier_initializer())
        bd00 = tf.get_variable("bd00", shape=[96], initializer=tf.zeros_initializer())
        conv0a = tf.nn.conv2d(input, wd00, strides=[1, 1, 1, 1], padding='SAME')
        conv0a = tf.nn.bias_add(conv0a, bd00)

        wd01 = tf.get_variable("wd01", shape=[3, 3, 96, 96], initializer=tf.contrib.layers.xavier_initializer())
        bd01 = tf.get_variable("bd01", shape=[96], initializer=tf.zeros_initializer())
        conv0b = tf.nn.conv2d(conv0a, wd01, strides=[1, 1, 1, 1], padding='SAME')
        conv0b = tf.nn.bias_add(conv0b, bd01)

        pool1 = tf.nn.max_pool(conv0b, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool1)

        conv1 = encoder_residual_block(pool1, layer_id, 2, 128)
        pool2 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool2)
        layer_id += 1

        conv2 = encoder_residual_block(pool2, layer_id, 2, 256)
        pool3 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool3)
        layer_id += 1

        wd30 = tf.get_variable("wd30", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bd30 = tf.get_variable("bd30", shape=[256], initializer=tf.zeros_initializer())
        conv3a = tf.nn.conv2d(pool3, wd30, strides=[1, 1, 1, 1], padding='SAME')
        conv3a = tf.nn.bias_add(conv3a, bd30)

        wd31 = tf.get_variable("wd31", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bd31 = tf.get_variable("bd31", shape=[256], initializer=tf.zeros_initializer())
        conv3b = tf.nn.conv2d(conv3a, wd31, strides=[1, 1, 1, 1], padding='SAME')
        conv3b = tf.nn.bias_add(conv3b, bd31)

        pool4 = tf.nn.max_pool(conv3b, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool4)
        layer_id += 1

        conv4 = encoder_residual_block(pool4, layer_id, 2, 256)
        pool5 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool5)
        layer_id += 1

        conv5 = encoder_residual_block(pool5, layer_id, 2, 256)
        pool6 = tf.nn.max_pool(conv5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        feature_map = pool6

        pool6 = tf.reduce_mean(pool6, [1, 2])
        wfc = tf.get_variable("wfc", shape=[256, 1024], initializer=tf.contrib.layers.xavier_initializer())
        feature = tf.matmul(pool6, wfc)

        w_e = tf.get_variable("w_euler", shape=[1024, 3], initializer=tf.contrib.layers.xavier_initializer())
        euler_angle = tf.matmul(feature, w_e)

        w_st = tf.get_variable('w_ft', shape=[1024, 3], initializer=tf.contrib.layers.xavier_initializer())
        st = tf.matmul(feature, w_st)

        print('pool1', pool1)
        print('pool2', pool2)
        print('pool3', pool3)
        print('pool4', pool4)
        print('pool5', pool5)
        print('pool6', pool6)
        print('feature', feature)
        print('feature_map', feature_map)

        return feature, feature_map, euler_angle, st, shortcuts

def encoder_angle(input, reuse=False):
    batch_size = int(input.get_shape()[0])
    input = tf.reshape(input, shape=[batch_size, img_h, img_w, 3])
    layer_id = 1
    shortcuts = []
    eulers_cates = {}
    st_cates = {}
    with tf.variable_scope("encoder", reuse=reuse):
        wd00 = tf.get_variable("wd00", shape=[7, 7, 3, 96], initializer=tf.contrib.layers.xavier_initializer())
        bd00 = tf.get_variable("bd00", shape=[96], initializer=tf.zeros_initializer())
        conv0a = tf.nn.conv2d(input, wd00, strides=[1, 1, 1, 1], padding='SAME')
        conv0a = tf.nn.bias_add(conv0a, bd00)

        wd01 = tf.get_variable("wd01", shape=[3, 3, 96, 96], initializer=tf.contrib.layers.xavier_initializer())
        bd01 = tf.get_variable("bd01", shape=[96], initializer=tf.zeros_initializer())
        conv0b = tf.nn.conv2d(conv0a, wd01, strides=[1, 1, 1, 1], padding='SAME')
        conv0b = tf.nn.bias_add(conv0b, bd01)

        pool1 = tf.nn.max_pool(conv0b, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool1)

        conv1 = encoder_residual_block(pool1, layer_id, 2, 128)
        pool2 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool2)
        layer_id += 1

        conv2 = encoder_residual_block(pool2, layer_id, 2, 256)
        pool3 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool3)
        layer_id += 1

        wd30 = tf.get_variable("wd30", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bd30 = tf.get_variable("bd30", shape=[256], initializer=tf.zeros_initializer())
        conv3a = tf.nn.conv2d(pool3, wd30, strides=[1, 1, 1, 1], padding='SAME')
        conv3a = tf.nn.bias_add(conv3a, bd30)

        wd31 = tf.get_variable("wd31", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bd31 = tf.get_variable("bd31", shape=[256], initializer=tf.zeros_initializer())
        conv3b = tf.nn.conv2d(conv3a, wd31, strides=[1, 1, 1, 1], padding='SAME')
        conv3b = tf.nn.bias_add(conv3b, bd31)

        pool4 = tf.nn.max_pool(conv3b, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool4)
        layer_id += 1

        conv4 = encoder_residual_block(pool4, layer_id, 2, 256)
        pool5 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool5)
        layer_id += 1

        conv5 = encoder_residual_block(pool5, layer_id, 2, 256)
        pool6 = tf.nn.max_pool(conv5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        feature_map = pool6

        pool6 = tf.reduce_mean(pool6, [1, 2])
        wfc = tf.get_variable("wfc", shape=[256, 1024], initializer=tf.contrib.layers.xavier_initializer())
        feature = tf.matmul(pool6, wfc)

        print('pool1', pool1)
        print('pool2', pool2)
        print('pool3', pool3)
        print('pool4', pool4)
        print('pool5', pool5)
        print('pool6', pool6)
        print('feature', feature)
        print('feature_map', feature_map)

        return feature, feature_map, shortcuts

def generator(input, shortcuts, reuse=False):
    batch_size = int(input.shape[0])
    strides = [[1, 2, 2, 1],  # 4
               [1, 2, 2, 1],  # 8
               [1, 2, 2, 1],  # 16
               [1, 2, 2, 1],  # 32
               [1, 2, 2, 1],  # 64
               [1, 2, 2, 1]]  # 127

    print(input)

    with tf.variable_scope("ge", reuse=reuse):
        wg1 = tf.get_variable('wg1', shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bg1 = tf.get_variable('bg1', shape=[256], initializer=tf.zeros_initializer())
        g_1 = tf.nn.conv2d_transpose(input, wg1, [batch_size, 4, 4, 256], strides=strides[0], padding='SAME')
        g_1 = tf.nn.bias_add(g_1, bg1)
        g_1 = lrelu(g_1)
        g_1 = tf.add(g_1, shortcuts[4])

        wg2 = tf.get_variable('wg2', shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bg2 = tf.get_variable('bg2', shape=[256], initializer=tf.zeros_initializer())
        g_2 = tf.nn.conv2d_transpose(g_1, wg2, [batch_size, 8, 8, 256], strides=strides[1], padding='SAME')
        g_2 = tf.nn.bias_add(g_2, bg2)
        g_2 = lrelu(g_2)
        g_2 = tf.add(g_2, shortcuts[3])

        wg3 = tf.get_variable('wg3', shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bg3 = tf.get_variable('bg3', shape=[256], initializer=tf.zeros_initializer())
        g_3 = tf.nn.conv2d_transpose(g_2, wg3, [batch_size, 16, 16, 256], strides=strides[2], padding='SAME')
        g_3 = tf.nn.bias_add(g_3, bg3)
        g_3 = lrelu(g_3)
        g_3 = tf.add(g_3, shortcuts[2])

        wg4 = tf.get_variable('wg4', shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
        bg4 = tf.get_variable('bg4', shape=[128], initializer=tf.zeros_initializer())
        g_4 = tf.nn.conv2d_transpose(g_3, wg4, [batch_size, 32, 32, 128], strides=strides[3], padding='SAME')
        g_4 = tf.nn.bias_add(g_4, bg4)
        g_4 = lrelu(g_4)
        g_4 = tf.add(g_4, shortcuts[1])

        wg5 = tf.get_variable('wg5', shape=[4, 4, 96, 128], initializer=tf.contrib.layers.xavier_initializer())
        bg5 = tf.get_variable('bg5', shape=[96], initializer=tf.zeros_initializer())
        g_5 = tf.nn.conv2d_transpose(g_4, wg5, [batch_size, 64, 64, 96], strides=strides[4], padding='SAME')
        g_5 = tf.nn.bias_add(g_5, bg5)
        g_5 = lrelu(g_5)
        g_5 = tf.add(g_5, shortcuts[0])

        wg6 = tf.get_variable('wg6', shape=[4, 4, 2, 96], initializer=tf.contrib.layers.xavier_initializer())
        g_6 = tf.nn.conv2d_transpose(g_5, wg6, [batch_size, img_h, img_w, 2], strides=strides[5], padding='SAME')
        mask_softmax = tf.nn.softmax(g_6)
    return g_6, mask_softmax

def residual_block(input, layer_id, num_layers=2, div=2, unpool=True):
    input_shape = input.get_shape()
    last_channel = input_shape[-1]
    current_channel = int(int(last_channel) / div)

    # reg = tf.contrib.layers.l2_regularizer(scale=0.1)
    # upsampling

    strides = [1, 1, 1, 1, 1]
    output_shape = [int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3]),
                    current_channel]
    if unpool:
        strides = [1, 2, 2, 2, 1]
        output_shape = [int(input_shape[0]), int(input_shape[1]) * 2, int(input_shape[2]) * 2, int(input_shape[3]) * 2,
                        current_channel]

    wd_0 = tf.get_variable("wd%d_0" % layer_id, shape=[3, 3, 3, current_channel, last_channel],
                           initializer=tf.contrib.layers.xavier_initializer())
    bd_0 = tf.get_variable("bd%d_0" % layer_id, shape=[current_channel], initializer=tf.zeros_initializer())
    d_0 = tf.nn.conv3d_transpose(value=input, filter=wd_0, output_shape=output_shape, strides=strides, padding='SAME')
    d_0 = tf.nn.bias_add(d_0, bd_0)
    d_0 = tf.nn.relu(d_0)

    last_layer = d_0

    for i in range(1, num_layers + 1):
        wd = tf.get_variable("wd%d_%d" % (layer_id, i), shape=[3, 3, 3, current_channel, current_channel],
                             initializer=tf.contrib.layers.xavier_initializer())
        bd = tf.get_variable("bd%d_%d" % (layer_id, i), shape=[current_channel], initializer=tf.zeros_initializer())
        d = tf.nn.conv3d(last_layer, filter=wd, strides=[1, 1, 1, 1, 1], padding='SAME')
        d = tf.nn.bias_add(d, bd)
        d = tf.nn.relu(d)
        last_layer = d

    return tf.add(last_layer, d_0)

def decoder(input, reuse=False):
    batch_size = int(input.get_shape()[0])
    strides = [1, 2, 2, 2, 1]
    layer_id = 2
    print(input)
    with tf.variable_scope("decoder", reuse=reuse):
        input = tf.reshape(input, (batch_size, 1, 1, 1, 1024))
        print(input)
        wd = tf.get_variable("wd1", shape=[4, 4, 4, 256, 1024],
                             initializer=tf.contrib.layers.xavier_initializer())
        bd = tf.get_variable("bd1", shape=[256], initializer=tf.zeros_initializer())

        d_1 = tf.nn.conv3d_transpose(input, wd, (batch_size, 4, 4, 4, 256), strides=[1, 1, 1, 1, 1], padding='VALID')
        d_1 = tf.nn.bias_add(d_1, bd)
        d_1 = tf.nn.relu(d_1)

        d_2 = residual_block(d_1, layer_id)
        layer_id += 1

        d_3 = residual_block(d_2, layer_id)
        layer_id += 1

        d_4 = residual_block(d_3, layer_id)
        layer_id += 1

        d_5 = residual_block(d_4, layer_id, 3, unpool=False)
        layer_id += 1

        last_channel = int(d_5.shape[-1])

        print('d1', d_1)
        print('d2', d_2)
        print('d3', d_3)
        print('d4', d_4)
        print('d5', d_5)

        wd = tf.get_variable("wd6", shape=[3, 3, 3, 2, last_channel],
                             initializer=tf.contrib.layers.xavier_initializer())

        res = tf.nn.conv3d_transpose(d_5, wd, (batch_size, 32, 32, 32, 2), strides=[1, 1, 1, 1, 1], padding='SAME')
        res_softmax = tf.nn.softmax(res)
        print('d6', res)
        return res, res_softmax

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
        if len(grads) != 0:
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads

#####################################################
# Begin Data IO functions
#####################################################

class data_fetch_worker(object):
    def __init__(self, io, num_gpu, batch_size=batchsize):
        # self.train_queue = queue.Queue(20)
        self.train_queue = {}  # queue.Queue(20)
        self.test_queue = {}
        self.io = io
        self.cates = io.cates
        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self.thread_stop = False
        self.thread_pool = []
        for cate in self.cates:
            self.test_queue[cate] = queue.Queue(11)
            self.train_queue[cate] = queue.Queue(20)

    def run(self, random_bg=True):
        for cate in self.cates:
            th = threading.Thread(target=self.fill_train, args=[cate, random_bg])
            self.thread_pool.append(th)
        for cate in self.cates:
            th = threading.Thread(target=self.fill_test, args=[cate, random_bg])
            self.thread_pool.append(th)
        for th in self.thread_pool:
            th.setDaemon(True)
            th.start()

        print('run over')

    def stop(self):
        self.thread_stop = True

    def fill_train(self, cate, random_bg=True):
        # print('fill training start')
        while not self.thread_stop:
            data = get_data(self.io, self.batch_size, self.num_gpu, train_phase=True, cates=[cate], random_bg=random_bg)
            # print('fill a training batch')
            self.train_queue[cate].put(data, block=True)

    def fill_test(self, cate, random_bg=True):
        # print('fill testing start')
        while not self.thread_stop:
            data = get_data(self.io, self.batch_size, self.num_gpu, train_phase=False, cates=[cate],
                            random_bg=random_bg)
            # print(dic[cate],data[0].shape)
            # print('fill a testing batch')
            self.test_queue[cate].put(data, block=True)

    def get_batch(self, cate=None, train_phase=True):
        if train_phase:
            return self.train_queue[cate].get(block=True)
        else:
            return self.test_queue[cate].get(block=True)


def load_batch(io, batch_size, train_phase, cates, random_bg=True):
    res = io.fetch_batch(batch_size, train_phase=train_phase, random_background=random_bg, cates=cates)
    x, y, m, e, st = res[0:5]
    x = data_transfer(x)
    m = data_transfer(m)
    return x, y, m, e, st

def get_data(io, num_gpu, batch_size, train_phase=True, cates=None, random_bg=True):
    # print('start fetching')
    thread_pool = []
    x_op = []
    y_op = []
    m_op = []
    e_op = []
    st_op = []

    for i in range(num_gpu):
        th = MyThread(load_batch, (io, batch_size, train_phase, cates, random_bg))
        thread_pool.append(th)
        th.start()

    for th in thread_pool:
        th.join()
        res = th.get_result()
        x, y, m, e, st = res[0:5]
        x_op.append(x)
        y_op.append(y)
        m_op.append(m)
        e_op.append(e)
        st_op.append(st)

    # print('fetch over')
    x_op = np.asarray(x_op).reshape([batch_size * num_gpu, voxel_size, voxel_size, voxel_size, 2])
    y_op = np.asarray(y_op).reshape([batch_size * num_gpu, img_w, img_h, 3])
    m_op = np.asarray(m_op).reshape([batch_size * num_gpu, img_w, img_h, 2])
    e_op = np.asarray(e_op).reshape([batch_size * num_gpu, 3])
    st_op = np.asarray(st_op).reshape([batch_size * num_gpu, 3])
    return x_op, y_op, m_op, e_op, st_op

def data_transfer(data):
    data = np.squeeze(data)
    ret = np.squeeze(np.ones_like(data, dtype=np.float))
    ret -= data
    ret = np.stack([data, ret], axis=-1)
    return ret

class Stacker(object):
    def __init__(self, worker):
        self.train_queue = queue.Queue(11)
        self.test_queue = queue.Queue(11)
        self.worker = worker
        self.cates = worker.cates
        # self.thread_pool = []
        self.thread_stop = False

    def run(self, random_bg=True):
        self.worker.run(random_bg)

        th = threading.Thread(target=self.fill_train,args=[])
        th.setDaemon(True)
        th.start()

        th = threading.Thread(target=self.fill_test,args=[])
        th.setDaemon(True)
        th.start()


    def stop(self):
        self.thread_stop = True

    def fill_train(self):
        while not self.thread_stop:
            x_stack = []
            y_stack = []
            m_stack = []
            e_stack = []
            st_stack = []
            for cate in self.cates:
                x_op, y_op, m_op, e_op, st_op = self.worker.get_batch(cate,train_phase=True)
                x_stack.append(x_op)
                y_stack.append(y_op)
                m_stack.append(m_op)
                e_stack.append(e_op)
                st_stack.append(st_op)
            x_stack = np.stack(x_stack, 1)
            y_stack = np.stack(y_stack, 1)
            m_stack = np.stack(m_stack, 1)
            e_stack = np.stack(e_stack, 1)
            st_stack = np.stack(st_stack, 1)
            data = [x_stack, y_stack, m_stack, e_stack, st_stack]
            self.train_queue.put(data)

    def fill_test(self):
        while not self.thread_stop:
            x_stack = []
            y_stack = []
            m_stack = []
            e_stack = []
            st_stack = []
            for cate in self.cates:
                x_op, y_op, m_op, e_op, st_op = self.worker.get_batch(cate, train_phase=False)
                x_stack.append(x_op)
                y_stack.append(y_op)
                m_stack.append(m_op)
                e_stack.append(e_op)
                st_stack.append(st_op)
            x_stack = np.stack(x_stack, 1)
            y_stack = np.stack(y_stack, 1)
            m_stack = np.stack(m_stack, 1)
            e_stack = np.stack(e_stack, 1)
            st_stack = np.stack(st_stack, 1)
            data = [x_stack, y_stack, m_stack, e_stack, st_stack]
            self.test_queue.put(data)

    def get_batch(self,train_phase=True):
        if train_phase:
            return self.train_queue.get()
        else:
            return self.test_queue.get()


#####################################################
# End Data IO functions
#####################################################

def train_end2end(weight_path=None, continue_train_path=None, add_regulizer=False):
    tensorboard_path = 'tensorboard_cross_categories_4_Adam_end2end'
    checkpoint_path = 'checkpoint_cross_categories_4_Adam_end2end'
    lrs = {60000: 1e-6}

    cates = small_set


    def calc_loss(input, gt, reuse=True):
        # feature, _, __ = resnet_50(input, global_pool=True, reuse=reuse)
        feature, shortcuts = refine_encoder(input, reuse=reuse)
        voxels, voxels_softmax = refine_decoder(feature, shortcuts, reuse=reuse)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=voxels))

        return loss, voxels_softmax



    with tf.device('/cpu:0'):
        num_gpu = 1  # len(available_gpu)
        # num_gpu = 1
        # tower_grads = []
        # tower_loss = []
        # tower_iou = []
        # tower_loss_reg = []
        # tower_iou_before = []
        tower_grads_cates = {}
        tower_loss_cates = {}
        tower_loss_reg_cates = {}
        tower_loss_e_0_cates = {}
        tower_loss_e_1_cates = {}
        tower_loss_e_2_cates = {}
        tower_loss_dist_cates = {}
        tower_loss_transx_cates = {}
        tower_loss_transy_cates = {}
        tower_err_e_0_cates = {}
        tower_err_e_1_cates = {}
        tower_voxel_iou_before_cates = {}
        tower_voxel_iou_after_cates = {}
        tower_mask_iou_cates = {}
        tower_loss_mask_cates = {}

        apply_gradient_op_cates = {}

        io = dataIO(data_path, bg_path, vox_path, cates)
        worker = data_fetch_worker(io, num_gpu)
        worker.run()
        worker_uniform = data_fetch_worker(io, num_gpu)
        worker_uniform.run(False)

        for cate in cates:
            tower_grads_cates[cate] = []
            tower_loss_cates[cate] = []
            tower_loss_reg_cates[cate] = []
            tower_loss_e_0_cates[cate] = []
            tower_loss_e_1_cates[cate] = []
            tower_loss_e_2_cates[cate] = []
            tower_loss_dist_cates[cate] = []
            tower_loss_transx_cates[cate] = []
            tower_loss_transy_cates[cate] = []
            tower_err_e_0_cates[cate] = []
            tower_err_e_1_cates[cate] = []
            tower_voxel_iou_after_cates[cate] = []
            tower_voxel_iou_before_cates[cate] = []
            tower_mask_iou_cates[cate] = []
            tower_loss_mask_cates[cate] = []



        learning_rate = tf.placeholder(dtype=tf.float32)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


        with tf.device('/gpu:0'):
            x_vectors = tf.placeholder(shape=[batchsize * num_gpu, voxel_size, voxel_size, voxel_size, 2],
                                       dtype=tf.float32, name='all_Voxels')
            y_vectors = tf.placeholder(shape=[batchsize * num_gpu, img_w, img_h, 3], dtype=tf.float32,
                                       name='all_Images')
            m_vectors = tf.placeholder(shape=[batchsize * num_gpu, img_w, img_h, 2], dtype=tf.float32, name='all_masks')

            e_vectors = tf.placeholder(shape=[batchsize * num_gpu, 3], dtype=tf.float32, name='all_angles')

            st_vectors = tf.placeholder(shape=[batchsize * num_gpu, 3], dtype=tf.float32, name='all_translation')
            #x_vectors_ = down_sample(x_vectors)
            x_vectors_ = x_vectors

        reuse_dict = {}
        reuse_dict['encoder'] = False
        for cate in cates:
            reuse_dict[cate] = False

        for i in range(num_gpu):
            with tf.device('/gpu:%d' % i):
                cur_x = x_vectors_[i * batchsize:(i + 1) * batchsize]
                cur_y = y_vectors[i * batchsize:(i + 1) * batchsize]
                cur_m = m_vectors[i * batchsize:(i + 1) * batchsize]
                cur_e = e_vectors[i * batchsize:(i + 1) * batchsize]
                cur_st = st_vectors[i * batchsize:(i + 1) * batchsize]

                reuse = reuse_dict['encoder']

                with tf.variable_scope("voxel"):
                    feature, feature_map, euler, st, shortcuts = encoder(cur_y, reuse=reuse)
                    voxels, voxels_softmax = decoder(feature, reuse=reuse)

                with tf.variable_scope("mask"):
                    feature, feature_map, euler, st, shortcuts = encoder(cur_y, reuse=reuse)
                    mask, mask_softmax = generator(feature_map, shortcuts, reuse=reuse)

                for cate in cates:
                    reuse = reuse_dict['encoder']
                    reuse_dict['encoder'] = True

                    reuse_fc = reuse_dict[cate]
                    reuse_dict[cate] = True

                    feature, feature_map, shortcuts = encoder_angle(cur_y, reuse=reuse)

                    with tf.variable_scope("angles_trans", reuse=reuse_fc):
                        w_e_1 = tf.get_variable("w_euler_0_%s" % cate, shape=[1024, 512],
                                                initializer=tf.contrib.layers.xavier_initializer())
                        e_1 = lrelu(tf.matmul(feature, w_e_1))
                        w_e_2 = tf.get_variable("w_euler_1_%s" % cate, shape=[512, 3],
                                                initializer=tf.contrib.layers.xavier_initializer())
                        euler = tf.matmul(e_1, w_e_2)

                        w_st = tf.get_variable('w_ft_%s' % cate, shape=[1024, 3],
                                               initializer=tf.contrib.layers.xavier_initializer())
                        st = tf.matmul(feature, w_st)

                    loss_e_0 = tf.reduce_mean(tf.abs(euler[..., 0] - cur_e[..., 0]))
                    loss_e_1 = tf.reduce_mean(tf.abs(euler[..., 1] - cur_e[..., 1]))
                    loss_e_2 = tf.reduce_mean(tf.abs(euler[..., 2] - cur_e[..., 2]))

                    loss_dist = tf.reduce_mean(tf.abs(st[..., 0] - cur_st[..., 0]))
                    loss_transx = tf.reduce_mean(tf.abs(st[..., 1] - cur_st[..., 1]))
                    loss_transy = tf.reduce_mean(tf.abs(st[..., 2] - cur_st[..., 2]))

                    err_e_0 = tf.reduce_mean(
                        tf.minimum(tf.abs(euler[..., 0] - cur_e[..., 0]),
                                   tf.abs(1 - tf.abs(euler[..., 0] - cur_e[..., 0])))) * 360
                    err_e_1 = tf.reduce_mean(
                        tf.minimum(tf.abs(euler[..., 1] - cur_e[..., 1]),
                                   tf.abs(1 - tf.abs(euler[..., 1] - cur_e[..., 1])))) * 360

                    # feature, feature_map, euler, st, shortcuts = encoder(cur_y, reuse=reuse)
                    #
                    # mask, mask_softmax = generator(feature_map, shortcuts, reuse=reuse)
                    # voxels, voxels_softmax = decoder(feature, reuse=reuse)

                    IoU_before = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(voxels_softmax[:, :, :, :, 0] > thresh_hold,
                                               cur_x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32),axis=[1,2,3]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(voxels_softmax[:, :, :, :, 0] > thresh_hold,
                                              cur_x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32),axis=[1,2,3]))



                    loss_mask = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=cur_m, logits=mask))

                    IoU_mask = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(mask_softmax[..., 0] > thresh_hold,
                                               cur_m[..., 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(mask_softmax[..., 0] > thresh_hold,
                                              cur_m[..., 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2]))

                    rotation_matrices = get_rotation_matrix(euler)
                    mask_indexs = scale_trans(st)
                    projection = cast(mask_softmax[..., 0], mask_indexs, rotation_matrices=rotation_matrices)

                    pc = rotate_and_translate(rotation_matrices, st)
                    dvx = DVX(projection)
                    b_falpha = tf.reduce_sum(pc*tf.stop_gradient(dvx),axis=-1)
                    projection = b_falpha - tf.stop_gradient(b_falpha) + tf.stop_gradient(projection)

                    c1 = voxels_softmax[..., 0]
                    c2 = projection
                    c3 = c1 - c1 * c2
                    c4 = c2 - c1 * c2

                    feedin = tf.stack([c1, c2, c3, c4], axis=4)

                    loss_, voxels_softmax_after = calc_loss(feedin, cur_x, reuse)
                    loss = loss_

                    IoU = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                               cur_x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32),axis=[1,2,3]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                              cur_x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32),axis=[1,2,3]))

                    params = tf.trainable_variables()
                    # params = [para for para in params if 'refine' in para.name]

                    if add_regulizer:
                        for para in params:
                            loss_ += beta * tf.reduce_mean(tf.square(para))

                    grads = optimizer.compute_gradients(loss_, params)
                    tower_grads_cates[cate].append(grads)
                    tower_loss_cates[cate].append(loss)
                    tower_loss_reg_cates[cate].append(loss_)
                    tower_voxel_iou_after_cates[cate].append(IoU)
                    tower_voxel_iou_before_cates[cate].append(IoU_before)
                    tower_loss_e_0_cates[cate].append(loss_e_0)
                    tower_loss_e_1_cates[cate].append(loss_e_1)
                    tower_loss_e_2_cates[cate].append(loss_e_2)
                    tower_loss_dist_cates[cate].append(loss_dist)
                    tower_loss_transx_cates[cate].append(loss_transx)
                    tower_loss_transy_cates[cate].append(loss_transy)
                    tower_err_e_0_cates[cate].append(err_e_0)
                    tower_err_e_1_cates[cate].append(err_e_1)
                    tower_mask_iou_cates[cate].append(IoU_mask)
                    tower_loss_mask_cates[cate].append(loss_mask)



        with tf.device('/gpu:0'):

            total_loss_reg_cates = {}
            total_loss_cates = {}

            total_loss_e_0_cates = {}
            total_loss_e_1_cates = {}
            total_loss_e_2_cates = {}

            total_err_e_0_cates = {}
            total_err_e_1_cates = {}

            total_loss_dist_cates = {}
            total_loss_transx_cates = {}
            total_loss_transy_cates = {}

            total_voxel_iou_before_cates = {}
            total_voxel_iou_after_cates = {}
            total_mask_iou_cates = {}
            total_loss_mask_cates = {}

            total_iou_diff_cates = {}

            summary_list_cates = {}
            summary_merge_cates = {}

            for cate in cates:
                grads = average_gradients(tower_grads_cates[cate])
                apply_gradient_op_cates[cate] = optimizer.apply_gradients(grads)

                total_loss_reg_cates[cate] = 0
                total_loss_cates[cate] = 0

                total_loss_e_0_cates[cate] = 0
                total_loss_e_1_cates[cate] = 0
                total_loss_e_2_cates[cate] = 0

                total_err_e_0_cates[cate] = 0
                total_err_e_1_cates[cate] = 0

                total_loss_dist_cates[cate] = 0
                total_loss_transx_cates[cate] = 0
                total_loss_transy_cates[cate] = 0

                total_voxel_iou_before_cates[cate] = 0
                total_voxel_iou_after_cates[cate] = 0
                total_mask_iou_cates[cate] = 0
                total_loss_mask_cates[cate] = 0

                summary_list_cates[cate] = []

                for _ in tower_voxel_iou_before_cates[cate]:
                    total_voxel_iou_before_cates[cate] += _
                for _ in tower_voxel_iou_after_cates[cate]:
                    total_voxel_iou_after_cates[cate] += _
                for _ in tower_loss_cates[cate]:
                    total_loss_cates[cate] += _
                for _ in tower_loss_reg_cates[cate]:
                    total_loss_reg_cates[cate] += _
                for _ in tower_loss_e_0_cates[cate]:
                    total_loss_e_0_cates[cate] += _
                for _ in tower_loss_e_1_cates[cate]:
                    total_loss_e_1_cates[cate] += _
                for _ in tower_loss_e_2_cates[cate]:
                    total_loss_e_2_cates[cate] += _

                for _ in tower_err_e_0_cates[cate]:
                    total_err_e_0_cates[cate] += _
                for _ in tower_err_e_1_cates[cate]:
                    total_err_e_1_cates[cate] += _
                for _ in tower_loss_dist_cates[cate]:
                    total_loss_dist_cates[cate] += _
                for _ in tower_loss_transx_cates[cate]:
                    total_loss_transx_cates[cate] += _
                for _ in tower_loss_transy_cates[cate]:
                    total_loss_transy_cates[cate] += _
                for _ in tower_mask_iou_cates[cate]:
                    total_mask_iou_cates[cate] += _
                for _ in tower_loss_mask_cates[cate]:
                    total_loss_mask_cates[cate] += _

                total_voxel_iou_before_cates[cate] /= num_gpu
                total_voxel_iou_after_cates[cate] /= num_gpu
                total_loss_cates[cate] /= num_gpu
                total_loss_reg_cates[cate] /= num_gpu
                total_loss_e_0_cates[cate] /= num_gpu
                total_loss_e_1_cates[cate] /= num_gpu
                total_loss_e_2_cates[cate] /= num_gpu
                total_err_e_0_cates[cate] /= num_gpu
                total_err_e_1_cates[cate] /= num_gpu
                total_loss_dist_cates[cate] /= num_gpu
                total_loss_transx_cates[cate] /= num_gpu
                total_loss_transy_cates[cate] /= num_gpu
                total_mask_iou_cates[cate] /= num_gpu
                total_loss_mask_cates[cate] /= num_gpu


                total_iou_diff_cates[cate] = total_voxel_iou_after_cates[cate] - total_voxel_iou_before_cates[cate]

                # summary_loss = tf.summary.scalar("total_loss", total_loss_cates[cate])
                summary_IoU = tf.summary.scalar("%s_IoU_after"%dic[cate], total_voxel_iou_after_cates[cate])
                # summary_loss_ = tf.summary.scalar("total_loss_reg", total_loss_reg_cates[cate])
                summary_IoU_before = tf.summary.scalar("%s_IoU_before"%dic[cate], total_voxel_iou_before_cates[cate])
                summary_IoU_diff = tf.summary.scalar("%s_IoU_difference"%dic[cate], total_iou_diff_cates[cate])

                summary_loss = tf.summary.scalar("%s_loss" % dic[cate], total_loss_cates[cate])
                summary_loss_ = tf.summary.scalar("%s_loss_reg" % dic[cate], total_loss_reg_cates[cate])
                summary_loss_e_0 = tf.summary.scalar("%s_loss_e_0" % dic[cate], total_loss_e_0_cates[cate])
                summary_loss_e_1 = tf.summary.scalar("%s_loss_e_1" % dic[cate], total_loss_e_1_cates[cate])
                summary_loss_e_2 = tf.summary.scalar("%s_loss_e_2" % dic[cate], total_loss_e_2_cates[cate])
                summary_err_e_0 = tf.summary.scalar("%s_err_e_0" % dic[cate], total_err_e_0_cates[cate])
                summary_err_e_1 = tf.summary.scalar("%s_err_e_1" % dic[cate], total_err_e_1_cates[cate])
                summary_loss_dist = tf.summary.scalar("%s_loss_dist" % dic[cate], total_loss_dist_cates[cate])
                summary_loss_transx = tf.summary.scalar("%s_loss_transx" % dic[cate], total_loss_transx_cates[cate])
                summary_loss_transy = tf.summary.scalar("%s_loss_transy" % dic[cate], total_loss_transy_cates[cate])

                summary_mask_iou = tf.summary.scalar("%s_mask_iou"%dic[cate], total_mask_iou_cates[cate])
                summary_loss_mask = tf.summary.scalar('%s_loss_mask'%dic[cate], total_loss_mask_cates[cate])

                summary_list_cates[cate].append(summary_IoU)
                summary_list_cates[cate].append(summary_IoU_before)
                summary_list_cates[cate].append(summary_IoU_diff)

                summary_list_cates[cate].append(summary_loss)
                summary_list_cates[cate].append(summary_loss_)
                summary_list_cates[cate].append(summary_loss_e_0)
                summary_list_cates[cate].append(summary_loss_e_1)
                summary_list_cates[cate].append(summary_loss_e_2)
                summary_list_cates[cate].append(summary_err_e_0)
                summary_list_cates[cate].append(summary_err_e_1)
                summary_list_cates[cate].append(summary_loss_dist)
                summary_list_cates[cate].append(summary_loss_transx)
                summary_list_cates[cate].append(summary_loss_transy)
                summary_list_cates[cate].append(summary_mask_iou)
                summary_list_cates[cate].append(summary_loss_mask)

                summary_merge_cates[cate] =  tf.summary.merge(summary_list_cates[cate])
                # summary_merge = tf.summary.merge(
                #     [summary_loss, summary_IoU, summary_loss_, summary_IoU_before, summary_IoU_diff])

        weight_angle = 'angle/399501.cptk'
        weight_voxel = 'voxel/100001.cptk'
        # weight_dist_trans = 'dt_trans/179001.cptk'
        weight_mask = 'mask/195501.cptk'
        weight_refine = 'refine/92001.cptk'
        path_weight = []
        path_weight.append(weight_angle)
        path_weight.append(weight_voxel)
        # path_weight.append(weight_dist_trans)
        path_weight.append(weight_mask)
        path_weight.append(weight_refine)


        saver = tf.train.Saver(max_to_keep=25)

        scalarsaver = ScalarSaver()



        angle_params = [para for para in tf.trainable_variables() if 'voxel' not in para.name and 'mask' not in para.name and 'refine' not in para.name ]
        voxel_params = [para for para in tf.trainable_variables() if 'voxel' in para.name]
        mask_params = [para for para in tf.trainable_variables() if 'mask' in para.name]
        refine_params = [para for para in tf.trainable_variables() if 'refine' in para.name]

        previous_params = []
        previous_params.append(angle_params)
        previous_params.append(voxel_params)
        previous_params.append(mask_params)
        previous_params.append(refine_params)

        num_cates = len(cates)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            sess.run(tf.global_variables_initializer())
            writer_train = tf.summary.FileWriter(os.path.join(tensorboard_path, 'train'), sess.graph)
            writer_test = tf.summary.FileWriter(os.path.join(tensorboard_path, 'test'), sess.graph)

            for i,p in enumerate(previous_params):
                loader = tf.train.Saver(var_list=p)
                loader.restore(sess,path_weight[i])
            print('restore OK')

            if not continue_train_path is None:
                saver.restore(sess, continue_train_path)

            lr = 1e-5
            for step in range(0, iterations):
                t = time.time()

                if step in lrs:
                    lr = lrs[step]

                for cate in cates:
                    x_op, y_op, m_op, e_op, st_op = worker.get_batch(cate)  # get_data()
                    sess.run(apply_gradient_op_cates[cate], feed_dict={x_vectors: x_op, y_vectors: y_op, learning_rate: lr})
                print('SHAPE IS ', x_op.shape, y_op.shape, 'optimize:', time.time() - t)

                if step % 100 == 1 or step == 0:
                    for cate in cates:
                        x_op, y_op, m_op, e_op, st_op = worker.get_batch(cate)
                        summary_train = sess.run(
                            summary_merge_cates[cate],
                            feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op,
                                       st_vectors: st_op})

                        writer_train.add_summary(summary_train, step)

                    loss_test_total = 0
                    loss_reg_test_total = 0
                    # loss_v_test_total = 0
                    loss_e_0_test_total = 0
                    loss_e_1_test_total = 0
                    loss_e_2_test_total = 0
                    err_e_0_test_total = 0
                    err_e_1_test_total = 0

                    loss_dist_test_total = 0
                    loss_transx_test_total = 0
                    loss_transy_test_total = 0

                    voxel_iou_before_test_total = 0
                    voxel_iou_after_test_total = 0
                    mask_iou_test_total = 0
                    loss_mask_test_total = 0
                    iou_diff_test_total = 0

                    num_iter = 5
                    for cate in cates:
                        loss_test = 0
                        loss_reg_test = 0

                        loss_e_0_test = 0
                        loss_e_1_test = 0
                        loss_e_2_test = 0

                        err_e_0_test = 0
                        err_e_1_test = 0

                        loss_dist_test = 0
                        loss_transx_test = 0
                        loss_transy_test = 0

                        voxel_iou_before_test = 0
                        voxel_iou_after_test = 0
                        mask_iou_test = 0
                        loss_mask_test = 0
                        iou_diff_test = 0

                        for _ in range(num_iter):
                            # print('i',_)
                            x_op, y_op, m_op, e_op, st_op = worker_uniform.get_batch(cate, False)
                            loss, loss_reg, loss_e_0, loss_e_1, loss_e_2, loss_dist, loss_transx, loss_transy, err_e_0, err_e_1, iou_before, iou_after, mask_iou, loss_mask, iou_diff = sess.run(
                                [total_loss_cates[cate], total_loss_reg_cates[cate], total_loss_e_0_cates[cate],
                                 total_loss_e_1_cates[cate], total_loss_e_2_cates[cate], total_loss_dist_cates[cate],
                                 total_loss_transx_cates[cate], total_loss_transy_cates[cate],
                                 total_err_e_0_cates[cate], total_err_e_1_cates[cate], total_voxel_iou_before_cates[cate], total_voxel_iou_after_cates[cate],
                                 total_mask_iou_cates[cate], total_loss_mask_cates[cate], total_iou_diff_cates[cate]
                                 ],
                                feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op,
                                           st_vectors: st_op})

                            loss_test += loss
                            loss_reg_test += loss_reg
                            loss_e_0_test += loss_e_0
                            loss_e_1_test += loss_e_1
                            loss_e_2_test += loss_e_2
                            err_e_0_test += err_e_0
                            err_e_1_test += err_e_1

                            loss_dist_test += loss_dist
                            loss_transx_test += loss_transx
                            loss_transy_test += loss_transy

                            voxel_iou_before_test += iou_before
                            voxel_iou_after_test += iou_after
                            mask_iou_test += mask_iou
                            loss_mask_test += loss_mask
                            iou_diff_test += iou_diff

                        loss_test /= num_iter
                        loss_reg_test /= num_iter
                        # iou_test /= num_iter
                        # loss_v_test /= num_iter

                        loss_e_0_test /= num_iter  # loss_e
                        loss_e_1_test /= num_iter  # loss_e
                        loss_e_2_test /= num_iter  # loss_e

                        err_e_0_test /= num_iter
                        err_e_1_test /= num_iter

                        loss_dist_test /= num_iter
                        loss_transx_test /= num_iter
                        loss_transy_test /= num_iter

                        voxel_iou_after_test /= num_iter
                        voxel_iou_before_test /= num_iter
                        mask_iou_test /= num_iter
                        loss_mask_test /= num_iter
                        iou_diff_test /= num_iter

                        summt = tf.Summary()
                        summt.value.add(tag='%s_loss' % dic[cate], simple_value=loss_test)
                        # summt.value.add(tag='%s_IoU' % dic[cate], simple_value=iou_test)
                        summt.value.add(tag='%s_loss_reg' % dic[cate], simple_value=loss_reg_test)
                        # summt.value.add(tag='%s_loss_v' % dic[cate], simple_value=loss_v_test)

                        summt.value.add(tag='%s_loss_e_0' % dic[cate], simple_value=loss_e_0_test)
                        summt.value.add(tag='%s_loss_e_1' % dic[cate], simple_value=loss_e_1_test)
                        summt.value.add(tag='%s_loss_e_2' % dic[cate], simple_value=loss_e_2_test)

                        summt.value.add(tag='%s_err_e_0' % dic[cate], simple_value=err_e_0_test)
                        summt.value.add(tag='%s_err_e_1' % dic[cate], simple_value=err_e_1_test)

                        summt.value.add(tag='%s_loss_dist' % dic[cate], simple_value=loss_dist_test)
                        summt.value.add(tag='%s_loss_transx' % dic[cate], simple_value=loss_transx_test)
                        summt.value.add(tag='%s_loss_transy' % dic[cate], simple_value=loss_transy_test)

                        summt.value.add(tag='%s_IoU_after' % dic[cate], simple_value=voxel_iou_after_test)
                        summt.value.add(tag='%s_IoU_before' % dic[cate], simple_value=voxel_iou_before_test)
                        summt.value.add(tag='%s_IoU_difference' % dic[cate], simple_value=iou_diff_test)
                        summt.value.add(tag='%s_loss_mask' % dic[cate], simple_value=loss_mask_test)
                        summt.value.add(tag='%s_mask_iou' % dic[cate], simple_value=mask_iou_test)


                        writer_test.add_summary(summt, step)

                        loss_test_total += loss_test
                        # iou_test_total += iou_test
                        loss_reg_test_total += loss_reg_test
                        loss_e_0_test_total += loss_e_0_test
                        loss_e_1_test_total += loss_e_1_test
                        loss_e_2_test_total += loss_e_2_test

                        err_e_0_test_total += err_e_0_test
                        err_e_1_test_total += err_e_1_test

                        loss_dist_test_total += loss_dist_test
                        loss_transx_test_total += loss_transx_test
                        loss_transy_test_total += loss_transy_test

                        voxel_iou_after_test_total += voxel_iou_after_test
                        voxel_iou_before_test_total += voxel_iou_before_test
                        mask_iou_test_total += mask_iou_test
                        loss_mask_test_total += loss_mask_test
                        iou_diff_test_total += iou_diff_test

                    loss_test_total /= num_cates
                    # iou_test_total /= num_cates
                    loss_reg_test_total /= num_cates
                    # loss_v_test_total /= num_cates
                    loss_e_0_test_total /= num_cates
                    loss_e_1_test_total /= num_cates
                    loss_e_2_test_total /= num_cates
                    err_e_0_test_total /= num_cates
                    err_e_1_test_total /= num_cates

                    loss_dist_test_total /= num_cates
                    loss_transx_test_total /= num_cates
                    loss_transy_test_total /= num_cates

                    voxel_iou_after_test_total /= num_cates
                    voxel_iou_before_test_total /= num_cates
                    mask_iou_test_total /= num_cates
                    loss_mask_test_total /= num_cates
                    iou_diff_test_total /= num_cates

                    summt = tf.Summary()
                    summt.value.add(tag='total_loss', simple_value=loss_test_total)
                    # summt.value.add(tag='total_IoU', simple_value=iou_test_total)
                    summt.value.add(tag='total_loss_reg', simple_value=loss_reg_test_total)
                    # summt.value.add(tag='total_loss_v', simple_value=loss_v_test_total)

                    summt.value.add(tag='total_loss_e_0', simple_value=loss_e_0_test_total)
                    summt.value.add(tag='total_loss_e_1', simple_value=loss_e_1_test_total)
                    summt.value.add(tag='total_loss_e_2', simple_value=loss_e_2_test_total)
                    summt.value.add(tag='total_err_e_0', simple_value=err_e_0_test_total)
                    summt.value.add(tag='total_err_e_1', simple_value=err_e_1_test_total)

                    summt.value.add(tag='total_loss_dist', simple_value=loss_dist_test_total)
                    summt.value.add(tag='total_loss_transx', simple_value=loss_transx_test_total)
                    summt.value.add(tag='total_loss_transy', simple_value=loss_transy_test_total)

                    summt.value.add(tag='total_IoU_after', simple_value=voxel_iou_after_test_total)
                    summt.value.add(tag='total_IoU_before', simple_value=voxel_iou_before_test_total)
                    summt.value.add(tag='total_IoU_difference', simple_value=iou_diff_test_total)
                    summt.value.add(tag='total_loss_mask', simple_value=loss_mask_test_total)
                    summt.value.add(tag='total_mask_iou', simple_value=mask_iou_test_total)

                    writer_test.add_summary(summt, step)

                    # print("Steps:", step, "training loss:", loss_train, "iou:", iou_train, "testing loss:",
                    #       sum_loss, "iou:", sum_iou)
                    print("Steps:", step, "testing loss:",
                          loss_test_total)

                if step % 500 == 1 and step > 500:
                    saver.save(sess, save_path=os.path.join(checkpoint_path, "%d.cptk" % step))
                    with open(os.path.join(checkpoint_path, '%d.pkl' % step), 'wb') as fp:
                        pickle.dump(scalarsaver, fp, -1)

                print('Time consume:', time.time() - t)

def train_end2end_noposebackprop(weight_path=None, continue_train_path=None, add_regulizer=False):
    tensorboard_path = 'tensorboard_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel'
    checkpoint_path = 'checkpoint_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel'
    lrs = {60000: 1e-6}

    cates = small_set

    def calc_loss(input, gt, reuse=True):
        # feature, _, __ = resnet_50(input, global_pool=True, reuse=reuse)
        feature, shortcuts = refine_encoder(input, reuse=reuse)
        voxels, voxels_softmax = refine_decoder(feature, shortcuts, reuse=reuse)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=voxels))

        return loss, voxels_softmax

    with tf.device('/cpu:0'):
        num_gpu = 1  # len(available_gpu)
        # num_gpu = 1
        # tower_grads = []
        # tower_loss = []
        # tower_iou = []
        # tower_loss_reg = []
        # tower_iou_before = []
        tower_grads = []
        tower_loss = []
        tower_loss_reg = []
        tower_loss_e_0_cates = {}
        tower_loss_e_1_cates = {}
        tower_loss_e_2_cates = {}
        tower_loss_dist_cates = {}
        tower_loss_transx_cates = {}
        tower_loss_transy_cates = {}
        tower_err_e_0_cates = {}
        tower_err_e_1_cates = {}
        tower_voxel_iou_before_cates = {}
        tower_voxel_iou_after_cates = {}
        tower_mask_iou_cates = {}
        tower_loss_mask_cates = {}


        io = dataIO(data_path, bg_path, vox_path, cates)
        worker = data_fetch_worker(io, num_gpu)
        stacker = Stacker(worker)
        stacker.run(True)
        # worker.run()
        worker_uniform = data_fetch_worker(io, num_gpu)
        stacker_uniform = Stacker(worker_uniform)
        stacker_uniform.run(False)
        # worker_uniform.run(False)

        for cate in cates:
            # tower_grads_cates[cate] = []
            # tower_loss_cates[cate] = []
            # tower_loss_reg_cates[cate] = []
            tower_loss_e_0_cates[cate] = []
            tower_loss_e_1_cates[cate] = []
            tower_loss_e_2_cates[cate] = []
            tower_loss_dist_cates[cate] = []
            tower_loss_transx_cates[cate] = []
            tower_loss_transy_cates[cate] = []
            tower_err_e_0_cates[cate] = []
            tower_err_e_1_cates[cate] = []
            tower_voxel_iou_after_cates[cate] = []
            tower_voxel_iou_before_cates[cate] = []
            tower_mask_iou_cates[cate] = []
            tower_loss_mask_cates[cate] = []



        learning_rate = tf.placeholder(dtype=tf.float32)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        num_cates = len(cates)

        with tf.device('/gpu:0'):
            x_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, voxel_size, voxel_size, voxel_size, 2],
                                       dtype=tf.float32, name='all_Voxels')
            y_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, img_w, img_h, 3], dtype=tf.float32,
                                       name='all_Images')
            m_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, img_w, img_h, 2], dtype=tf.float32, name='all_masks')

            e_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, 3], dtype=tf.float32, name='all_angles')

            st_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, 3], dtype=tf.float32, name='all_translation')
            #x_vectors_ = down_sample(x_vectors)
            x_vectors_ = x_vectors

        reuse_dict = {}
        reuse_dict['encoder'] = False
        for cate in cates:
            reuse_dict[cate] = False

        for i in range(num_gpu):
            with tf.device('/gpu:%d' % i):
                cur_x = x_vectors_[i * batchsize:(i + 1) * batchsize]
                cur_y = y_vectors[i * batchsize:(i + 1) * batchsize]
                cur_m = m_vectors[i * batchsize:(i + 1) * batchsize]
                cur_e = e_vectors[i * batchsize:(i + 1) * batchsize]
                cur_st = st_vectors[i * batchsize:(i + 1) * batchsize]

                all_loss = 0
                all_loss_reg = 0
                for j,cate in enumerate(cates):
                    reuse = reuse_dict['encoder']
                    reuse_dict['encoder'] = True

                    reuse_fc = reuse_dict[cate]
                    reuse_dict[cate] = True

                    _x = cur_x[:,j,...]
                    _y = cur_y[:,j,...]
                    _m = cur_m[:,j,...]
                    _e = cur_e[:,j,...]
                    _st = cur_st[:,j,...]

                    with tf.variable_scope("voxel"):
                        feature, feature_map, euler, st, shortcuts = encoder(_y, reuse=reuse)
                        voxels, voxels_softmax = decoder(feature, reuse=reuse)

                    with tf.variable_scope("mask"):
                        feature, feature_map, euler, st, shortcuts = encoder(_y, reuse=reuse)
                        mask, mask_softmax = generator(feature_map, shortcuts, reuse=reuse)

                    loss_voxel = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_x, logits=voxels))

                    feature, feature_map, shortcuts = encoder_angle(_y, reuse=reuse)

                    with tf.variable_scope("angles_trans", reuse=reuse_fc):
                        w_e_1 = tf.get_variable("w_euler_0_%s" % cate, shape=[1024, 512],
                                                initializer=tf.contrib.layers.xavier_initializer())
                        e_1 = lrelu(tf.matmul(feature, w_e_1))
                        w_e_2 = tf.get_variable("w_euler_1_%s" % cate, shape=[512, 3],
                                                initializer=tf.contrib.layers.xavier_initializer())
                        euler = tf.matmul(e_1, w_e_2)

                        w_st = tf.get_variable('w_ft_%s' % cate, shape=[1024, 3],
                                               initializer=tf.contrib.layers.xavier_initializer())
                        st = tf.matmul(feature, w_st)

                    loss_e_0 = tf.reduce_mean(tf.abs(euler[..., 0] - _e[..., 0]))
                    loss_e_1 = tf.reduce_mean(tf.abs(euler[..., 1] - _e[..., 1]))
                    loss_e_2 = tf.reduce_mean(tf.abs(euler[..., 2] - _e[..., 2]))

                    loss_dist = tf.reduce_mean(tf.abs(st[..., 0] - _st[..., 0]))
                    loss_transx = tf.reduce_mean(tf.abs(st[..., 1] - _st[..., 1]))
                    loss_transy = tf.reduce_mean(tf.abs(st[..., 2] - _st[..., 2]))

                    err_e_0 = tf.reduce_mean(
                        tf.minimum(tf.abs(euler[..., 0] - _e[..., 0]),
                                   tf.abs(1 - tf.abs(euler[..., 0] - _e[..., 0])))) * 360
                    err_e_1 = tf.reduce_mean(
                        tf.minimum(tf.abs(euler[..., 1] - _e[..., 1]),
                                   tf.abs(1 - tf.abs(euler[..., 1] - _e[..., 1])))) * 360

                    # feature, feature_map, euler, st, shortcuts = encoder(cur_y, reuse=reuse)
                    #
                    # mask, mask_softmax = generator(feature_map, shortcuts, reuse=reuse)
                    # voxels, voxels_softmax = decoder(feature, reuse=reuse)

                    IoU_before = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(voxels_softmax[:, :, :, :, 0] > thresh_hold,
                                               _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32),axis=[1,2,3]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(voxels_softmax[:, :, :, :, 0] > thresh_hold,
                                              _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32),axis=[1,2,3]))



                    loss_mask = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_m, logits=mask))

                    IoU_mask = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(mask_softmax[..., 0] > thresh_hold,
                                               _m[..., 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(mask_softmax[..., 0] > thresh_hold,
                                              _m[..., 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2]))

                    rotation_matrices = get_rotation_matrix(euler)
                    mask_indexs = scale_trans(st)
                    projection = cast(mask_softmax[..., 0], mask_indexs, rotation_matrices=rotation_matrices)

                    # pc = rotate_and_translate(rotation_matrices, st)
                    # dvx = DVX(projection)
                    # b_falpha = tf.reduce_sum(pc*tf.stop_gradient(dvx),axis=-1)
                    # projection = b_falpha - tf.stop_gradient(b_falpha) + tf.stop_gradient(projection)

                    c1 = voxels_softmax[..., 0]
                    c2 = projection
                    c3 = c1 - c1 * c2
                    c4 = c2 - c1 * c2

                    feedin = tf.stack([c1, c2, c3, c4], axis=4)

                    loss_, voxels_softmax_after = calc_loss(feedin, _x, reuse)
                    # loss_ += 0.1*(loss_e_0 + loss_e_1 + loss_e_2 + loss_dist + loss_transx + loss_transy + loss_mask + loss_voxel)
                    loss_ += 0.1 * (loss_mask + loss_voxel)
                    loss = loss_

                    IoU = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                               _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32),axis=[1,2,3]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                              _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32),axis=[1,2,3]))

                    params = tf.trainable_variables()
                    # params = [para for para in params if 'refine' in para.name]

                    if add_regulizer:
                        for para in params:
                            loss_ += beta * tf.reduce_mean(tf.square(para))

                    all_loss += loss
                    all_loss_reg += loss_


                    tower_voxel_iou_after_cates[cate].append(IoU)
                    tower_voxel_iou_before_cates[cate].append(IoU_before)
                    tower_loss_e_0_cates[cate].append(loss_e_0)
                    tower_loss_e_1_cates[cate].append(loss_e_1)
                    tower_loss_e_2_cates[cate].append(loss_e_2)
                    tower_loss_dist_cates[cate].append(loss_dist)
                    tower_loss_transx_cates[cate].append(loss_transx)
                    tower_loss_transy_cates[cate].append(loss_transy)
                    tower_err_e_0_cates[cate].append(err_e_0)
                    tower_err_e_1_cates[cate].append(err_e_1)
                    tower_mask_iou_cates[cate].append(IoU_mask)
                    tower_loss_mask_cates[cate].append(loss_mask)

                    print(cate, 'build over')

                params = tf.trainable_variables()
                grads = optimizer.compute_gradients(all_loss_reg, params)
                tower_grads.append(grads)
                tower_loss.append(all_loss)
                tower_loss_reg.append(all_loss_reg)


        with tf.device('/cpu:0'):

            total_loss_reg = 0
            total_loss = 0

            total_loss_e_0_cates = {}
            total_loss_e_1_cates = {}
            total_loss_e_2_cates = {}

            total_err_e_0_cates = {}
            total_err_e_1_cates = {}

            total_loss_dist_cates = {}
            total_loss_transx_cates = {}
            total_loss_transy_cates = {}

            total_voxel_iou_before_cates = {}
            total_voxel_iou_after_cates = {}
            total_mask_iou_cates = {}
            total_loss_mask_cates = {}

            total_iou_diff_cates = {}

            summary_list = []

            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads)
            for _ in tower_loss:
                total_loss += _
            for _ in tower_loss_reg:
                total_loss_reg += _

            total_loss /= num_gpu
            total_loss_reg /= num_gpu
            summary_loss = tf.summary.scalar("total_loss", total_loss)
            summary_loss_reg = tf.summary.scalar("total_loss_reg", total_loss_reg)

            summary_list.append(summary_loss)
            summary_list.append(summary_loss_reg)

            for cate in cates:


                # total_loss_reg_cates[cate] = 0
                # total_loss_cates[cate] = 0

                total_loss_e_0_cates[cate] = 0
                total_loss_e_1_cates[cate] = 0
                total_loss_e_2_cates[cate] = 0

                total_err_e_0_cates[cate] = 0
                total_err_e_1_cates[cate] = 0

                total_loss_dist_cates[cate] = 0
                total_loss_transx_cates[cate] = 0
                total_loss_transy_cates[cate] = 0

                total_voxel_iou_before_cates[cate] = 0
                total_voxel_iou_after_cates[cate] = 0
                total_mask_iou_cates[cate] = 0
                total_loss_mask_cates[cate] = 0

                # summary_list_cates[cate] = []

                for _ in tower_voxel_iou_before_cates[cate]:
                    total_voxel_iou_before_cates[cate] += _
                for _ in tower_voxel_iou_after_cates[cate]:
                    total_voxel_iou_after_cates[cate] += _

                for _ in tower_loss_e_0_cates[cate]:
                    total_loss_e_0_cates[cate] += _
                for _ in tower_loss_e_1_cates[cate]:
                    total_loss_e_1_cates[cate] += _
                for _ in tower_loss_e_2_cates[cate]:
                    total_loss_e_2_cates[cate] += _

                for _ in tower_err_e_0_cates[cate]:
                    total_err_e_0_cates[cate] += _
                for _ in tower_err_e_1_cates[cate]:
                    total_err_e_1_cates[cate] += _
                for _ in tower_loss_dist_cates[cate]:
                    total_loss_dist_cates[cate] += _
                for _ in tower_loss_transx_cates[cate]:
                    total_loss_transx_cates[cate] += _
                for _ in tower_loss_transy_cates[cate]:
                    total_loss_transy_cates[cate] += _
                for _ in tower_mask_iou_cates[cate]:
                    total_mask_iou_cates[cate] += _
                for _ in tower_loss_mask_cates[cate]:
                    total_loss_mask_cates[cate] += _

                total_voxel_iou_before_cates[cate] /= num_gpu
                total_voxel_iou_after_cates[cate] /= num_gpu
                # total_loss_cates[cate] /= num_gpu
                # total_loss_reg_cates[cate] /= num_gpu
                total_loss_e_0_cates[cate] /= num_gpu
                total_loss_e_1_cates[cate] /= num_gpu
                total_loss_e_2_cates[cate] /= num_gpu
                total_err_e_0_cates[cate] /= num_gpu
                total_err_e_1_cates[cate] /= num_gpu
                total_loss_dist_cates[cate] /= num_gpu
                total_loss_transx_cates[cate] /= num_gpu
                total_loss_transy_cates[cate] /= num_gpu
                total_mask_iou_cates[cate] /= num_gpu
                total_loss_mask_cates[cate] /= num_gpu


                total_iou_diff_cates[cate] = total_voxel_iou_after_cates[cate] - total_voxel_iou_before_cates[cate]

                # summary_loss = tf.summary.scalar("total_loss", total_loss_cates[cate])
                summary_IoU = tf.summary.scalar("%s_IoU_after"%dic[cate], total_voxel_iou_after_cates[cate])
                # summary_loss_ = tf.summary.scalar("total_loss_reg", total_loss_reg_cates[cate])
                summary_IoU_before = tf.summary.scalar("%s_IoU_before"%dic[cate], total_voxel_iou_before_cates[cate])
                summary_IoU_diff = tf.summary.scalar("%s_IoU_difference"%dic[cate], total_iou_diff_cates[cate])

                # summary_loss = tf.summary.scalar("%s_loss" % dic[cate], total_loss_cates[cate])
                # summary_loss_ = tf.summary.scalar("%s_loss_reg" % dic[cate], total_loss_reg_cates[cate])
                summary_loss_e_0 = tf.summary.scalar("%s_loss_e_0" % dic[cate], total_loss_e_0_cates[cate])
                summary_loss_e_1 = tf.summary.scalar("%s_loss_e_1" % dic[cate], total_loss_e_1_cates[cate])
                summary_loss_e_2 = tf.summary.scalar("%s_loss_e_2" % dic[cate], total_loss_e_2_cates[cate])
                summary_err_e_0 = tf.summary.scalar("%s_err_e_0" % dic[cate], total_err_e_0_cates[cate])
                summary_err_e_1 = tf.summary.scalar("%s_err_e_1" % dic[cate], total_err_e_1_cates[cate])
                summary_loss_dist = tf.summary.scalar("%s_loss_dist" % dic[cate], total_loss_dist_cates[cate])
                summary_loss_transx = tf.summary.scalar("%s_loss_transx" % dic[cate], total_loss_transx_cates[cate])
                summary_loss_transy = tf.summary.scalar("%s_loss_transy" % dic[cate], total_loss_transy_cates[cate])

                summary_mask_iou = tf.summary.scalar("%s_mask_iou"%dic[cate], total_mask_iou_cates[cate])
                summary_loss_mask = tf.summary.scalar('%s_loss_mask'%dic[cate], total_loss_mask_cates[cate])

                summary_list.append(summary_IoU)
                summary_list.append(summary_IoU_before)
                summary_list.append(summary_IoU_diff)

                # summary_list.append(summary_loss)
                # summary_list.append(summary_loss_)
                summary_list.append(summary_loss_e_0)
                summary_list.append(summary_loss_e_1)
                summary_list.append(summary_loss_e_2)
                summary_list.append(summary_err_e_0)
                summary_list.append(summary_err_e_1)
                summary_list.append(summary_loss_dist)
                summary_list.append(summary_loss_transx)
                summary_list.append(summary_loss_transy)
                summary_list.append(summary_mask_iou)
                summary_list.append(summary_loss_mask)

            summary_merge = tf.summary.merge(summary_list)
            print('gradient average over')


        weight_angle = 'angle/745501.cptk'
        weight_voxel = 'voxel/100001.cptk'
        # weight_dist_trans = 'dt_trans/179001.cptk'
        weight_mask = 'mask/195501.cptk'
        weight_refine = 'refine/92001.cptk'
        path_weight = []
        path_weight.append(weight_angle)
        path_weight.append(weight_voxel)
        # path_weight.append(weight_dist_trans)
        path_weight.append(weight_mask)
        path_weight.append(weight_refine)


        saver = tf.train.Saver(max_to_keep=25,var_list=tf.trainable_variables())

        scalarsaver = ScalarSaver()

        angle_params = [para for para in tf.trainable_variables() if 'voxel' not in para.name and 'mask' not in para.name and 'refine' not in para.name ]
        voxel_params = [para for para in tf.trainable_variables() if 'voxel' in para.name]
        mask_params = [para for para in tf.trainable_variables() if 'mask' in para.name]
        refine_params = [para for para in tf.trainable_variables() if 'refine' in para.name]

        previous_params = []
        previous_params.append(angle_params)
        previous_params.append(voxel_params)
        previous_params.append(mask_params)
        previous_params.append(refine_params)

        num_cates = len(cates)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            sess.run(tf.global_variables_initializer())
            writer_train = tf.summary.FileWriter(os.path.join(tensorboard_path, 'train'), sess.graph)
            writer_test = tf.summary.FileWriter(os.path.join(tensorboard_path, 'test'), sess.graph)

            for i,p in enumerate(previous_params):
                loader = tf.train.Saver(var_list=p)
                loader.restore(sess,path_weight[i])
            print('restore OK')

            if not continue_train_path is None:
                saver.restore(sess, continue_train_path)

            lr = 1e-5
            for step in range(0, iterations):
                t = time.time()

                if step in lrs:
                    lr = lrs[step]

                x_stack, y_stack, m_stack, e_stack, st_stack = stacker.get_batch()
                print('Data fetch',time.time() - t)
                sess.run(apply_gradient_op,feed_dict={x_vectors: x_stack, y_vectors: y_stack, m_vectors: m_stack, e_vectors: e_stack,
                                    st_vectors: st_stack, learning_rate: lr})
                # for cate in cates:
                #     x_op, y_op, m_op, e_op, st_op = worker.get_batch(cate)  # get_data()
                #     sess.run(apply_gradient_op_cates[cate], feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op,
                #                        st_vectors: st_op, learning_rate: lr})
                print('optimize:', time.time() - t)

                if step % 100 == 1 or step == 0:

                    summary_train = sess.run(
                        summary_merge,
                        feed_dict={x_vectors: x_stack, y_vectors: y_stack, m_vectors: m_stack, e_vectors: e_stack,
                                   st_vectors: st_stack})
                    # for cate in cates:
                    #     x_op, y_op, m_op, e_op, st_op = worker.get_batch(cate)
                    #     summary_train = sess.run(
                    #         summary_merge_cates[cate],
                    #         feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op,
                    #                    st_vectors: st_op})

                    writer_train.add_summary(summary_train, step)

                    # loss_test_total = 0
                    # loss_reg_test_total = 0
                    # loss_v_test_total = 0
                    loss_e_0_test_total = 0
                    loss_e_1_test_total = 0
                    loss_e_2_test_total = 0
                    err_e_0_test_total = 0
                    err_e_1_test_total = 0

                    loss_dist_test_total = 0
                    loss_transx_test_total = 0
                    loss_transy_test_total = 0

                    voxel_iou_before_test_total = 0
                    voxel_iou_after_test_total = 0
                    mask_iou_test_total = 0
                    loss_mask_test_total = 0
                    iou_diff_test_total = 0

                    num_iter = 5
                    loss_test = 0
                    loss_reg_test = 0

                    loss_e_0_test = {}
                    loss_e_1_test = {}
                    loss_e_2_test = {}

                    err_e_0_test = {}
                    err_e_1_test = {}

                    loss_dist_test = {}
                    loss_transx_test = {}
                    loss_transy_test = {}

                    voxel_iou_before_test = {}
                    voxel_iou_after_test = {}
                    mask_iou_test = {}
                    loss_mask_test = {}
                    iou_diff_test = {}
                    
                    for cate in cates:
                        loss_e_0_test[cate] = 0
                        loss_e_1_test[cate] = 0
                        loss_e_2_test[cate] = 0

                        err_e_0_test[cate] = 0
                        err_e_1_test[cate] = 0

                        loss_dist_test[cate] = 0
                        loss_transx_test[cate] = 0
                        loss_transy_test[cate] = 0

                        voxel_iou_before_test[cate] = 0
                        voxel_iou_after_test[cate] = 0
                        mask_iou_test[cate] = 0
                        loss_mask_test[cate] = 0
                        iou_diff_test[cate] = 0

                    for _ in range(num_iter):
                        # print('i',_)
                        # x_op, y_op, m_op, e_op, st_op = worker_uniform.get_batch(cate, False)
                        x_stack, y_stack, m_stack, e_stack, st_stack = stacker_uniform.get_batch(False)
                        res_list = []

                        for cate in cates:
                            res_list.extend([total_loss_e_0_cates[cate],
                             total_loss_e_1_cates[cate], total_loss_e_2_cates[cate], total_loss_dist_cates[cate],
                             total_loss_transx_cates[cate], total_loss_transy_cates[cate],
                             total_err_e_0_cates[cate], total_err_e_1_cates[cate], total_voxel_iou_before_cates[cate], total_voxel_iou_after_cates[cate],
                             total_mask_iou_cates[cate], total_loss_mask_cates[cate], total_iou_diff_cates[cate]
                             ])
                        res_list.append(total_loss)
                        res_list.append(total_loss_reg)


                        res = sess.run(res_list,
                                       feed_dict={x_vectors: x_stack, y_vectors: y_stack, m_vectors: m_stack,
                                                  e_vectors: e_stack,
                                                  st_vectors: st_stack})

                        for i,cate in enumerate(cates):
                            loss_e_0, loss_e_1, loss_e_2, loss_dist, loss_transx, loss_transy, err_e_0, err_e_1, iou_before, iou_after, mask_iou, loss_mask, iou_diff = \
                            res[i * 13:(i + 1) * 13]
                            loss_e_0_test[cate] += loss_e_0
                            loss_e_1_test[cate] += loss_e_1
                            loss_e_2_test[cate] += loss_e_2
                            err_e_0_test[cate] += err_e_0
                            err_e_1_test[cate] += err_e_1

                            loss_dist_test[cate] += loss_dist
                            loss_transx_test[cate] += loss_transx
                            loss_transy_test[cate] += loss_transy

                            voxel_iou_before_test[cate] += iou_before
                            voxel_iou_after_test[cate] += iou_after
                            mask_iou_test[cate] += mask_iou
                            loss_mask_test[cate] += loss_mask
                            iou_diff_test[cate] += iou_diff

                        loss_test += res[-2]
                        loss_reg_test += res[-1]

                    loss_test /= num_iter
                    loss_reg_test /= num_iter

                    for cate in cates:

                        # iou_test /= num_iter
                        # loss_v_test /= num_iter

                        loss_e_0_test[cate] /= num_iter  # loss_e
                        loss_e_1_test[cate] /= num_iter  # loss_e
                        loss_e_2_test[cate] /= num_iter  # loss_e

                        err_e_0_test[cate] /= num_iter
                        err_e_1_test[cate] /= num_iter

                        loss_dist_test[cate] /= num_iter
                        loss_transx_test[cate] /= num_iter
                        loss_transy_test[cate] /= num_iter

                        voxel_iou_after_test[cate] /= num_iter
                        voxel_iou_before_test[cate] /= num_iter
                        mask_iou_test[cate] /= num_iter
                        loss_mask_test[cate] /= num_iter
                        iou_diff_test[cate] /= num_iter

                        summt = tf.Summary()

                        # summt.value.add(tag='%s_loss_v' % dic[cate], simple_value=loss_v_test)

                        summt.value.add(tag='%s_loss_e_0' % dic[cate], simple_value=loss_e_0_test[cate])
                        summt.value.add(tag='%s_loss_e_1' % dic[cate], simple_value=loss_e_1_test[cate])
                        summt.value.add(tag='%s_loss_e_2' % dic[cate], simple_value=loss_e_2_test[cate])

                        summt.value.add(tag='%s_err_e_0' % dic[cate], simple_value=err_e_0_test[cate])
                        summt.value.add(tag='%s_err_e_1' % dic[cate], simple_value=err_e_1_test[cate])

                        summt.value.add(tag='%s_loss_dist' % dic[cate], simple_value=loss_dist_test[cate])
                        summt.value.add(tag='%s_loss_transx' % dic[cate], simple_value=loss_transx_test[cate])
                        summt.value.add(tag='%s_loss_transy' % dic[cate], simple_value=loss_transy_test[cate])

                        summt.value.add(tag='%s_IoU_after' % dic[cate], simple_value=voxel_iou_after_test[cate])
                        summt.value.add(tag='%s_IoU_before' % dic[cate], simple_value=voxel_iou_before_test[cate])
                        summt.value.add(tag='%s_IoU_difference' % dic[cate], simple_value=iou_diff_test[cate])
                        summt.value.add(tag='%s_loss_mask' % dic[cate], simple_value=loss_mask_test[cate])
                        summt.value.add(tag='%s_mask_iou' % dic[cate], simple_value=mask_iou_test[cate])


                        writer_test.add_summary(summt, step)

                        # loss_test_total += loss_test
                        # # iou_test_total += iou_test
                        # loss_reg_test_total += loss_reg_test
                        loss_e_0_test_total += loss_e_0_test[cate]
                        loss_e_1_test_total += loss_e_1_test[cate]
                        loss_e_2_test_total += loss_e_2_test[cate]

                        err_e_0_test_total += err_e_0_test[cate]
                        err_e_1_test_total += err_e_1_test[cate]

                        loss_dist_test_total += loss_dist_test[cate]
                        loss_transx_test_total += loss_transx_test[cate]
                        loss_transy_test_total += loss_transy_test[cate]

                        voxel_iou_after_test_total += voxel_iou_after_test[cate]
                        voxel_iou_before_test_total += voxel_iou_before_test[cate]
                        mask_iou_test_total += mask_iou_test[cate]
                        loss_mask_test_total += loss_mask_test[cate]
                        iou_diff_test_total += iou_diff_test[cate]

                    # loss_test_total /= num_cates
                    # # iou_test_total /= num_cates
                    # loss_reg_test_total /= num_cates
                    # loss_v_test_total /= num_cates
                    loss_e_0_test_total /= num_cates
                    loss_e_1_test_total /= num_cates
                    loss_e_2_test_total /= num_cates
                    err_e_0_test_total /= num_cates
                    err_e_1_test_total /= num_cates

                    loss_dist_test_total /= num_cates
                    loss_transx_test_total /= num_cates
                    loss_transy_test_total /= num_cates

                    voxel_iou_after_test_total /= num_cates
                    voxel_iou_before_test_total /= num_cates
                    mask_iou_test_total /= num_cates
                    loss_mask_test_total /= num_cates
                    iou_diff_test_total /= num_cates

                    summt = tf.Summary()
                    summt.value.add(tag='total_loss', simple_value=loss_test)
                    # summt.value.add(tag='total_IoU', simple_value=iou_test_total)
                    summt.value.add(tag='total_loss_reg', simple_value=loss_reg_test)
                    # summt.value.add(tag='total_loss_v', simple_value=loss_v_test_total)

                    summt.value.add(tag='total_loss_e_0', simple_value=loss_e_0_test_total)
                    summt.value.add(tag='total_loss_e_1', simple_value=loss_e_1_test_total)
                    summt.value.add(tag='total_loss_e_2', simple_value=loss_e_2_test_total)
                    summt.value.add(tag='total_err_e_0', simple_value=err_e_0_test_total)
                    summt.value.add(tag='total_err_e_1', simple_value=err_e_1_test_total)

                    summt.value.add(tag='total_loss_dist', simple_value=loss_dist_test_total)
                    summt.value.add(tag='total_loss_transx', simple_value=loss_transx_test_total)
                    summt.value.add(tag='total_loss_transy', simple_value=loss_transy_test_total)

                    summt.value.add(tag='total_IoU_after', simple_value=voxel_iou_after_test_total)
                    summt.value.add(tag='total_IoU_before', simple_value=voxel_iou_before_test_total)
                    summt.value.add(tag='total_IoU_difference', simple_value=iou_diff_test_total)
                    summt.value.add(tag='total_loss_mask', simple_value=loss_mask_test_total)
                    summt.value.add(tag='total_mask_iou', simple_value=mask_iou_test_total)

                    writer_test.add_summary(summt, step)

                    # print("Steps:", step, "training loss:", loss_train, "iou:", iou_train, "testing loss:",
                    #       sum_loss, "iou:", sum_iou)
                    print("Steps:", step, "testing loss:",
                          loss_test)

                if step % 500 == 1 and step > 500:
                    saver.save(sess, save_path=os.path.join(checkpoint_path, "%d.cptk" % step))
                    with open(os.path.join(checkpoint_path, '%d.pkl' % step), 'wb') as fp:
                        pickle.dump(scalarsaver, fp, -1)

                print('Time consume:', time.time() - t)


def train_end2end_noposebackprop_r2n2(weight_path=None, continue_train_path=None, add_regulizer=False):
    # tensorboard_path = 'tensorboard_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel_r2n2_fixmask'
    # checkpoint_path = 'checkpoint_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel_r2n2_fixmask'
    # tensorboard_path = 'tensorboard_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel_r2n2_fixmask_413501_voxel'
    # checkpoint_path = 'checkpoint_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel_r2n2_fixmask_413501_voxel'
    tensorboard_path = 'tensorboard_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel_r2n2_413501_voxel'
    checkpoint_path = 'checkpoint_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel_r2n2_413501_voxel'
    tensorboard_path = 'tensorboard_end2end_nopbmap_from_scratch'
    checkpoint_path = 'checkpoint_end2end_nopbmap_from_scratch'
    data_path = '../dataset/ShapeNetRendering'
    data_path = '../dataset/ShapeNet/ShapeNetRendering'
    vox_path = '../dataset/ShapeNetVox32'
    bg_path = '../dataset/bg_crop'
    lrs = {60000: 1e-6}
    #continue_train_path = './end2end/23501.cptk'
    #continue_train_path = './end2end/23501.cptk'
    # continue_train_path = os.path.join('checkpoint_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel_r2n2_fixmask','10001.cptk')
    from dataIO_r2n2 import dataIO_r2n2
    cates = small_set

    def calc_loss(input, gt, reuse=True):
        # feature, _, __ = resnet_50(input, global_pool=True, reuse=reuse)
        feature, shortcuts = refine_encoder(input, reuse=reuse)
        voxels, voxels_softmax = refine_decoder(feature, shortcuts, reuse=reuse)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=voxels))

        return loss, voxels_softmax

    with tf.device('/cpu:0'):
        num_gpu = 4  # len(available_gpu)
        # num_gpu = 1
        # tower_grads = []
        # tower_loss = []
        # tower_iou = []
        # tower_loss_reg = []
        # tower_iou_before = []
        tower_grads = []
        tower_loss = []
        tower_loss_reg = []
        tower_loss_e_0_cates = {}
        tower_loss_e_1_cates = {}
        tower_loss_e_2_cates = {}
        tower_loss_dist_cates = {}
        tower_loss_transx_cates = {}
        tower_loss_transy_cates = {}
        tower_err_e_0_cates = {}
        tower_err_e_1_cates = {}
        tower_voxel_iou_before_cates = {}
        tower_voxel_iou_after_cates = {}
        tower_mask_iou_cates = {}
        tower_loss_mask_cates = {}

        io = dataIO_r2n2(data_path, bg_path, vox_path, cates)
        worker = data_fetch_worker(io, num_gpu)
        stacker = Stacker(worker)
        stacker.run(True)
        # worker.run()
        worker_uniform = data_fetch_worker(io, num_gpu)
        stacker_uniform = Stacker(worker_uniform)
        stacker_uniform.run(False)
        # worker_uniform.run(False)

        for cate in cates:
            # tower_grads_cates[cate] = []
            # tower_loss_cates[cate] = []
            # tower_loss_reg_cates[cate] = []
            tower_loss_e_0_cates[cate] = []
            tower_loss_e_1_cates[cate] = []
            tower_loss_e_2_cates[cate] = []
            tower_loss_dist_cates[cate] = []
            tower_loss_transx_cates[cate] = []
            tower_loss_transy_cates[cate] = []
            tower_err_e_0_cates[cate] = []
            tower_err_e_1_cates[cate] = []
            tower_voxel_iou_after_cates[cate] = []
            tower_voxel_iou_before_cates[cate] = []
            tower_mask_iou_cates[cate] = []
            tower_loss_mask_cates[cate] = []

        learning_rate = tf.placeholder(dtype=tf.float32)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        num_cates = len(cates)

        with tf.device('/gpu:0'):
            x_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, voxel_size, voxel_size, voxel_size, 2],
                                       dtype=tf.float32, name='all_Voxels')
            y_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, img_w, img_h, 3], dtype=tf.float32,
                                       name='all_Images')
            m_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, img_w, img_h, 2], dtype=tf.float32,
                                       name='all_masks')

            e_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, 3], dtype=tf.float32, name='all_angles')

            st_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, 3], dtype=tf.float32,
                                        name='all_translation')
            # x_vectors_ = down_sample(x_vectors)
            x_vectors_ = x_vectors

        reuse_dict = {}
        reuse_dict['encoder'] = False
        for cate in cates:
            reuse_dict[cate] = False

        for i in range(num_gpu):
            with tf.device('/gpu:%d' % i):
                cur_x = x_vectors_[i * batchsize:(i + 1) * batchsize]
                cur_y = y_vectors[i * batchsize:(i + 1) * batchsize]
                cur_m = m_vectors[i * batchsize:(i + 1) * batchsize]
                cur_e = e_vectors[i * batchsize:(i + 1) * batchsize]
                cur_st = st_vectors[i * batchsize:(i + 1) * batchsize]

                all_loss = 0
                all_loss_reg = 0
                for j, cate in enumerate(cates):
                    reuse = reuse_dict['encoder']
                    reuse_dict['encoder'] = True

                    reuse_fc = reuse_dict[cate]
                    reuse_dict[cate] = True

                    _x = cur_x[:, j, ...]
                    _y = cur_y[:, j, ...]
                    _m = cur_m[:, j, ...]
                    _e = cur_e[:, j, ...]
                    _st = cur_st[:, j, ...]

                    with tf.variable_scope("voxel"):
                        feature, feature_map, _, _, shortcuts = encoder(_y, reuse=reuse)
                        voxels, voxels_softmax = decoder(feature, reuse=reuse)

                    with tf.variable_scope("mask"):
                        feature, feature_map, _, _, shortcuts = encoder(_y, reuse=reuse)
                        mask, mask_softmax = generator(feature_map, shortcuts, reuse=reuse)
                        # mask_softmax = tf.stop_gradient(mask_softmax)

                    loss_voxel = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_x, logits=voxels))

                    feature, feature_map, shortcuts = encoder_angle(_y, reuse=reuse)

                    with tf.variable_scope("angles_trans", reuse=reuse_fc):
                        w_e_1 = tf.get_variable("w_euler_0_%s" % cate, shape=[1024, 512],
                                                initializer=tf.contrib.layers.xavier_initializer())
                        e_1 = lrelu(tf.matmul(feature, w_e_1))
                        w_e_2 = tf.get_variable("w_euler_1_%s" % cate, shape=[512, 3],
                                                initializer=tf.contrib.layers.xavier_initializer())
                        euler = tf.matmul(e_1, w_e_2)

                        w_st = tf.get_variable('w_ft_%s' % cate, shape=[1024, 3],
                                               initializer=tf.contrib.layers.xavier_initializer())
                        st = tf.matmul(feature, w_st)

                    loss_e_0 = tf.reduce_mean(tf.abs(euler[..., 0] - _e[..., 0]))
                    loss_e_1 = tf.reduce_mean(tf.abs(euler[..., 1] - _e[..., 1]))
                    loss_e_2 = tf.reduce_mean(tf.abs(euler[..., 2] - _e[..., 2]))

                    loss_dist = tf.reduce_mean(tf.abs(st[..., 0] - _st[..., 0]))
                    loss_transx = tf.reduce_mean(tf.abs(st[..., 1] - _st[..., 1]))
                    loss_transy = tf.reduce_mean(tf.abs(st[..., 2] - _st[..., 2]))

                    err_e_0 = tf.reduce_mean(
                        tf.minimum(tf.abs(euler[..., 0] - _e[..., 0]),
                                   tf.abs(1 - tf.abs(euler[..., 0] - _e[..., 0])))) * 360
                    err_e_1 = tf.reduce_mean(
                        tf.minimum(tf.abs(euler[..., 1] - _e[..., 1]),
                                   tf.abs(1 - tf.abs(euler[..., 1] - _e[..., 1])))) * 360

                    # feature, feature_map, euler, st, shortcuts = encoder(cur_y, reuse=reuse)
                    #
                    # mask, mask_softmax = generator(feature_map, shortcuts, reuse=reuse)
                    # voxels, voxels_softmax = decoder(feature, reuse=reuse)

                    IoU_before = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(voxels_softmax[:, :, :, :, 0] > thresh_hold,
                                               _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(voxels_softmax[:, :, :, :, 0] > thresh_hold,
                                              _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2, 3]))

                    loss_mask = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_m, logits=mask))

                    IoU_mask = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(mask_softmax[..., 0] > thresh_hold,
                                               _m[..., 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(mask_softmax[..., 0] > thresh_hold,
                                              _m[..., 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2]))

                    rotation_matrices = get_rotation_matrix_r2n2(euler)
                    mask_indexs = scale_trans_r2n2(st)
                    projection = cast(mask_softmax[..., 0], mask_indexs, rotation_matrices=rotation_matrices)

                    # pc = rotate_and_translate(rotation_matrices, st)
                    # dvx = DVX(projection)
                    # b_falpha = tf.reduce_sum(pc*tf.stop_gradient(dvx),axis=-1)
                    # projection = b_falpha - tf.stop_gradient(b_falpha) + tf.stop_gradient(projection)

                    c1 = voxels_softmax[..., 0]
                    c2 = projection
                    c3 = tf.zeros_like(c1)#c1 - c1 * c2
                    c4 = tf.zeros_like(c1)#c2 - c1 * c2

                    feedin = tf.stack([c1, c2, c3, c4], axis=4)

                    loss_, voxels_softmax_after = calc_loss(feedin, _x, reuse)
                    # loss_ += 0.1*(loss_e_0 + loss_e_1 + loss_e_2 + loss_dist + loss_transx + loss_transy + loss_mask + loss_voxel)
                    # loss_ += 0.1 * (loss_mask + loss_voxel)
                    loss_ += 0.1 * (loss_voxel)
                    loss = loss_

                    IoU = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                               _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                              _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2, 3]))

                    params = tf.trainable_variables()
                    # params = [para for para in params if 'refine' in para.name]

                    if add_regulizer:
                        for para in params:
                            loss_ += beta * tf.reduce_mean(tf.square(para))

                    all_loss += loss
                    all_loss_reg += loss_

                    tower_voxel_iou_after_cates[cate].append(IoU)
                    tower_voxel_iou_before_cates[cate].append(IoU_before)
                    tower_loss_e_0_cates[cate].append(loss_e_0)
                    tower_loss_e_1_cates[cate].append(loss_e_1)
                    tower_loss_e_2_cates[cate].append(loss_e_2)
                    tower_loss_dist_cates[cate].append(loss_dist)
                    tower_loss_transx_cates[cate].append(loss_transx)
                    tower_loss_transy_cates[cate].append(loss_transy)
                    tower_err_e_0_cates[cate].append(err_e_0)
                    tower_err_e_1_cates[cate].append(err_e_1)
                    tower_mask_iou_cates[cate].append(IoU_mask)
                    tower_loss_mask_cates[cate].append(loss_mask)

                    print(cate, 'build over')

                params = tf.trainable_variables()
                grads = optimizer.compute_gradients(all_loss_reg, params)
                tower_grads.append(grads)
                tower_loss.append(all_loss)
                tower_loss_reg.append(all_loss_reg)

        with tf.device('/gpu:0'):

            total_loss_reg = 0
            total_loss = 0

            total_loss_e_0_cates = {}
            total_loss_e_1_cates = {}
            total_loss_e_2_cates = {}

            total_err_e_0_cates = {}
            total_err_e_1_cates = {}

            total_loss_dist_cates = {}
            total_loss_transx_cates = {}
            total_loss_transy_cates = {}

            total_voxel_iou_before_cates = {}
            total_voxel_iou_after_cates = {}
            total_mask_iou_cates = {}
            total_loss_mask_cates = {}

            total_iou_diff_cates = {}

            summary_list = []

            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads)
            for _ in tower_loss:
                total_loss += _
            for _ in tower_loss_reg:
                total_loss_reg += _

            total_loss /= num_gpu
            total_loss_reg /= num_gpu
            summary_loss = tf.summary.scalar("total_loss", total_loss)
            summary_loss_reg = tf.summary.scalar("total_loss_reg", total_loss_reg)

            summary_list.append(summary_loss)
            summary_list.append(summary_loss_reg)

            for cate in cates:

                # total_loss_reg_cates[cate] = 0
                # total_loss_cates[cate] = 0

                total_loss_e_0_cates[cate] = 0
                total_loss_e_1_cates[cate] = 0
                total_loss_e_2_cates[cate] = 0

                total_err_e_0_cates[cate] = 0
                total_err_e_1_cates[cate] = 0

                total_loss_dist_cates[cate] = 0
                total_loss_transx_cates[cate] = 0
                total_loss_transy_cates[cate] = 0

                total_voxel_iou_before_cates[cate] = 0
                total_voxel_iou_after_cates[cate] = 0
                total_mask_iou_cates[cate] = 0
                total_loss_mask_cates[cate] = 0

                # summary_list_cates[cate] = []

                for _ in tower_voxel_iou_before_cates[cate]:
                    total_voxel_iou_before_cates[cate] += _
                for _ in tower_voxel_iou_after_cates[cate]:
                    total_voxel_iou_after_cates[cate] += _

                for _ in tower_loss_e_0_cates[cate]:
                    total_loss_e_0_cates[cate] += _
                for _ in tower_loss_e_1_cates[cate]:
                    total_loss_e_1_cates[cate] += _
                for _ in tower_loss_e_2_cates[cate]:
                    total_loss_e_2_cates[cate] += _

                for _ in tower_err_e_0_cates[cate]:
                    total_err_e_0_cates[cate] += _
                for _ in tower_err_e_1_cates[cate]:
                    total_err_e_1_cates[cate] += _
                for _ in tower_loss_dist_cates[cate]:
                    total_loss_dist_cates[cate] += _
                for _ in tower_loss_transx_cates[cate]:
                    total_loss_transx_cates[cate] += _
                for _ in tower_loss_transy_cates[cate]:
                    total_loss_transy_cates[cate] += _
                for _ in tower_mask_iou_cates[cate]:
                    total_mask_iou_cates[cate] += _
                for _ in tower_loss_mask_cates[cate]:
                    total_loss_mask_cates[cate] += _

                total_voxel_iou_before_cates[cate] /= num_gpu
                total_voxel_iou_after_cates[cate] /= num_gpu
                # total_loss_cates[cate] /= num_gpu
                # total_loss_reg_cates[cate] /= num_gpu
                total_loss_e_0_cates[cate] /= num_gpu
                total_loss_e_1_cates[cate] /= num_gpu
                total_loss_e_2_cates[cate] /= num_gpu
                total_err_e_0_cates[cate] /= num_gpu
                total_err_e_1_cates[cate] /= num_gpu
                total_loss_dist_cates[cate] /= num_gpu
                total_loss_transx_cates[cate] /= num_gpu
                total_loss_transy_cates[cate] /= num_gpu
                total_mask_iou_cates[cate] /= num_gpu
                total_loss_mask_cates[cate] /= num_gpu

                total_iou_diff_cates[cate] = total_voxel_iou_after_cates[cate] - total_voxel_iou_before_cates[cate]

                # summary_loss = tf.summary.scalar("total_loss", total_loss_cates[cate])
                summary_IoU = tf.summary.scalar("%s_IoU_after" % dic[cate], total_voxel_iou_after_cates[cate])
                # summary_loss_ = tf.summary.scalar("total_loss_reg", total_loss_reg_cates[cate])
                summary_IoU_before = tf.summary.scalar("%s_IoU_before" % dic[cate], total_voxel_iou_before_cates[cate])
                summary_IoU_diff = tf.summary.scalar("%s_IoU_difference" % dic[cate], total_iou_diff_cates[cate])

                # summary_loss = tf.summary.scalar("%s_loss" % dic[cate], total_loss_cates[cate])
                # summary_loss_ = tf.summary.scalar("%s_loss_reg" % dic[cate], total_loss_reg_cates[cate])
                summary_loss_e_0 = tf.summary.scalar("%s_loss_e_0" % dic[cate], total_loss_e_0_cates[cate])
                summary_loss_e_1 = tf.summary.scalar("%s_loss_e_1" % dic[cate], total_loss_e_1_cates[cate])
                summary_loss_e_2 = tf.summary.scalar("%s_loss_e_2" % dic[cate], total_loss_e_2_cates[cate])
                summary_err_e_0 = tf.summary.scalar("%s_err_e_0" % dic[cate], total_err_e_0_cates[cate])
                summary_err_e_1 = tf.summary.scalar("%s_err_e_1" % dic[cate], total_err_e_1_cates[cate])
                summary_loss_dist = tf.summary.scalar("%s_loss_dist" % dic[cate], total_loss_dist_cates[cate])
                summary_loss_transx = tf.summary.scalar("%s_loss_transx" % dic[cate], total_loss_transx_cates[cate])
                summary_loss_transy = tf.summary.scalar("%s_loss_transy" % dic[cate], total_loss_transy_cates[cate])

                summary_mask_iou = tf.summary.scalar("%s_mask_iou" % dic[cate], total_mask_iou_cates[cate])
                summary_loss_mask = tf.summary.scalar('%s_loss_mask' % dic[cate], total_loss_mask_cates[cate])

                summary_list.append(summary_IoU)
                summary_list.append(summary_IoU_before)
                summary_list.append(summary_IoU_diff)

                # summary_list.append(summary_loss)
                # summary_list.append(summary_loss_)
                summary_list.append(summary_loss_e_0)
                summary_list.append(summary_loss_e_1)
                summary_list.append(summary_loss_e_2)
                summary_list.append(summary_err_e_0)
                summary_list.append(summary_err_e_1)
                summary_list.append(summary_loss_dist)
                summary_list.append(summary_loss_transx)
                summary_list.append(summary_loss_transy)
                summary_list.append(summary_mask_iou)
                summary_list.append(summary_loss_mask)

            summary_merge = tf.summary.merge(summary_list)
            print('gradient average over')

        weight_angle = 'angle_r2n2/131501.cptk'
        # weight_voxel = 'voxel_r2n2/87501.cptk'
        weight_voxel = 'checkpoint_cross_categories_4_Adam_voxel_r2n2finetune/51501.cptk'
        # weight_voxel = 'checkpoint_cross_categories_4_Adam_voxel_r2n2finetune/413501.cptk'
        weight_voxel = 'voxel_r2n2/413501.cptk'
        # weight_dist_trans = 'dt_trans/179001.cptk'
        weight_mask = 'mask_r2n2/180501.cptk'
        weight_refine = 'refine_r2n2/8001.cptk'
        path_weight = []
        path_weight.append(weight_angle)
        path_weight.append(weight_voxel)
        # path_weight.append(weight_dist_trans)
        path_weight.append(weight_mask)
        path_weight.append(weight_refine)

        saver = tf.train.Saver(max_to_keep=25, var_list=tf.trainable_variables())

        scalarsaver = ScalarSaver()

        angle_params = [para for para in tf.trainable_variables() if
                        'voxel' not in para.name and 'mask' not in para.name and 'refine' not in para.name]
        voxel_params = [para for para in tf.trainable_variables() if 'voxel' in para.name]
        mask_params = [para for para in tf.trainable_variables() if 'mask' in para.name]
        refine_params = [para for para in tf.trainable_variables() if 'refine' in para.name]

        previous_params = []
        previous_params.append(angle_params)
        previous_params.append(voxel_params)
        previous_params.append(mask_params)
        previous_params.append(refine_params)

        num_cates = len(cates)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            sess.run(tf.global_variables_initializer())
            writer_train = tf.summary.FileWriter(os.path.join(tensorboard_path, 'train'), sess.graph)
            writer_test = tf.summary.FileWriter(os.path.join(tensorboard_path, 'test'), sess.graph)

            for i, p in enumerate(previous_params):
                print('load from', path_weight[i])
                loader = tf.train.Saver(var_list=p)
                loader.restore(sess, path_weight[i])
            print('restore OK')

            if not continue_train_path is None:
                saver.restore(sess, continue_train_path)
                loader = tf.train.Saver(var_list=previous_params[1])
                loader.restore(sess, path_weight[1])
                loader = tf.train.Saver(var_list=previous_params[2])
                loader.restore(sess, path_weight[2])

            lr = 1e-6
            for step in range(10002, iterations):
                t = time.time()

                if step in lrs:
                    lr = lrs[step]

                x_stack, y_stack, m_stack, e_stack, st_stack = stacker.get_batch()
                print('Data fetch', time.time() - t)
                sess.run(apply_gradient_op,
                         feed_dict={x_vectors: x_stack, y_vectors: y_stack, m_vectors: m_stack, e_vectors: e_stack,
                                    st_vectors: st_stack, learning_rate: lr})
                # for cate in cates:
                #     x_op, y_op, m_op, e_op, st_op = worker.get_batch(cate)  # get_data()
                #     sess.run(apply_gradient_op_cates[cate], feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op,
                #                        st_vectors: st_op, learning_rate: lr})
                print('optimize:', time.time() - t)

                if step % 100 == 1 or step == 0:

                    summary_train = sess.run(
                        summary_merge,
                        feed_dict={x_vectors: x_stack, y_vectors: y_stack, m_vectors: m_stack, e_vectors: e_stack,
                                   st_vectors: st_stack})
                    # for cate in cates:
                    #     x_op, y_op, m_op, e_op, st_op = worker.get_batch(cate)
                    #     summary_train = sess.run(
                    #         summary_merge_cates[cate],
                    #         feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op,
                    #                    st_vectors: st_op})

                    writer_train.add_summary(summary_train, step)

                    # loss_test_total = 0
                    # loss_reg_test_total = 0
                    # loss_v_test_total = 0
                    loss_e_0_test_total = 0
                    loss_e_1_test_total = 0
                    loss_e_2_test_total = 0
                    err_e_0_test_total = 0
                    err_e_1_test_total = 0

                    loss_dist_test_total = 0
                    loss_transx_test_total = 0
                    loss_transy_test_total = 0

                    voxel_iou_before_test_total = 0
                    voxel_iou_after_test_total = 0
                    mask_iou_test_total = 0
                    loss_mask_test_total = 0
                    iou_diff_test_total = 0

                    num_iter = 5
                    loss_test = 0
                    loss_reg_test = 0

                    loss_e_0_test = {}
                    loss_e_1_test = {}
                    loss_e_2_test = {}

                    err_e_0_test = {}
                    err_e_1_test = {}

                    loss_dist_test = {}
                    loss_transx_test = {}
                    loss_transy_test = {}

                    voxel_iou_before_test = {}
                    voxel_iou_after_test = {}
                    mask_iou_test = {}
                    loss_mask_test = {}
                    iou_diff_test = {}

                    for cate in cates:
                        loss_e_0_test[cate] = 0
                        loss_e_1_test[cate] = 0
                        loss_e_2_test[cate] = 0

                        err_e_0_test[cate] = 0
                        err_e_1_test[cate] = 0

                        loss_dist_test[cate] = 0
                        loss_transx_test[cate] = 0
                        loss_transy_test[cate] = 0

                        voxel_iou_before_test[cate] = 0
                        voxel_iou_after_test[cate] = 0
                        mask_iou_test[cate] = 0
                        loss_mask_test[cate] = 0
                        iou_diff_test[cate] = 0

                    for _ in range(num_iter):
                        # print('i',_)
                        # x_op, y_op, m_op, e_op, st_op = worker_uniform.get_batch(cate, False)
                        x_stack, y_stack, m_stack, e_stack, st_stack = stacker_uniform.get_batch(False)
                        res_list = []

                        for cate in cates:
                            res_list.extend([total_loss_e_0_cates[cate],
                                             total_loss_e_1_cates[cate], total_loss_e_2_cates[cate],
                                             total_loss_dist_cates[cate],
                                             total_loss_transx_cates[cate], total_loss_transy_cates[cate],
                                             total_err_e_0_cates[cate], total_err_e_1_cates[cate],
                                             total_voxel_iou_before_cates[cate], total_voxel_iou_after_cates[cate],
                                             total_mask_iou_cates[cate], total_loss_mask_cates[cate],
                                             total_iou_diff_cates[cate]
                                             ])
                        res_list.append(total_loss)
                        res_list.append(total_loss_reg)

                        res = sess.run(res_list,
                                       feed_dict={x_vectors: x_stack, y_vectors: y_stack, m_vectors: m_stack,
                                                  e_vectors: e_stack,
                                                  st_vectors: st_stack})

                        for i, cate in enumerate(cates):
                            loss_e_0, loss_e_1, loss_e_2, loss_dist, loss_transx, loss_transy, err_e_0, err_e_1, iou_before, iou_after, mask_iou, loss_mask, iou_diff = \
                                res[i * 13:(i + 1) * 13]
                            loss_e_0_test[cate] += loss_e_0
                            loss_e_1_test[cate] += loss_e_1
                            loss_e_2_test[cate] += loss_e_2
                            err_e_0_test[cate] += err_e_0
                            err_e_1_test[cate] += err_e_1

                            loss_dist_test[cate] += loss_dist
                            loss_transx_test[cate] += loss_transx
                            loss_transy_test[cate] += loss_transy

                            voxel_iou_before_test[cate] += iou_before
                            voxel_iou_after_test[cate] += iou_after
                            mask_iou_test[cate] += mask_iou
                            loss_mask_test[cate] += loss_mask
                            iou_diff_test[cate] += iou_diff

                        loss_test += res[-2]
                        loss_reg_test += res[-1]

                    loss_test /= num_iter
                    loss_reg_test /= num_iter

                    for cate in cates:
                        # iou_test /= num_iter
                        # loss_v_test /= num_iter

                        loss_e_0_test[cate] /= num_iter  # loss_e
                        loss_e_1_test[cate] /= num_iter  # loss_e
                        loss_e_2_test[cate] /= num_iter  # loss_e

                        err_e_0_test[cate] /= num_iter
                        err_e_1_test[cate] /= num_iter

                        loss_dist_test[cate] /= num_iter
                        loss_transx_test[cate] /= num_iter
                        loss_transy_test[cate] /= num_iter

                        voxel_iou_after_test[cate] /= num_iter
                        voxel_iou_before_test[cate] /= num_iter
                        mask_iou_test[cate] /= num_iter
                        loss_mask_test[cate] /= num_iter
                        iou_diff_test[cate] /= num_iter

                        summt = tf.Summary()

                        # summt.value.add(tag='%s_loss_v' % dic[cate], simple_value=loss_v_test)

                        summt.value.add(tag='%s_loss_e_0' % dic[cate], simple_value=loss_e_0_test[cate])
                        summt.value.add(tag='%s_loss_e_1' % dic[cate], simple_value=loss_e_1_test[cate])
                        summt.value.add(tag='%s_loss_e_2' % dic[cate], simple_value=loss_e_2_test[cate])

                        summt.value.add(tag='%s_err_e_0' % dic[cate], simple_value=err_e_0_test[cate])
                        summt.value.add(tag='%s_err_e_1' % dic[cate], simple_value=err_e_1_test[cate])

                        summt.value.add(tag='%s_loss_dist' % dic[cate], simple_value=loss_dist_test[cate])
                        summt.value.add(tag='%s_loss_transx' % dic[cate], simple_value=loss_transx_test[cate])
                        summt.value.add(tag='%s_loss_transy' % dic[cate], simple_value=loss_transy_test[cate])

                        summt.value.add(tag='%s_IoU_after' % dic[cate], simple_value=voxel_iou_after_test[cate])
                        summt.value.add(tag='%s_IoU_before' % dic[cate], simple_value=voxel_iou_before_test[cate])
                        summt.value.add(tag='%s_IoU_difference' % dic[cate], simple_value=iou_diff_test[cate])
                        summt.value.add(tag='%s_loss_mask' % dic[cate], simple_value=loss_mask_test[cate])
                        summt.value.add(tag='%s_mask_iou' % dic[cate], simple_value=mask_iou_test[cate])

                        writer_test.add_summary(summt, step)

                        # loss_test_total += loss_test
                        # # iou_test_total += iou_test
                        # loss_reg_test_total += loss_reg_test
                        loss_e_0_test_total += loss_e_0_test[cate]
                        loss_e_1_test_total += loss_e_1_test[cate]
                        loss_e_2_test_total += loss_e_2_test[cate]

                        err_e_0_test_total += err_e_0_test[cate]
                        err_e_1_test_total += err_e_1_test[cate]

                        loss_dist_test_total += loss_dist_test[cate]
                        loss_transx_test_total += loss_transx_test[cate]
                        loss_transy_test_total += loss_transy_test[cate]

                        voxel_iou_after_test_total += voxel_iou_after_test[cate]
                        voxel_iou_before_test_total += voxel_iou_before_test[cate]
                        mask_iou_test_total += mask_iou_test[cate]
                        loss_mask_test_total += loss_mask_test[cate]
                        iou_diff_test_total += iou_diff_test[cate]

                    # loss_test_total /= num_cates
                    # # iou_test_total /= num_cates
                    # loss_reg_test_total /= num_cates
                    # loss_v_test_total /= num_cates
                    loss_e_0_test_total /= num_cates
                    loss_e_1_test_total /= num_cates
                    loss_e_2_test_total /= num_cates
                    err_e_0_test_total /= num_cates
                    err_e_1_test_total /= num_cates

                    loss_dist_test_total /= num_cates
                    loss_transx_test_total /= num_cates
                    loss_transy_test_total /= num_cates

                    voxel_iou_after_test_total /= num_cates
                    voxel_iou_before_test_total /= num_cates
                    mask_iou_test_total /= num_cates
                    loss_mask_test_total /= num_cates
                    iou_diff_test_total /= num_cates

                    summt = tf.Summary()
                    summt.value.add(tag='total_loss', simple_value=loss_test)
                    # summt.value.add(tag='total_IoU', simple_value=iou_test_total)
                    summt.value.add(tag='total_loss_reg', simple_value=loss_reg_test)
                    # summt.value.add(tag='total_loss_v', simple_value=loss_v_test_total)

                    summt.value.add(tag='total_loss_e_0', simple_value=loss_e_0_test_total)
                    summt.value.add(tag='total_loss_e_1', simple_value=loss_e_1_test_total)
                    summt.value.add(tag='total_loss_e_2', simple_value=loss_e_2_test_total)
                    summt.value.add(tag='total_err_e_0', simple_value=err_e_0_test_total)
                    summt.value.add(tag='total_err_e_1', simple_value=err_e_1_test_total)

                    summt.value.add(tag='total_loss_dist', simple_value=loss_dist_test_total)
                    summt.value.add(tag='total_loss_transx', simple_value=loss_transx_test_total)
                    summt.value.add(tag='total_loss_transy', simple_value=loss_transy_test_total)

                    summt.value.add(tag='total_IoU_after', simple_value=voxel_iou_after_test_total)
                    summt.value.add(tag='total_IoU_before', simple_value=voxel_iou_before_test_total)
                    summt.value.add(tag='total_IoU_difference', simple_value=iou_diff_test_total)
                    summt.value.add(tag='total_loss_mask', simple_value=loss_mask_test_total)
                    summt.value.add(tag='total_mask_iou', simple_value=mask_iou_test_total)

                    writer_test.add_summary(summt, step)

                    # print("Steps:", step, "training loss:", loss_train, "iou:", iou_train, "testing loss:",
                    #       sum_loss, "iou:", sum_iou)
                    print("Steps:", step, "testing loss:",
                          loss_test)

                if step % 500 == 1 and step > 500:
                    saver.save(sess, save_path=os.path.join(checkpoint_path, "%d.cptk" % step))
                    with open(os.path.join(checkpoint_path, '%d.pkl' % step), 'wb') as fp:
                        pickle.dump(scalarsaver, fp, -1)

                print('Time consume:', time.time() - t)

def train_end2end_noposebackprop_voc(weight_path=None, continue_train_path=None, add_regulizer=False):
    tensorboard_path = 'tensorboard_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel_voc'
    checkpoint_path = 'checkpoint_cross_categories_4_Adam_end2end_nopose_fixedpose_parallel_voc'
    data_path = '../dataset/voc3d/imgs'
    vox_path = '../dataset/voc3d/Vox'
    bg_path = '../dataset/bg_crop'
    lrs = {60000: 1e-6}
    from dataIO_voc import dataIO_voc

    # cates = ['aeroplane', 'car', 'chair', 'sofa']
    cates = ['sofa', 'aeroplane', 'chair', 'car']
    cates_name = small_set
    def calc_loss(input, gt, reuse=True):
        # feature, _, __ = resnet_50(input, global_pool=True, reuse=reuse)
        feature, shortcuts = refine_encoder(input, reuse=reuse)
        voxels, voxels_softmax = refine_decoder(feature, shortcuts, reuse=reuse)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=voxels))

        return loss, voxels_softmax

    with tf.device('/cpu:0'):
        num_gpu = 4  # len(available_gpu)
        # num_gpu = 1
        # tower_grads = []
        # tower_loss = []
        # tower_iou = []
        # tower_loss_reg = []
        # tower_iou_before = []
        tower_grads = []
        tower_loss = []
        tower_loss_reg = []
        tower_loss_e_0_cates = {}
        tower_loss_e_1_cates = {}
        tower_loss_e_2_cates = {}
        tower_loss_dist_cates = {}
        tower_loss_transx_cates = {}
        tower_loss_transy_cates = {}
        tower_err_e_0_cates = {}
        tower_err_e_1_cates = {}
        tower_err_e_2_cates = {}
        tower_voxel_iou_before_cates = {}
        tower_voxel_iou_after_cates = {}
        tower_mask_iou_cates = {}
        tower_loss_mask_cates = {}

        io = dataIO_voc(data_path, bg_path, vox_path, cates)
        worker = data_fetch_worker(io, num_gpu)
        stacker = Stacker(worker)
        stacker.run(True)
        # worker.run()
        worker_uniform = data_fetch_worker(io, num_gpu)
        stacker_uniform = Stacker(worker_uniform)
        stacker_uniform.run(False)
        # worker_uniform.run(False)

        for cate in cates:
            # tower_grads_cates[cate] = []
            # tower_loss_cates[cate] = []
            # tower_loss_reg_cates[cate] = []
            tower_loss_e_0_cates[cate] = []
            tower_loss_e_1_cates[cate] = []
            tower_loss_e_2_cates[cate] = []
            tower_loss_dist_cates[cate] = []
            tower_loss_transx_cates[cate] = []
            tower_loss_transy_cates[cate] = []
            tower_err_e_0_cates[cate] = []
            tower_err_e_1_cates[cate] = []
            tower_err_e_2_cates[cate] = []
            tower_voxel_iou_after_cates[cate] = []
            tower_voxel_iou_before_cates[cate] = []
            tower_mask_iou_cates[cate] = []
            tower_loss_mask_cates[cate] = []

        learning_rate = tf.placeholder(dtype=tf.float32)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        num_cates = len(cates)

        with tf.device('/gpu:0'):
            x_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, voxel_size, voxel_size, voxel_size, 2],
                                       dtype=tf.float32, name='all_Voxels')
            y_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, img_w, img_h, 3], dtype=tf.float32,
                                       name='all_Images')
            m_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, img_w, img_h, 2], dtype=tf.float32,
                                       name='all_masks')

            e_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, 3], dtype=tf.float32, name='all_angles')

            st_vectors = tf.placeholder(shape=[batchsize * num_gpu, num_cates, 3], dtype=tf.float32,
                                        name='all_translation')
            # x_vectors_ = down_sample(x_vectors)
            x_vectors_ = x_vectors

        reuse_dict = {}
        reuse_dict['encoder'] = False
        for cate in cates:
            reuse_dict[cate] = False

        for i in range(num_gpu):
            with tf.device('/gpu:%d' % i):
                cur_x = x_vectors_[i * batchsize:(i + 1) * batchsize]
                cur_y = y_vectors[i * batchsize:(i + 1) * batchsize]
                cur_m = m_vectors[i * batchsize:(i + 1) * batchsize]
                cur_e = e_vectors[i * batchsize:(i + 1) * batchsize]
                cur_st = st_vectors[i * batchsize:(i + 1) * batchsize]

                all_loss = 0
                all_loss_reg = 0
                for j, cate in enumerate(cates):
                    reuse = reuse_dict['encoder']
                    reuse_dict['encoder'] = True

                    reuse_fc = reuse_dict[cate]
                    reuse_dict[cate] = True

                    _x = cur_x[:, j, ...]
                    _y = cur_y[:, j, ...]
                    _m = cur_m[:, j, ...]
                    _e = cur_e[:, j, ...]
                    _st = cur_st[:, j, ...]

                    with tf.variable_scope("voxel"):
                        feature, feature_map, _, _, shortcuts = encoder(_y, reuse=reuse)
                        voxels, voxels_softmax = decoder(feature, reuse=reuse)

                    with tf.variable_scope("mask"):
                        feature, feature_map, _, _, shortcuts = encoder(_y, reuse=reuse)
                        mask, mask_softmax = generator(feature_map, shortcuts, reuse=reuse)

                    loss_voxel = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_x, logits=voxels))

                    feature, feature_map, shortcuts = encoder_angle(_y, reuse=reuse)

                    with tf.variable_scope("angles_trans", reuse=reuse_fc):
                        w_e_1 = tf.get_variable("w_euler_0_%s" % cates_name[j], shape=[1024, 512],
                                                initializer=tf.contrib.layers.xavier_initializer())
                        e_1 = lrelu(tf.matmul(feature, w_e_1))
                        w_e_2 = tf.get_variable("w_euler_1_%s" % cates_name[j], shape=[512, 3],
                                                initializer=tf.contrib.layers.xavier_initializer())
                        euler = tf.matmul(e_1, w_e_2)

                        w_st = tf.get_variable('w_ft_%s' % cates_name[j], shape=[1024, 3],
                                               initializer=tf.contrib.layers.xavier_initializer())
                        st = tf.matmul(feature, w_st)
                        st = tf.stack([st[..., 0]*10,st[...,1],st[...,2]],axis=-1)

                    loss_e_0 = tf.reduce_mean(tf.abs(euler[..., 0] - _e[..., 0]))
                    loss_e_1 = tf.reduce_mean(tf.abs(euler[..., 1] - _e[..., 1]))
                    loss_e_2 = tf.reduce_mean(tf.abs(euler[..., 2] - _e[..., 2]))

                    loss_dist = tf.reduce_mean(tf.abs(st[..., 0] - _st[..., 0]))
                    loss_transx = tf.reduce_mean(tf.abs(st[..., 1] - _st[..., 1]))
                    loss_transy = tf.reduce_mean(tf.abs(st[..., 2] - _st[..., 2]))

                    err_e_0 = tf.reduce_mean(
                        tf.minimum(tf.abs(euler[..., 0] - _e[..., 0]),
                                   tf.abs(1 - tf.abs(euler[..., 0] - _e[..., 0])))) * 360
                    err_e_1 = tf.reduce_mean(
                        tf.minimum(tf.abs(euler[..., 1] - _e[..., 1]),
                                   tf.abs(1 - tf.abs(euler[..., 1] - _e[..., 1])))) * 360

                    err_e_2 = tf.reduce_mean(
                        tf.minimum(tf.abs(euler[..., 2] - _e[..., 2]),
                                   tf.abs(1 - tf.abs(euler[..., 2] - _e[..., 2])))) * 360

                    # feature, feature_map, euler, st, shortcuts = encoder(cur_y, reuse=reuse)
                    #
                    # mask, mask_softmax = generator(feature_map, shortcuts, reuse=reuse)
                    # voxels, voxels_softmax = decoder(feature, reuse=reuse)

                    IoU_before = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(voxels_softmax[:, :, :, :, 0] > thresh_hold,
                                               _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(voxels_softmax[:, :, :, :, 0] > thresh_hold,
                                              _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2, 3]))

                    loss_mask = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_m, logits=mask))

                    IoU_mask = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(mask_softmax[..., 0] > thresh_hold,
                                               _m[..., 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(mask_softmax[..., 0] > thresh_hold,
                                              _m[..., 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2]))

                    rotation_matrices = get_rotation_matrix_voc(euler)
                    mask_indexs = scale_trans_voc(st)
                    masks = rotate_mask_voc(mask_softmax[..., 0], euler)
                    projection = cast(masks, mask_indexs, rotation_matrices=rotation_matrices)

                    # pc = rotate_and_translate(rotation_matrices, st)
                    # dvx = DVX(projection)
                    # b_falpha = tf.reduce_sum(pc*tf.stop_gradient(dvx),axis=-1)
                    # projection = b_falpha - tf.stop_gradient(b_falpha) + tf.stop_gradient(projection)

                    c1 = voxels_softmax[..., 0]
                    c2 = projection
                    c3 = c1 - c1 * c2
                    c4 = c2 - c1 * c2

                    feedin = tf.stack([c1, c2, c3, c4], axis=4)

                    loss_, voxels_softmax_after = calc_loss(feedin, _x, reuse)
                    # loss_ += 0.1*(loss_e_0 + loss_e_1 + loss_e_2 + loss_dist + loss_transx + loss_transy + loss_mask + loss_voxel)
                    loss_ += 0.1 * (loss_mask + loss_voxel)
                    loss = loss_

                    IoU = tf.reduce_mean(tf.reduce_sum(
                        tf.cast(tf.logical_and(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                               _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
                        tf.cast(tf.logical_or(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                              _x[:, :, :, :, 0] > thresh_hold),
                                dtype=tf.float32), axis=[1, 2, 3]))

                    params = tf.trainable_variables()
                    # params = [para for para in params if 'refine' in para.name]

                    if add_regulizer:
                        for para in params:
                            loss_ += beta * tf.reduce_mean(tf.square(para))

                    all_loss += loss
                    all_loss_reg += loss_

                    tower_voxel_iou_after_cates[cate].append(IoU)
                    tower_voxel_iou_before_cates[cate].append(IoU_before)
                    tower_loss_e_0_cates[cate].append(loss_e_0)
                    tower_loss_e_1_cates[cate].append(loss_e_1)
                    tower_loss_e_2_cates[cate].append(loss_e_2)
                    tower_loss_dist_cates[cate].append(loss_dist)
                    tower_loss_transx_cates[cate].append(loss_transx)
                    tower_loss_transy_cates[cate].append(loss_transy)
                    tower_err_e_0_cates[cate].append(err_e_0)
                    tower_err_e_1_cates[cate].append(err_e_1)
                    tower_err_e_2_cates[cate].append(err_e_2)
                    tower_mask_iou_cates[cate].append(IoU_mask)
                    tower_loss_mask_cates[cate].append(loss_mask)

                    print(cate, 'build over')

                params = tf.trainable_variables()
                grads = optimizer.compute_gradients(all_loss_reg, params)
                tower_grads.append(grads)
                tower_loss.append(all_loss)
                tower_loss_reg.append(all_loss_reg)

        with tf.device('/gpu:0'):

            total_loss_reg = 0
            total_loss = 0

            total_loss_e_0_cates = {}
            total_loss_e_1_cates = {}
            total_loss_e_2_cates = {}

            total_err_e_0_cates = {}
            total_err_e_1_cates = {}
            total_err_e_2_cates = {}

            total_loss_dist_cates = {}
            total_loss_transx_cates = {}
            total_loss_transy_cates = {}

            total_voxel_iou_before_cates = {}
            total_voxel_iou_after_cates = {}
            total_mask_iou_cates = {}
            total_loss_mask_cates = {}

            total_iou_diff_cates = {}

            summary_list = []

            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads)
            for _ in tower_loss:
                total_loss += _
            for _ in tower_loss_reg:
                total_loss_reg += _

            total_loss /= num_gpu
            total_loss_reg /= num_gpu
            summary_loss = tf.summary.scalar("total_loss", total_loss)
            summary_loss_reg = tf.summary.scalar("total_loss_reg", total_loss_reg)

            summary_list.append(summary_loss)
            summary_list.append(summary_loss_reg)

            for cate in cates:

                # total_loss_reg_cates[cate] = 0
                # total_loss_cates[cate] = 0

                total_loss_e_0_cates[cate] = 0
                total_loss_e_1_cates[cate] = 0
                total_loss_e_2_cates[cate] = 0

                total_err_e_0_cates[cate] = 0
                total_err_e_1_cates[cate] = 0
                total_err_e_2_cates[cate] = 0

                total_loss_dist_cates[cate] = 0
                total_loss_transx_cates[cate] = 0
                total_loss_transy_cates[cate] = 0

                total_voxel_iou_before_cates[cate] = 0
                total_voxel_iou_after_cates[cate] = 0
                total_mask_iou_cates[cate] = 0
                total_loss_mask_cates[cate] = 0

                # summary_list_cates[cate] = []

                for _ in tower_voxel_iou_before_cates[cate]:
                    total_voxel_iou_before_cates[cate] += _
                for _ in tower_voxel_iou_after_cates[cate]:
                    total_voxel_iou_after_cates[cate] += _

                for _ in tower_loss_e_0_cates[cate]:
                    total_loss_e_0_cates[cate] += _
                for _ in tower_loss_e_1_cates[cate]:
                    total_loss_e_1_cates[cate] += _
                for _ in tower_loss_e_2_cates[cate]:
                    total_loss_e_2_cates[cate] += _

                for _ in tower_err_e_0_cates[cate]:
                    total_err_e_0_cates[cate] += _
                for _ in tower_err_e_1_cates[cate]:
                    total_err_e_1_cates[cate] += _
                for _ in tower_err_e_2_cates[cate]:
                    total_err_e_2_cates[cate] += _
                for _ in tower_loss_dist_cates[cate]:
                    total_loss_dist_cates[cate] += _
                for _ in tower_loss_transx_cates[cate]:
                    total_loss_transx_cates[cate] += _
                for _ in tower_loss_transy_cates[cate]:
                    total_loss_transy_cates[cate] += _
                for _ in tower_mask_iou_cates[cate]:
                    total_mask_iou_cates[cate] += _
                for _ in tower_loss_mask_cates[cate]:
                    total_loss_mask_cates[cate] += _

                total_voxel_iou_before_cates[cate] /= num_gpu
                total_voxel_iou_after_cates[cate] /= num_gpu
                # total_loss_cates[cate] /= num_gpu
                # total_loss_reg_cates[cate] /= num_gpu
                total_loss_e_0_cates[cate] /= num_gpu
                total_loss_e_1_cates[cate] /= num_gpu
                total_loss_e_2_cates[cate] /= num_gpu
                total_err_e_0_cates[cate] /= num_gpu
                total_err_e_1_cates[cate] /= num_gpu
                total_err_e_2_cates[cate] /= num_gpu
                total_loss_dist_cates[cate] /= num_gpu
                total_loss_transx_cates[cate] /= num_gpu
                total_loss_transy_cates[cate] /= num_gpu
                total_mask_iou_cates[cate] /= num_gpu
                total_loss_mask_cates[cate] /= num_gpu

                total_iou_diff_cates[cate] = total_voxel_iou_after_cates[cate] - total_voxel_iou_before_cates[cate]

                # summary_loss = tf.summary.scalar("total_loss", total_loss_cates[cate])
                summary_IoU = tf.summary.scalar("%s_IoU_after" % cate, total_voxel_iou_after_cates[cate])
                # summary_loss_ = tf.summary.scalar("total_loss_reg", total_loss_reg_cates[cate])
                summary_IoU_before = tf.summary.scalar("%s_IoU_before" % cate, total_voxel_iou_before_cates[cate])
                summary_IoU_diff = tf.summary.scalar("%s_IoU_difference" % cate, total_iou_diff_cates[cate])

                # summary_loss = tf.summary.scalar("%s_loss" % cate, total_loss_cates[cate])
                # summary_loss_ = tf.summary.scalar("%s_loss_reg" % cate, total_loss_reg_cates[cate])
                summary_loss_e_0 = tf.summary.scalar("%s_loss_e_0" % cate, total_loss_e_0_cates[cate])
                summary_loss_e_1 = tf.summary.scalar("%s_loss_e_1" % cate, total_loss_e_1_cates[cate])
                summary_loss_e_2 = tf.summary.scalar("%s_loss_e_2" % cate, total_loss_e_2_cates[cate])
                summary_err_e_0 = tf.summary.scalar("%s_err_e_0" % cate, total_err_e_0_cates[cate])
                summary_err_e_1 = tf.summary.scalar("%s_err_e_1" % cate, total_err_e_1_cates[cate])
                summary_err_e_2 = tf.summary.scalar("%s_err_e_2" % cate, total_err_e_1_cates[cate])
                summary_loss_dist = tf.summary.scalar("%s_loss_dist" % cate, total_loss_dist_cates[cate])
                summary_loss_transx = tf.summary.scalar("%s_loss_transx" % cate, total_loss_transx_cates[cate])
                summary_loss_transy = tf.summary.scalar("%s_loss_transy" % cate, total_loss_transy_cates[cate])

                summary_mask_iou = tf.summary.scalar("%s_mask_iou" % cate, total_mask_iou_cates[cate])
                summary_loss_mask = tf.summary.scalar('%s_loss_mask' % cate, total_loss_mask_cates[cate])

                summary_list.append(summary_IoU)
                summary_list.append(summary_IoU_before)
                summary_list.append(summary_IoU_diff)

                # summary_list.append(summary_loss)
                # summary_list.append(summary_loss_)
                summary_list.append(summary_loss_e_0)
                summary_list.append(summary_loss_e_1)
                summary_list.append(summary_loss_e_2)
                summary_list.append(summary_err_e_0)
                summary_list.append(summary_err_e_1)
                summary_list.append(summary_err_e_2)
                summary_list.append(summary_loss_dist)
                summary_list.append(summary_loss_transx)
                summary_list.append(summary_loss_transy)
                summary_list.append(summary_mask_iou)
                summary_list.append(summary_loss_mask)

            summary_merge = tf.summary.merge(summary_list)
            print('gradient average over')

        weight_angle = 'angle_voc/27001.cptk'
        weight_voxel = 'voxel_voc/11501.cptk'
        # weight_dist_trans = 'dt_trans/179001.cptk'
        weight_mask = 'mask_voc/23001.cptk'
        weight_refine = 'refine_r2n2/8001.cptk'
        path_weight = []
        path_weight.append(weight_angle)
        path_weight.append(weight_voxel)
        # path_weight.append(weight_dist_trans)
        path_weight.append(weight_mask)
        path_weight.append(weight_refine)

        saver = tf.train.Saver(max_to_keep=25, var_list=tf.trainable_variables())

        scalarsaver = ScalarSaver()

        angle_params = [para for para in tf.trainable_variables() if
                        'voxel' not in para.name and 'mask' not in para.name and 'refine' not in para.name]
        voxel_params = [para for para in tf.trainable_variables() if 'voxel' in para.name]
        mask_params = [para for para in tf.trainable_variables() if 'mask' in para.name]
        refine_params = [para for para in tf.trainable_variables() if 'refine' in para.name]

        previous_params = []
        previous_params.append(angle_params)
        previous_params.append(voxel_params)
        previous_params.append(mask_params)
        previous_params.append(refine_params)

        num_cates = len(cates)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            sess.run(tf.global_variables_initializer())
            writer_train = tf.summary.FileWriter(os.path.join(tensorboard_path, 'train'), sess.graph)
            writer_test = tf.summary.FileWriter(os.path.join(tensorboard_path, 'test'), sess.graph)

            for i, p in enumerate(previous_params):
                print('load from', path_weight[i])
                loader = tf.train.Saver(var_list=p)
                loader.restore(sess, path_weight[i])
            print('restore OK')

            if not continue_train_path is None:
                saver.restore(sess, continue_train_path)

            lr = 1e-5
            for step in range(0, iterations):
                t = time.time()

                if step in lrs:
                    lr = lrs[step]

                x_stack, y_stack, m_stack, e_stack, st_stack = stacker.get_batch()
                print('Data fetch', time.time() - t)
                sess.run(apply_gradient_op,
                         feed_dict={x_vectors: x_stack, y_vectors: y_stack, m_vectors: m_stack, e_vectors: e_stack,
                                    st_vectors: st_stack, learning_rate: lr})
                # for cate in cates:
                #     x_op, y_op, m_op, e_op, st_op = worker.get_batch(cate)  # get_data()
                #     sess.run(apply_gradient_op_cates[cate], feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op,
                #                        st_vectors: st_op, learning_rate: lr})
                print('optimize:', time.time() - t)

                if step % 100 == 1 or step == 0:

                    summary_train = sess.run(
                        summary_merge,
                        feed_dict={x_vectors: x_stack, y_vectors: y_stack, m_vectors: m_stack, e_vectors: e_stack,
                                   st_vectors: st_stack})
                    # for cate in cates:
                    #     x_op, y_op, m_op, e_op, st_op = worker.get_batch(cate)
                    #     summary_train = sess.run(
                    #         summary_merge_cates[cate],
                    #         feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op,
                    #                    st_vectors: st_op})

                    writer_train.add_summary(summary_train, step)

                    # loss_test_total = 0
                    # loss_reg_test_total = 0
                    # loss_v_test_total = 0
                    loss_e_0_test_total = 0
                    loss_e_1_test_total = 0
                    loss_e_2_test_total = 0
                    err_e_0_test_total = 0
                    err_e_1_test_total = 0
                    err_e_2_test_total = 0

                    loss_dist_test_total = 0
                    loss_transx_test_total = 0
                    loss_transy_test_total = 0

                    voxel_iou_before_test_total = 0
                    voxel_iou_after_test_total = 0
                    mask_iou_test_total = 0
                    loss_mask_test_total = 0
                    iou_diff_test_total = 0

                    num_iter = 5
                    loss_test = 0
                    loss_reg_test = 0

                    loss_e_0_test = {}
                    loss_e_1_test = {}
                    loss_e_2_test = {}

                    err_e_0_test = {}
                    err_e_1_test = {}
                    err_e_2_test = {}

                    loss_dist_test = {}
                    loss_transx_test = {}
                    loss_transy_test = {}

                    voxel_iou_before_test = {}
                    voxel_iou_after_test = {}
                    mask_iou_test = {}
                    loss_mask_test = {}
                    iou_diff_test = {}

                    for cate in cates:
                        loss_e_0_test[cate] = 0
                        loss_e_1_test[cate] = 0
                        loss_e_2_test[cate] = 0

                        err_e_0_test[cate] = 0
                        err_e_1_test[cate] = 0
                        err_e_2_test[cate] = 0

                        loss_dist_test[cate] = 0
                        loss_transx_test[cate] = 0
                        loss_transy_test[cate] = 0

                        voxel_iou_before_test[cate] = 0
                        voxel_iou_after_test[cate] = 0
                        mask_iou_test[cate] = 0
                        loss_mask_test[cate] = 0
                        iou_diff_test[cate] = 0

                    for _ in range(num_iter):
                        # print('i',_)
                        # x_op, y_op, m_op, e_op, st_op = worker_uniform.get_batch(cate, False)
                        x_stack, y_stack, m_stack, e_stack, st_stack = stacker_uniform.get_batch(False)
                        res_list = []

                        for cate in cates:
                            res_list.extend([total_loss_e_0_cates[cate],
                                             total_loss_e_1_cates[cate], total_loss_e_2_cates[cate],
                                             total_loss_dist_cates[cate],
                                             total_loss_transx_cates[cate], total_loss_transy_cates[cate],
                                             total_err_e_0_cates[cate], total_err_e_1_cates[cate], total_err_e_2_cates[cate],
                                             total_voxel_iou_before_cates[cate], total_voxel_iou_after_cates[cate],
                                             total_mask_iou_cates[cate], total_loss_mask_cates[cate],
                                             total_iou_diff_cates[cate]
                                             ])
                        res_list.append(total_loss)
                        res_list.append(total_loss_reg)

                        res = sess.run(res_list,
                                       feed_dict={x_vectors: x_stack, y_vectors: y_stack, m_vectors: m_stack,
                                                  e_vectors: e_stack,
                                                  st_vectors: st_stack})

                        for i, cate in enumerate(cates):
                            loss_e_0, loss_e_1, loss_e_2, loss_dist, loss_transx, loss_transy, err_e_0, err_e_1, err_e_2, iou_before, iou_after, mask_iou, loss_mask, iou_diff = \
                                res[i * 14:(i + 1) * 14]
                            loss_e_0_test[cate] += loss_e_0
                            loss_e_1_test[cate] += loss_e_1
                            loss_e_2_test[cate] += loss_e_2
                            err_e_0_test[cate] += err_e_0
                            err_e_1_test[cate] += err_e_1
                            err_e_2_test[cate] += err_e_2

                            loss_dist_test[cate] += loss_dist
                            loss_transx_test[cate] += loss_transx
                            loss_transy_test[cate] += loss_transy

                            voxel_iou_before_test[cate] += iou_before
                            voxel_iou_after_test[cate] += iou_after
                            mask_iou_test[cate] += mask_iou
                            loss_mask_test[cate] += loss_mask
                            iou_diff_test[cate] += iou_diff

                        loss_test += res[-2]
                        loss_reg_test += res[-1]

                    loss_test /= num_iter
                    loss_reg_test /= num_iter

                    for cate in cates:
                        # iou_test /= num_iter
                        # loss_v_test /= num_iter

                        loss_e_0_test[cate] /= num_iter  # loss_e
                        loss_e_1_test[cate] /= num_iter  # loss_e
                        loss_e_2_test[cate] /= num_iter  # loss_e

                        err_e_0_test[cate] /= num_iter
                        err_e_1_test[cate] /= num_iter
                        err_e_2_test[cate] /= num_iter

                        loss_dist_test[cate] /= num_iter
                        loss_transx_test[cate] /= num_iter
                        loss_transy_test[cate] /= num_iter

                        voxel_iou_after_test[cate] /= num_iter
                        voxel_iou_before_test[cate] /= num_iter
                        mask_iou_test[cate] /= num_iter
                        loss_mask_test[cate] /= num_iter
                        iou_diff_test[cate] /= num_iter

                        summt = tf.Summary()

                        # summt.value.add(tag='%s_loss_v' % cate, simple_value=loss_v_test)

                        summt.value.add(tag='%s_loss_e_0' % cate, simple_value=loss_e_0_test[cate])
                        summt.value.add(tag='%s_loss_e_1' % cate, simple_value=loss_e_1_test[cate])
                        summt.value.add(tag='%s_loss_e_2' % cate, simple_value=loss_e_2_test[cate])

                        summt.value.add(tag='%s_err_e_0' % cate, simple_value=err_e_0_test[cate])
                        summt.value.add(tag='%s_err_e_1' % cate, simple_value=err_e_1_test[cate])
                        summt.value.add(tag='%s_err_e_2' % cate, simple_value=err_e_2_test[cate])

                        summt.value.add(tag='%s_loss_dist' % cate, simple_value=loss_dist_test[cate])
                        summt.value.add(tag='%s_loss_transx' % cate, simple_value=loss_transx_test[cate])
                        summt.value.add(tag='%s_loss_transy' % cate, simple_value=loss_transy_test[cate])

                        summt.value.add(tag='%s_IoU_after' % cate, simple_value=voxel_iou_after_test[cate])
                        summt.value.add(tag='%s_IoU_before' % cate, simple_value=voxel_iou_before_test[cate])
                        summt.value.add(tag='%s_IoU_difference' % cate, simple_value=iou_diff_test[cate])
                        summt.value.add(tag='%s_loss_mask' % cate, simple_value=loss_mask_test[cate])
                        summt.value.add(tag='%s_mask_iou' % cate, simple_value=mask_iou_test[cate])

                        writer_test.add_summary(summt, step)

                        # loss_test_total += loss_test
                        # # iou_test_total += iou_test
                        # loss_reg_test_total += loss_reg_test
                        loss_e_0_test_total += loss_e_0_test[cate]
                        loss_e_1_test_total += loss_e_1_test[cate]
                        loss_e_2_test_total += loss_e_2_test[cate]

                        err_e_0_test_total += err_e_0_test[cate]
                        err_e_1_test_total += err_e_1_test[cate]
                        err_e_2_test_total += err_e_2_test[cate]

                        loss_dist_test_total += loss_dist_test[cate]
                        loss_transx_test_total += loss_transx_test[cate]
                        loss_transy_test_total += loss_transy_test[cate]

                        voxel_iou_after_test_total += voxel_iou_after_test[cate]
                        voxel_iou_before_test_total += voxel_iou_before_test[cate]
                        mask_iou_test_total += mask_iou_test[cate]
                        loss_mask_test_total += loss_mask_test[cate]
                        iou_diff_test_total += iou_diff_test[cate]

                    # loss_test_total /= num_cates
                    # # iou_test_total /= num_cates
                    # loss_reg_test_total /= num_cates
                    # loss_v_test_total /= num_cates
                    loss_e_0_test_total /= num_cates
                    loss_e_1_test_total /= num_cates
                    loss_e_2_test_total /= num_cates
                    err_e_0_test_total /= num_cates
                    err_e_1_test_total /= num_cates
                    err_e_2_test_total /= num_cates

                    loss_dist_test_total /= num_cates
                    loss_transx_test_total /= num_cates
                    loss_transy_test_total /= num_cates

                    voxel_iou_after_test_total /= num_cates
                    voxel_iou_before_test_total /= num_cates
                    mask_iou_test_total /= num_cates
                    loss_mask_test_total /= num_cates
                    iou_diff_test_total /= num_cates

                    summt = tf.Summary()
                    summt.value.add(tag='total_loss', simple_value=loss_test)
                    # summt.value.add(tag='total_IoU', simple_value=iou_test_total)
                    summt.value.add(tag='total_loss_reg', simple_value=loss_reg_test)
                    # summt.value.add(tag='total_loss_v', simple_value=loss_v_test_total)

                    summt.value.add(tag='total_loss_e_0', simple_value=loss_e_0_test_total)
                    summt.value.add(tag='total_loss_e_1', simple_value=loss_e_1_test_total)
                    summt.value.add(tag='total_loss_e_2', simple_value=loss_e_2_test_total)
                    summt.value.add(tag='total_err_e_0', simple_value=err_e_0_test_total)
                    summt.value.add(tag='total_err_e_1', simple_value=err_e_1_test_total)
                    summt.value.add(tag='total_err_e_2', simple_value=err_e_2_test_total)

                    summt.value.add(tag='total_loss_dist', simple_value=loss_dist_test_total)
                    summt.value.add(tag='total_loss_transx', simple_value=loss_transx_test_total)
                    summt.value.add(tag='total_loss_transy', simple_value=loss_transy_test_total)

                    summt.value.add(tag='total_IoU_after', simple_value=voxel_iou_after_test_total)
                    summt.value.add(tag='total_IoU_before', simple_value=voxel_iou_before_test_total)
                    summt.value.add(tag='total_IoU_difference', simple_value=iou_diff_test_total)
                    summt.value.add(tag='total_loss_mask', simple_value=loss_mask_test_total)
                    summt.value.add(tag='total_mask_iou', simple_value=mask_iou_test_total)

                    writer_test.add_summary(summt, step)

                    # print("Steps:", step, "training loss:", loss_train, "iou:", iou_train, "testing loss:",
                    #       sum_loss, "iou:", sum_iou)
                    print("Steps:", step, "testing loss:",
                          loss_test)

                if step % 500 == 1 and step > 500:
                    saver.save(sess, save_path=os.path.join(checkpoint_path, "%d.cptk" % step))
                    with open(os.path.join(checkpoint_path, '%d.pkl' % step), 'wb') as fp:
                        pickle.dump(scalarsaver, fp, -1)

                print('Time consume:', time.time() - t)


def evaluation_random():
    data_path = '../dataset/ShapeNet/ShapeNetRendering'
    vox_path = '../dataset/ShapeNet/ShapeNetVox32'
    bg_path = '../dataset/bg_crop'

    cates = small_set
    cates = [small_set[0]]
    cates = [small_set[1]]
    cates = [small_set[2]]
    cates = [small_set[3]]

    # weight_path = 'voxel/100001.cptk'
    # refine_path = os.path.join('checkpoint_cross_categories_4_Adam_refine','70501.cptk')
    weight_path = os.path.join('end2end','23501.cptk')

    batchsize = 20
    from dataIO_r2n2 import dataIO_r2n2
    # io = dataIO(data_path, bg_path, vox_path, cates, ratio=0.8)
    io = dataIO_r2n2(data_path, bg_path, vox_path, small_set)
    x_vectors = tf.placeholder(shape=[batchsize, voxel_size, voxel_size, voxel_size, 2],
                               dtype=tf.float32, name='all_Voxels')
    y_vectors = tf.placeholder(shape=[batchsize, img_w, img_h, 3], dtype=tf.float32, name='all_Images')
    m_vectors = tf.placeholder(shape=[batchsize, img_w, img_h, 2], dtype=tf.float32, name='all_Masks')
    e_vectors = tf.placeholder(shape=[batchsize, 3], dtype=tf.float32, name='all_Angles')
    st_vectors = tf.placeholder(shape=[batchsize, 3], dtype=tf.float32, name='all_scale_translation')

    with tf.variable_scope('voxel'):
        feature, feature_map, euler, st, shortcuts = encoder(y_vectors, reuse=False)
        voxels, voxels_softmax_before = decoder(feature, reuse=False)

    with tf.variable_scope("mask"):
        feature, feature_map, _, _, shortcuts = encoder(y_vectors, reuse=False)
        mask, mask_softmax = generator(feature_map, shortcuts, reuse=False)

    feature, feature_map, shortcuts = encoder_angle(y_vectors, reuse=False)

    IoU_before = tf.reduce_sum(
        tf.cast(tf.logical_and(voxels_softmax_before[:, :, :, :, 0] > thresh_hold,
                               x_vectors[:, :, :, :, 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
        tf.cast(tf.logical_or(voxels_softmax_before[:, :, :, :, 0] > thresh_hold,
                              x_vectors[:, :, :, :, 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2, 3])

    IoU_mask = tf.reduce_sum(
        tf.cast(tf.logical_and(mask_softmax[..., 0] > thresh_hold,
                               m_vectors[..., 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2]) / tf.reduce_sum(
        tf.cast(tf.logical_or(mask_softmax[..., 0] > thresh_hold,
                              m_vectors[..., 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2])

    iou_after_dic = {}
    voxel_after_dic = {}
    angle_dic = {}
    st_dic = {}
    visual_hull_dic = {}

    for i,cate in enumerate(cates):
        if i == 0:
            reuse = False
        else:
            reuse = True

        with tf.variable_scope("angles_trans", reuse=False):
            w_e_1 = tf.get_variable("w_euler_0_%s" % cate, shape=[1024, 512],
                                    initializer=tf.contrib.layers.xavier_initializer())
            e_1 = lrelu(tf.matmul(feature, w_e_1))
            w_e_2 = tf.get_variable("w_euler_1_%s" % cate, shape=[512, 3],
                                    initializer=tf.contrib.layers.xavier_initializer())
            euler = tf.matmul(e_1, w_e_2)

            w_st = tf.get_variable('w_ft_%s' % cate, shape=[1024, 3],
                                   initializer=tf.contrib.layers.xavier_initializer())
            st = tf.matmul(feature, w_st)

        rotation_matrices = get_rotation_matrix_r2n2(euler)
        mask_indexs = scale_trans_r2n2(st)
        projection = cast(mask_softmax[..., 0], mask_indexs, rotation_matrices=rotation_matrices)
        c1 = voxels_softmax_before[..., 0]
        c2 = projection
        c3 = c1 - c1 * c2
        c4 = c2 - c1 * c2

        feedin = tf.stack([c1, c2, c3, c4], axis=4)

        feature_vector, shortcuts = refine_encoder(feedin, reuse=reuse)
        voxels, voxels_softmax_after = refine_decoder(feature_vector, shortcuts, reuse=reuse)

        IoU_after = tf.reduce_sum(
            tf.cast(tf.logical_and(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                   x_vectors[:, :, :, :, 0] > thresh_hold),
                    dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
            tf.cast(tf.logical_or(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                  x_vectors[:, :, :, :, 0] > thresh_hold),
                    dtype=tf.float32), axis=[1, 2, 3])

        iou_after_dic[cate] = IoU_after
        angle_dic[cate] = euler
        st_dic[cate] = st
        voxel_after_dic[cate] = voxels_softmax_after[..., 0]
        visual_hull_dic[cate] = projection

    params = tf.trainable_variables()
    saver = tf.train.Saver(var_list=params)

    save_path = 'random_bg'
    with tf.Session() as sess:
        print('evaluate random_bg',cates)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,weight_path)
        for cate in small_set:
            if not os.path.exists(os.path.join(save_path,cate)):
                os.makedirs(os.path.join(save_path,cate))

            iou_test = 0
            idx = 1
            num_batch = int(len(io.testlist_cates[cate]) * 24 / batchsize) + 1
            angles_gt = []
            angles_pt = []
            st_gt = []
            st_pt = []
            iou_before_after_mask = []
            print(cate,num_batch)
            for _ in range(num_batch):
                print(_,'/',num_batch)
                x_op, y_op, m_op, e_op, st_op, s_op = io.fetch_evaluation(batchsize,train_phase=False,random_background=True,cates=[cate])
                # print(x_op.shape)
                if cate != cates[0]:
                    continue
                x_op = data_transfer(x_op)
                m_op = data_transfer(m_op)
                v_before,v_after,iou_before,iou_after,iou_mask,masks,angles,sts, visual_hull = sess.run([voxels_softmax_before[...,0],voxel_after_dic[cate],IoU_before,iou_after_dic[cate],IoU_mask,mask_softmax[...,0],angle_dic[cate],st_dic[cate], visual_hull_dic[cate]], feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op, st_vectors: st_op})
                iou_test += np.mean(iou_after)
                for i,v in enumerate(v_before):

                    # iou_diff = iou_after[i] - iou_before[i]
                    # input_img = Image.fromarray(np.array(y_op[i]*255).astype(np.uint8))
                    # mask_gt = Image.fromarray(np.array(m_op[i,...,0]*255).astype(np.uint8))
                    # mask_pt = Image.fromarray(np.array(masks[i]*255).astype(np.uint8))
                    # input_img.save(os.path.join(save_path,cate,'%f_%d_input.jpg'%(iou_diff,idx)))
                    # mask_gt.save(os.path.join(save_path,cate,'%f_%d_mask_gt.gif'%(iou_diff,idx)))
                    # mask_pt.save(os.path.join(save_path,cate,'%f_%d_mask_pt.gif'%(iou_diff,idx)))
                    # with open(os.path.join(save_path,cate,'%f_%d_voxel_before.txt'%(iou_diff,idx)),'w') as fp:
                    #     for ___ in v_before[i]:
                    #         for __ in ___:
                    #             for _ in __:
                    #                 fp.write('%f\n'%_)
                    #
                    # with open(os.path.join(save_path,cate,'%f_%d_voxel_after.txt'%(iou_diff,idx)),'w') as fp:
                    #     for ___ in v_after[i]:
                    #         for __ in ___:
                    #             for _ in __:
                    #                 fp.write('%f\n'%_)
                    # with open(os.path.join(save_path,cate,'%f_%d_voxel_gt.txt'%(iou_diff,idx)),'w') as fp:
                    #     for ___ in x_op[i,...,0]:
                    #         for __ in ___:
                    #             for _ in __:
                    #                 fp.write('%f\n'%_)
                    # with open(os.path.join(save_path,cate,'%f_%d_visual_hull.txt'%(iou_diff,idx)),'w') as fp:
                    #     for ___ in visual_hull[i]:
                    #         for __ in ___:
                    #             for _ in __:
                    #                 fp.write('%f\n'%_)

                    angles_pt.append(angles[i])
                    angles_gt.append(e_op[i])
                    st_pt.append(sts[i])
                    st_gt.append(st_op[i])
                    iou_before_after_mask.append([iou_before[i],iou_after[i],iou_mask[i]])

                    idx += 1

            if cate == cates[0]:
                with open(os.path.join(save_path,cate+'_angles_st_iou_pt_24.txt'),'w') as fp:
                    for i,a in enumerate(angles_pt):
                        for _ in a:
                            fp.write('%f '%_)
                        for _ in st_pt[i]:
                            fp.write('%f '%_)
                        for _ in iou_before_after_mask[i]:
                            fp.write('%f '%_)
                        fp.write('\n')

                with open(os.path.join(save_path,cate+'_angles_st_gt_24.txt'),'w') as fp:
                    for i,a in enumerate(angles_gt):
                        for _ in a:
                            fp.write('%f '%_)
                        for _ in st_gt[i]:
                            fp.write('%f '%_)
                        fp.write('\n')

                iou_test /= num_batch
                xx = '%s %f' % (dic[cate], iou_test)
                print(xx)

def evaluation_transparent():
    data_path = '../dataset/ShapeNet/ShapeNetRendering'
    vox_path = '../dataset/ShapeNet/ShapeNetVox32'
    bg_path = '../dataset/bg_crop'

    cates = small_set
    # cates = [small_set[0]]
    # cates = [small_set[1]]
    # cates = [small_set[2]]
    # cates = [small_set[3]]
    # weight_path = 'voxel/100001.cptk'
    # refine_path = os.path.join('checkpoint_cross_categories_4_Adam_refine','70501.cptk')
    weight_path = os.path.join('end2end','23501.cptk')
    weight_path = os.path.join('end2end','20501.cptk')
    # weight_path = os.path.join('end2end','28501.cptk')

    batchsize = 20
    from dataIO_r2n2 import dataIO_r2n2
    # io = dataIO(data_path, bg_path, vox_path, cates, ratio=0.8)
    io = dataIO_r2n2(data_path, bg_path, vox_path, small_set)
    x_vectors = tf.placeholder(shape=[batchsize, voxel_size, voxel_size, voxel_size, 2],
                               dtype=tf.float32, name='all_Voxels')
    y_vectors = tf.placeholder(shape=[batchsize, img_w, img_h, 3], dtype=tf.float32, name='all_Images')
    m_vectors = tf.placeholder(shape=[batchsize, img_w, img_h, 2], dtype=tf.float32, name='all_Masks')
    e_vectors = tf.placeholder(shape=[batchsize, 3], dtype=tf.float32, name='all_Angles')
    st_vectors = tf.placeholder(shape=[batchsize, 3], dtype=tf.float32, name='all_scale_translation')

    with tf.variable_scope('voxel'):
        feature, feature_map, euler, st, shortcuts = encoder(y_vectors, reuse=False)
        voxels, voxels_softmax_before = decoder(feature, reuse=False)

    with tf.variable_scope("mask"):
        feature, feature_map, _, _, shortcuts = encoder(y_vectors, reuse=False)
        mask, mask_softmax = generator(feature_map, shortcuts, reuse=False)

    feature, feature_map, shortcuts = encoder_angle(y_vectors, reuse=False)

    IoU_before = tf.reduce_sum(
        tf.cast(tf.logical_and(voxels_softmax_before[:, :, :, :, 0] > thresh_hold,
                               x_vectors[:, :, :, :, 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
        tf.cast(tf.logical_or(voxels_softmax_before[:, :, :, :, 0] > thresh_hold,
                              x_vectors[:, :, :, :, 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2, 3])

    IoU_mask = tf.reduce_sum(
        tf.cast(tf.logical_and(mask_softmax[..., 0] > thresh_hold,
                               m_vectors[..., 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2]) / tf.reduce_sum(
        tf.cast(tf.logical_or(mask_softmax[..., 0] > thresh_hold,
                              m_vectors[..., 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2])

    iou_after_dic = {}
    voxel_after_dic = {}
    angle_dic = {}
    st_dic = {}
    visual_hull_dic = {}

    for i,cate in enumerate(cates):
        if i == 0:
            reuse = False
        else:
            reuse = True

        with tf.variable_scope("angles_trans", reuse=False):
            w_e_1 = tf.get_variable("w_euler_0_%s" % cate, shape=[1024, 512],
                                    initializer=tf.contrib.layers.xavier_initializer())
            e_1 = lrelu(tf.matmul(feature, w_e_1))
            w_e_2 = tf.get_variable("w_euler_1_%s" % cate, shape=[512, 3],
                                    initializer=tf.contrib.layers.xavier_initializer())
            euler = tf.matmul(e_1, w_e_2)

            w_st = tf.get_variable('w_ft_%s' % cate, shape=[1024, 3],
                                   initializer=tf.contrib.layers.xavier_initializer())
            st = tf.matmul(feature, w_st)

        rotation_matrices = get_rotation_matrix_r2n2(euler)
        mask_indexs = scale_trans_r2n2(st)
        projection = cast(mask_softmax[..., 0], mask_indexs, rotation_matrices=rotation_matrices)
        c1 = voxels_softmax_before[..., 0]
        c2 = projection
        c3 = c1 - c1 * c2
        c4 = c2 - c1 * c2

        feedin = tf.stack([c1, c2, c3, c4], axis=4)

        feature_vector, shortcuts = refine_encoder(feedin, reuse=reuse)
        voxels, voxels_softmax_after = refine_decoder(feature_vector, shortcuts, reuse=reuse)

        IoU_after = tf.reduce_sum(
            tf.cast(tf.logical_and(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                   x_vectors[:, :, :, :, 0] > thresh_hold),
                    dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
            tf.cast(tf.logical_or(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                  x_vectors[:, :, :, :, 0] > thresh_hold),
                    dtype=tf.float32), axis=[1, 2, 3])

        iou_after_dic[cate] = IoU_after
        angle_dic[cate] = euler
        st_dic[cate] = st
        voxel_after_dic[cate] = voxels_softmax_after[..., 0]
        visual_hull_dic[cate] = projection

    params = tf.trainable_variables()
    saver = tf.train.Saver(var_list=params)

    save_path = 'transparent_bg'
    with tf.Session() as sess:
        print('evaluate transparent',cates,'restore',weight_path)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,weight_path)
        for cate in small_set:
            if not os.path.exists(os.path.join(save_path,cate)):
                os.makedirs(os.path.join(save_path,cate))
            print(cate)
            iou_test = 0
            idx = 1
            num_batch = int(len(io.testlist_cates[cate]) * 24 / batchsize) + 1
            angles_gt = []
            angles_pt = []
            st_gt = []
            st_pt = []
            iou_before_after_mask = []
            for _ in range(num_batch):
                x_op, y_op, m_op, e_op, st_op, s_op = io.fetch_evaluation(batchsize,train_phase=False,random_background=False,cates=[cate])
                # print(x_op.shape)
                # if cate != cates[0]:
                #     continue
                x_op = data_transfer(x_op)
                m_op = data_transfer(m_op)
                v_before,v_after,iou_before,iou_after,iou_mask,masks,angles,sts,visual_hull = sess.run([voxels_softmax_before,voxel_after_dic[cate],IoU_before,iou_after_dic[cate],IoU_mask,mask_softmax,angle_dic[cate],st_dic[cate],visual_hull_dic[cate]], feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op, st_vectors: st_op})
                iou_test += np.mean(iou_after)
                for i,v in enumerate(v_before[...,0]):
                    # iou_diff = iou_after[i] - iou_before[i]
                    # input_img = Image.fromarray(np.array(y_op[i]*255).astype(np.uint8))
                    # mask_gt = Image.fromarray(np.array(m_op[i,...,0]*255).astype(np.uint8))
                    # mask_pt = Image.fromarray(np.array(masks[i,...,0]*255).astype(np.uint8))
                    # input_img.save(os.path.join(save_path,cate,'%f_%d_input.jpg'%(iou_diff,idx)))
                    # mask_gt.save(os.path.join(save_path,cate,'%f_%d_mask_gt.gif'%(iou_diff,idx)))
                    # mask_pt.save(os.path.join(save_path,cate,'%f_%d_mask_pt.gif'%(iou_diff,idx)))
                    # with open(os.path.join(save_path,cate,'%f_%d_voxel_before.txt'%(iou_diff,idx)),'w') as fp:
                    #     for ___ in v_before[i]:
                    #         for __ in ___:
                    #             for _ in __:
                    #                 fp.write('%f\n'%_)
                    #
                    # with open(os.path.join(save_path,cate,'%f_%d_voxel_after.txt'%(iou_diff,idx)),'w') as fp:
                    #     for ___ in v_after[i]:
                    #         for __ in ___:
                    #             for _ in __:
                    #                 fp.write('%f\n'%_)
                    # with open(os.path.join(save_path,cate,'%f_%d_voxel_gt.txt'%(iou_diff,idx)),'w') as fp:
                    #     for ___ in x_op[i,...,0]:
                    #         for __ in ___:
                    #             for _ in __:
                    #                 fp.write('%f\n'%_)

                    # with open(os.path.join(save_path,cate,'%f_%d_visual_hull.txt'%(iou_diff,idx)),'w') as fp:
                    #     for ___ in visual_hull[i]:
                    #         for __ in ___:
                    #             for _ in __:
                    #                 fp.write('%f\n'%_)

                    angles_pt.append(angles[i])
                    angles_gt.append(e_op[i])
                    st_pt.append(sts[i])
                    st_gt.append(st_op[i])
                    iou_before_after_mask.append([iou_before[i],iou_after[i],iou_mask[i]])

                    idx += 1
            # if cate == cates[0]:
            with open(os.path.join(save_path,cate+'_angles_st_iou_pt_24.txt'),'w') as fp:
                for i,a in enumerate(angles_pt):
                    for _ in a:
                        fp.write('%f '%_)
                    for _ in st_pt[i]:
                        fp.write('%f '%_)
                    for _ in iou_before_after_mask[i]:
                        fp.write('%f '%_)
                    fp.write('\n')

            with open(os.path.join(save_path,cate+'_angles_st_gt_24.txt'),'w') as fp:
                for i,a in enumerate(angles_gt):
                    for _ in a:
                        fp.write('%f '%_)
                    for _ in st_gt[i]:
                        fp.write('%f '%_)
                    fp.write('\n')

            iou_test /= num_batch
            xx = '%s %f' % (dic[cate], iou_test)
            print(xx)

def evaluation_voc():
    data_path = '../dataset/voc3d/imgs'
    data_path = '../dataset/voc3dori/imgs'
    vox_path = '../dataset/voc3d/Vox'
    bg_path = '../dataset/bg_crop'

    cates = ['sofa', 'aeroplane', 'chair', 'car']
    cates_name = small_set

    # weight_path = 'voxel/100001.cptk'
    # refine_path = os.path.join('checkpoint_cross_categories_4_Adam_refine','70501.cptk')
    weight_path = os.path.join('end2end_voc','2001.cptk')
    weight_path = os.path.join('end2end_voc','6001.cptk')

    batchsize = 20
    from dataIO_voc import dataIO_voc
    # io = dataIO(data_path, bg_path, vox_path, cates, ratio=0.8)
    io = dataIO_voc(data_path, bg_path, vox_path, cates)
    x_vectors = tf.placeholder(shape=[batchsize, voxel_size, voxel_size, voxel_size, 2],
                               dtype=tf.float32, name='all_Voxels')
    y_vectors = tf.placeholder(shape=[batchsize, img_w, img_h, 3], dtype=tf.float32, name='all_Images')
    m_vectors = tf.placeholder(shape=[batchsize, img_w, img_h, 2], dtype=tf.float32, name='all_Masks')
    e_vectors = tf.placeholder(shape=[batchsize, 3], dtype=tf.float32, name='all_Angles')
    st_vectors = tf.placeholder(shape=[batchsize, 3], dtype=tf.float32, name='all_scale_translation')

    with tf.variable_scope('voxel'):
        feature, feature_map, euler, st, shortcuts = encoder(y_vectors, reuse=False)
        voxels, voxels_softmax_before = decoder(feature, reuse=False)

    with tf.variable_scope("mask"):
        feature, feature_map, _, _, shortcuts = encoder(y_vectors, reuse=False)
        mask, mask_softmax = generator(feature_map, shortcuts, reuse=False)

    feature, feature_map, shortcuts = encoder_angle(y_vectors, reuse=False)

    IoU_before = tf.reduce_sum(
        tf.cast(tf.logical_and(voxels_softmax_before[:, :, :, :, 0] > thresh_hold,
                               x_vectors[:, :, :, :, 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
        tf.cast(tf.logical_or(voxels_softmax_before[:, :, :, :, 0] > thresh_hold,
                              x_vectors[:, :, :, :, 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2, 3])

    IoU_mask = tf.reduce_sum(
        tf.cast(tf.logical_and(mask_softmax[..., 0] > thresh_hold,
                               m_vectors[..., 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2]) / tf.reduce_sum(
        tf.cast(tf.logical_or(mask_softmax[..., 0] > thresh_hold,
                              m_vectors[..., 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2])

    iou_after_dic = {}
    voxel_after_dic = {}
    angle_dic = {}
    st_dic = {}
    visual_hull_dic = {}

    for i,cate in enumerate(cates):
        if i == 0:
            reuse = False
        else:
            reuse = True

        with tf.variable_scope("angles_trans", reuse=False):
            w_e_1 = tf.get_variable("w_euler_0_%s" % cates_name[i], shape=[1024, 512],
                                    initializer=tf.contrib.layers.xavier_initializer())
            e_1 = lrelu(tf.matmul(feature, w_e_1))
            w_e_2 = tf.get_variable("w_euler_1_%s" % cates_name[i], shape=[512, 3],
                                    initializer=tf.contrib.layers.xavier_initializer())
            euler = tf.matmul(e_1, w_e_2)

            w_st = tf.get_variable('w_ft_%s' % cates_name[i], shape=[1024, 3],
                                   initializer=tf.contrib.layers.xavier_initializer())
            st = tf.matmul(feature, w_st)
            st = tf.stack([st[..., 0] * 10, st[..., 1], st[..., 2]], axis=-1)

        rotation_matrices = get_rotation_matrix_voc(euler)
        mask_indexs = scale_trans_voc(st)
        masks = rotate_mask_voc(mask_softmax[..., 0], euler)
        projection = cast(masks, mask_indexs, rotation_matrices=rotation_matrices)
        c1 = voxels_softmax_before[..., 0]
        c2 = projection
        c3 = c1 - c1 * c2
        c4 = c2 - c1 * c2

        feedin = tf.stack([c1, c2, c3, c4], axis=4)

        feature_vector, shortcuts = refine_encoder(feedin, reuse=reuse)
        voxels, voxels_softmax_after = refine_decoder(feature_vector, shortcuts, reuse=reuse)

        IoU_after = tf.reduce_sum(
            tf.cast(tf.logical_and(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                   x_vectors[:, :, :, :, 0] > thresh_hold),
                    dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
            tf.cast(tf.logical_or(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                  x_vectors[:, :, :, :, 0] > thresh_hold),
                    dtype=tf.float32), axis=[1, 2, 3])

        iou_after_dic[cate] = IoU_after
        angle_dic[cate] = euler
        st_dic[cate] = st
        voxel_after_dic[cate] = voxels_softmax_after[..., 0]
        visual_hull_dic[cate] = projection

    params = tf.trainable_variables()
    saver = tf.train.Saver(var_list=params)

    save_path = 'voc_result'
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,weight_path)
        for cate in cates:
            if not os.path.exists(os.path.join(save_path,cate)):
                os.makedirs(os.path.join(save_path,cate))

            iou_test = 0
            idx = 1
            num_batch = int(len(io.testlist_cates[cate]) / batchsize) + 1
            angles_gt = []
            angles_pt = []
            st_gt = []
            st_pt = []
            iou_before_after_mask = []
            for _ in range(num_batch):
                x_op, y_op, m_op, e_op, st_op = io.fetch_evaluation(batchsize,train_phase=False,random_background=False,cates=[cate])
                # print(x_op.shape)
                x_op = data_transfer(x_op)
                m_op = data_transfer(m_op)
                v_before,v_after,iou_before,iou_after,iou_mask,masks,angles,sts,visual_hulls = sess.run([voxels_softmax_before[...,0],voxel_after_dic[cate],IoU_before,iou_after_dic[cate],IoU_mask,mask_softmax[...,0],angle_dic[cate],st_dic[cate],visual_hull_dic[cate]], feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op, st_vectors: st_op})
                iou_test += np.mean(iou_after)
                for i,v in enumerate(v_before):
                    iou_diff = iou_after[i] - iou_before[i]
                    input_img = Image.fromarray(np.array(y_op[i]*255).astype(np.uint8))
                    mask_gt = Image.fromarray(np.array(m_op[i,...,0]*255).astype(np.uint8))
                    mask_pt = Image.fromarray(np.array(masks[i]*255).astype(np.uint8))
                    input_img.save(os.path.join(save_path,cate,'%f_%d_input.jpg'%(iou_diff,idx)))
                    mask_gt.save(os.path.join(save_path,cate,'%f_%d_mask_gt.gif'%(iou_diff,idx)))
                    mask_pt.save(os.path.join(save_path,cate,'%f_%d_mask_pt.gif'%(iou_diff,idx)))
                    # sio.savemat(os.path.join(save_path,cate,'%f_%d_voxel_before.txt'%(iou_diff,idx)),mdict={'data':v_before[i]})
                    # sio.savemat(os.path.join(save_path,cate,'%f_%d_voxel_after.txt'%(iou_diff,idx)), mdict={'data':v_after[i]})
                    # sio.savemat(os.path.join(save_path,cate,'%f_%d_voxel_gt.txt'%(iou_diff,idx)), mdict={'data':x_op[i,...,0]})
                    with open(os.path.join(save_path,cate,'%f_%d_voxel_before.txt'%(iou_diff,idx)),'w') as fp:
                        for ___ in v_before[i]:
                            for __ in ___:
                                for _ in __:
                                    fp.write('%f\n'%_)

                    with open(os.path.join(save_path,cate,'%f_%d_voxel_after.txt'%(iou_diff,idx)),'w') as fp:
                        for ___ in v_after[i]:
                            for __ in ___:
                                for _ in __:
                                    fp.write('%f\n'%_)
                    with open(os.path.join(save_path,cate,'%f_%d_voxel_gt.txt'%(iou_diff,idx)),'w') as fp:
                        for ___ in x_op[i,...,0]:
                            for __ in ___:
                                for _ in __:
                                    fp.write('%f\n'%_)
                    with open(os.path.join(save_path,cate,'%f_%d_visual_hull.txt'%(iou_diff,idx)),'w') as fp:
                        for ___ in visual_hulls[i]:
                            for __ in ___:
                                for _ in __:
                                    fp.write('%f\n'%_)
                    angles_pt.append(angles[i])
                    angles_gt.append(e_op[i])
                    st_pt.append(sts[i])
                    st_gt.append(st_op[i])
                    iou_before_after_mask.append([iou_before[i],iou_after[i],iou_mask[i]])

                    idx += 1

            with open(os.path.join(save_path,cate+'_angles_st_iou_pt.txt'),'w') as fp:
                for i,a in enumerate(angles_pt):
                    for _ in a:
                        fp.write('%f '%_)
                    for _ in st_pt[i]:
                        fp.write('%f '%_)
                    for _ in iou_before_after_mask[i]:
                        fp.write('%f '%_)
                    fp.write('\n')

            with open(os.path.join(save_path,cate+'_angles_st_gt.txt'),'w') as fp:
                for i,a in enumerate(angles_gt):
                    for _ in a:
                        fp.write('%f '%_)
                    for _ in st_gt[i]:
                        fp.write('%f '%_)
                    fp.write('\n')

            iou_test /= num_batch
            xx = '%s %f' % (cate, iou_test)
            print(xx)

def test_forward():
    from dataIO_r2n2 import dataIO_r2n2
    cates = small_set
    data_path = '../dataset/ShapeNet/ShapeNetRendering'
    vox_path = '../dataset/ShapeNetVox32'
    bg_path = '../dataset/bg_crop'
    io = dataIO_r2n2(data_path, bg_path, vox_path, cates)
    batchsize = 24
    x_vectors = tf.placeholder(shape=[batchsize, voxel_size, voxel_size, voxel_size, 2],
                               dtype=tf.float32, name='all_Voxels')
    y_vectors = tf.placeholder(shape=[batchsize, img_w, img_h, 3], dtype=tf.float32, name='all_Images')
    m_vectors = tf.placeholder(shape=[batchsize, img_w, img_h, 2], dtype=tf.float32, name='all_Masks')
    e_vectors = tf.placeholder(shape=[batchsize, 3], dtype=tf.float32, name='all_Angles')
    st_vectors = tf.placeholder(shape=[batchsize, 3], dtype=tf.float32, name='all_scale_translation')

    with tf.variable_scope('voxel'):
        feature, feature_map, euler, st, shortcuts = encoder(y_vectors, reuse=False)
        voxels, voxels_softmax_before = decoder(feature, reuse=False)

    with tf.variable_scope("mask"):
        feature, feature_map, _, _, shortcuts = encoder(y_vectors, reuse=False)
        mask, mask_softmax = generator(feature_map, shortcuts, reuse=False)

    feature, feature_map, shortcuts = encoder_angle(y_vectors, reuse=False)

    IoU_before = tf.reduce_sum(
        tf.cast(tf.logical_and(voxels_softmax_before[:, :, :, :, 0] > thresh_hold,
                               x_vectors[:, :, :, :, 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
        tf.cast(tf.logical_or(voxels_softmax_before[:, :, :, :, 0] > thresh_hold,
                              x_vectors[:, :, :, :, 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2, 3])

    IoU_mask = tf.reduce_sum(
        tf.cast(tf.logical_and(mask_softmax[..., 0] > thresh_hold,
                               m_vectors[..., 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2]) / tf.reduce_sum(
        tf.cast(tf.logical_or(mask_softmax[..., 0] > thresh_hold,
                              m_vectors[..., 0] > thresh_hold),
                dtype=tf.float32), axis=[1, 2])

    iou_after_dic = {}
    voxel_after_dic = {}
    angle_dic = {}
    st_dic = {}
    visual_hull_dic = {}

    for i, cate in enumerate(cates):
        if i == 0:
            reuse = False
        else:
            reuse = True

        with tf.variable_scope("angles_trans", reuse=False):
            w_e_1 = tf.get_variable("w_euler_0_%s" % cate, shape=[1024, 512],
                                    initializer=tf.contrib.layers.xavier_initializer())
            e_1 = lrelu(tf.matmul(feature, w_e_1))
            w_e_2 = tf.get_variable("w_euler_1_%s" % cate, shape=[512, 3],
                                    initializer=tf.contrib.layers.xavier_initializer())
            euler = tf.matmul(e_1, w_e_2)

            w_st = tf.get_variable('w_ft_%s' % cate, shape=[1024, 3],
                                   initializer=tf.contrib.layers.xavier_initializer())
            st = tf.matmul(feature, w_st)

        rotation_matrices = get_rotation_matrix_r2n2(euler)
        mask_indexs = scale_trans_r2n2(st)
        projection = cast(mask_softmax[..., 0], mask_indexs, rotation_matrices=rotation_matrices)
        c1 = voxels_softmax_before[..., 0]
        c2 = projection
        c3 = c1 - c1 * c2
        c4 = c2 - c1 * c2

        feedin = tf.stack([c1, c2, c3, c4], axis=4)

        feature_vector, shortcuts = refine_encoder(feedin, reuse=reuse)
        voxels, voxels_softmax_after = refine_decoder(feature_vector, shortcuts, reuse=reuse)

        IoU_after = tf.reduce_sum(
            tf.cast(tf.logical_and(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                   x_vectors[:, :, :, :, 0] > thresh_hold),
                    dtype=tf.float32), axis=[1, 2, 3]) / tf.reduce_sum(
            tf.cast(tf.logical_or(voxels_softmax_after[:, :, :, :, 0] > thresh_hold,
                                  x_vectors[:, :, :, :, 0] > thresh_hold),
                    dtype=tf.float32), axis=[1, 2, 3])

        iou_after_dic[cate] = IoU_after
        angle_dic[cate] = euler
        st_dic[cate] = st
        voxel_after_dic[cate] = voxels_softmax_after[..., 0]
        visual_hull_dic[cate] = projection
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total = 0
        for i in range(10):
            x_op, y_op, m_op, e_op, st_op, s_op = io.fetch_evaluation(batchsize, train_phase=False,
                                                                      random_background=False,
                                                                      cates=[cates[0]])
            x_op = data_transfer(x_op)
            m_op = data_transfer(m_op)
            t = time.time()
            sess.run(voxel_after_dic[cates[0]],
                     feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op, st_vectors: st_op})
            xx = time.time() - t

        for i in range(100):
            x_op, y_op, m_op, e_op, st_op, s_op = io.fetch_evaluation(batchsize, train_phase=False, random_background=False,
                                                                      cates=[cates[0]])
            x_op = data_transfer(x_op)
            m_op = data_transfer(m_op)
            t = time.time()
            sess.run(voxel_after_dic[cates[0]],feed_dict={x_vectors: x_op, y_vectors: y_op, m_vectors: m_op, e_vectors: e_op, st_vectors: st_op})
            xx = time.time()-t
            total += xx
            print('time consume',time.time()-t)
        print('average time',total / 100)

def main():
    if len(sys.argv) < 3:
        # train_gen(add_regulizer=True)
        # pre_train(add_regulizer=True)
        train_end2end_noposebackprop_r2n2(add_regulizer=False)
        # train_end2end_noposebackprop_voc(add_regulizer=False)
    elif len(sys.argv) < 4:
        weights_path = sys.argv[2]
        train_end2end(weights_path, add_regulizer=True)
        #train_gen(weights_path, add_regulizer=True)
    else:
        weights_path = sys.argv[2]
        # test3(weights_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    # os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    main()
    # test_forward()
    # evaluation_random()
    # evaluation_transparent()
    # evaluation_voc()
    #debug()
    #test_angles()
    # see_weight()