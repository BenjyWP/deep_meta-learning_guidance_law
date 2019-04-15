"""

Chen Liang, Beihang University
Code accompanying the paper
"Learing to guide: Guidance Law Based on Deep Meta-learning and Model Predictive Path Integral Control"


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf


class model_dense():
    def __init__(self, states, actions,dense_size):#, target_size):

        dimOut = 7#states.get_shape()[1].value
        reshape = tf.concat([states, actions], 1)

        fc1 = tf.layers.dense(reshape, dense_size, tf.nn.relu)
        fc2 = tf.layers.dense(fc1, dense_size, tf.nn.relu)
        fc3 = tf.layers.dense(fc2, dense_size, tf.nn.relu)
        self.outputs = tf.layers.dense(fc3, dimOut, None)

    def meta_train(self, deltas, fc2, learning_rate):
        deltas = tf.cast(deltas, tf.float32)
        loss_function = tf.abs(deltas - fc2, name='loss_per_example')   # /FLAGS.batch_size
        loss_mean = tf.reduce_mean(loss_function, name='loss')
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_mean)

        return train_op, loss_mean





class dense_dynamics_model():
    def __init__(self,
                 env
                 ):

        self.states = tf.placeholder(shape = [None, 10], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None, 2], dtype = tf.float32)
        self.deltas = tf.placeholder(shape = [None, 7], dtype = tf.float32)

        self.meanstddata = np.load('npys/meanstdobs.npy')
        self.meanstddelta = np.load('npys/meanstddlt.npy')

        self.meanstddelta = np.hstack((self.meanstddelta[:,0:5],self.meanstddelta[:,8:10]))


        self.model = model_dense(self.states, self.actions, 512)
        self.train_op, self.loss = self.model.meta_train(self.deltas, self.model.outputs, 0.001)

        gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction = 0.15)
        config = tf.ConfigProto(gpu_options = gpu_opt)

        self.sess = tf.Session(config = config)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        self.ckpt = tf.train.get_checkpoint_state('model')
        if self.ckpt and self.ckpt.model_checkpoint_path:
            # Restores from checkpoint
            tf.reset_default_graph()
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            global_step = self.ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]


    def restore_default(self):
        tf.reset_default_graph()
        self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        global_step = self.ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

    def predict(self, states, actions):

        if len(states.shape) == 1:
            states = states.reshape((1, states.shape[0]))
        if len(actions.shape) == 1:
            actions = actions.reshape((1, actions.shape[0]))

        statesNorm = (states - self.meanstddata[0,:]) / (self.meanstddata[1,:] + 1e-7)

        next_observationsNorm = self.sess.run(self.model.outputs, feed_dict={self.states: statesNorm,
                                                                            self.actions: actions})
        next_observations = next_observationsNorm * self.meanstddelta[1, :] + self.meanstddelta[0, :]

        return next_observations

    def fit(self,states,actions,deltas, next):
        #in_state = np.hstack((states,actions))
        statesNorm = (states - self.meanstddata[0, :]) / (self.meanstddata[1, :] + 1e-7)
        nextNorm = (next - self.meanstddata[0, :]) / (self.meanstddata[1, :] + 1e-7)
        # The 1.732 is for denormalizing the action since the initial
        # distribution for sampled action is from a uniform distribution in [-1,1]
        actionsNorm = (actions * 1.732)
        deltasNorm = (deltas - self.meanstddelta[0, :]) / (self.meanstddelta[1, :] + 1e-7)

        feed = {self.states: statesNorm, self.actions: actionsNorm, self.deltas: deltasNorm}
        loss, _ = self.sess.run([self.loss, self.train_op],feed)
        return loss


