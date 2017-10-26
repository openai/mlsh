import rl_algs.common.tf_util as U
import tensorflow as tf
import numpy as np
import gym


class Features(object):
    def __init__(self, name, ob):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name

            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=(ob.get_shape()[1],))
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            x = tf.nn.relu(U.conv2d(obz, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 16, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 64, 'lin', U.normc_initializer(1.0)))

            self.ob = x

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
