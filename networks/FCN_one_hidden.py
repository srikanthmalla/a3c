import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper_funcs import print_shape

def FCN_one_hidden(inputs,no_of_hidden,no_of_out,scope):
    with slim.arg_scope([slim.fully_connected],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(0.0,0.01),
                weights_regularizer=slim.l2_regularizer(0.0005)) and tf.variable_scope(scope):
        net=slim.fully_connected(inputs,no_of_hidden,scope='hidden1')
        net=slim.fully_connected(net,no_of_out,activation_fn=None,scope='out')
        return net
