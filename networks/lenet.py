import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper_funcs import print_shape

def lenet(inputs,no_of_hidden,no_of_out,scope):
    layers=[]
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
            weights_regularizer=slim.l2_regularizer(0.0005)) and tf.variable_scope(scope):
        net = slim.conv2d(inputs, 20, [5,5], scope='conv1')
        layers.append(net)
        net = slim.max_pool2d(net, [2,2], scope='pool1')
        layers.append(net)
        net = slim.conv2d(net, 50, [5,5], scope='conv2')
        layers.append(net)
        net = slim.max_pool2d(net, [2,2], scope='pool2')
        layers.append(net)
        net = slim.flatten(net, scope='flatten3')
        # layers.append(net)
        net = slim.fully_connected(net, 500, scope='fc4')
        # layers.append(net)
        net = slim.fully_connected(net, no_of_out, activation_fn=None, scope='fc5')
        return net, layers

