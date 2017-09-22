import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper_funcs import print_shape

def VGG(inputs,no_of_out,scope):
	# with tf.name_scope(scope):
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
			activation_fn=tf.nn.relu,
			weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
			weights_regularizer=slim.l2_regularizer(0.0005)):
		net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope=scope+'conv1')
		net = slim.max_pool2d(net, [2, 2], scope=scope+'pool1')
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope=scope+'conv2')
		net = slim.max_pool2d(net, [2, 2], scope=scope+'pool2')
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope=scope+'conv3')
		net = slim.max_pool2d(net, [2, 2], scope=scope+'pool3')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope=scope+'conv4')
		net = slim.max_pool2d(net, [2, 2], scope=scope+'pool4')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope=scope+'conv5')
		net = slim.max_pool2d(net, [2, 2], scope=scope+'pool5')
		net = slim.fully_connected(net, 4096, scope=scope+'fc6')
		net = slim.dropout(net, 0.5, scope=scope+'dropout6')
		net = slim.fully_connected(net, 4096, scope=scope+'fc7')
		net = slim.dropout(net, 0.5, scope=scope+'dropout7')
		net=slim.flatten(net, scope=scope+'flatten') #flatten just to get only four actions
		net = slim.fully_connected(net, no_of_out, activation_fn=None, scope=scope+'fc8')
		print_shape(net)
		return net
