import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper_funcs import print_shape

def VGG(inputs,no_of_out,scope):
	# with tf.name_scope(scope):
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
			activation_fn=tf.nn.relu,
			weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
			weights_regularizer=slim.l2_regularizer(0.0005)) and tf.variable_scope(scope):
		net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1')
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		net = slim.max_pool2d(net, [2, 2], scope='pool3')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		net = slim.max_pool2d(net, [2, 2], scope='pool5')
		net = slim.fully_connected(net, 4096, scope='fc6')
		net = slim.dropout(net, 0.5, scope='dropout6')
		net = slim.fully_connected(net, 4096, scope='fc7')
		net = slim.dropout(net, 0.5, scope='dropout7')
		net=slim.flatten(net, scope='flatten') #flatten just to get only four actions
		net = slim.fully_connected(net, no_of_out, activation_fn=None, scope='fc8')
		print_shape(net)
		return net
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

def FCN_one_hidden(inputs,no_of_hidden,no_of_out,scope):
	with slim.arg_scope([slim.fully_connected],
				activation_fn=tf.nn.relu,
				weights_initializer=tf.truncated_normal_initializer(0.0,0.01),
				weights_regularizer=slim.l2_regularizer(0.0005)) and tf.variable_scope(scope):
		net=slim.fully_connected(inputs,no_of_hidden,scope='hidden1')
		net=slim.fully_connected(net,no_of_out,scope='out')
		return net
