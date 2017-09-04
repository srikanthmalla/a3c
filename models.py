import tensorflow as tf
import tensorflow.contrib.slim as slim
from params import input_shape
from helper_funcs import *
from params import tf_logdir

class a3c:
	def __init__(self):
		self.observation=tf.placeholder(tf.float32, shape=input_shape)
		self.p= tf.nn.softmax(actor(self.observation), name='action_probability')
		# self.v= critic(self.observation)
		self.reward= tf.placeholder(tf.float32,shape=[1,1])
		# self.logp = tf.log(tf.reduce_sum(self.p*a_t, axis=1, keep_dims=True) + 1e-10)
		# self.advantage= self.reward - critic(state)
		# self.loss = - self.logp * tf.stop_gradient(self.advantage)
		self.sess=tf.Session()
		self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.writer = tf.summary.FileWriter(tf_logdir, self.sess.graph)
		self.log_reward=tf.summary.scalar("reward", self.reward)
		self.summary=tf.summary.merge_all()

	def train(self):
		self.error=0
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.error)

def actor(inputs): #modified vgg net
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
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
	    net = slim.fully_connected(net, 4, activation_fn=None, scope='fc8')
	    print_shape(net)
	    return net
def critic(inputs):
	#modified vgg net
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
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
	    net = slim.fully_connected(net, 4, activation_fn=None, scope='fc8')
	    print_shape(net)
	    return net