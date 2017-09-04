import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper_funcs import *
from params import *
import gym, time, threading
from gym import wrappers
import numpy as np

class a3c:
	def __init__(self):
		# <s,a,r> information collection by acting in environment
		self.observation=tf.placeholder(tf.float32, shape=input_shape)
		self.reward= tf.placeholder(tf.float32,shape=[None,1]) #not immediate but n step discounted
		self.a_t=tf.placeholder(tf.float32,shape=[None,4])
		
		# act in environment and critisize the actions
		self.p= tf.nn.softmax(actor(self.observation), name='action_probability')#probabilities of action predicted
		self.v= critic(self.observation) #value predicted
				
		#session and initialization
		self.sess=tf.Session()
		self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		
		#logging data to tensorboard
		self.writer = tf.summary.FileWriter(tf_logdir, self.sess.graph)
		self.log_reward=tf.summary.scalar("reward", self.reward)
		self.summary=tf.summary.merge_all()

		#advantage and losses
		self.logp = tf.log(tf.reduce_sum(self.p*self.a_t, axis=1, keep_dims=True) + 1e-10)
		self.advantage= self.reward - self.v
		self.loss_policy = - self.logp * tf.stop_gradient(self.advantage)
		self.loss_value  = LOSS_V * tf.square(self.advantage)												# minimize value error
		self.entropy = LOSS_ENTROPY * tf.reduce_sum(self.p * tf.log(self.p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)
		self.loss_total = tf.reduce_mean(self.loss_policy + self.loss_value + self.entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_total)
class Environment(threading.Thread):
	stop_signal = False
	def __init__(self):
		threading.Thread.__init__(self)
		self.env = gym.make(environment)
		self.model=a3c()
		self.episode=1
		self.memory=[]
		self.R=0
		if create_video:	
			self.env = wrappers.Monitor(self.env, './tmp',force=True)
		print('observation space:',self.env.observation_space)
		print('action space:',self.env.action_space)
		print('action details',self.env.unwrapped.get_action_meanings())
	def run_episode(self):
		with self.model.sess.as_default() as sess:
			sess.run(model.init_op)
			observation = self.env.reset()
			t = 0
			while True:
				if (threads==1):
					self.env.render()#show the game, xlib error with multiple threads
				action = self.env.action_space.sample() # your agent here (this takes random actions)
				observation=np.expand_dims(observation,axis=0)
				observation_new, reward, done, info = self.env.step(action)
				t=t+1
				self.R+=reward
				self.memory.append((observation,action,observation_new,reward))
				observation=observation_new
				if done:
					print(threading.current_thread().name," episode:",self.episode, " reward:",self.R," took {} steps".format(t))
					self.memory=[]
					break
	def run(self):
		while not self.stop_signal:
			self.R=0
			self.run_episode()
			self.episode+=1
			if self.episode>max_no_episodes:
				self.stop()
	def stop(self):
		self.stop_signal = True
	# def rollout_reward(self):

def actor(inputs): #modified vgg net
	actions=VGG_modified(inputs,4,'actor/')
	return actions
def critic(inputs):
	value=VGG_modified(inputs,4,'value/')
	return value

def VGG_modified(inputs,no_of_out,scope):
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