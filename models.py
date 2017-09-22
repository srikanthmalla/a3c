import tensorflow as tf
from params import *
import gym, time, threading, random
from gym import wrappers
import numpy as np
from networks import VGG
# ---------------------------
class a3c:
	train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()	
	def __init__(self):
		# <s,a,r,s_> information collection by acting in environment
		self.observation=tf.placeholder(tf.float32, shape=input_shape)
		self.reward= tf.placeholder(tf.float32,shape=[None,1]) #not immediate but n step discounted
		self.a_t=tf.placeholder(tf.float32,shape=[None,4])
		self.s_=tf.placeholder(tf.float32, shape=input_shape)

		# act in environment and critisize the actions
		self.p= tf.nn.softmax(self.actor(self.observation), name='action_probability')#probabilities of action predicted
		self.v= self.critic(self.observation) #value predicted

		#saver of the model
		self.saver = tf.train.Saver()

		#advantage and losses
		self.logp = tf.log(tf.reduce_sum(self.p*self.a_t, axis=1, keep_dims=True) + 1e-10)
		self.advantage= self.reward - self.v
		self.loss_policy = - self.logp * tf.stop_gradient(self.advantage)
		self.loss_value  = LOSS_V * tf.square(self.advantage)												# minimize value error
		self.entropy = LOSS_ENTROPY * tf.reduce_sum(self.p * tf.log(self.p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)
		self.loss_total = tf.reduce_mean(self.loss_policy + self.loss_value + self.entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_total)

		#session and initialization
		self.sess=tf.Session()
		self.writer = tf.summary.FileWriter(tf_logdir, self.sess.graph)
		self.log_reward=tf.summary.scalar("reward", self.reward)
		self.summary=tf.summary.merge_all()

		self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(self.init_op)
		self.default_graph = tf.get_default_graph()
		self.default_graph.finalize() # avoid modifications
	def save_model(self,step):
		self.saver.save(self.sess, './tmp/model.ckpt', global_step=step)

	def predict_v(self,observation):
		v=self.sess.run(self.v,feed_dict={self.observation:observation})
		return v
	def train_push(self, s, a, r, s_):
		with self.lock_queue:
			self.train_queue[0].append(s)
			self.train_queue[1].append(a)
			self.train_queue[2].append(r)
			if s_ is None:
				self.train_queue[3].append(NONE_STATE)
				self.train_queue[4].append(0.)
			else:	
				self.train_queue[3].append(s_)
				self.train_queue[4].append(1.)
	def optimize(self):
		if len(self.train_queue[0]) < MIN_BATCH:
			time.sleep(0)	# yield
			return

		with self.lock_queue:
			if len(self.train_queue[0]) < MIN_BATCH:	# more thread could have passed without lock
				return 									# we can't yield inside lock

			s, a, r, s_, s_mask = self.train_queue
			self.train_queue = [ [], [], [], [], [] ]

		s = np.vstack(s)
		a = np.vstack(a)
		r = np.vstack(r)
		s_ = np.vstack(s_)
		s_mask = np.vstack(s_mask)

		if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

		v = self.predict_v(s_)
		r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state
		self.sess.run(self.optimizer, feed_dict={self.observation: s, self.a_t: a, self.reward: r})

	def actor(self,inputs): #modified vgg net
		actions=VGG(inputs,4,'actor/')
		return actions
	
	def critic(self,inputs):
		value=VGG(inputs,1,'value/')
		return value

