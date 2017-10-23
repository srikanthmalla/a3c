import tensorflow as tf
from params import *
import gym, time, threading, random
from gym import wrappers
import numpy as np
import os.path
from networks import *
if use_net=='fc_1':
	net=FCN_one_hidden
elif use_net=='lenet':
	net=lenet
elif use_net=='VGG':
	net=VGG
	
class a2c():
	def __init__(self):
		# <s,a,R,s_> information collection by acting in environment
		self.observation=tf.placeholder(tf.float32, shape=input_shape)
		self.R= tf.placeholder(tf.float32,shape=[None,1]) #not immediate but n step discounted
		self.a_t=tf.placeholder(tf.float32,shape=[None,no_of_actions]) #which action was taken 
		# self.total_reward=tf.placeholder(tf.float32,shape=[None,1])
		# act in environment and critisize the actions
		self.p= tf.nn.softmax(self.actor(self.observation), name='action_probability')#probabilities of action predicted
		self.V= self.critic(self.observation) #value predicted

		#saver 
		self.saver = tf.train.Saver()

		#advantage and losses
		self.exec_prob=self.p*self.a_t
		self.logp = tf.log(tf.reduce_sum(self.exec_prob, axis=1, keep_dims=True) + 1e-10)
		self.advantage= self.R - self.V
		self.loss_policy = - tf.reduce_sum(self.logp * tf.stop_gradient(self.advantage))
		self.loss_value  = LOSS_V * tf.nn.l2_loss(self.advantage)				# minimize value error
		#will try entropy loss afterwards
		self.entropy =  tf.reduce_sum(self.p * tf.log(self.p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)
		#self.loss_total = tf.reduce_mean(self.loss_policy + self.loss_value + self.entropy)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(self.loss_policy+self.entropy)
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.loss_value)
		
		#session and initialization
		self.sess=tf.Session()
				
		self.writer = tf.summary.FileWriter(tf_logdir, self.sess.graph)
		# self.log_reward=tf.summary.scalar(self.id+"totalreward", tf.reduce_sum(self.total_reward))
		# self.log_policyloss=tf.summary.scalar("actor_loss",self.loss_policy)
		# self.log_criticloss=tf.summary.scalar("critic_loss",self.loss_value)
		#self.summary=tf.summary.merge_all()

		self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(self.init_op)
		self.default_graph = tf.get_default_graph()
		self.default_graph.finalize() # avoid modifications
	
	def save(self,step):
		self.saver.save(self.sess, ckpt_dir)
		
	def load(self):
		if os.path.isfile(ckpt_dir):
			self.saver.restore(self.sess,ckpt_dir)
			print("model restored")
		else: 
			print("nothing to load")

	def predict_value(self,observation):
		v=self.sess.run(self.V,feed_dict={self.observation:observation})
		return v

	def predict_action_prob(self,observation):
		a=self.sess.run(self.p,feed_dict={self.observation:observation})
		return a

	def actor(self,inputs): #modified vgg net
		actions=net(inputs,10,no_of_actions,'actor/')
		return actions
	
	def critic(self,inputs):
		value=net(inputs,10,1,'value/')
		return value
		
	def train_actor(self, observations, actions, R):
		[_, loss_policy]=self.sess.run([self.actor_optimizer, self.loss_policy], feed_dict={self.observation:observations, self.a_t:actions, self.R:R})	
# 		policyloss= tf.Summary(value=[tf.Summary.Value(tag=tag+'/actorloss',
# simple_value=np.float32(loss_policy))])
# 		self.writer.add_summary(policyloss, step)

	def train_critic(self, observations, R):      
		[_, loss_value]=self.sess.run([self.critic_optimizer, self.loss_value], feed_dict={self.observation:observations, self.R:R})
# 		criticloss= tf.Summary(value=[tf.Summary.Value(tag=tag+'/criticloss',
# simple_value=np.float32(loss_value))])
# 		self.writer.add_summary(criticloss, step)

	#useful to log other details like features, rewards
	def log_details(self, total_reward, observations, actions, R, step,tag):
		tot_r = tf.Summary(value=[tf.Summary.Value(tag=tag+'/reward',
simple_value=total_reward)])
		self.writer.add_summary(tot_r, step) 
		[_, loss_policy]=self.sess.run([self.actor_optimizer, self.loss_policy], feed_dict={self.observation:observations, self.a_t:actions, self.R:R})	
		policyloss= tf.Summary(value=[tf.Summary.Value(tag=tag+'/actorloss',
simple_value=np.float32(loss_policy))])
		self.writer.add_summary(policyloss, step)
		[_, loss_value]=self.sess.run([self.critic_optimizer, self.loss_value], feed_dict={self.observation:observations, self.R:R})
		criticloss= tf.Summary(value=[tf.Summary.Value(tag=tag+'/criticloss',
simple_value=np.float32(loss_value))])
		self.writer.add_summary(criticloss, step)


class a3c(a2c):
	def __init__(self):
		a2c.__init__(self)
		self.threads = THREADS

class trpo():
	pass

		
		

