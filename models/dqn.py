import tensorflow as tf
import os
import numpy as np
import shutil
from params import lr,net
class dqn():
	def __init__(self):
		self.observation=tf.placeholder(tf.float32,shape=[None,4])
		self.action=tf.placeholder(tf.float32,shape=[None,1])
		self.inputs=tf.concat([self.observation,self.action],axis=1)
		self.qpred=net(self.inputs,10,1,'dqn/')
		self.q=tf.placeholder(tf.float32,shape=[None,1])

		dir_="./tmp/cart_pole/dqn/"
		self.ckpt_dir=dir_+"model.ckpt"
		if not os.path.exists(dir_):
			os.makedirs(dir_)

		self.saver=tf.train.Saver()
		self.sess=tf.Session()

		#losses
		self.loss=tf.nn.l2_loss(self.q-self.qpred)
		self.optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

		tf_logdir="./graphs/dqn/"
		if os.path.exists(tf_logdir):
			shutil.rmtree(tf_logdir)

		self.writer=tf.summary.FileWriter(tf_logdir,self.sess.graph)
		#initializations$
		self.init_op=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(self.init_op)
		self.default_graph = tf.get_default_graph()
		self.default_graph.finalize()

	def save(self, step):
		self.saver.save(self.sess, self.ckpt_dir)

	def load(self):
		self.saver.restore(self.sess,self.ckpt_dir)

	def predict_Q(self, observation,action):
		return self.sess.run(self.qpred,feed_dict={self.observation:observation,self.action:action})

	def train(self,observation,action,qval,step):
		# action=np.expand_dims(action,axis=1)
		# print(np.shape(observation),np.shape(action),np.shape(qval))
		[_,loss]=self.sess.run([self.optimizer,self.loss],feed_dict={self.observation:observation, self.action:action, self.q:qval})

	def test(self,observation,action,qval):
		# action=np.expand_dims(action,axis=1)
		[predq,val_loss]=self.sess.run([self.qpred,self.loss],feed_dict={self.observation:observation,self.action:action,self.q:qval}) 
		print("current:",observation,"action:",action,"predicted qval:",predq,"qval:",qval,"loss:",val_loss,"\n")

	def log_details(self, total_reward, observations, actions, qval, step,tag):	
		# print(np.shape(observations),np.shape(actions),np.shape(qval))
		tot_r = tf.Summary(value=[tf.Summary.Value(tag=tag+'/reward',
simple_value=total_reward)])
		self.writer.add_summary(tot_r, step) 
		loss=self.sess.run(self.loss, feed_dict={self.observation:observations, self.action:actions, self.q:qval}) 
		lossV= tf.Summary(value=[tf.Summary.Value(tag=tag+'/loss',
simple_value=float(loss))])
		self.writer.add_summary(lossV, step)