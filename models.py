import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper_funcs import *
from params import *
import gym, time, threading, random
from gym import wrappers
import numpy as np
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
		actions=self.VGG_modified(inputs,4,'actor/')
		return actions
	
	def critic(self,inputs):
		value=self.VGG_modified(inputs,1,'value/')
		return value

	def VGG_modified(self,inputs,no_of_out,scope):
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
# ---------------------------
model=a3c()
class Environment(threading.Thread):
	stop_signal = False
	def __init__(self,render=False,eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
		threading.Thread.__init__(self)
		self.env = gym.make(environment)
		self.render=render
		self.episode=1
		self.R=0
		self.agent = Agent(eps_start, eps_end, eps_steps)
		if create_video:	
			self.env = wrappers.Monitor(self.env, './tmp',force=True)
		print('observation space:',self.env.observation_space)
		print('action space:',self.env.action_space)
		print('action details',self.env.unwrapped.get_action_meanings())
	def run_episode(self):
		observation = self.env.reset()
		t = 0
		while True:
			if (THREADS==1)&(self.render):
				self.env.render()#show the game, xlib error with multiple threads
			# action = self.env.action_space.sample() # your agent here (this takes random actions)
			# observation=np.expand_dims(observation,axis=0)
			action=self.agent.act(observation)
			observation_new, reward, done, info = self.env.step(action)
			t=t+1
			self.R+=reward
			if done:
				observation_new_=None
			self.agent.train(observation, action, reward, observation_new)
			observation=observation_new
			if done:
				print(threading.current_thread().name," episode:",self.episode, " reward:",self.R," took {} steps".format(t))
				# self.agent.memory=[]
				self.R=0
				t=0
				break
	def run(self):
		while not self.stop_signal:
			self.run_episode()
			self.episode+=1
			if self.episode>max_no_episodes:
				self.stop()
			if self.episode%ckpt_episode==0:
				model.save(self.episode)
				print("saved model at episode {}".format(self.episode))

	def stop(self):
		self.stop_signal = True
	# def rollout_reward(self):
# ---------------------------
class Optimizer(threading.Thread):
	stop_signal = False

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while not self.stop_signal:
			model.optimize()

	def stop(self):
		self.stop_signal = True

# ---------------------------
frames=0
class Agent:
	def __init__(self, eps_start, eps_end, eps_steps):
		self.eps_start = eps_start
		self.eps_end   = eps_end
		self.eps_steps = eps_steps

		self.memory = []	# used for n_step return
		self.R = 0.

	def getEpsilon(self):
		if(frames >= self.eps_steps):
			return self.eps_end
		else:
			return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

	def act(self, s):
		eps = self.getEpsilon()			
		global frames; frames = frames + 1

		if random.random() < eps:
			return random.randint(0, NUM_ACTIONS-1)

		else:
			s = np.array([s])
			p=model.sess.run(model.p,feed_dict={model.observation:s})
			a = np.random.choice(NUM_ACTIONS, p=p[0])
			return a
	
	def train(self, s, a, r, s_):
		def get_sample(memory, n):
			s, a, _, _  = memory[0]
			_, _, _, s_ = memory[n-1]

			return s, a, self.R, s_

		a_cats = np.zeros(NUM_ACTIONS)	# turn action into one-hot representation
		a_cats[a] = 1 
		self.memory.append( ([s], a_cats, r, [s_]) )

		self.R = ( self.R + r * GAMMA_N ) / GAMMA

		if s_ is None:
			while len(self.memory) > 0:
				n = len(self.memory)
				s, a, r, s_ = get_sample(self.memory, n)
				print(np.shape(a))
				model.train_push(s, a, r, s_)

				self.R = ( self.R - self.memory[0][2] ) / GAMMA
				self.memory.pop(0)		

			self.R = 0

		if len(self.memory) >= N_STEP_RETURN:
			s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
			# a=np.reshape(a,[1,4])
			model.train_push(s, a, r, s_)

			self.R = self.R - self.memory[0][2]
			self.memory.pop(0)		