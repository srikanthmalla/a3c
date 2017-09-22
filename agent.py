from models import a3c
import threading, time, random
import numpy as np
from params import *
import gym
from gym import wrappers

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
