from models import *
import numpy as np
import time
from params import *
from helper_funcs import *
from gym import wrappers
from threading import Thread
from gym.envs.classic_control import rendering
#to get same random for rerun
np.random.seed(1234)
if (use_model=='a2c') or (use_model=='human'):
	model = a2c()
elif use_model=='a3c':
	model = a3c()
else:
	model = trpo()

class a2c_agent():
	def __init__(self):
		self.env=gym.make(env_name)
		self.viewer = rendering.SimpleImageViewer()
		self.render=render
		self.episode=1
		self.total_reward=0
		self.clear()
		self.EPS=eps_start
		if create_video:	
			self.env = wrappers.Monitor(self.env, dir_,force=True)
		print('observations shape:',observation_shape)
		#print('action details', action_details)
		print('no of actions:', no_of_actions)
	def getName(self):
			return self.__class__.__name__
	def run_episode(self):
		observation = self.env.reset()
		t = 0   #to calculate time steps
		while True:
			if (self.render):
				rgb=self.env.render('rgb_array')#show the game, xlib error with multiple threads
				# print(np.shape(rgb))
				upscaled= np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1)
				self.viewer.imshow(upscaled)
				if (mode=='test') or (use_model=='human'):
					time.sleep(0.2)
			if use_model =='human':
				self.record()
				action=self.action
				print("step:",t,end="\r")
				# self.action=self.env.action_space.sample()
				# print('action:',self.action)
			else:
				action_prob=model.predict_action_prob([observation])
				action=self.predict_action(action_prob,t)
			self.observations.append(observation)
			self.actions.append(onehot(action,no_of_actions))
			observation_new, reward, done, info = self.env.step(action)
			self.total_reward+=reward	
			self.r.append(reward)	
			t=t+1
			if done:
				self.R_terminal=0
				self.bellman_update() #can be used for batch
				print(" episode:",self.episode,"eps:","{0:.2f}".format(self.EPS), " reward:",self.total_reward," took {} steps".format(t))
				self.R=np.reshape(self.R,(np.shape(self.R)[0],1))
				model.log_details(self.total_reward,self.observations,self.actions,self.R,self.episode,self.getName())
				self.train()
				self.total_reward=0
				break
			else:
				if (t%10 == 0):
					self.R_terminal=model.predict_value([observation_new])
					self.bellman_update()
					self.train()
				observation=observation_new

	def run(self):
		start = time.time()
		while self.episode<max_no_episodes:
			self.run_episode()
			# self.train()#for a batch of complete episode
			self.EPS-=d_eps
			self.episode+=1
			if self.episode%ckpt_episode==0:
				model.save(self.episode)
				print("saved model at episode {}".format(self.episode))
		end = time.time()
		print("took:",end - start)

	def train(self):
		self.R=np.reshape(self.R,[np.shape(self.R)[0],1])
		model.train_actor(self.observations,self.actions,self.R)
		model.train_critic(self.observations,self.R)
		self.clear()

	def clear(self):
		self.observations=[]
		self.actions=[]
		self.R=[]
		self.r=[]
		self.R_terminal=0

	def bellman_update(self):
		self.R=[]
		for i in range(len(self.r),0,-1):
			t=self.r[i-1]+self.R_terminal
			self.R.append([t])
			# self.R_terminal=t #normal bellman update
			self.R_terminal=model.predict_value([self.observations[i-1]]) #batch bootstrap
		self.R=np.flip(self.R,axis=0)	
	
	def predict_action(self,prob,t):
		#here we use epsilon greedy exploration by tossing a coin
		action=np.argmax(prob)
		if np.random.uniform() < self.EPS:
			action=self.env.action_space.sample()
		return action
	def test(self):
		model.load()
		self.render=True
		self.run_episode()
		print("done..")
class a3c_agent(a2c_agent,Thread):
	def __init__(self):
		a2c_agent.__init__(self)
		Thread.__init__(self)

import keyboard
#this human agent mapping is for breakout
class human_agent(a2c_agent):
	def __init__(self):
		a2c_agent.__init__(self)
		self.render=True
		self.action=0
	def record(self):
		keyboard.start_recording()
		time.sleep(0.2)
		recorded=keyboard.stop_recording()
		try:
			action=recorded[0].name
		except IndexError:
			action=None

		if action=='right':
			self.action=2
		elif action=='left':
			self.action=3
		elif action=='up':
			self.action=1
		else: 
			self.action=0
	def run(self):
		start = time.time()
		while self.episode<max_no_episodes:
			self.run_episode()
			print('h')
			self.EPS-=d_eps
			self.episode+=1
			if self.episode%ckpt_episode==0:
				model.save(self.episode)
				print("saved model at episode {}".format(self.episode))
		end = time.time()
		print("took:",end - start)
	
class trpo_agent():
	pass
