from models import a2c
import numpy as np
from params import *
from helper_funcs import onehot
from gym import wrappers

class RL_Agent():
	def __init__(self):
		self.env = Environment
		self.render=render
		if use_model=='a2c': 
			self.model=a2c()
		self.episode=1
		self.total_reward=0
		self.episode_over()
		self.EPS=eps_start
		if create_video:	
			self.env = wrappers.Monitor(self.env, './tmp',force=True)
		print('observations shape:',observation_shape)
		#print('action details', action_details)
		print('no of actions:', no_of_actions)
	def run_episode(self):
		observation = self.env.reset()
		t = 0   #to calculate time steps
		while True:
			if (self.render):
				self.env.render()#show the game, xlib error with multiple threads
			# action = self.env.action_space.sample() # your agent here (this takes random actions)
			action_prob=self.model.predict_action_prob([observation])
			action=self.predict_action(action_prob)
			self.observations.append(observation)
			self.actions.append(onehot(action,no_of_actions))
			observation_new, reward, done, info = self.env.step(action)
			self.r.append(reward)
			t=t+1
			self.total_reward+=reward
			observation=observation_new
			if done:
				print(" episode:",self.episode, " reward:",self.total_reward," took {} steps".format(t))
				self.model.log_details(np.array([[self.total_reward]]),self.episode)
				self.total_reward=0
				break
			#TODO:makesure that memory dont overflow by stopping for n steps
			#if (t>max_no_steps):
				#self.R=critic   #for terminal state give R as 0
	def run(self):
		while self.episode<max_no_episodes:
			self.run_episode()
			self.bellman_update()
			self.model.train_actor(self.observations,self.actions,self.R,self.episode)
			self.model.train_critic(self.observations,self.R,self.episode)
			self.EPS-=d_eps
			self.episode+=1
			self.episode_over()
			if self.episode%ckpt_episode==0:
				self.model.save(self.episode)
				print("saved model at episode {}".format(self.episode))
	def episode_over(self):
		self.observations=[]
		self.actions=[]
		self.R=[]
		self.r=[]
		self.R_terminal=0

	def bellman_update(self):
		for i in range(len(self.r),0,-1):
			t=self.r[i-1]+GAMMA*self.R_terminal
			self.R.append([t])
			self.R_terminal=t
	def predict_action(self,prob):
		#here we use epsilon greedy exploration by tossing a coin
		action=np.amax(prob)
		if np.random.uniform > self.EPS:
			action=self.env.action_space.sample()
		return action
